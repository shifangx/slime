"""Nemotron 3.5 Super VL with a frozen HF RADIO and trainable projector/LLM.

Reuses Shifang\'s provider from shifangx/slime@328760de. Vision/projector are
checkpoint-owned Transformers modules; the language tower uses MCore\'s hybrid
stack. Image features are injected before sequence-parallel scatter.
"""

from __future__ import annotations

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from transformers import AutoConfig

from slime.utils import accelerator

from .nemotron_h import get_nemotron_h_spec

# Written for Qwen3.5-VL but neither function mentions it: both are pure
# arithmetic over `cu_seqlens` and a context-parallel group. They are imported
# rather than copied so a fix to the packing arithmetic reaches both models.
from .qwen3_5_vl_utils import gather_packed_input_ids, get_packed_cp_local_indices

__all__ = [
    "NemotronOmniVLModel",
    "get_nemotron_35_super_vl_model_provider",
    "project_images",
]


def project_images(pixel_values, vision_model, vision_projector) -> torch.Tensor:
    """Run the frozen tower plus the projector, and return ``(tokens, hidden)``.

    One call covers the whole vision stack, because
    ``NemotronH_Omni_Reasoning_V3VisionProjector.forward`` does: tower -> drop
    the 10 class/register tokens (inside ``RadioModel``, via
    ``num_summary_tokens``) -> ``vision_final_layernorm`` -> 2x2 pixel shuffle
    (1280 -> 5120, N -> N/4) -> ``mlp1`` (5120 -> 20480 -> 4096, squared ReLU).

    WHY THIS DOES NOT JUST HAND THE LIST TO THE PROJECTOR
    ----------------------------------------------------
    It has a list branch -- ``torch.cat([self.forward(pv, ...) for pv in ...])``
    -- but that branch cannot serve this processor. Two things are wrong with it
    here, and both come from dynamic resolution:

    * the elements the image processor puts in the list are ``(3, H, W)``, and
      ``forward`` immediately does ``_, _, H, W = pixel_values.shape``, which
      needs four dimensions;
    * even given four, each image projects to ``(1, tokens_i, hidden)`` with a
      *different* ``tokens_i``, and ``torch.cat(..., dim=0)`` on those raises.

    So the list is iterated here instead, each image is projected on its own and
    flattened, and the results concatenate on the token axis. That is also the
    shape the placeholder scatter needs: one row per ``<image>`` token, in image
    order. A stacked ``(N, 3, H, W)`` tensor -- what the processor returns when
    every image in the sample happens to be the same size -- takes the same path
    through the projector's own batch dimension and flattens to the same thing.

    The tower is frozen, but the projector is not, so this is deliberately not
    wrapped in ``torch.no_grad()``: the gradient has to reach ``mlp1``.
    """

    def flatten(features: torch.Tensor) -> torch.Tensor:
        # (images, tokens, hidden) -> (images * tokens, hidden)
        return features.reshape(-1, features.shape[-1]) if features.dim() == 3 else features

    parameter = next(vision_model.parameters(), None) if isinstance(vision_model, torch.nn.Module) else None

    def prepare(image):
        if parameter is not None:
            image = image.to(device=parameter.device, dtype=parameter.dtype)
        return image if image.dim() == 4 else image.unsqueeze(0)

    if isinstance(pixel_values, (list, tuple)):
        return torch.cat(
            [flatten(vision_projector(prepare(image), vision_model)) for image in pixel_values],
            dim=0,
        )

    return flatten(vision_projector(prepare(pixel_values), vision_model))


def _load_remote_class(hf_checkpoint: str, class_reference: str):
    """Import one class out of the checkpoint's own modelling code.

    The vision half of this model is remote code -- ``modeling_radio.py`` and
    ``modeling_nemotron_h_omni.py`` ship inside the checkpoint directory -- so
    it cannot be imported from an installed ``transformers``. Going through
    ``get_class_from_dynamic_module`` rather than ``AutoModel.from_config`` is
    what keeps this to the two vision modules: ``from_config`` on the omni
    config would build the 124 B language tower in Transformers as well.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(class_reference, hf_checkpoint, trust_remote_code=True)


def _build_vision_modules(args, hf_config, config):
    """Instantiate the RADIO tower and the ``mlp1`` projector, replicated.

    Weights are *not* loaded here. Either slime's HF loader fills them
    (``hf_to_megatron/nemotron_h.py``, which carries the vision name map) or,
    on the normal path, they arrive from the torch_dist checkpoint stage 04
    wrote. Both need the modules to exist first.
    """
    radio_cls = _load_remote_class(args.hf_checkpoint, "modeling_radio.RadioModel")
    projector_cls = _load_remote_class(
        args.hf_checkpoint,
        "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3VisionProjector",
    )

    device = torch.device("cpu") if config.use_cpu_initialization else accelerator.current_device()
    with torch.device(device):
        vision_model = radio_cls(hf_config.vision_config)
        # The image processor already normalizes (preprocessor_config.json
        # carries norm_mean/norm_std), so the in-model conditioner is replaced
        # by an Identity. Its two buffers are in the checkpoint and have no
        # module afterwards -- deliberately unloaded, and the loader says so
        # rather than dropping them silently.
        vision_model.make_preprocessor_external()
        vision_projector = projector_cls(hf_config)

    vision_model.to(dtype=config.params_dtype)
    vision_projector.to(dtype=config.params_dtype)

    # Bridge's recipe choice, copied verbatim: tower frozen, projector trained.
    vision_model.requires_grad_(False)
    vision_model.eval()

    # Transformers modules are replicated on every TP rank. Marking them says so
    # explicitly, so slime's sharding path returns the whole tensor instead of
    # trying to all-gather a parameter that was never split.
    for module in (vision_model, vision_projector):
        for parameter in module.parameters():
            parameter.tensor_model_parallel = False
            parameter.partition_dim = -1
            parameter.partition_stride = 1

    return vision_model, vision_projector


class NemotronOmniVLModel(MegatronModule):
    """Megatron Nemotron-H hybrid language tower with a replicated RADIO tower."""

    def __init__(
        self,
        config,
        hf_config,
        args,
        *,
        pre_process: bool,
        post_process: bool,
        vp_stage: int | None,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process

        # Token id 18 in this checkpoint, and the name is `img_context_token_id`
        # rather than Qwen's `image_token_id`. It is the id of the *expanded*
        # placeholder, not of the literal "<image>" the dataset row carries:
        # the processor rewrites one "<image>" into
        # "<img>" + "<image>" * n_tokens + "</img>", with n_tokens decided
        # per image by the image processor.
        self.image_token_id = hf_config.img_context_token_id

        language_provider = get_nemotron_h_spec(
            args,
            config,
            vp_stage,
            # Injection happens on whole-sequence token positions, so the
            # embedding must not already be split across sequence-parallel
            # ranks. This module scatters, below, once the features are in.
            scatter_embedding_sequence_parallel=False,
        )
        self.language_model = language_provider(
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

        if pre_process:
            self.vision_model, self.vision_projector = _build_vision_modules(args, hf_config, config)
        else:
            self.vision_model, self.vision_projector = None, None

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    # -- the handful of attributes megatron's pipeline plumbing reaches for ----
    @property
    def decoder(self):
        return self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def train(self, mode=True):
        super().train(mode)
        if self.vision_model is not None:
            self.vision_model.eval()
        return self

    # ------------------------------------------------------------------------
    def _inject_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        full_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cp_group,
        pixel_values,
        num_tokens=None,
    ) -> torch.Tensor:
        embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None).clone()
        embeddings_bsh = embeddings.transpose(0, 1).contiguous()

        if pixel_values is not None:
            # Match Qwen's CP1 path: identity indexing permits odd-length packed
            # samples, which need not satisfy the two-chunk CP divisibility rule.
            local_indices = (
                None
                if cp_group is None or cp_group.size() == 1
                else get_packed_cp_local_indices(cu_seqlens, cp_group.size(), cp_group.rank(), input_ids.device)
            )

            vision_embeddings = project_images(pixel_values, self.vision_model, self.vision_projector).to(
                device=embeddings.device, dtype=embeddings.dtype
            )
            full_vision_positions = (full_input_ids[0] == self.image_token_id).nonzero(as_tuple=False).flatten()
            # The classic VLM trap: the processor decides how many placeholders
            # a given image expands to, and the tower decides how many features
            # it produces. A disagreement misaligns every token after the image
            # and the loss mask with it, and nothing downstream would object.
            if full_vision_positions.numel() != vision_embeddings.shape[0]:
                raise ValueError(
                    f"Nemotron 3.5 VL token/features mismatch: "
                    f"{full_vision_positions.numel()} <image> placeholders, "
                    f"{vision_embeddings.shape[0]} projected features"
                )

            feature_indices = torch.full((full_input_ids.shape[1],), -1, dtype=torch.long, device=input_ids.device)
            feature_indices[full_vision_positions] = torch.arange(vision_embeddings.shape[0], device=input_ids.device)
            local_feature_indices = feature_indices if local_indices is None else feature_indices[local_indices]
            local_vision_mask = local_feature_indices >= 0
            if not torch.equal(local_vision_mask, input_ids[0] == self.image_token_id):
                raise ValueError("Nemotron 3.5 VL CP token layout does not match its full packed sequence")
            embeddings_bsh[0, local_vision_mask] = vision_embeddings[local_feature_indices[local_vision_mask]]

        embeddings = embeddings_bsh.transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings).contiguous()
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        loss_mask: torch.Tensor | None = None,
        pixel_values=None,
        # The processor returns these next to `pixel_values` and slime forwards
        # whatever it returned. They are metadata -- the projector reads the
        # grid off `pixel_values.shape` -- so they are named here only so they
        # do not fall into **kwargs and get passed down to MambaModel, which
        # would raise on the first microbatch.
        num_patches=None,
        # Processor metadata, consumed here rather than forwarded to MCore.
        num_tokens=None,
        pixel_values_videos=None,
        sound_clips=None,
        **kwargs,
    ) -> torch.Tensor:
        if packed_seq_params is None:
            raise ValueError("Nemotron 3.5 VL native training currently requires packed sequences")
        if pixel_values_videos is not None or sound_clips is not None:
            raise ValueError(
                "Nemotron 3.5 VL: only the image path is supported here; "
                "video and audio inputs reached the model provider"
            )

        cp_group = mpu.get_context_parallel_group()
        full_input_ids = gather_packed_input_ids(input_ids, packed_seq_params.cu_seqlens_q, cp_group)

        decoder_input = None
        if self.pre_process:
            decoder_input = self._inject_vision_embeddings(
                input_ids,
                full_input_ids,
                packed_seq_params.cu_seqlens_q,
                cp_group,
                pixel_values,
                num_tokens,
            )

        # position_ids stays whatever it was -- NoPE, so MambaModel's embedding
        # never reads it and there is no mrope construction to do. This is the
        # exact opposite of Qwen3.5-VL, which has to build interleaved T/H/W
        # position ids here.
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
        )


def get_nemotron_35_super_vl_model_provider(args, config, vp_stage):
    """Return the Nemotron 3.5 Super VL model provider."""

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    _assert_is_nemotron_omni(hf_config)

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
    ) -> NemotronOmniVLModel:
        return NemotronOmniVLModel(
            config,
            hf_config,
            args,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    return model_provider


def _assert_is_nemotron_omni(hf_config) -> None:
    """Refuse a config this file cannot build, where the message can still say why.

    The failure it prevents is the expensive one: pointed at the text-only
    Nemotron-3 checkpoint, everything below would build an 88-layer tower with
    a vision half bolted to it and then fail to find a single `vision_model.*`
    tensor -- an hour into a conversion.
    """
    if getattr(hf_config, "model_type", None) != "nemotron_h_omni":
        raise ValueError(
            f"nemotron_35_super_vl: expected an HF config with model_type 'nemotron_h_omni', got "
            f"{getattr(hf_config, 'model_type', None)!r}. For the text-only Nemotron-3 tower use "
            f"--spec slime_plugins.models.nemotron_h get_nemotron_h_spec instead."
        )
    missing = [
        field for field in ("vision_config", "llm_config", "img_context_token_id") if not hasattr(hf_config, field)
    ]
    if missing:
        raise ValueError(f"nemotron_35_super_vl: the HF config is missing {missing}")
