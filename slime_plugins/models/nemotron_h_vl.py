"""NVIDIA Nemotron 3.5 Super VL (`nemotron_h_omni`) for slime's Megatron backend.

The language half needs nothing new. Nemotron 3.5 Super's ``llm_config`` is
field-for-field identical to Nemotron 3 Super 120B-A12B -- 88 layers, hidden
4096, 512 experts at top-k 22, ``moe_latent_size`` 1024, one MTP layer -- and
its ``layers_block_type`` derives to exactly the ``hybrid_override_pattern``
that ``scripts/models/nemotron3-super-120b-a12b.sh`` already carries. So the
decoder here is MCore's ``HybridModel``, built the same way
``slime_plugins/models/nemotron_h.py`` builds it.

What this module adds is the vision half:

    RADIO ViT-H tower  ->  LayerNorm  ->  pixel shuffle  ->  projector MLP
                                             |
                                    scattered into the decoder
                                    embedding at <image> positions

Both vision modules are instantiated from the *checkpoint's own remote code*
rather than reimplemented:

    modeling_radio.RadioModel
    modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3VisionProjector

That is the same idiom ``qwen3_5_vl.py`` uses for its Transformers ViT, and it
is deliberate: the projector owns a pixel-shuffle whose version flag matters and
an ``mlp1`` whose activation is squared-ReLU. Reimplementing any of that here
would be transcribing semantics that the checkpoint already ships. The weight
names then match the checkpoint too, which is what lets
``hf_to_megatron/nemotron_h_vl.py`` stay a thin dispatcher.

Two of those reference modules are then removed again, by
``_align_vision_modules_with_vllm``: the projector's ``vision_final_layernorm``
and the tower's 64 ``RadioLayerScale`` gates. Both are parameters vLLM's
implementation does not have, and the rollout engine is pinned to vLLM's
semantics, so the actor has to drop them too or it would score rollouts with a
vision tower the engine never ran. See that function and
``Scripts-Slime/grpo_vlm_geo3k_nemotron3.5/docs/08_vllm_parity_vision_path.md``.

Unlike Qwen3.5-VL this model does *not* use mrope, and it does not use rope
either: the 8 attention layers in the hybrid stack are NoPE. SGLang's
``nemotron_h.py`` has no rotary embedding at all -- ``NemotronHAttention.forward``
is ``qkv_proj -> RadixAttention -> o_proj`` and takes no ``positions`` argument
-- and neither does the checkpoint's ``modeling_nemotron_h.py``. ``config.json``
still carries ``rope_theta`` and ``partial_rotary_factor``; nothing reads them,
and configuring ``--position-embedding-type rope`` off the back of them is what
put ``train_rollout_logprob_abs_diff`` at 2.3 against slime's 0.1 bound (see
``Scripts-Slime/docs/06_train_rollout_logprob_abs_diff_debug_plan.md``). So
there is no position-id rebuild here -- ``position_ids`` passes straight
through, into an embedding that ignores it under ``none``.

Usage (see scripts/models/nemotron3.5-super-vl.sh):

    --spec slime_plugins.models.nemotron_h_vl get_nemotron_h_vl_model_provider
"""

from __future__ import annotations

import json
import os

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from transformers import AutoConfig

from slime.utils.accelerator import current_device

from .qwen3_5_vl_utils import gather_packed_input_ids, get_packed_cp_local_indices


def _align_vision_modules_with_vllm(vision_model, vision_projector) -> None:
    """Strip the two vision parameters vLLM's implementation does not have.

    The GRPO rollout engine is pinned to vLLM's semantics (see
    ``Scripts-Slime/grpo_vlm_geo3k_nemotron3.5/docs/08_vllm_parity_vision_path.md``),
    and the actor has to match it or every rollout log-prob is computed against a
    different vision tower than the one that produced the tokens. Both edits are
    made here, on the reference modules, rather than reimplemented downstream, so
    that the trainer and the engine differ from the released checkpoint in
    exactly the same two places and nowhere else.

    1. ``vision_projector.vision_final_layernorm``. The checkpoint carries it and
       the reference applies it at the top of ``_project``; vLLM's
       ``nano_nemotron_vl.py`` has no such module at all (the architecture is
       aliased onto ``NemotronH_Nano_VL_V2``, whose prefix-routing
       ``load_weights`` has no else branch, so the two tensors are dropped).
       Setting it to ``None`` is the reference's own "no norm" path --
       ``_project`` is written as ``if self.vision_final_layernorm is not None``
       -- so this removes the op without editing remote code.

    2. ``encoder.layer.{i}.layer_scale{1,2}``. C-RADIO ships no LayerScale; the
       module is inherited and ``layerscale_value`` is 1.0, so ``x * lambda1`` is
       the identity at load time. vLLM builds ``ls1``/``ls2`` as ones and then
       refuses to load them (``radio.py:733-734`` skips the suffix outright), so
       its gate is a construction-time constant. Ours was a live parameter --
       this tower is not frozen in the GRPO recipe, so the optimizer moved it off
       1.0 from the first step. ``nn.Identity`` pins it the way vLLM pins it, and
       it is exact rather than approximate: ``x * 1.0 == x``.
    """
    if getattr(vision_projector, "vision_final_layernorm", None) is None:
        raise ValueError(
            "Nemotron 3.5 Super VL projector has no vision_final_layernorm to strip. "
            "Either the checkpoint is not a Super VL one (llm_config.num_nextn_predict_layers "
            "must be > 0) or the remote code changed; vLLM parity cannot be asserted blind."
        )
    vision_projector.vision_final_layernorm = None

    encoder = getattr(vision_model, "encoder", None)
    if encoder is None:
        raise ValueError("RadioModel has no .encoder; cannot locate the LayerScale modules")
    replaced = 0
    for layer in encoder.layer:
        for attribute in ("layer_scale1", "layer_scale2"):
            if isinstance(getattr(layer, attribute, None), torch.nn.Identity):
                continue
            setattr(layer, attribute, torch.nn.Identity())
            replaced += 1
    expected = 2 * len(encoder.layer)
    if replaced != expected:
        raise ValueError(f"expected to replace {expected} RADIO LayerScale modules, replaced {replaced}")


def _load_vision_modules(hf_checkpoint: str, hf_config, dtype: torch.dtype, use_cpu_initialization: bool):
    """Build the RADIO tower and the projector from the checkpoint's remote code."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    radio_cls = get_class_from_dynamic_module("modeling_radio.RadioModel", hf_checkpoint)
    projector_cls = get_class_from_dynamic_module(
        "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3VisionProjector", hf_checkpoint
    )

    device = torch.device("cpu") if use_cpu_initialization else current_device()
    with torch.device(device):
        vision_model = radio_cls(hf_config.vision_config)
        vision_projector = projector_cls(hf_config)

    # The reference model calls this right after construction: it moves image
    # normalization out of the tower, because the processor has already done it.
    # Leaving it in would normalize twice.
    if hasattr(vision_model, "make_preprocessor_external"):
        vision_model.make_preprocessor_external()

    _align_vision_modules_with_vllm(vision_model, vision_projector)

    vision_model.to(dtype=dtype)
    vision_projector.to(dtype=dtype)

    # HF modules are replicated across TP ranks. Mark them explicitly so slime's
    # direct weight exporter does not try to tensor-parallel all-gather them.
    for module in (vision_model, vision_projector):
        for parameter in module.parameters():
            parameter.tensor_model_parallel = False
            parameter.partition_dim = -1
            parameter.partition_stride = 1
    return vision_model, vision_projector


class NemotronHVLModel(MegatronModule):
    """MCore HybridModel decoder with a replicated RADIO tower and projector."""

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

        # <image> in this checkpoint; the processor expands it to one token per
        # projected patch, so this id marks every position a vision feature goes.
        self.img_context_token_id = hf_config.img_context_token_id
        if self.img_context_token_id is None:
            raise ValueError("Nemotron 3.5 VL needs img_context_token_id in the HF config")

        hybrid_override_pattern = get_hybrid_override_pattern(args, hf_config)

        self.language_model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            hybrid_override_pattern=hybrid_override_pattern,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            # Only the 8 attention layers use positions; HybridModel defaults to
            # 'none' for a pure-Mamba stack, so a hybrid that has attention has
            # to say so. Same reasoning as slime_plugins/models/nemotron_h.py.
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            # The embedding must NOT shard the sequence itself. It does by
            # default under --sequence-parallel (LanguageModelEmbedding:143-145),
            # and _inject_vision_embeddings() builds its <image> mask from the
            # full packed sequence -- so with TP2 the mask is twice the length of
            # what it indexes:
            #
            #   IndexError: The shape of the mask [2304] at index 0 does not
            #               match the shape of the indexed tensor [1152, 4096]
            #
            # (job 19051632). Vision injection has to happen on the whole
            # sequence, so the scatter is deferred to the end of that method.
            # HybridModel re-scatters for a standalone LM forward
            # (hybrid_model.py:501), but only on the branch that builds
            # decoder_input from the embedding; this wrapper passes its own, so
            # that branch is skipped and the sequence is scattered exactly once.
            # ../qwen3_5_vl.py passes this flag for the same reason.
            scatter_embedding_sequence_parallel=False,
            vp_stage=vp_stage,
        )

        # Only the first pipeline stage embeds tokens, so only it needs a tower.
        self.vision_model = None
        self.vision_projector = None
        if pre_process:
            self.vision_model, self.vision_projector = _load_vision_modules(
                args.hf_checkpoint,
                hf_config,
                dtype=config.params_dtype,
                use_cpu_initialization=args.use_cpu_initialization,
            )

    @property
    def decoder(self):
        return self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def _inject_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        full_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cp_group,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None).clone()
        embeddings_bsh = embeddings.transpose(0, 1).contiguous()

        # Without CP the local token stream *is* the full packed stream, so the
        # mapping is the identity. Skipping the lookup is not just an
        # optimization: get_packed_cp_local_indices() requires every packed
        # sequence to divide by 2 * cp_size, which at cp_size=1 degenerates into
        # "must be even" and rejects any odd-length packed sequence.
        if cp_group is None or cp_group.size() == 1:
            local_indices = None
        else:
            local_indices = get_packed_cp_local_indices(
                cu_seqlens,
                cp_group.size(),
                cp_group.rank(),
                input_ids.device,
            )

        # The projector owns the whole pipeline -- tower, LayerNorm, pixel
        # shuffle, mlp1 -- and takes the tower as an argument, exactly as the
        # reference model calls it.
        #
        # pixel_values arrives as a list when the micro-batch holds images of
        # different resolutions, because this tower keeps them as pictures
        # rather than ragged-packed patches and they do not concatenate
        # (backends/megatron_utils/data.py).
        #
        # The list is iterated HERE rather than handed to the projector whole,
        # and that is the whole point of this block. The reference projector
        # does recurse on list/tuple, but it combines the per-image results with
        #
        #     torch.cat([self.forward(pv, vision_model) for pv in pixel_values], dim=0)
        #
        # (modeling_nemotron_h_omni.py:231) over tensors still shaped
        # [n, tokens_i, hidden]. A cat on dim 0 requires every other dimension to
        # agree, so that only works when every image in the list has the same
        # resolution. This processor is aspect-ratio-preserving and
        # variable-resolution (preprocessor_config.json: min_num_patches 1024,
        # max_num_patches 13312), so tokens_i differs per image and the cat
        # raises:
        #
        #   RuntimeError: Sizes of tensors must match except in dimension 0.
        #                 Expected size 286 but got size 270 for tensor number 1
        #
        # -- jobs 19050220 (416 vs 384) and 19051011 (286 vs 270), both in
        # train_one_step. The reference branch is fine for its own use, which is
        # generate() on one image at a time.
        #
        # Flattening each image to [tokens_i, hidden] first makes the
        # concatenation well defined for any mix of resolutions, and it is what
        # sglang does for this same model on the same checkpoints
        # (srt/models/nano_nemotron_vl.py:extract_feature_dynamic, which ends in
        # `img_feats.view(-1, hidden)` per image and then one cat).
        #
        # For a list of equal-resolution images this is bit-identical to the old
        # path -- cat-then-flatten and flatten-then-cat produce the same row
        # order -- so it is a strict generalisation, not a behaviour change.
        # Calling the projector per single tensor also keeps its own _project,
        # which is where pixel shuffle and `mlp1` live. (_project's
        # vision_final_layernorm branch is inert here --
        # _align_vision_modules_with_vllm set it to None.)
        projector_dtype = next(self.vision_projector.parameters()).dtype
        if isinstance(pixel_values, (list, tuple)):
            per_image = []
            for image in pixel_values:
                features = self.vision_projector(image.to(dtype=projector_dtype), self.vision_model)
                per_image.append(features.reshape(-1, features.shape[-1]))
            vision_output = torch.cat(per_image, dim=0)
        else:
            vision_output = self.vision_projector(
                pixel_values.to(dtype=projector_dtype), self.vision_model
            )
        vision_embeddings = vision_output.reshape(-1, vision_output.shape[-1]).to(
            device=embeddings.device, dtype=embeddings.dtype
        )

        full_vision_positions = (
            (full_input_ids[0] == self.img_context_token_id).nonzero(as_tuple=False).flatten()
        )
        if full_vision_positions.numel() != vision_embeddings.shape[0]:
            raise ValueError(
                f"Nemotron 3.5 VL token/feature mismatch: {full_vision_positions.numel()} "
                f"<image> tokens, {vision_embeddings.shape[0]} projected features"
            )

        feature_indices = torch.full(
            (full_input_ids.shape[1],),
            -1,
            dtype=torch.long,
            device=input_ids.device,
        )
        feature_indices[full_vision_positions] = torch.arange(
            vision_embeddings.shape[0], device=input_ids.device
        )
        local_feature_indices = feature_indices if local_indices is None else feature_indices[local_indices]
        local_vision_mask = local_feature_indices >= 0
        if not torch.equal(local_vision_mask, input_ids[0] == self.img_context_token_id):
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
        pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
        # The processor also emits num_patches / num_tokens / imgs_sizes, which
        # slime forwards verbatim from multimodal_train_inputs. The projector
        # derives the same facts from pixel_values itself, so they are accepted
        # and ignored rather than left to land in **kwargs and reach HybridModel.
        num_patches: torch.Tensor | None = None,
        num_tokens: torch.Tensor | None = None,
        imgs_sizes: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if packed_seq_params is None:
            raise ValueError("Nemotron 3.5 VL native training currently requires packed sequences")

        decoder_input = None
        if self.pre_process:
            if pixel_values is None:
                raise ValueError(
                    "Nemotron 3.5 VL received no pixel_values. A text-only batch would need the "
                    "vision branch skipped; this recipe's dataset carries an image per sample."
                )
            cp_group = mpu.get_context_parallel_group()
            full_input_ids = gather_packed_input_ids(input_ids, packed_seq_params.cu_seqlens_q, cp_group)
            decoder_input = self._inject_vision_embeddings(
                input_ids,
                full_input_ids,
                packed_seq_params.cu_seqlens_q,
                cp_group,
                pixel_values,
            )

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            **kwargs,
        )


# 'M' Mamba2, 'E' MoE, '*' attention -- MCore's Symbols, and the three block
# types this family's layers_block_type uses.
_BLOCK_TYPE_SYMBOLS = {"mamba": "M", "moe": "E", "attention": "*"}

# MCore's separator between the main stack and each MTP depth in the unified
# pattern (hybrid_layer_allocation.Symbols.MTP_SEPARATOR).
_MTP_SEPARATOR = "/"

# The block layout of one MTP depth for this family: one attention layer then
# one MoE layer.
#
# Unlike the main pattern this is NOT derivable from the config. Nemotron 3's
# text config carries `mtp_hybrid_override_pattern`, but the 3.5 Omni config
# carries neither that key nor an MTP entry in `layers_block_type` -- checked
# against NVIDIA-Nemotron-3.5-Super-EA-09112026/config.json, which has
# `num_nextn_predict_layers: 1` and nothing else about the MTP block's shape.
#
# So it is a literal, and the thing that keeps a literal honest is checking it:
# validate_mtp_pattern() below reads the checkpoint's own tensor names and
# refuses a pattern that does not match them. The same "*E" is hard-required by
# Megatron-Bridge's bridge for this checkpoint
# (megatron/bridge/models/nemotron_omni/nemotron_omni_bridge.py:401).
_MTP_BLOCK_PATTERN = "*E"


def get_mtp_num_layers(args) -> int:
    """MTP prediction depths, 0 when MTP is off.

    ``--mtp-num-layers`` is a reset_arg on Megatron's own (slime
    ``utils/arguments.py:1514``), so it defaults to None rather than 0.
    """
    return int(getattr(args, "mtp_num_layers", None) or 0)


def get_hybrid_override_pattern(args, hf_config) -> str:
    """The M/E/* string MCore needs, from --hybrid-override-pattern or the config.

    Nemotron 3's HF config spells this out as ``hybrid_override_pattern``; the
    3.5 Omni config nests the language model and spells it as a
    ``layers_block_type`` list instead. Both derive to the same 88-character
    pattern for the Super models, so accept either rather than making the model
    config carry a literal that can drift from the checkpoint.

    With ``--mtp-num-layers`` set the return value is MCore's *unified* pattern:
    the main stack, then one ``/``-separated segment per prediction depth
    (``"<88 chars>/*E"`` for one depth). ``HybridModel`` copies the value into
    ``hybrid_layer_pattern`` and splits it on the separator
    (``hybrid_model.py:162-170``, ``hybrid_layer_allocation.py:213``), and both
    Nemotron converters already index main-stack layers through
    ``pattern.split("/")[0]``, so the composite is what every consumer wants.

    The ``--num-layers`` check applies to the main segment only: the MTP layers
    are extra and are not counted by ``--num-layers``.
    """
    pattern = getattr(args, "hybrid_override_pattern", None)
    if not pattern:
        llm_config = getattr(hf_config, "llm_config", hf_config)
        pattern = getattr(llm_config, "hybrid_override_pattern", None)
    if not pattern:
        llm_config = getattr(hf_config, "llm_config", hf_config)
        block_types = getattr(llm_config, "layers_block_type", None)
        if block_types:
            try:
                pattern = "".join(_BLOCK_TYPE_SYMBOLS[block] for block in block_types)
            except KeyError as exc:
                raise ValueError(f"unknown layers_block_type entry {exc.args[0]!r}") from exc
    if not pattern:
        raise ValueError(
            "Nemotron-H needs a hybrid layer pattern: pass --hybrid-override-pattern, or use a "
            "checkpoint whose config carries hybrid_override_pattern or layers_block_type."
        )

    # Before appending: --num-layers counts the main stack, not the MTP depths.
    main_pattern = pattern.split(_MTP_SEPARATOR)[0]
    if len(main_pattern) != args.num_layers:
        raise ValueError(
            f"hybrid layer pattern is {len(main_pattern)} characters but --num-layers is "
            f"{args.num_layers}; every layer needs a symbol."
        )

    mtp_num_layers = get_mtp_num_layers(args)
    if not mtp_num_layers:
        return main_pattern

    # An explicit pattern may already carry its own MTP segments; respect it.
    if _MTP_SEPARATOR in pattern:
        return pattern

    llm_config = getattr(hf_config, "llm_config", hf_config)
    mtp_pattern = getattr(args, "mtp_hybrid_override_pattern", None) or getattr(
        llm_config, "mtp_hybrid_override_pattern", None
    )
    if not mtp_pattern:
        mtp_pattern = _MTP_BLOCK_PATTERN
    pattern = main_pattern + (_MTP_SEPARATOR + mtp_pattern) * mtp_num_layers

    # Write it back, because this function is not the only reader. The
    # Megatron->HF converter takes its pattern from args
    # (megatron_to_hf/nemotron_h.py), and it is what tells the weight sync
    # which MTP block is which. Leaving the composite only in this function's
    # return value gives the converter the bare main pattern and a
    # "hybrid_override_pattern carries no MTP segment" failure on the first
    # sync -- job 19535251 died of exactly that, one level up, in the loading
    # direction. Idempotent: the separator check above short-circuits on the
    # second call.
    args.hybrid_override_pattern = pattern
    return pattern


# Which HF tensor suffixes identify each block symbol inside an MTP layer. Read
# off NVIDIA-Nemotron-3.5-Super-EA-09112026's index: an attention layer carries
# mixer.{q,k,v,o}_proj, an MoE layer carries mixer.gate and mixer.experts.
_MTP_SYMBOL_EVIDENCE = {
    "*": "mixer.q_proj.weight",
    "E": "mixer.gate.weight",
    "M": "mixer.A_log",
}


def validate_mtp_pattern(args, hf_checkpoint: str, pattern: str) -> None:
    """Check the MTP segments of ``pattern`` against the checkpoint's tensor names.

    ``_MTP_BLOCK_PATTERN`` is a literal because the 3.5 Omni config does not
    carry the MTP block layout (see its comment). A wrong literal would not
    fail here -- it would build the wrong layer types, load nothing into them
    because no HF name matches, and surface much later as a draft head that is
    silently untrained. The checkpoint index is the cheap ground truth: it
    needs no GPU and no weights, only ``model.safetensors.index.json``.

    Skipped without complaint when the index is absent (a Megatron-format
    ``--load`` resume has no HF index, and by then the pattern came from the
    checkpoint's own args anyway).
    """
    if _MTP_SEPARATOR not in pattern:
        return
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return
    with open(index_path) as handle:
        names = json.load(handle).get("weight_map", {})

    # HF flattens every depth into one list, so depth d's j-th block is at
    # d*L+j -- EXCEPT under --mtp-use-repeated-layer, where MCore builds one
    # layer object and calls it once per depth
    # (multi_token_prediction.py:2314-2320, :2414). Every depth then reads the
    # same serialized block, HF serializes only that one, and there is exactly
    # one depth's worth of tensors to check.
    segments = pattern.split(_MTP_SEPARATOR)[1:]
    if getattr(args, "mtp_use_repeated_layer", False):
        segments = segments[:1]
    for depth, segment in enumerate(segments):
        stride = len(segment)
        for j, symbol in enumerate(segment):
            evidence = _MTP_SYMBOL_EVIDENCE.get(symbol)
            if evidence is None:
                raise ValueError(f"unknown MTP block symbol {symbol!r} in pattern {pattern!r}")
            expected = f"mtp.layers.{depth * stride + j}.{evidence}"
            if not any(name == expected or name.endswith("." + expected) for name in names):
                raise ValueError(
                    f"MTP pattern {segment!r} says depth {depth} block {j} is {symbol!r}, but "
                    f"{hf_checkpoint} has no tensor {expected!r}. The checkpoint's MTP block "
                    f"layout differs from this model plugin's; fix _MTP_BLOCK_PATTERN or pass "
                    f"--mtp-hybrid-override-pattern."
                )


def get_nemotron_h_vl_model_provider(args, config, vp_stage=None):
    """Return the native Nemotron 3.5 Super VL model provider."""

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    if not hasattr(hf_config, "vision_config") or not hasattr(hf_config, "llm_config"):
        raise ValueError(
            f"{args.hf_checkpoint} is not a Nemotron omni checkpoint "
            "(expected both vision_config and llm_config)"
        )
    # Fail here rather than inside the first model_provider call, which on a
    # pipeline-parallel run happens once per stage. With MTP on, this is also
    # the cheapest place the literal MTP block layout gets checked against the
    # checkpoint -- before the 235 GB load, not after it.
    validate_mtp_pattern(args, args.hf_checkpoint, get_hybrid_override_pattern(args, hf_config))

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
    ) -> NemotronHVLModel:
        return NemotronHVLModel(
            config,
            hf_config,
            args,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    return model_provider
