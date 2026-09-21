"""Nemotron-H/VL HF mappings reused from shifangx/slime@328760de.

The VL model shares its language mapping with Nemotron-H and adds RADIO and
projector tensors. Import at TP=ETP=1: generic HF sharding does not implement
the ungated FFN and component-packed Mamba layouts. MCore DCP then reshards
the checkpoint for training. MTP training is outside this preview.
"""

from __future__ import annotations

import re

import torch

from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader, merge_qkv, strip_mcore_wrappers

# The four tables below are public because `megatron_to_hf/nemotron_h.py` -- the
# export direction RL needs -- reads them instead of keeping a second copy. Both
# directions key on the Megatron name and want the HF name, so there is nothing
# to invert, and one table cannot drift from the other.

# Megatron name (after wrapper stripping) -> HF name, for the handful of
# parameters that live outside a layer.
TOP_LEVEL = {
    "embedding.word_embeddings.weight": "backbone.embeddings.weight",
    # MambaStack names its trailing norm `final_norm` (ssm/mamba_block.py:166);
    # `final_layernorm` is the GPTModel spelling and is accepted so a future
    # spec swap does not land here as a KeyError.
    "decoder.final_norm.weight": "backbone.norm_f.weight",
    "decoder.final_layernorm.weight": "backbone.norm_f.weight",
    "output_layer.weight": "lm_head.weight",
}

# Per-layer Megatron suffix -> HF suffix, for everything that is a plain rename.
# `{i}` is filled in by the caller. Ordered by layer type for readability; the
# lookup is a single dict because the layer type is implied by which names the
# model actually asks for.
PER_LAYER = {
    # -- Mamba-2 mixer ------------------------------------------------------
    "mixer.A_log": "mixer.A_log",
    "mixer.D": "mixer.D",
    "mixer.dt_bias": "mixer.dt_bias",
    # The mixer's internal gated RMSNorm (8192-wide), NOT the layer input norm.
    "mixer.norm.weight": "mixer.norm.weight",
    "mixer.in_proj.weight": "mixer.in_proj.weight",
    "mixer.out_proj.weight": "mixer.out_proj.weight",
    # megatron.core has spelled the convolution both ways across releases; the
    # Bridge registry carries both for the same reason.
    "mixer.conv1d.weight": "mixer.conv1d.weight",
    "mixer.conv1d.bias": "mixer.conv1d.bias",
    "mixer.conv1d_weight": "mixer.conv1d.weight",
    "mixer.conv1d_bias": "mixer.conv1d.bias",
    # -- attention ----------------------------------------------------------
    "self_attention.linear_proj.weight": "mixer.o_proj.weight",
    # -- latent MoE ---------------------------------------------------------
    "mlp.router.weight": "mixer.gate.weight",
    "mlp.router.expert_bias": "mixer.gate.e_score_correction_bias",
    "mlp.fc1_latent_proj.weight": "mixer.fc1_latent_proj.weight",
    "mlp.fc2_latent_proj.weight": "mixer.fc2_latent_proj.weight",
    # Non-gated: linear_fc1 is up_proj alone, with no gate half to concatenate.
    "mlp.shared_experts.linear_fc1.weight": "mixer.shared_experts.up_proj.weight",
    "mlp.shared_experts.linear_fc2.weight": "mixer.shared_experts.down_proj.weight",
    # -- the layer input norm, under each of its four spellings -------------
    # Every layer type in this stack fuses its input norm into the first TE
    # linear, so which of these occurs depends only on the layer type; all four
    # resolve to the same HF tensor.
    "mixer.in_proj.layer_norm_weight": "norm.weight",
    "self_attention.linear_qkv.layer_norm_weight": "norm.weight",
    "mlp.linear_fc1.layer_norm_weight": "norm.weight",
    "pre_mlp_layernorm.weight": "norm.weight",
    "input_layernorm.weight": "norm.weight",
    "norm.weight": "norm.weight",
    # -- dense MLP layers ('-' in the pattern). Nemotron-3 Super has none, but
    # the sibling Nano/Ultra configs do, and these two lines make this file
    # work for them unchanged.
    "mlp.linear_fc1.weight": "mixer.up_proj.weight",
    "mlp.linear_fc2.weight": "mixer.down_proj.weight",
}

# TEGroupedMLP stores the routed experts as `...linear_fc1.weight{global_index}`;
# slime's named_params_and_buffers() has already added the expert-parallel
# offset, so the index here is the global one and indexes HF directly.
EXPERT_RE = re.compile(r"mlp\.experts\.linear_fc(1|2)\.weight(\d+)")
EXPERT_HF = {"1": "up_proj", "2": "down_proj"}

# --------------------------------------------------------------------------
# Nemotron 3.5 Super VL only: the vision half.
#
# Left column is the module tree `slime_plugins/models/nemotron_35_super_vl.py`
# builds; right column is the released checkpoint. The names differ because the
# checkpoint is in timm layout and `modeling_radio.py` rewrites it on load.
# --------------------------------------------------------------------------
_RADIO_PREFIX = "vision_model.radio_model.model"

_VISION_TOP_LEVEL = {
    "vision_model.embeddings.patch_projection.weight": f"{_RADIO_PREFIX}.patch_generator.embedder.weight",
    "vision_model.embeddings.video_patch_projection.weight": f"{_RADIO_PREFIX}.patch_generator.video_embedder.weight",
    "vision_model.embeddings.position_embedding": f"{_RADIO_PREFIX}.patch_generator.pos_embed",
    "vision_model.embeddings.cls_register_token": f"{_RADIO_PREFIX}.patch_generator.cls_token.token",
    # The projector. `mlp1` is an nn.Sequential in the released checkpoint --
    # 0 = RMSNorm, 1 = Linear, 2 = the activation (no weights), 3 = Linear --
    # and named submodules in the module tree, so this is a rename and not a
    # reshape. The same three renames are what
    # `register_nemotron_h_omni_conversion_mapping` declares.
    "vision_projector.mlp1.norm.weight": "mlp1.0.weight",
    "vision_projector.mlp1.linear1.weight": "mlp1.1.weight",
    "vision_projector.mlp1.linear2.weight": "mlp1.3.weight",
    # Specific to 3.5 Super VL: it exists only because the language config has
    # num_nextn_predict_layers > 0, which in Megatron puts a final LayerNorm on
    # every block built from the shared config -- including the vision tower.
    "vision_projector.vision_final_layernorm.weight": "vision_projector.vision_final_layernorm.weight",
    "vision_projector.vision_final_layernorm.bias": "vision_projector.vision_final_layernorm.bias",
}

# Per-RADIO-block renames. `{i}` is the block index, identical on both sides.
_VISION_PER_LAYER = {
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "mlp.fc1.weight": "mlp.fc1.weight",
    "mlp.fc1.bias": "mlp.fc1.bias",
    "mlp.fc2.weight": "mlp.fc2.weight",
    "mlp.fc2.bias": "mlp.fc2.bias",
    "attention.output.dense.weight": "attn.proj.weight",
    "attention.output.dense.bias": "attn.proj.bias",
}

_VISION_LAYER_RE = re.compile(r"vision_model\.encoder\.layer\.(\d+)\.(.+)")
# The one real transform on this side, and it is `merge_qkv` run backwards: the
# checkpoint fuses q/k/v into one [3*hidden, hidden] tensor and the module holds
# three separate Linears. Equal thirds -- RADIO is plain MHA, 16 heads of 80, no
# grouped-query asymmetry to respect.
_VISION_QKV_RE = re.compile(r"attention\.attention\.(query|key|value)\.(weight|bias)")
_VISION_QKV_CHUNK = {"query": 0, "key": 1, "value": 2}
# The digit is captured because the exporter needs it: `layer_scale1` is the
# attention residual's scale and `layer_scale2` the MLP's, and SGLang calls
# them `ls1` / `ls2`.
_VISION_LAYER_SCALE_RE = re.compile(r"layer_scale([12])\.lambda1")


def _assert_no_tensor_parallel() -> None:
    """Refuse the layouts the generic sharding path gets wrong for this model."""
    from megatron.core import mpu

    tp = mpu.get_tensor_model_parallel_world_size()
    etp = mpu.get_expert_tensor_parallel_world_size()
    if tp != 1 or etp != 1:
        raise ValueError(
            f"nemotron_h: loading HF weights needs TP=1 and ETP=1 (got TP={tp}, ETP={etp}). "
            "Convert once with tools/convert_hf_to_torch_dist.py and train from the "
            "torch_dist checkpoint instead -- see this module's docstring for the three "
            "transforms that would otherwise be silently wrong."
        )


def _language_config(hf_config):
    """The sub-config carrying the language tower's shapes.

    Nemotron-3 puts them at the top level; Nemotron 3.5 Super VL puts them in
    `llm_config`. `common.py::text_config` looks for `text_config`, which is
    the Qwen spelling and which neither of these configs has, so `merge_qkv`
    has to be handed the right object rather than left to find it.
    """
    return getattr(hf_config, "llm_config", hf_config)


def _hf_prefix(hf_config) -> str:
    """Where the language tensors live in the checkpoint.

    Measured, not guessed: every one of Nemotron-3's 42,683 names appears in the
    3.5 Super VL index under exactly this prefix and nothing else moves.
    """
    return "language_model." if getattr(hf_config, "model_type", None) == "nemotron_h_omni" else ""


def _vision_hf_tensor(name: str, reader: SafetensorReader, hf_config) -> torch.Tensor:
    """Return the full tensor for one vision-half parameter of Nemotron 3.5 VL."""
    vision_config = hf_config.vision_config

    if name == "vision_model.summary_idxs":
        # A persistent buffer that is NOT in the checkpoint -- the released
        # index has no `summary_idxs` key at all, which is why Megatron-Bridge
        # has to add it back on export. RadioModel registers it from the config,
        # so the config is the source of truth here too. It selects which class
        # tokens form the `summary` output, and the projector reads `.features`,
        # so it does not affect anything this run computes.
        return torch.tensor(vision_config.summary_idxs, dtype=torch.long)

    if name in _VISION_TOP_LEVEL:
        return reader.get_tensor(_VISION_TOP_LEVEL[name])

    layer_match = _VISION_LAYER_RE.fullmatch(name)
    if not layer_match:
        raise KeyError(f"Unsupported Nemotron 3.5 VL vision parameter {name!r}")
    layer_idx, rest = layer_match.groups()
    hf_block = f"{_RADIO_PREFIX}.blocks.{layer_idx}"

    layer_scale = _VISION_LAYER_SCALE_RE.fullmatch(rest)
    if layer_scale:
        # Absent from the RELEASED checkpoint. C-RADIO has no LayerScale; the
        # module is inherited and `layerscale_value` is 1.0, which makes it an
        # identity op. Synthesizing ones is therefore exactly what
        # `from_pretrained` leaves in place, not an approximation of it.
        #
        # A checkpoint written by this tree's exporter DOES carry it, under
        # SGLang's spelling (`blocks.<i>.ls{1,2}`), because dropping it on
        # export is what left the engine's copy zeroed after a sync. Prefer the
        # file when it is there: synthesizing over a real value would make the
        # round trip lossy in the one direction that was already lossy once.
        exported = f"{hf_block}.ls{layer_scale.group(1)}"
        if reader is not None and exported in reader:
            return reader.get_tensor(exported)
        return torch.full((vision_config.hidden_size,), float(vision_config.layerscale_value))

    qkv_match = _VISION_QKV_RE.fullmatch(rest)
    if qkv_match:
        which, suffix = qkv_match.groups()
        fused = reader.get_tensor(f"{hf_block}.attn.qkv.{suffix}")
        return fused.chunk(3, dim=0)[_VISION_QKV_CHUNK[which]].contiguous()

    if rest in _VISION_PER_LAYER:
        return reader.get_tensor(f"{hf_block}.{_VISION_PER_LAYER[rest]}")

    raise KeyError(f"Unsupported Nemotron 3.5 VL vision parameter {name!r} (block suffix {rest!r})")


def nemotron_h_hf_tensor(name: str, reader: SafetensorReader, hf_config) -> torch.Tensor:
    """Return the full, unsharded MCore tensor for a Nemotron-3 / 3.5 parameter name."""
    _assert_no_tensor_parallel()

    # This also strips the `language_model.` wrapper the VL model gives its
    # language tower, so from here on both models look the same.
    name = strip_mcore_wrappers(name)

    if name.startswith(("vision_model.", "vision_projector.")):
        return _vision_hf_tensor(name, reader, hf_config)

    prefix = _hf_prefix(hf_config)

    if name in TOP_LEVEL:
        return reader.get_tensor(f"{prefix}{TOP_LEVEL[name]}")

    layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not layer_match:
        raise KeyError(f"Unsupported Nemotron-3 Megatron parameter {name!r}")
    layer_idx, rest = layer_match.groups()
    hf_layer = f"{prefix}backbone.layers.{layer_idx}"

    expert_match = EXPERT_RE.fullmatch(rest)
    if expert_match:
        which, expert_idx = expert_match.groups()
        return reader.get_tensor(f"{hf_layer}.mixer.experts.{expert_idx}.{EXPERT_HF[which]}.weight")

    # The only transform. HF keeps q/k/v apart; Megatron wants one interleaved
    # [q k v] per KV group. merge_qkv reads num_key_value_heads / head_dim off
    # the HF config, which for this checkpoint is 2 groups of 128 -- so the
    # result is [32*128 + 2*128 + 2*128, 4096] = [4608, 4096].
    #
    # Unlike Qwen3.5 there is no RoPE-pair interleave to undo: this model is
    # NoPE, so the plain merge is the right one.
    if rest == "self_attention.linear_qkv.weight":
        return merge_qkv(
            reader.get_tensor(f"{hf_layer}.mixer.q_proj.weight"),
            reader.get_tensor(f"{hf_layer}.mixer.k_proj.weight"),
            reader.get_tensor(f"{hf_layer}.mixer.v_proj.weight"),
            _language_config(hf_config),
        )

    if rest in PER_LAYER:
        return reader.get_tensor(f"{hf_layer}.{PER_LAYER[rest]}")

    raise KeyError(f"Unsupported Nemotron-3 Megatron parameter {name!r} (layer suffix {rest!r})")
