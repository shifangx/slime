"""Nemotron-H/VL export mappings reused from shifangx/slime@328760de.

Shares the HF import tables. TP gathering must already reconstruct Mamba
components and ungated FFNs. Full vision refit includes RADIO LayerScale,
including parameters synthesized from config rather than stored in source HF.
"""

from __future__ import annotations

import re

import torch

from slime.backends.megatron_utils.hf_to_megatron.common import strip_mcore_wrappers
from slime.backends.megatron_utils.hf_to_megatron.nemotron_h import (
    _RADIO_PREFIX,
    _VISION_LAYER_RE,
    _VISION_LAYER_SCALE_RE,
    _VISION_PER_LAYER,
    _VISION_QKV_CHUNK,
    _VISION_QKV_RE,
    _VISION_TOP_LEVEL,
    EXPERT_HF,
    EXPERT_RE,
    PER_LAYER,
    TOP_LEVEL,
)

_LAYER_RE = re.compile(r"decoder\.layers\.(\d+)\.(.+)")

# Where the language tensors live in the checkpoint, and the mirror of the
# loader's `_hf_prefix`. Nemotron-3 is a bare causal LM and its tensors sit at
# the root; 3.5 Super VL wraps the identical tower under `language_model.`.
#
# This is not cosmetic. SGLang's NemotronH_Nano_VL_V2.load_weights sorts every
# incoming tensor by prefix -- `language_model` / `mlp1` /
# `vision_model.radio_model.` / `sound` -- and has **no else branch**. An
# unprefixed `backbone.layers.0...` matches none of them and is dropped without
# a word, so getting this wrong does not raise: it syncs nothing and serves the
# weights the engine loaded at startup, forever.
_VL_LANGUAGE_PREFIX = "language_model."


def _is_vl(model_name: str) -> bool:
    """True for Nemotron 3.5 Super VL, false for the text Nemotron-3.

    `_convert_to_hf_core` has already lowercased the name and stripped `_`/`-`,
    so both the config class (`NemotronH_Omni_Reasoning_V3_Config`) and an
    explicit `--model-name nemotron_h_omni` normalise to something containing
    `omni`, while `NemotronHConfig` / `nemotron_h` do not.
    """
    return "omni" in model_name


# The three Megatron parameters that fuse into one HF tensor, held until the
# last of them arrives. `convert_to_hf` is called one parameter at a time, so a
# many-to-one mapping has nowhere else to live; `megatron_to_hf/__init__.py`
# already does the same thing for DeepSeek's q_a_proj / kv_a_proj pair.
_vision_qkv_buffer: dict[str, dict[str, torch.Tensor]] = {}


def pending_vision_qkv() -> list[str]:
    """Report incomplete QKV fusions after a weight update/export."""
    return [f"{key} (have {sorted(slot)})" for key, slot in sorted(_vision_qkv_buffer.items())]


def _split_qkv(args, param: torch.Tensor) -> list[torch.Tensor]:
    """Undo the per-KV-group ``[q k v]`` fusion that ``merge_qkv`` applied.

    For Nemotron-3 Super that is 2 groups of ``(16 q + 1 k + 1 v) * 128`` rows,
    i.e. ``[4608, 4096]`` back into ``[4096, 4096]`` and two ``[256, 4096]``.
    """
    head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    heads_per_group = args.num_attention_heads // args.num_query_groups
    grouped = param.view(args.num_query_groups, -1, *param.shape[1:])
    q, k, v = grouped.split([heads_per_group * head_dim, head_dim, head_dim], dim=1)
    return [t.reshape(-1, *param.shape[1:]).contiguous() for t in (q, k, v)]


def _convert_vision(name: str, param: torch.Tensor):
    """Restore HF vision names, fused QKV, and the runtime LayerScale parameters."""
    if name == "vision_model.summary_idxs":
        return []

    if name in _VISION_TOP_LEVEL:
        return [(_VISION_TOP_LEVEL[name], param)]

    layer_match = _VISION_LAYER_RE.fullmatch(name)
    if not layer_match:
        raise ValueError(f"Unknown Nemotron 3.5 VL vision parameter: {name}")
    layer_idx, rest = layer_match.groups()
    hf_block = f"{_RADIO_PREFIX}.blocks.{layer_idx}"

    layer_scale = _VISION_LAYER_SCALE_RE.fullmatch(rest)
    if layer_scale:
        return [(f"{hf_block}.ls{layer_scale.group(1)}", param)]

    qkv_match = _VISION_QKV_RE.fullmatch(rest)
    if qkv_match:
        which, suffix = qkv_match.groups()
        slot = _vision_qkv_buffer.setdefault(f"{hf_block}.{suffix}", {})
        if which in slot:
            raise ValueError(
                f"nemotron_h: {name!r} arrived twice before its q/k/v siblings completed a "
                f"fused tensor (have {sorted(slot)}). The buffer assumes one pass over the "
                "parameters per sync; something is iterating them more than once."
            )
        slot[which] = param
        if len(slot) < 3:
            return []
        # HF fuses in q, k, v order and RADIO is plain MHA -- 16 heads of 80, no
        # grouped-query asymmetry -- so this is a plain cat, the exact inverse of
        # the loader's `fused.chunk(3, dim=0)[...]`.
        parts = sorted(slot.items(), key=lambda kv: _VISION_QKV_CHUNK[kv[0]])
        del _vision_qkv_buffer[f"{hf_block}.{suffix}"]
        fused = torch.cat([tensor for _, tensor in parts], dim=0)
        return [(f"{hf_block}.attn.qkv.{suffix}", fused)]

    if rest in _VISION_PER_LAYER:
        return [(f"{hf_block}.{_VISION_PER_LAYER[rest]}", param)]

    raise ValueError(f"Unknown Nemotron 3.5 VL vision parameter: {name} (block suffix {rest!r})")


def convert_nemotron_h_to_hf(args, name, param, model_name=""):
    """Convert one Nemotron-3 / 3.5 Super VL Megatron parameter to HF tensors."""
    is_vl = _is_vl(model_name) or "language_model." in name
    name = strip_mcore_wrappers(name)

    # The vision half exists only on 3.5 Super VL, and its names are unambiguous,
    # so it is dispatched before anything else -- exactly as the loader does.
    if name.startswith(("vision_model.", "vision_projector.")):
        converted = _convert_vision(name, param)
        return converted

    prefix = _VL_LANGUAGE_PREFIX if is_vl else ""

    if name in TOP_LEVEL:
        if name in ("embedding.word_embeddings.weight", "output_layer.weight"):
            param = param[: getattr(args, "vocab_size", None)]
        return [(f"{prefix}{TOP_LEVEL[name]}", param)]

    if name.startswith("mtp."):
        raise ValueError(
            f"nemotron_h: {name!r} -- this stack builds no MTP block, so an MTP parameter here "
            "means the hybrid layer pattern grew a '/' segment that the HF mapping does not cover."
        )

    layer_match = _LAYER_RE.fullmatch(name)
    if not layer_match:
        raise ValueError(f"Unknown parameter name: {name}")
    layer_idx, rest = layer_match.groups()
    hf_layer = f"{prefix}backbone.layers.{layer_idx}"

    expert_match = EXPERT_RE.fullmatch(rest)
    if expert_match:
        which, expert_idx = expert_match.groups()
        # slime's named_params_and_buffers() has already added the expert-parallel
        # offset, so this index is the global one and addresses HF directly.
        return [(f"{hf_layer}.mixer.experts.{expert_idx}.{EXPERT_HF[which]}.weight", param)]

    if rest == "self_attention.linear_qkv.weight":
        q, k, v = _split_qkv(args, param)
        return [
            (f"{hf_layer}.mixer.q_proj.weight", q),
            (f"{hf_layer}.mixer.k_proj.weight", k),
            (f"{hf_layer}.mixer.v_proj.weight", v),
        ]

    if rest in PER_LAYER:
        return [(f"{hf_layer}.{PER_LAYER[rest]}", param)]

    raise ValueError(f"Unknown parameter name: {name} (layer suffix {rest!r})")
