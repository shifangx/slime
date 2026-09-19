"""Megatron hybrid stack -> HuggingFace ``nemotron_h`` weight mapping.

The exporter contract is one function: given a *Megatron* parameter name and the
full, TP-gathered tensor it holds, return the ``(HF name, tensor)`` pairs that
carry it. RL needs it -- every GRPO rollout pushes the trained weights into the
SGLang engine through ``convert_to_hf()`` -- while SFT does not, because
``--debug-train-only`` starts no engine and ``--save`` writes torch_dist.

This is the inverse of ``hf_to_megatron/nemotron_h.py`` and shares that module's
two tables rather than restating them. Both directions look up the same key (the
Megatron name) and want the same value (the HF name), so there is nothing to
invert and no second copy to keep in step -- which matters here more than usual,
because the map has ~25 entries and a single wrong one is a silently corrupted
weight sync rather than a crash. See that module's docstring for where the map
came from (Megatron-Bridge's verified bidirectional ``mapping_registry``).

WHAT IS NOT A PLAIN RENAME
--------------------------
Only the attention QKV. HF keeps ``q_proj`` / ``k_proj`` / ``v_proj`` apart and
Megatron fuses them per KV group, so ``linear_qkv.weight`` splits back into
three. There is no RoPE-pair interleave to undo: Nemotron-3 is NoPE, so this is
the plain split, and ``merge_qkv`` on the loader side is the plain merge.

Everything else that *could* have needed a transform is handled before this file
sees the tensor, in ``update_weight/common.py::merge_tp_partitions``:

* **the squared-ReLU experts.** ``linear_fc1`` here is ``up_proj`` alone -- this
  model is ungated, so there is no gate half to de-interleave across TP ranks.
  The gather only does that de-interleave when ``has_gated_linear_unit(args)``.
* **the Mamba tensors.** ``in_proj`` packs ``[z, x, B, C, dt]`` and ``conv1d``
  packs ``[x, B, C]``, and each TP rank holds a slice of *every* component. The
  gather reassembles them from ``param.partition_sizes``, so what arrives here is
  already the HF layout.

Putting both in the gather rather than here keeps one implementation of the TP
arithmetic for every model that packs components into one column-parallel
tensor, instead of a per-exporter copy.

WHAT IS NOT EXPORTED
--------------------
The released checkpoints ship a Multi-Token-Prediction block (``mtp.layers.0.*``,
512 more experts) that the stack this tree builds does not have, so those tensors
are never produced and the engine keeps whatever it loaded for them. That is the
same subset the loader reads, so a round trip through both is closed.
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
    """Slots still holding a partial q/k/v. Empty is the only correct end state.

    `_convert_vision` returns ``[]`` for the first two of a block's three qkv
    parameters and emits the fused tensor on the third. That is right when all
    three arrive and **silent** when they do not: the ones that did arrive are
    held here, never returned, never sent. The engine then keeps whatever it
    loaded at startup for that block's attention while every other block is
    updated -- a half-synced tower, with nothing in any log to say so.

    The conversion cannot notice that itself. It is called one parameter at a
    time and has no way to know which one is last. The caller does: when a sync
    ends, every slot must have been consumed. See
    `megatron_to_hf.assert_conversion_buffers_drained`.

    Note what this is *not* for, because the buffer already handles it: the slot
    key is ``f"{hf_block}.{suffix}"`` and `hf_block` carries the layer index, so
    a `q` from one block can never be fused with a `k` from another. The two
    failure modes are a slot that never completes -- silent, and what this
    catches -- and a slot that fills twice, which `_convert_vision` already
    raises on.
    """
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
    """Convert one Nemotron 3.5 Super VL vision-half parameter.

    The exact inverse of the loader's `_vision_hf_tensor`, reusing its tables
    rather than restating them, and returning the names the released checkpoint
    uses -- which is the point: those are the names SGLang already loads at
    startup, so a sync that reproduces them travels a path that is known to work.

    Two parameters are deliberately NOT exported, and neither is an
    approximation:

    * `vision_model.summary_idxs` -- a config-derived buffer that the released
      index does not contain at all. The loader synthesizes it from
      `vision_config.summary_idxs`; SGLang's RadioModel registers it the same
      way and has no parameter to receive it.
    * `layer_scale[12].lambda1` -- also absent from the checkpoint. C-RADIO has
      no LayerScale, `layerscale_value` is 1.0, and SGLang's radio.py has no
      such parameter either (`grep layer_scale` finds nothing).

    Both would be a problem if training could move them. It cannot:
    `slime_plugins/models/nemotron_35_super_vl.py:159` calls
    `vision_model.requires_grad_(False)`, so the whole vision tower is frozen
    and every tensor here is byte-identical to what the engine loaded at
    startup. Exporting it at all is redundant rather than wrong -- kept because
    a sync that silently omits half a model is exactly the failure this file's
    docstring warns about, and because the freeze is a property of one plugin
    line that could change.
    """
    if name == "vision_model.summary_idxs":
        return []

    if name in _VISION_TOP_LEVEL:
        return [(_VISION_TOP_LEVEL[name], param)]

    layer_match = _VISION_LAYER_RE.fullmatch(name)
    if not layer_match:
        raise ValueError(f"Unknown Nemotron 3.5 VL vision parameter: {name}")
    layer_idx, rest = layer_match.groups()
    hf_block = f"{_RADIO_PREFIX}.blocks.{layer_idx}"

    if _VISION_LAYER_SCALE_RE.fullmatch(rest):
        return []

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
    name = strip_mcore_wrappers(name)

    # The vision half exists only on 3.5 Super VL, and its names are unambiguous,
    # so it is dispatched before anything else -- exactly as the loader does.
    if name.startswith(("vision_model.", "vision_projector.")):
        return _convert_vision(name, param)

    prefix = _VL_LANGUAGE_PREFIX if _is_vl(model_name) else ""

    if name in TOP_LEVEL:
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
