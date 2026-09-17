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
from slime.backends.megatron_utils.hf_to_megatron.nemotron_h import EXPERT_HF, EXPERT_RE, PER_LAYER, TOP_LEVEL

_LAYER_RE = re.compile(r"decoder\.layers\.(\d+)\.(.+)")


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


def convert_nemotron_h_to_hf(args, name, param):
    """Convert one Nemotron-3 Megatron parameter to its HuggingFace tensors."""
    name = strip_mcore_wrappers(name)

    if name in TOP_LEVEL:
        return [(TOP_LEVEL[name], param)]

    if name.startswith("mtp."):
        raise ValueError(
            f"nemotron_h: {name!r} -- this stack builds no MTP block, so an MTP parameter here "
            "means the hybrid layer pattern grew a '/' segment that the HF mapping does not cover."
        )

    layer_match = _LAYER_RE.fullmatch(name)
    if not layer_match:
        raise ValueError(f"Unknown parameter name: {name}")
    layer_idx, rest = layer_match.groups()
    hf_layer = f"backbone.layers.{layer_idx}"

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
