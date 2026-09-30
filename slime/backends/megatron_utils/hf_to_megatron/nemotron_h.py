"""HuggingFace -> Megatron weight mapping for NVIDIA Nemotron-H (`nemotron_h`).

Nemotron 3 Super is an 88-layer hybrid built from three layer types, chosen per
layer by ``config.hybrid_override_pattern``:

    'M'  Mamba2      40 layers
    'E'  MoE         40 layers   (512 routed experts in a 1024-wide latent space)
    '*'  attention    8 layers   (GQA, 32 heads / 2 KV heads, head_dim 128)

MCore's HybridModel builds exactly those three, in the same order, so layer
index i on the HF side is layer index i on the MCore side. The pattern is what
tells this module which mapping to apply.

HF names everything under a layer ``mixer``, whatever the layer type, and puts
the pre-mixer norm at ``backbone.layers.{i}.norm``. MCore instead folds that
norm into whichever linear comes first (TELayerNormColumnParallelLinear) for
Mamba and attention layers, and keeps it as ``pre_mlp_layernorm`` for MoE
layers -- which is why one HF tensor is reached by three different MCore names
below.

The Mamba tensors are the only ones that are not a rename. See _interleave_tp.
"""

from __future__ import annotations

import re

import torch

from .common import SafetensorReader, strip_mcore_wrappers


def _tp_size() -> int:
    """Tensor-parallel width, or 1 outside an initialised MCore process."""
    try:
        from megatron.core import mpu

        return mpu.get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _interleave_tp(components: list[torch.Tensor], tp_size: int) -> torch.Tensor:
    """Lay components out so a plain TP chunk gives each rank all of them.

    This is the one place the mapping is not a rename, and MCore says why at
    ssm/mamba_mixer.py:300-303:

        in_proj packs [z, x, B, C, dt] into one ColumnParallelLinear.  Each
        component is independently TP-sharded but with different sizes.  When
        resharding across different TP sizes the planner must interleave
        per-component blocks rather than doing a contiguous concat.

    MCore records that as a ``partition_sizes`` attribute on the parameter and
    its own checkpoint planner honours it. slime's loader does not: it shards
    with a generic ``torch.chunk`` over partition_dim
    (hf_to_megatron/common.py:_tensor_parallel_shard). So a flat HF
    ``[z | x | B | C | dt]`` would hand rank 0 all of z and none of dt.

    Building the interleaved layout here instead means the generic chunk is
    correct by construction:

        [z_0, x_0, B_0, C_0, dt_0, z_1, x_1, B_1, C_1, dt_1, ...]

    At tp_size == 1 this is the identity, which is why a TP1 run cannot tell
    the two layouts apart -- and why this needs a TP>1 test.
    """
    if tp_size == 1:
        return torch.cat(components, dim=0).contiguous()

    blocks = []
    for rank in range(tp_size):
        for component in components:
            if component.shape[0] % tp_size != 0:
                raise ValueError(
                    f"Mamba component of size {component.shape[0]} does not divide across TP{tp_size}"
                )
            blocks.append(component.chunk(tp_size, dim=0)[rank])
    return torch.cat(blocks, dim=0).contiguous()


def _mamba_dims(hf_config) -> tuple[int, int, int]:
    """(d_inner, groups_state, nheads) -- the component sizes of in_proj."""
    d_inner = hf_config.expand * hf_config.hidden_size
    groups_state = hf_config.n_groups * hf_config.ssm_state_size
    return d_inner, groups_state, hf_config.mamba_num_heads


def _split_in_proj(tensor: torch.Tensor, hf_config) -> list[torch.Tensor]:
    """HF in_proj is [z | xBC | dt]; return [z, x, B, C, dt].

    Both sides order the components the same way -- HF at
    modeling_nemotron_h.py:400 (`[d_mlp, d_mlp, intermediate_size, conv_dim,
    num_heads]`, with d_mlp 0 for this checkpoint) and MCore at
    ssm/mamba_mixer.py:591 (`z, xBC, dt`) -- so only the TP layout differs.
    """
    d_inner, groups_state, nheads = _mamba_dims(hf_config)
    z, x, b, c, dt = torch.split(tensor, [d_inner, d_inner, groups_state, groups_state, nheads], dim=0)
    return [z, x, b, c, dt]


def _split_conv1d(tensor: torch.Tensor, hf_config) -> list[torch.Tensor]:
    """conv1d covers [x | B | C] -- d_inner + 2 * n_groups * d_state rows."""
    d_inner, groups_state, _ = _mamba_dims(hf_config)
    x, b, c = torch.split(tensor, [d_inner, groups_state, groups_state], dim=0)
    return [x, b, c]


def _merge_qkv(reader: SafetensorReader, prefix: str, hf_config) -> torch.Tensor:
    """HF keeps q/k/v apart; MCore's linear_qkv wants them grouped per KV head."""
    q = reader.get_tensor(f"{prefix}.q_proj.weight")
    k = reader.get_tensor(f"{prefix}.k_proj.weight")
    v = reader.get_tensor(f"{prefix}.v_proj.weight")
    num_groups = hf_config.num_key_value_heads
    queries_per_group = hf_config.num_attention_heads // num_groups
    head_dim = hf_config.head_dim

    trailing = q.shape[1:]
    q = q.reshape(num_groups, queries_per_group * head_dim, *trailing)
    k = k.reshape(num_groups, head_dim, *trailing)
    v = v.reshape(num_groups, head_dim, *trailing)
    return torch.cat((q, k, v), dim=1).reshape(-1, *trailing).contiguous()


def nemotron_h_hf_tensor(name: str, reader: SafetensorReader, hf_config) -> torch.Tensor:
    """Return the full, unsharded MCore tensor for a Nemotron-H parameter name."""

    name = strip_mcore_wrappers(name)

    direct = {
        "embedding.word_embeddings.weight": "backbone.embeddings.weight",
        "decoder.final_norm.weight": "backbone.norm_f.weight",
        # HybridModel inherits GPTModel's name for the post-decoder norm in some
        # configurations; accept both rather than depend on which.
        "decoder.final_layernorm.weight": "backbone.norm_f.weight",
        "output_layer.weight": (
            "backbone.embeddings.weight"
            if getattr(hf_config, "tie_word_embeddings", False)
            else "lm_head.weight"
        ),
    }
    if name in direct:
        return reader.get_tensor(direct[name])

    mtp_match = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", name)
    if mtp_match:
        return _convert_mtp(reader, name, hf_config, int(mtp_match.group(1)), mtp_match.group(2))

    layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not layer_match:
        raise KeyError(f"Unsupported Nemotron-H Megatron parameter {name!r}")
    layer_idx, rest = layer_match.groups()
    layer_idx = int(layer_idx)

    pattern = hf_config.hybrid_override_pattern
    # The MTP pattern is appended after a separator in MCore's spelling; the
    # main stack is what layer indices address.
    pattern = pattern.split("/")[0]
    if layer_idx >= len(pattern):
        raise KeyError(f"Layer {layer_idx} is past the end of hybrid_override_pattern ({len(pattern)} layers)")
    symbol = pattern[layer_idx]

    return _convert_block(reader, name, hf_config, symbol, rest, f"backbone.layers.{layer_idx}")


def _mtp_patterns(hf_config) -> list[str]:
    """The ``/``-separated MTP segments of the unified hybrid pattern, one per depth."""
    segments = str(hf_config.hybrid_override_pattern).split("/")[1:]
    if not segments:
        raise KeyError(
            "an mtp.* parameter was requested but hybrid_override_pattern carries no MTP "
            "segment; pass --mtp-num-layers so the provider appends one"
        )
    return segments


def _convert_mtp(reader, name: str, hf_config, depth: int, rest: str):
    """Map one ``mtp.layers.{depth}.…`` MCore name onto its HF tensor.

    MCore nests an MTP depth as a wrapper (``enorm`` / ``hnorm`` / ``eh_proj``)
    around an inner stack, ``mtp.layers.{d}.mtp_model_layer.layers.{j}.…``.
    HF flattens both into one list, so depth ``d``'s ``j``-th block lands at
    ``d*L + j`` where ``L`` is the length of that depth's pattern segment --
    the same arithmetic SGLang does at ``nemotron_h_mtp.py:253-270``. Verified
    against NVIDIA-Nemotron-3.5-Super-EA-09112026's index, where the single
    depth of ``*E`` gives ``mtp.layers.0`` the attention tensors and
    ``mtp.layers.1`` the MoE ones, with ``final_layernorm`` on the last block.

    The per-symbol bodies are the main stack's, unchanged: ``_convert_block``
    is shared with the decoder branch rather than copied, because two copies
    would be free to drift and this model family punishes exactly that.
    """
    segments = _mtp_patterns(hf_config)
    if depth >= len(segments):
        raise KeyError(f"MTP depth {depth} is past the end of the pattern ({len(segments)} depths)")
    segment = segments[depth]
    stride = len(segment)
    base = depth * stride

    # Wrapper tensors: HF puts them on the depth's first block, and the trailing
    # final_layernorm on its last.
    if rest in {"enorm.weight", "hnorm.weight", "eh_proj.weight"}:
        return reader.get_tensor(f"mtp.layers.{base}.{rest}")
    if rest == "final_layernorm.weight":
        return reader.get_tensor(f"mtp.layers.{base + stride - 1}.final_layernorm.weight")

    inner = re.fullmatch(r"mtp_model_layer\.layers\.(\d+)\.(.+)", rest)
    if not inner:
        raise KeyError(f"Unsupported Nemotron-H MTP parameter {name!r}")
    j, inner_rest = int(inner.group(1)), inner.group(2)
    if j >= stride:
        raise KeyError(f"MTP block {j} is past the end of depth {depth}'s pattern {segment!r}")

    return _convert_block(reader, name, hf_config, segment[j], inner_rest, f"mtp.layers.{base + j}")


def _convert_block(reader, name: str, hf_config, symbol: str, rest: str, hf: str):
    """The per-symbol HF->MCore mapping for one hybrid block.

    Shared verbatim by the main decoder stack and by the inner layers of every
    MTP depth: an MTP block of symbol ``s`` is the same kind of layer as a main
    stack block of symbol ``s``, and its HF tensors are spelled the same way
    under a different prefix.

    Args:
        hf: the HF prefix for this block -- ``backbone.layers.{i}`` for the main
            stack, ``mtp.layers.{d*L+j}`` for an MTP block.
    """
    # Every layer type has the same pre-mixer norm on the HF side; MCore reaches
    # it under three different names depending on what the layer is.
    pre_norm_names = {
        "mixer.in_proj.layer_norm_weight",            # M: folded into in_proj
        "self_attention.linear_qkv.layer_norm_weight",  # *: folded into linear_qkv
        "pre_mlp_layernorm.weight",                   # E: kept separate
        "input_layernorm.weight",
    }
    if rest in pre_norm_names:
        return reader.get_tensor(f"{hf}.norm.weight")

    if symbol == "M":
        tp = _tp_size()
        if rest == "mixer.in_proj.weight":
            components = _split_in_proj(reader.get_tensor(f"{hf}.mixer.in_proj.weight"), hf_config)
            return _interleave_tp(components, tp)
        # MCore registers these as FLAT parameters -- conv1d_weight, not a
        # conv1d submodule with a .weight (ssm/mamba_mixer.py:329,338) -- while
        # HF has a real nn.Conv1d. Same shapes either way, [conv_dim, 1, d_conv]
        # and [conv_dim]; only the name differs. Job 19048940 died here.
        if rest in {"mixer.conv1d_weight", "mixer.conv1d_bias"}:
            suffix = "weight" if rest.endswith("_weight") else "bias"
            tensor = reader.get_tensor(f"{hf}.mixer.conv1d.{suffix}")
            return _interleave_tp(_split_conv1d(tensor, hf_config), tp)
        # A_log, D and dt_bias are one value per Mamba head, so a plain chunk
        # over heads is already the right split and no interleaving applies.
        if rest in {"mixer.A_log", "mixer.D", "mixer.dt_bias"}:
            return reader.get_tensor(f"{hf}.{rest}")
        if rest in {"mixer.norm.weight", "mixer.out_proj.weight"}:
            return reader.get_tensor(f"{hf}.{rest}")
        raise KeyError(f"Unsupported Nemotron-H Mamba parameter {name!r}")

    if symbol == "*":
        if rest == "self_attention.linear_qkv.weight":
            return _merge_qkv(reader, f"{hf}.mixer", hf_config)
        if rest == "self_attention.linear_proj.weight":
            return reader.get_tensor(f"{hf}.mixer.o_proj.weight")
        raise KeyError(f"Unsupported Nemotron-H attention parameter {name!r}")

    if symbol == "E":
        if rest == "mlp.router.weight":
            return reader.get_tensor(f"{hf}.mixer.gate.weight")
        if rest == "mlp.router.expert_bias":
            return reader.get_tensor(f"{hf}.mixer.gate.e_score_correction_bias")
        if rest in {"mlp.fc1_latent_proj.weight", "mlp.fc2_latent_proj.weight"}:
            suffix = rest.removeprefix("mlp.")
            return reader.get_tensor(f"{hf}.mixer.{suffix}")

        # Routed experts. MCore's grouped GEMM exposes one tensor per expert as
        # linear_fc1.weight{e}; the un-suffixed form is the whole stack.
        # Nemotron's experts are NOT gated -- `mlp_hidden_act: relu2`, and the
        # checkpoint has no gate_proj -- so linear_fc1 is up_proj alone rather
        # than cat(gate, up) the way every gated model in this directory does.
        expert_match = re.fullmatch(r"mlp\.experts\.linear_fc([12])(?:\.weight)?(\d+)?", rest)
        if expert_match:
            projection, expert_idx = expert_match.groups()
            hf_proj = "up_proj" if projection == "1" else "down_proj"
            if expert_idx is None:
                raise KeyError(
                    f"{name!r} addresses the whole expert stack; Nemotron-H stores experts "
                    "individually, so --moe-grouped-gemm must expose per-expert weights"
                )
            return reader.get_tensor(f"{hf}.mixer.experts.{int(expert_idx)}.{hf_proj}.weight")

        shared = {
            "mlp.shared_experts.linear_fc1.weight": "up_proj",
            "mlp.shared_experts.linear_fc2.weight": "down_proj",
        }
        if rest in shared:
            return reader.get_tensor(f"{hf}.mixer.shared_experts.{shared[rest]}.weight")
        raise KeyError(f"Unsupported Nemotron-H MoE parameter {name!r}")

    raise KeyError(f"{hf} has unknown hybrid symbol {symbol!r} for parameter {name!r}")
