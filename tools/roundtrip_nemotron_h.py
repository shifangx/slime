#!/usr/bin/env python3
"""HF -> Megatron -> HF for Nemotron-3 / 3.5 Super VL, tensor by tensor.

WHY THIS EXISTS
---------------
Three RL jobs on Nemotron 3.5 Super VL produced rollouts that are visibly worse
than the same checkpoint served directly: `zero_std/count_1` 3-4 of 64 against
16-17, truncation 0.65 against 0.26, and the model emitting stray `<image>`
tokens into its own responses. The only difference between the two is the path
the weights took --

    serving   HF -> SGLang
    RL        HF -> convert_hf_to_torch_dist -> Megatron -> convert_to_hf -> SGLang

-- so the round trip is the suspect, and nothing had ever checked it on real
tensors. `tests/test_hf_to_megatron.py` checks it on synthetic ones, which
catches a wrong name and not a wrong value.

`train_rollout_logprob_abs_diff` is NOT the test for this, which is worth saying
because it is the obvious candidate: it compares what Megatron recomputes
against what SGLang returned, and both hold the *same* post-round-trip weights.
It measures sync fidelity, not conversion fidelity. This does.

WHAT IT COVERS, AND WHAT IT DOES NOT
------------------------------------
It runs slime's own two converters back to back, on CPU:

    nemotron_h_hf_tensor(name, reader, config)      HF safetensors -> Megatron
    convert_nemotron_h_to_hf(args, name, tensor)    Megatron -> HF named tensors

and compares the result against the original safetensors entry. A pure rename
must come back bit-identical; the QKV split/merge and the vision QKV fusion are
permutations, so they must too. Any difference at all is a bug, not a tolerance
question -- there is no arithmetic in this path that could round.

What it does **not** cover is `torch_dist` itself: the save/load and the
tensor-parallel sharding between the two converters. So a clean run here narrows
the suspect to that layer rather than clearing the round trip entirely, and a
dirty run localizes it to a named tensor without needing a GPU at all.

USAGE
-----
    python tools/roundtrip_nemotron_h.py --hf-checkpoint /path/to/ckpt
    python tools/roundtrip_nemotron_h.py --hf-checkpoint ... --experts-per-layer 2
    python tools/roundtrip_nemotron_h.py --hf-checkpoint ... --full

Needs torch and safetensors and nothing else -- neither converter imports
megatron, and the loader's one `mpu` call is stubbed below. On a login node:

    uv venv --python 3.12 .venv_cpu_torch
    VIRTUAL_ENV=.venv_cpu_torch uv pip install torch safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slime.backends.megatron_utils.hf_to_megatron import nemotron_h as loader  # noqa: E402
from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader  # noqa: E402
from slime.backends.megatron_utils.megatron_to_hf.nemotron_h import convert_nemotron_h_to_hf  # noqa: E402

# The loader refuses TP > 1 by asking megatron's mpu, which a CPU box has no
# copy of. TP is 1 here by construction: nothing is sharded.
loader._assert_no_tensor_parallel = lambda: None


def _namespace(obj):
    """config.json -> attribute access, without importing transformers."""
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: _namespace(v) for k, v in obj.items()})
    return obj


def megatron_names(config, layer_types, experts_per_layer: int | None):
    """Every Megatron parameter name this stack holds, in checkpoint order.

    Built from the loader's own tables crossed with `layers_block_type`, because
    which names a layer has is decided by its type. Where several Megatron
    spellings map to one HF tensor -- the input norm has six -- one is picked per
    layer type, so a tensor is compared once rather than six times.
    """
    names = list(loader.TOP_LEVEL)

    norm_for = {
        "mamba": "mixer.in_proj.layer_norm_weight",
        "attention": "self_attention.linear_qkv.layer_norm_weight",
        "moe": "pre_mlp_layernorm.weight",
    }
    per_type = {
        "mamba": [
            "mixer.A_log",
            "mixer.D",
            "mixer.dt_bias",
            "mixer.norm.weight",
            "mixer.in_proj.weight",
            "mixer.out_proj.weight",
            "mixer.conv1d.weight",
            "mixer.conv1d.bias",
        ],
        "attention": ["self_attention.linear_qkv.weight", "self_attention.linear_proj.weight"],
        "moe": [
            "mlp.router.weight",
            "mlp.router.expert_bias",
            "mlp.fc1_latent_proj.weight",
            "mlp.fc2_latent_proj.weight",
            "mlp.shared_experts.linear_fc1.weight",
            "mlp.shared_experts.linear_fc2.weight",
        ],
    }

    num_experts = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", 0)
    for index, kind in enumerate(layer_types):
        names.append(f"decoder.layers.{index}.{norm_for[kind]}")
        names += [f"decoder.layers.{index}.{suffix}" for suffix in per_type[kind]]
        if kind == "moe" and num_experts:
            take = num_experts if experts_per_layer is None else min(experts_per_layer, num_experts)
            for expert in range(take):
                names.append(f"decoder.layers.{index}.mlp.experts.linear_fc1.weight{expert}")
                names.append(f"decoder.layers.{index}.mlp.experts.linear_fc2.weight{expert}")
    return names


def vision_names(config, num_blocks: int):
    names = list(loader._VISION_TOP_LEVEL)
    for block in range(num_blocks):
        names += [f"vision_model.encoder.layer.{block}.{suffix}" for suffix in loader._VISION_PER_LAYER]
        for which in ("query", "key", "value"):
            for suffix in ("weight", "bias"):
                names.append(f"vision_model.encoder.layer.{block}.attention.attention.{which}.{suffix}")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--experts-per-layer",
        type=int,
        default=2,
        help="routed experts to check per MoE layer (default 2; --full checks all)",
    )
    parser.add_argument("--full", action="store_true", help="check every routed expert -- reads the whole checkpoint")
    parser.add_argument("--max-report", type=int, default=25)
    args_cli = parser.parse_args()

    raw = json.loads((args_cli.hf_checkpoint / "config.json").read_text())
    config = _namespace(raw)
    language = getattr(config, "llm_config", config)
    is_vl = raw.get("model_type") == "nemotron_h_omni"
    model_name = "nemotronhomnireasoningv3config" if is_vl else "nemotronhconfig"

    # What the exporter reads off slime's args. Megatron's own names for the
    # same numbers the HF config carries.
    args = types.SimpleNamespace(
        kv_channels=getattr(language, "head_dim", None),
        hidden_size=language.hidden_size,
        num_attention_heads=language.num_attention_heads,
        num_query_groups=language.num_key_value_heads,
    )

    reader = SafetensorReader(args_cli.hf_checkpoint)
    hf_names = set(reader.weight_map)

    names = megatron_names(language, language.layers_block_type, None if args_cli.full else args_cli.experts_per_layer)
    if is_vl:
        names += vision_names(config, config.vision_config.num_hidden_layers)

    print(f"checkpoint      : {args_cli.hf_checkpoint}")
    print(f"model_type      : {raw.get('model_type')}  (VL={is_vl})")
    print(f"HF tensors      : {len(hf_names)}")
    print(f"megatron names  : {len(names)}"
          + ("" if args_cli.full else f"  (experts sampled: {args_cli.experts_per_layer}/layer -- --full for all)"))
    print()

    checked = skipped = 0
    mismatches: list[tuple[str, str, str]] = []
    load_errors: list[tuple[str, str]] = []
    covered: set[str] = set()

    for name in names:
        try:
            mega = loader.nemotron_h_hf_tensor(name, reader, config)
        except KeyError:
            # A name this stack does not actually hold for this config.
            skipped += 1
            continue
        except Exception as exc:  # noqa: BLE001
            load_errors.append((name, f"{type(exc).__name__}: {exc}"))
            continue

        try:
            pairs = convert_nemotron_h_to_hf(args, name, mega, model_name)
        except Exception as exc:  # noqa: BLE001
            load_errors.append((name, f"export {type(exc).__name__}: {exc}"))
            continue

        for hf_name, got in pairs:
            covered.add(hf_name)
            if hf_name not in hf_names:
                mismatches.append((name, hf_name, "exported a name the checkpoint does not have"))
                continue
            want = reader.get_tensor(hf_name)
            if got.shape != want.shape:
                mismatches.append((name, hf_name, f"shape {tuple(got.shape)} != {tuple(want.shape)}"))
            elif not torch.equal(got, want):
                delta = (got.float() - want.float()).abs()
                mismatches.append(
                    (name, hf_name, f"values differ: max |d| {delta.max().item():.3e}, "
                                    f"{int((delta > 0).sum())}/{delta.numel()} elements")
                )
            checked += 1

    print(f"round-tripped   : {checked} tensors")
    print(f"skipped         : {skipped} names this config does not hold")
    print(f"load/export err : {len(load_errors)}")
    print(f"MISMATCHES      : {len(mismatches)}")
    print()

    for name, hf_name, why in mismatches[: args_cli.max_report]:
        print(f"  MISMATCH {name}\n           -> {hf_name}: {why}")
    if len(mismatches) > args_cli.max_report:
        print(f"  ... and {len(mismatches) - args_cli.max_report} more")
    for name, why in load_errors[: args_cli.max_report]:
        print(f"  ERROR    {name}: {why}")

    if args_cli.full:
        # Only meaningful when nothing was sampled away.
        uncovered = sorted(hf_names - covered)
        print()
        print(f"HF tensors never produced by the round trip: {len(uncovered)}")
        for hf_name in uncovered[: args_cli.max_report]:
            print(f"  UNCOVERED {hf_name}")
        if len(uncovered) > args_cli.max_report:
            print(f"  ... and {len(uncovered) - args_cli.max_report} more")

    return 1 if (mismatches or load_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
