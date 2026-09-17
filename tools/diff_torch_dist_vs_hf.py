#!/usr/bin/env python3
"""torch_dist -> HF, diffed against the HF checkpoint it was converted from.

WHY THIS EXISTS, AND HOW IT DIFFERS FROM `roundtrip_nemotron_h.py`
------------------------------------------------------------------
The RL arm's weights travel

    HF --[convert_hf_to_torch_dist]--> torch_dist --[training]--> Megatron
       --[convert_to_hf]--> SGLang

and the rollouts it produces are visibly worse than the same checkpoint served
straight from HF. Two tools bracket that chain, and the pair is the experiment:

    roundtrip_nemotron_h.py    HF -> Megatron -> HF, all in memory
                               exercises the two *converters* and nothing else
    this tool                  torch_dist -> HF, against the original HF
                               exercises the converters PLUS the on-disk
                               checkpoint: the sharding, the write, the read

So the reading is a subtraction:

    both clean          the weights survive the round trip; look elsewhere
    first clean, this   dirty in `convert_hf_to_torch_dist` or in torch_dist
      dirty             itself -- the layer the first tool cannot see
    both dirty          a converter, and the first tool names which tensor

It deliberately writes nothing. `convert_torch_dist_to_hf.py` exists and would
produce a full HF copy, but that is ~240 GB of scratch for this checkpoint and
the copy is not the artifact anyone wants -- the diff is. This reuses that
script's own loader and its `get_named_params` so the two cannot drift, converts
each parameter in turn, and compares it against the original immediately.

Comparison is exact. Every step between the two sides is a rename, a reshape or
a permutation; nothing rounds. A difference is a bug, not a tolerance question.

USAGE
-----
    python tools/diff_torch_dist_vs_hf.py \\
        --torch-dist-dir /.../MODEL_torch_dist_mcore9e68a11bd \\
        --origin-hf-dir  /.../MODEL

    --max-report N   how many differing tensors to print (default 25)
    --sample N       compare only every Nth tensor, for a fast first pass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_torch_dist_to_hf import (  # noqa: E402
    EmptyStateDictLoadPlanner,
    WrappedStorageReader,
    get_named_params,
)

from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader  # noqa: E402
from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf  # noqa: E402


def _resolve_iteration_dir(root: Path) -> Path:
    """Megatron writes `<root>/<iteration>/`, with the name in a pointer file.

    `release` for a converted checkpoint, `iter_0000007` for a trained one. The
    root itself is accepted too, so a path straight to the shards works.
    """
    if (root / ".metadata").is_file():
        return root
    pointer = root / "latest_checkpointed_iteration.txt"
    if not pointer.is_file():
        raise SystemExit(f"{root} has neither .metadata nor latest_checkpointed_iteration.txt")
    tag = pointer.read_text().strip()
    resolved = root / (tag if tag == "release" else f"iter_{int(tag):07d}")
    if not (resolved / ".metadata").is_file():
        raise SystemExit(f"{resolved} has no .metadata (pointer said {tag!r})")
    return resolved


def _args_from_hf_config(hf_dir: Path, vocab_size_override: int | None):
    """The six scalars the conversion path reads off megatron's args.

    Enumerated rather than guessed: `get_expert_param` wants `num_experts`,
    `get_layer_param` wants `num_layers`, `convert_to_hf` wants `vocab_size` and
    `q_lora_rank`, and the nemotron exporter wants `hidden_size`,
    `kv_channels`, `num_attention_heads` and `num_query_groups`. Every one of
    them is a property of the model, so the HF config is as authoritative a
    source as a serialized `args` namespace would be.
    """
    raw = json.loads((hf_dir / "config.json").read_text())
    language = raw.get("llm_config", raw)
    num_layers = language.get("num_hidden_layers") or len(language["layers_block_type"])
    vocab_size = language["vocab_size"] if vocab_size_override is None else vocab_size_override
    return types.SimpleNamespace(
        num_layers=num_layers,
        num_experts=language.get("n_routed_experts") or language.get("num_experts"),
        vocab_size=vocab_size,
        q_lora_rank=None,
        hidden_size=language["hidden_size"],
        kv_channels=language.get("head_dim"),
        num_attention_heads=language["num_attention_heads"],
        num_query_groups=language["num_key_value_heads"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--torch-dist-dir",
        required=True,
        type=Path,
        help="the checkpoint root; the iteration sub-directory is resolved from "
        "latest_checkpointed_iteration.txt, or pass it directly",
    )
    parser.add_argument("--origin-hf-dir", required=True, type=Path)
    parser.add_argument("--model-name", default=None, help="defaults to the HF config class name, as the converter does")
    parser.add_argument("--vocab-size", type=int, default=None, help="defaults to the HF config's; pass 0 to skip padding removal")
    parser.add_argument("--max-report", type=int, default=25)
    parser.add_argument("--sample", type=int, default=1, help="compare every Nth converted tensor (1 = all)")
    cli = parser.parse_args()

    model_name = cli.model_name
    if model_name is None:
        hf_config = AutoConfig.from_pretrained(str(cli.origin_hf_dir), trust_remote_code=True)
        model_name = type(hf_config).__name__.lower()

    print(f"torch_dist      : {cli.torch_dist_dir}")
    print(f"origin HF       : {cli.origin_hf_dir}")
    print(f"model_name      : {model_name}")
    print(f"sample          : every {cli.sample}")
    print(flush=True)

    reader = SafetensorReader(cli.origin_hf_dir)
    hf_names = set(reader.weight_map)
    print(f"HF tensors      : {len(hf_names)}", flush=True)

    # Exactly `convert_torch_dist_to_hf.py`'s load, reusing its reader and
    # planner so a fix there reaches here.
    checkpoint_dir = _resolve_iteration_dir(cli.torch_dist_dir)
    print(f"reading         : {checkpoint_dir}")

    common = checkpoint_dir / "common.pt"
    if common.is_file():
        megatron_args = torch.load(common, weights_only=False)["args"]
        print("megatron args   : from common.pt")
    else:
        # `convert_hf_to_torch_dist` does not write one, so the tool that reads
        # its output cannot require it. Everything the conversion path actually
        # touches -- six scalars, enumerated in `_args_from_hf_config` -- is in
        # the HF config, and taking them from there is the same numbers by a
        # shorter route.
        megatron_args = _args_from_hf_config(cli.origin_hf_dir, cli.vocab_size)
        print(f"megatron args   : derived from the HF config (no common.pt in {checkpoint_dir.name}/)")
    print(
        f"                  num_layers={megatron_args.num_layers} num_experts={megatron_args.num_experts} "
        f"vocab_size={megatron_args.vocab_size} kv_channels={megatron_args.kv_channels}"
    )

    started = time.time()
    print(f"loading {checkpoint_dir} ...", flush=True)
    state_dict: dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(str(checkpoint_dir)),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"loaded in {time.time() - started:.1f}s, {len(state_dict)} top-level entries", flush=True)
    print(flush=True)

    checked = 0
    index = 0
    mismatches: list[tuple[str, str, str]] = []
    errors: list[tuple[str, str]] = []
    produced: set[str] = set()

    # `convert_to_hf` already applies remove_padding from megatron_args.
    for name, param in get_named_params(megatron_args, state_dict):
        try:
            pairs = convert_to_hf(megatron_args, model_name, name, param)
        except Exception as exc:  # noqa: BLE001
            errors.append((name, f"{type(exc).__name__}: {exc}"))
            continue

        for hf_name, got in pairs:
            produced.add(hf_name)
            index += 1
            if cli.sample > 1 and index % cli.sample:
                continue
            if hf_name not in hf_names:
                mismatches.append((name, hf_name, "not in the origin HF checkpoint"))
                continue
            want = reader.get_tensor(hf_name)
            if got.shape != want.shape:
                mismatches.append((name, hf_name, f"shape {tuple(got.shape)} != {tuple(want.shape)}"))
            elif got.dtype != want.dtype:
                mismatches.append((name, hf_name, f"dtype {got.dtype} != {want.dtype}"))
            elif not torch.equal(got, want):
                delta = (got.float() - want.float()).abs()
                differing = int((delta > 0).sum())
                mismatches.append(
                    (
                        name,
                        hf_name,
                        f"values differ: max |d| {delta.max().item():.3e}, mean |d| {delta.mean().item():.3e}, "
                        f"{differing}/{delta.numel()} elements",
                    )
                )
            checked += 1
            if checked % 2000 == 0:
                print(f"  ... {checked} compared, {len(mismatches)} mismatches so far", flush=True)

    print()
    print(f"compared        : {checked} tensors")
    print(f"produced        : {len(produced)} distinct HF names")
    print(f"convert errors  : {len(errors)}")
    print(f"MISMATCHES      : {len(mismatches)}")
    print()

    for name, hf_name, why in mismatches[: cli.max_report]:
        print(f"  MISMATCH {name}\n           -> {hf_name}: {why}")
    if len(mismatches) > cli.max_report:
        print(f"  ... and {len(mismatches) - cli.max_report} more")
    for name, why in errors[: cli.max_report]:
        print(f"  ERROR    {name}: {why}")

    missing = sorted(hf_names - produced)
    print()
    print(f"HF tensors the export never produced: {len(missing)}")
    for hf_name in missing[: cli.max_report]:
        print(f"  MISSING  {hf_name}")
    if len(missing) > cli.max_report:
        print(f"  ... and {len(missing) - cli.max_report} more")

    return 1 if (mismatches or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
