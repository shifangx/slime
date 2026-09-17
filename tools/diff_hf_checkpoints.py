#!/usr/bin/env python3
"""Compare two HuggingFace checkpoints tensor by tensor.

Written for one question: an export produced by
`HF -> convert_hf_to_torch_dist -> torch_dist -> convert_to_hf -> HF` should be
the checkpoint it started from. Every step is a rename, a reshape or a
permutation; nothing in that path rounds, casts or reduces. So the two
directories should agree bit for bit, and any element that does not is a bug
with a name attached.

WHY BIT-EXACT IS THE RIGHT BAR, AND WHAT A NEAR-MISS WOULD MEAN
---------------------------------------------------------------
There is no arithmetic in the chain, so "close enough" is not a defensible
outcome -- but the *shape* of a near-miss is still informative and the report
distinguishes them:

  identical                 the chain preserves the tensor
  differs in the last bits  something cast through a narrower dtype and back
  differs structurally      a permutation or a packing order is wrong; the
                            values are all present and in the wrong places
  differs entirely          the wrong tensor is under this name

`max |d|` alone cannot tell those apart, so each mismatch also reports how many
elements moved and the largest relative error.

READING 43,078 MISMATCHES
-------------------------
Listing them would be useless, so they are grouped by a normalised name --
layer and expert indices collapsed to `N` -- and each group reports its count,
its worst delta and one example. "every `mixer.in_proj.weight`, 40 of 40 layers"
is a finding; four hundred individual lines are not.

USAGE
-----
    python tools/diff_hf_checkpoints.py --a /path/to/original --b /path/to/export
    python tools/diff_hf_checkpoints.py --a ... --b ... --sample 20
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

_INDEX = re.compile(r"\.(\d+)\.")
_EXPERT = re.compile(r"\.experts\.(\d+)\.")


def normalise(name: str) -> str:
    """Collapse layer and expert indices so one defect is one line."""
    return _INDEX.sub(".N.", _EXPERT.sub(".experts.N.", name))


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        index = path / "model.safetensors.index.json"
        if index.is_file():
            import json

            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            self.weight_map = {}
            for shard in sorted(path.glob("*.safetensors")):
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    self.weight_map.update(dict.fromkeys(handle.keys(), shard.name))
        self._open: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = safe_open(self.path / shard, framework="pt", device="cpu")
        return self._open[shard].get_tensor(name)


def describe(a: torch.Tensor, b: torch.Tensor) -> str:
    """One line saying not just how big the difference is but what shape it has."""
    delta = (a.float() - b.float()).abs()
    moved = int((delta > 0).sum())
    total = delta.numel()
    largest = delta.max().item()
    scale = a.float().abs().max().item()
    relative = largest / scale if scale else float("inf")
    if moved == total:
        kind = "every element"
    elif moved > total // 2:
        kind = f"{100 * moved // total}% of elements"
    else:
        kind = f"{moved}/{total} elements"
    return f"max |d| {largest:.3e} (rel {relative:.2e}), {kind} moved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, type=Path, help="the original checkpoint")
    parser.add_argument("--b", required=True, type=Path, help="the one to check against it")
    parser.add_argument("--sample", type=int, default=1, help="compare every Nth shared tensor (1 = all)")
    parser.add_argument("--max-groups", type=int, default=40)
    cli = parser.parse_args()

    a, b = Checkpoint(cli.a), Checkpoint(cli.b)
    names_a, names_b = set(a.weight_map), set(b.weight_map)
    shared = sorted(names_a & names_b)

    print(f"a (original)  : {cli.a}")
    print(f"b (to check)  : {cli.b}")
    print(f"tensors       : {len(names_a)} vs {len(names_b)}, {len(shared)} shared")
    if names_a - names_b:
        print(f"only in a     : {len(names_a - names_b)}  e.g. {sorted(names_a - names_b)[:3]}")
    if names_b - names_a:
        print(f"only in b     : {len(names_b - names_a)}  e.g. {sorted(names_b - names_a)[:3]}")
    print(f"sample        : every {cli.sample}")
    print(flush=True)

    identical = 0
    groups: dict[str, dict] = defaultdict(lambda: {"count": 0, "worst": -1.0, "example": None, "why": ""})
    started = time.time()

    for position, name in enumerate(shared):
        if cli.sample > 1 and position % cli.sample:
            continue
        left, right = a.get(name), b.get(name)

        if left.shape != right.shape:
            why = f"shape {tuple(left.shape)} != {tuple(right.shape)}"
            worst = float("inf")
        elif left.dtype != right.dtype:
            why = f"dtype {left.dtype} != {right.dtype}"
            worst = float("inf")
        elif torch.equal(left, right):
            identical += 1
            continue
        else:
            why = describe(left, right)
            worst = (left.float() - right.float()).abs().max().item()

        key = normalise(name)
        group = groups[key]
        group["count"] += 1
        if worst > group["worst"]:
            group["worst"], group["example"], group["why"] = worst, name, why

        if (identical + sum(g["count"] for g in groups.values())) % 5000 == 0:
            done = identical + sum(g["count"] for g in groups.values())
            print(f"  ... {done} compared, {sum(g['count'] for g in groups.values())} differing", flush=True)

    differing = sum(group["count"] for group in groups.values())
    print()
    print(f"compared      : {identical + differing} tensors in {time.time() - started:.0f}s")
    print(f"identical     : {identical}")
    print(f"DIFFERING     : {differing}, in {len(groups)} distinct parameter shapes")
    print()

    for key, group in sorted(groups.items(), key=lambda item: -item[1]["count"])[: cli.max_groups]:
        print(f"  {group['count']:6d}x  {key}")
        print(f"          worst: {group['example']}")
        print(f"                 {group['why']}")
    if len(groups) > cli.max_groups:
        print(f"  ... and {len(groups) - cli.max_groups} more shapes")

    return 1 if (differing or names_a != names_b) else 0


if __name__ == "__main__":
    sys.exit(main())
