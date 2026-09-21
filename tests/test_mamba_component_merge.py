"""`merge_mamba_components` -- reassembling a torch_dist checkpoint's Mamba tensors.

Megatron's dist-checkpoint stores the Mamba mixers' packed tensors one entry per
component, named by it (`mixer.in_proj.weight.z`, `.x`, `.B`, `.C`, `.dt`),
where the model holds a single packed tensor. Nothing downstream knows those
names, so without the merge every Mamba parameter of a torch_dist checkpoint
raises `Unknown parameter name` in the exporter -- 440 of them on the
Nemotron 3.5 Super VL checkpoint, which is how this was found.

The ordering is the part worth a test. `[z, x, B, C, dt]` and `[x, B, C]` are
what `update_weight/common.py::merge_tp_partitions` and
`megatron_to_hf/nemotron_h.py` both document, and getting them wrong produces a
tensor of the right shape holding the wrong thing.
"""

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from convert_torch_dist_to_hf import MAMBA_PACKED_COMPONENTS, merge_mamba_components  # noqa: E402

PREFIX = "language_model.decoder.layers.0.mixer"


@pytest.mark.unit
def test_mamba_tp_gather_preserves_component_order():
    pytest.importorskip("megatron.core")
    from slime.backends.megatron_utils.update_weight.common import merge_tp_partitions

    shards = [torch.tensor([0, 1, 10, 20]), torch.tensor([2, 3, 11, 21])]
    actual = merge_tp_partitions(Namespace(), "mixer.conv1d.bias", shards, 0, [2, 1, 1])
    assert torch.equal(actual, torch.tensor([0, 1, 2, 3, 10, 11, 20, 21]))


@pytest.mark.unit
def test_ungated_fc1_gather_does_not_reorder_rows():
    pytest.importorskip("megatron.core")
    from slime.backends.megatron_utils.update_weight.common import merge_tp_partitions

    shards = [torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])]
    actual = merge_tp_partitions(Namespace(squared_relu=True), "linear_fc1.weight", shards, 0)
    assert torch.equal(actual, torch.arange(8))
    gated = merge_tp_partitions(Namespace(swiglu=True), "linear_fc1.weight", shards, 0)
    assert torch.equal(gated, torch.tensor([0, 1, 4, 5, 2, 3, 6, 7]))


def _component(tag: int, width: int = 4) -> torch.Tensor:
    return torch.full((width, 2), float(tag))


@pytest.mark.unit
def test_in_proj_concatenates_in_z_x_b_c_dt_order():
    parts = {name: _component(index) for index, name in enumerate(("z", "x", "B", "C", "dt"))}
    state_dict = {f"{PREFIX}.in_proj.weight.{name}": tensor for name, tensor in parts.items()}

    merged = merge_mamba_components(state_dict)

    assert list(merged) == [f"{PREFIX}.in_proj.weight"]
    expected = torch.cat([parts[name] for name in ("z", "x", "B", "C", "dt")], dim=0)
    assert torch.equal(merged[f"{PREFIX}.in_proj.weight"], expected)
    # The order is the point: a contiguous concat in dict order would pass the
    # shape check and fail this one.
    assert merged[f"{PREFIX}.in_proj.weight"][0, 0] == 0  # z
    assert merged[f"{PREFIX}.in_proj.weight"][-1, 0] == 4  # dt


@pytest.mark.unit
def test_conv1d_weight_and_bias_use_x_b_c():
    state_dict = {}
    for suffix in ("weight", "bias"):
        for index, name in enumerate(("x", "B", "C")):
            state_dict[f"{PREFIX}.conv1d.{suffix}.{name}"] = _component(index)

    merged = merge_mamba_components(state_dict)

    assert sorted(merged) == [f"{PREFIX}.conv1d.bias", f"{PREFIX}.conv1d.weight"]
    for suffix in ("weight", "bias"):
        packed = merged[f"{PREFIX}.conv1d.{suffix}"]
        assert packed.shape == (12, 2)
        assert [packed[row, 0].item() for row in (0, 4, 8)] == [0.0, 1.0, 2.0]


@pytest.mark.unit
def test_everything_else_passes_through_untouched():
    """Including a tensor whose last segment looks like a component name."""
    state_dict = {
        f"{PREFIX}.A_log": _component(7),
        f"{PREFIX}.D": _component(8),
        f"{PREFIX}.dt_bias": _component(9),
        "language_model.decoder.layers.1.self_attention.linear_qkv.weight": _component(10),
        # Already packed -- a checkpoint that does not split them must survive.
        f"{PREFIX}.in_proj.weight": _component(11),
        f"{PREFIX}.conv1d.weight": _component(12),
    }

    merged = merge_mamba_components(state_dict)

    assert merged.keys() == state_dict.keys()
    for name, tensor in state_dict.items():
        assert torch.equal(merged[name], tensor)


@pytest.mark.unit
def test_an_incomplete_group_raises_rather_than_concatenating_what_it_has():
    state_dict = {f"{PREFIX}.in_proj.weight.{name}": _component(0) for name in ("z", "x", "B", "C")}

    with pytest.raises(ValueError, match=r"missing \['dt'\]"):
        merge_mamba_components(state_dict)


@pytest.mark.unit
def test_an_unexpected_component_raises():
    state_dict = {f"{PREFIX}.conv1d.weight.{name}": _component(0) for name in ("x", "B", "C")}
    state_dict[f"{PREFIX}.conv1d.weight.dt"] = _component(1)

    with pytest.raises(ValueError, match=r"unexpected \['dt'\]"):
        merge_mamba_components(state_dict)


@pytest.mark.unit
def test_the_documented_orders_are_the_ones_in_the_table():
    """A guard on the constant itself, since two other modules restate it."""
    assert MAMBA_PACKED_COMPONENTS["in_proj.weight"] == ("z", "x", "B", "C", "dt")
    assert MAMBA_PACKED_COMPONENTS["conv1d.weight"] == ("x", "B", "C")
    assert MAMBA_PACKED_COMPONENTS["conv1d.bias"] == ("x", "B", "C")
