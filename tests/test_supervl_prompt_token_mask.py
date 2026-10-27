"""CPU rank-by-rank layout tests; these do not replace distributed GPU tests."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core.models.hybrid.hybrid_model")

from slime.backends.megatron_utils import data
from slime_plugins.models import nemotron_h_vl as vl
from test_supervl_image_inputs import inject

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("cp_size", [1, 2, 16])
@pytest.mark.parametrize("allgather", [False, True])
@pytest.mark.parametrize("zero_loss", [False, True])
def test_mask_tracks_tokens_across_packing_padding_and_cp(monkeypatch, cp_size, allgather, zero_loss):
    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data.accelerator, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data.accelerator, "current_device", lambda: torch.device("cpu"))
    # IDs uniquely encode prompt ownership, independent of the implementation's slicing.
    sequences = [torch.tensor([11, 12, 13, 51, 52]), torch.tensor([21, 22, 61])]
    prompt_ids = torch.tensor([11, 12, 13, 21, 22])
    raw = dict(
        tokens=sequences,
        total_lengths=[5, 3],
        response_lengths=[2, 1],
        loss_masks=[torch.full((2,), int(not zero_loss)), torch.full((1,), int(not zero_loss))],
    )
    seen = []
    for rank in range(cp_size):
        monkeypatch.setattr(data.mpu, "get_context_parallel_rank", lambda rank=rank: rank)

        def batch(enabled):
            return data.get_batch(
                data.DataIterator(raw, [[0, 1]]),
                list(raw),
                pad_multiplier=8,
                allgather_cp=allgather,
                build_prompt_token_mask=enabled,
            )

        result, disabled = batch(True), batch(False)
        assert "prompt_token_mask" not in disabled
        torch.testing.assert_close(result["tokens"], disabled["tokens"])
        torch.testing.assert_close(result["full_loss_masks"], disabled["full_loss_masks"])
        assert result["prompt_token_mask"].dtype == torch.bool
        torch.testing.assert_close(result["prompt_token_mask"], torch.isin(result["tokens"], prompt_ids))
        seen.extend(result["tokens"][result["prompt_token_mask"]].tolist())
    assert sorted(seen) == sorted(prompt_ids.tolist())


def test_generated_image_marker_keeps_text_embedding():
    ids = torch.tensor([[1, 7, 7, 2, 7]])
    mask = torch.tensor([[True, True, True, False, False]])
    result = inject([torch.ones(1, 3, 2, 2)], ids, mask)
    assert result[:, 0, 0].tolist() == [1, 2, 2, 2, 7]
    with pytest.raises(ValueError, match="3 <image> tokens, 2 projected features"):
        inject([torch.ones(1, 3, 2, 2)], ids)


@pytest.mark.parametrize("mask", [torch.ones(1, 3), torch.ones(1, 2, dtype=torch.bool)])
def test_invalid_prompt_mask_rejected(mask):
    with pytest.raises(ValueError, match="boolean and match"):
        inject([torch.ones(1, 3, 2, 2)], torch.tensor([[7, 7, 1]]), mask)


def test_mask_does_not_hide_missing_prompt_features():
    with pytest.raises(ValueError, match="1 <image> tokens, 2 projected features"):
        inject([torch.ones(1, 3, 2, 2)], torch.tensor([[7, 7, 1]]), torch.tensor([[True, False, False]]))


@pytest.mark.parametrize("cp_size", [2, 16])
def test_cp_scatter_uses_prompt_features_not_generated_markers(cp_size):
    # Two packed sequences, each with one prompt image and a response image marker.
    width = 4 * cp_size
    ids = torch.ones(1, 2 * width, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for start in [0, width]:
        ids[0, start + 1 : start + 3] = 7
        ids[0, start + width - 1] = 7
        mask[0, start : start + width // 2] = True
    images = [torch.full((1, 3, 2, 2), value) for value in [3.0, 5.0]]
    reference = inject(images, ids, mask)
    for rank in range(cp_size):
        # Independent expected two-chunk layout: two tokens from each end per sequence.
        positions = []
        for start in [0, width]:
            positions.extend(
                [start + 2 * rank, start + 2 * rank + 1, start + width - 2 * rank - 2, start + width - 2 * rank - 1]
            )
        group = SimpleNamespace(size=lambda: cp_size, rank=lambda rank=rank: rank)
        result = inject(
            images,
            ids[:, positions],
            mask,
            full_ids=ids,
            boundaries=torch.tensor([0, width, 2 * width]),
            cp_group=group,
        )
        torch.testing.assert_close(result, reference[positions])


@pytest.mark.parametrize("pre_process", [True, False])
def test_forward_consumes_mask_without_forwarding_to_language_model(monkeypatch, pre_process):
    ids = torch.tensor([[1, 7, 7, 2, 7]])
    mask = torch.tensor([[True, True, True, False, False]])
    gathered = []
    injected = []
    forwarded = {}

    def gather(value, boundaries, group):
        gathered.append(value)
        return value

    def inject_embeddings(*args):
        injected.append(args[-1])
        return torch.ones(5, 1, 1)

    def language_model(**kwargs):
        forwarded.update(kwargs)
        return torch.tensor(1.0)

    monkeypatch.setattr(vl.mpu, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(vl, "gather_packed_input_ids", gather)
    model = SimpleNamespace(
        pre_process=pre_process, language_model=language_model, _inject_vision_embeddings=inject_embeddings
    )
    output = vl.NemotronHVLModel.forward(
        model,
        ids,
        prompt_token_mask=mask,
        pixel_values=[torch.ones(1, 3, 2, 2)],
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 5])),
    )
    assert output.item() == 1.0
    assert "prompt_token_mask" not in forwarded
    assert forwarded["input_ids"] is ids
    if pre_process:
        assert len(gathered) == 2 and gathered[0] is ids and gathered[1] is mask
        assert len(injected) == 1 and injected[0] is mask
        assert forwarded["decoder_input"] is not None
    else:
        assert gathered == [] and injected == []
        assert forwarded["decoder_input"] is None
