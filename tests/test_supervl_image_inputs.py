"""Vision input regression tests using the real plugin and a differentiable projector."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core.models.hybrid.hybrid_model")

from slime_plugins.models.nemotron_h_vl import NemotronHVLModel

NUM_GPUS = 0
pytestmark = pytest.mark.unit


class Projector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, image, vision_model):
        assert image.ndim == 4, "The vision encoder requires NCHW"
        return image.mean(dim=(1, 2)).unsqueeze(-1) * self.scale


def inject(images, ids, mask=None, projector=None, *, full_ids=None, boundaries=None, cp_group=None):
    model = SimpleNamespace(
        language_model=SimpleNamespace(embedding=lambda input_ids, position_ids: input_ids.T.unsqueeze(-1).float()),
        vision_projector=projector if projector is not None else Projector(),
        vision_model=None,
        img_context_token_id=7,
        config=SimpleNamespace(sequence_parallel=False),
    )
    full_ids = ids if full_ids is None else full_ids
    boundaries = torch.tensor([0, full_ids.numel()]) if boundaries is None else boundaries
    return NemotronHVLModel._inject_vision_embeddings(model, ids, full_ids, boundaries, cp_group, images, mask)


@pytest.mark.parametrize("container", [list, tuple])
def test_ragged_images_match_batched_inputs_and_gradients(container):
    ids = torch.tensor([[1, 7, 7, 2, 7, 7, 7, 3]])
    images = [torch.full((3, 2, 2), 3.0, requires_grad=True), torch.full((3, 4, 3), 5.0, requires_grad=True)]
    batched = [image.detach().unsqueeze(0).requires_grad_() for image in images]
    first, second = Projector(), Projector()
    output = inject(container(images), ids, projector=first)
    reference = inject(container(batched), ids, projector=second)
    torch.testing.assert_close(output, reference)
    assert output[:, 0, 0].tolist() == [1, 6, 6, 2, 10, 10, 10, 3]
    output.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(first.scale.grad, second.scale.grad)
    for image, batch in zip(images, batched, strict=True):
        torch.testing.assert_close(image.grad, batch.grad.squeeze(0))
        assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


def test_single_tensor_and_singleton_list_are_equivalent():
    image = torch.ones(1, 3, 2, 2)
    ids = torch.tensor([[7, 7, 1]])
    torch.testing.assert_close(inject(image, ids), inject([image], ids))


@pytest.mark.parametrize("shape", [(2, 2), (1, 1, 3, 2, 2)])
def test_invalid_image_rank_is_rejected(shape):
    with pytest.raises(ValueError, match="CHW or NCHW"):
        inject([torch.ones(shape)], torch.tensor([[7, 7]]))


def test_image_count_mismatch_is_not_silenced():
    with pytest.raises(ValueError, match="3 <image> tokens, 2 projected features"):
        inject([torch.ones(3, 2, 2)], torch.tensor([[7, 7, 7]]))
