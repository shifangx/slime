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
