"""`_concat_multimodal_field` -- joining one processor field across a micro-batch.

The function exists because dynamic-resolution image processors do not return
one uniform tensor per sample. Nemotron 3.5 Super VL's resizes every image to
its own aspect-preserving grid, so what arrives per sample is a tensor when that
sample's images happen to agree and a list when they do not -- and the join has
to cope with every combination of the two.

Job 18824653 is why the tensor/tensor case is tested at all: it got through the
weight sync and its first rollout and then died on the first micro-batch of
`actor_train` with

    RuntimeError: Sizes of tensors must match except in dimension 0.
    Expected size 480 but got size 384 for tensor number 1 in the list.

Two samples, each internally uniform, disagreeing with each other.
"""

import pytest
import torch

pytest.importorskip("megatron.core")

from slime.backends.megatron_utils.data import _concat_multimodal_field  # noqa: E402


def _images(count: int, size: int) -> torch.Tensor:
    """One sample's stacked `pixel_values`: (images, 3, size, size)."""
    return torch.arange(count * 3 * size * size, dtype=torch.float32).reshape(count, 3, size, size)


@pytest.mark.unit
def test_same_shape_tensors_still_cat():
    """The fixed-resolution path, unchanged -- this is the common case."""
    left, right = _images(2, 224), _images(3, 224)
    joined = _concat_multimodal_field(left, right)

    assert isinstance(joined, torch.Tensor)
    assert joined.shape == (5, 3, 224, 224)
    assert torch.equal(joined[:2], left)
    assert torch.equal(joined[2:], right)


@pytest.mark.unit
def test_different_shape_tensors_degrade_to_the_per_image_list():
    """Job 18824653's failure: a `cat` that cannot mean what it says.

    The result is the list form on purpose. `project_images` iterates it and
    projects each image on its own, which is the only thing that can work when
    the images have different token counts.
    """
    left, right = _images(2, 480), _images(1, 384)
    joined = _concat_multimodal_field(left, right)

    assert isinstance(joined, list)
    assert [tuple(image.shape) for image in joined] == [(3, 480, 480), (3, 480, 480), (3, 384, 384)]
    assert torch.equal(joined[0], left[0])
    assert torch.equal(joined[2], right[0])


@pytest.mark.unit
def test_mixed_tensor_and_list_join_in_order():
    """A sample that stacked next to one that could not, both ways round."""
    stacked = _images(2, 448)
    listed = [torch.zeros(3, 336, 336), torch.ones(3, 224, 224)]

    forward = _concat_multimodal_field(stacked, listed)
    assert [tuple(image.shape) for image in forward] == [(3, 448, 448), (3, 448, 448), (3, 336, 336), (3, 224, 224)]

    backward = _concat_multimodal_field(listed, stacked)
    assert [tuple(image.shape) for image in backward] == [(3, 336, 336), (3, 224, 224), (3, 448, 448), (3, 448, 448)]


@pytest.mark.unit
def test_plain_lists_concatenate():
    """`num_patches` / `num_tokens` come back as plain Python lists."""
    assert _concat_multimodal_field([4, 9], [16]) == [4, 9, 16]
    assert _concat_multimodal_field((4,), [9, 16]) == [4, 9, 16]


@pytest.mark.unit
def test_one_dimensional_fields_still_cat():
    """Trailing shapes agree trivially for 1-D tensors, so they must not degrade."""
    joined = _concat_multimodal_field(torch.tensor([4, 9]), torch.tensor([16]))

    assert isinstance(joined, torch.Tensor)
    assert torch.equal(joined, torch.tensor([4, 9, 16]))


@pytest.mark.unit
def test_ambiguous_rank_is_an_error_rather_than_a_guess():
    """A bare `(3, H, W)` next to a stack has no unambiguous image axis.

    `list()` on the 3-D one would yield 3 tensors of shape `(H, W)` -- which the
    projector would accept without complaint and turn into silent nonsense. No
    processor observed produces this, so it raises instead.
    """
    with pytest.raises(ValueError, match="ambiguous"):
        _concat_multimodal_field(_images(2, 480), torch.zeros(3, 384, 384))


@pytest.mark.unit
def test_unhandled_types_still_raise():
    with pytest.raises(TypeError):
        _concat_multimodal_field(torch.zeros(1), "not a processor output")
