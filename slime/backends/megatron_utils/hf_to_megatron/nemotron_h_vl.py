"""HF -> Megatron tensors for Nemotron 3.5 Super VL (``nemotron_h_omni``).

A dispatcher, not a second converter. The decoder is the same Nemotron-H stack
``nemotron_h.py`` already handles; the only things this file owns are the
prefix the omni checkpoint puts in front of it and the handful of vision keys:

    Megatron parameter                         HF checkpoint key
    ---------------------------------------    -------------------------------
    language_model.<anything>                  language_model.<nemotron_h name>
    vision_model.<anything>                    vision_model.<same>
    vision_projector.mlp1.norm.weight          mlp1.0.weight
    vision_projector.mlp1.linear1.weight       mlp1.1.weight
    vision_projector.mlp1.linear2.weight       mlp1.3.weight
    vision_projector.vision_final_layernorm.*  vision_projector.vision_final_layernorm.*

The three ``mlp1`` renames are not invented here: they are the checkpoint's own
conversion mapping, registered by ``register_nemotron_h_omni_conversion_mapping``
in its ``modeling_nemotron_h_omni.py``. The projector is built from that same
remote code (see ``slime_plugins/models/nemotron_h_vl.py``), so its parameter
names are the reference's and the mapping has to be too.

The vision tower and projector are replicated across TP ranks rather than
sharded, so their tensors pass through whole.
"""

from __future__ import annotations

import copy
import re

import torch

from .common import SafetensorReader, strip_mcore_wrappers
from .nemotron_h import nemotron_h_hf_tensor

# nn.Sequential indices in the checkpoint -> the reference module's attribute
# names. Index 2 is the activation and carries no weights.
_MLP1_RENAMES = {
    "mlp1.norm": "mlp1.0",
    "mlp1.linear1": "mlp1.1",
    "mlp1.linear2": "mlp1.3",
}

_BLOCK_TYPE_SYMBOLS = {"mamba": "M", "moe": "E", "attention": "*"}


def _language_config(hf_config):
    """The decoder's config, with the hybrid pattern the text converter expects.

    ``nemotron_h_hf_tensor`` reads ``hf_config.hybrid_override_pattern`` to know
    what each layer index is. Nemotron 3's config spells that out; the 3.5 omni
    config nests the language model and spells it as ``layers_block_type``
    instead, so derive it rather than requiring the caller to pass a literal.
    """
    llm_config = getattr(hf_config, "llm_config", hf_config)
    if getattr(llm_config, "hybrid_override_pattern", None):
        return llm_config

    block_types = getattr(llm_config, "layers_block_type", None)
    if not block_types:
        raise KeyError(
            "Nemotron omni llm_config has neither hybrid_override_pattern nor layers_block_type"
        )
    try:
        pattern = "".join(_BLOCK_TYPE_SYMBOLS[block] for block in block_types)
    except KeyError as exc:
        raise KeyError(f"unknown layers_block_type entry {exc.args[0]!r}") from exc

    # Do not mutate the caller's config object: it is shared with the model
    # provider, which derives the same pattern for MCore and would then disagree
    # about where it came from.
    llm_config = copy.copy(llm_config)
    llm_config.hybrid_override_pattern = pattern
    return llm_config


class _PrefixedReader:
    """A SafetensorReader view that prepends a fixed prefix to every key."""

    def __init__(self, reader: SafetensorReader, prefix: str) -> None:
        self._reader = reader
        self._prefix = prefix

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._reader.get_tensor(f"{self._prefix}{name}")

    def __getattr__(self, item):
        return getattr(self._reader, item)


def nemotron_h_vl_hf_tensor(name: str, reader: SafetensorReader, hf_config) -> torch.Tensor:
    """Return the full, unsharded MCore tensor for a Nemotron 3.5 VL parameter."""

    # Note this also removes the `language_model.` the VL module puts in front of
    # the decoder (common.py:60-63), so the decoder arrives here already looking
    # like a plain Nemotron-H name. The vision branches are therefore the ones
    # that have to be recognized, and the decoder is what is left over.
    name = strip_mcore_wrappers(name)

    if name.startswith("vision_model."):
        return reader.get_tensor(name)

    if name.startswith("vision_projector."):
        rest = name[len("vision_projector.") :]
        if rest.startswith("vision_final_layernorm."):
            return reader.get_tensor(name)
        mlp1_match = re.fullmatch(r"(mlp1\.(?:norm|linear1|linear2))\.(weight|bias)", rest)
        if mlp1_match:
            module, suffix = mlp1_match.groups()
            return reader.get_tensor(f"{_MLP1_RENAMES[module]}.{suffix}")
        raise KeyError(f"Unsupported Nemotron 3.5 VL projector parameter {name!r}")

    # The decoder. Its tensors live one level down in the omni checkpoint, under
    # `language_model.`, which is exactly the prefix strip_mcore_wrappers just
    # took off the Megatron side -- so it goes back on for the lookup.
    return nemotron_h_hf_tensor(
        name,
        _PrefixedReader(reader, "language_model."),
        _language_config(hf_config),
    )
