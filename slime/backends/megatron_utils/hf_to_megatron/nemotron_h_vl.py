"""HF -> Megatron tensors for Nemotron 3.5 Super VL (``nemotron_h_omni``).

A dispatcher, not a second converter. The decoder is the same Nemotron-H stack
``nemotron_h.py`` already handles; the only things this file owns are the
prefix the omni checkpoint puts in front of it and the handful of vision keys:

    Megatron parameter                         HF checkpoint key
    ---------------------------------------    -------------------------------
    language_model.<anything>                  language_model.<nemotron_h name>
    vision_model.<anything>                    see _VISION_RENAMES
    vision_projector.mlp1.norm.weight          mlp1.0.weight
    vision_projector.mlp1.linear1.weight       mlp1.1.weight
    vision_projector.mlp1.linear2.weight       mlp1.3.weight
    vision_projector.vision_final_layernorm.*  vision_projector.vision_final_layernorm.*

The three ``mlp1`` renames are not invented here: they are the checkpoint's own
conversion mapping, registered by ``register_nemotron_h_omni_conversion_mapping``
in its ``modeling_nemotron_h_omni.py``. The projector is built from that same
remote code (see ``slime_plugins/models/nemotron_h_vl.py``), so its parameter
names are the reference's and the mapping has to be too.

The same is true, and less obviously so, of the vision tower: it is
``modeling_radio.RadioModel``, whose module tree is *not* the one the released
weights use. The checkpoint is timm-style C-RADIO --
``radio_model.model.blocks.{N}.attn.qkv`` with a fused qkv -- and
``RadioModel.__init__`` calls ``register_radio_conversion_mapping()`` to rewrite
it on load. Treating the vision half as a pass-through is what killed job
19049784 on the first tensor it reached:

    KeyError: HuggingFace checkpoint does not contain
              'vision_model.embeddings.position_embedding'

_VISION_RENAMES below is that registered mapping, inverted. Nothing here is
guessed; see modeling_radio.py:43-75 for the forward form of every line.

The vision tower and projector are replicated across TP ranks rather than
sharded, so their tensors pass through whole.
"""

from __future__ import annotations

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

# RadioModel's module tree -> the released C-RADIO checkpoint's, i.e. every
# WeightRenaming in register_radio_conversion_mapping() read right-to-left.
# Applied as an ordered prefix substitution on the name below `vision_model.`,
# because that is how transformers applies the forward direction.
#
# `summary_idxs` is deliberately absent: the mapping renames
# `radio_model.summary_idxs` -> `summary_idxs`, but this checkpoint already
# stores it at `vision_model.summary_idxs`, so for these weights that rule is a
# no-op and the name passes through.
_VISION_RENAMES = (
    ("embeddings.video_patch_projection", "radio_model.model.patch_generator.video_embedder"),
    ("embeddings.patch_projection", "radio_model.model.patch_generator.embedder"),
    ("embeddings.position_embedding", "radio_model.model.patch_generator.pos_embed"),
    ("embeddings.cls_register_token", "radio_model.model.patch_generator.cls_token.token"),
    ("encoder.layer", "radio_model.model.blocks"),
    ("input_conditioner", "radio_model.input_conditioner"),
)

# Inside a block, the attention rewrite. norm1/norm2/mlp.fc1/mlp.fc2 are
# unrenamed on purpose -- RadioLayer spells them the same way timm does
# (modeling_radio.py:370-381), which is why the registered mapping lists no
# rule for them.
_VISION_BLOCK_RENAMES = (("attention.output.dense", "attn.proj"),)


def _vision_config(hf_config):
    vision_config = getattr(hf_config, "vision_config", None)
    if vision_config is None:
        raise KeyError("Nemotron omni config has no vision_config")
    return vision_config


def _vision_tensor(rest: str, reader: SafetensorReader, hf_config) -> torch.Tensor:
    """Resolve one RadioModel parameter name against the C-RADIO checkpoint."""

    # LayerScale is the one parameter with no checkpoint counterpart at all.
    # C-RADIO ViT-H has no layerscale, but RadioLayer builds one unconditionally
    # (modeling_radio.py:372,381) as `layerscale_value * ones(hidden_size)`, and
    # layerscale_value is 1.0 here -- so `hidden_state * lambda1` is the
    # identity and the reference model loads with these as missing keys left at
    # their init value. Synthesising that value is what keeps the two models
    # numerically equal; raising here, or loading zeros, would not.
    if re.fullmatch(r"encoder\.layer\.\d+\.layer_scale[12]\.lambda1", rest):
        vision_config = _vision_config(hf_config)
        return torch.full(
            (vision_config.hidden_size,),
            float(vision_config.layerscale_value),
            dtype=torch.float32,
        )

    for source, target in _VISION_RENAMES:
        if rest == source or rest.startswith(f"{source}."):
            rest = target + rest[len(source) :]
            break

    block_match = re.fullmatch(r"(radio_model\.model\.blocks\.\d+\.)(.+)", rest)
    if block_match:
        block, inner = block_match.groups()
        # Fused qkv: one checkpoint tensor feeds three Megatron parameters, so
        # this is a Chunk(dim=0) rather than a rename. Both .weight and .bias
        # split the same way (qkv_bias is true for this tower).
        qkv_match = re.fullmatch(r"attention\.attention\.(query|key|value)\.(weight|bias)", inner)
        if qkv_match:
            component, suffix = qkv_match.groups()
            fused = reader.get_tensor(f"vision_model.{block}attn.qkv.{suffix}")
            chunks = torch.chunk(fused, 3, dim=0)
            return chunks[("query", "key", "value").index(component)].contiguous()

        for source, target in _VISION_BLOCK_RENAMES:
            if inner == source or inner.startswith(f"{source}."):
                inner = target + inner[len(source) :]
                break
        rest = block + inner

    return reader.get_tensor(f"vision_model.{rest}")


def _mtp_suffix(llm_config) -> str:
    """The ``/``-separated MTP segments for this checkpoint, or ``""``.

    Derived from the checkpoint, not from ``args``: this module is reached
    through ``get_hf_tensor(name, reader, config)``, which carries no args.
    ``num_nextn_predict_layers`` is the number of prediction depths, and each
    depth's block layout is one attention layer then one MoE layer.

    That ``"*E"`` is a literal because no key in the 3.5 omni config states it
    -- ``mtp_hybrid_override_pattern`` is absent, and ``layers_block_type``
    covers the main stack only. What keeps the literal honest is that the model
    provider's ``validate_mtp_pattern()`` checks the same value against the
    checkpoint's own ``mtp.*`` tensor names before any of this runs, and
    Megatron-Bridge hard-requires the same ``"*E"`` for this checkpoint
    (``nemotron_omni_bridge.py:401``).

    Emitting it unconditionally is safe for a non-MTP run: MCore asks for no
    ``mtp.*`` names, and the decoder branch already indexes through
    ``pattern.split("/")[0]``.
    """
    depths = int(getattr(llm_config, "num_nextn_predict_layers", 0) or 0)
    return "/*E" * depths


def _language_config(hf_config):
    """The decoder's config, with the hybrid pattern the text converter expects.

    ``nemotron_h_hf_tensor`` reads ``hf_config.hybrid_override_pattern`` to know
    what each layer index is. Nemotron 3's config spells that out; the 3.5 omni
    config nests the language model and spells it as ``layers_block_type``
    instead, so derive it rather than requiring the caller to pass a literal.

    The MTP segments are appended here too, because the same string is what
    ``_convert_mtp`` uses to know each MTP block's type and its ``d*L+j`` HF
    index.
    """
    llm_config = getattr(hf_config, "llm_config", hf_config)
    existing = getattr(llm_config, "hybrid_override_pattern", None)
    if existing:
        if "/" in existing:
            return llm_config
        pattern = existing
    else:
        block_types = getattr(llm_config, "layers_block_type", None)
        if not block_types:
            raise KeyError(
                "Nemotron omni llm_config has neither hybrid_override_pattern nor layers_block_type"
            )
        try:
            pattern = "".join(_BLOCK_TYPE_SYMBOLS[block] for block in block_types)
        except KeyError as exc:
            raise KeyError(f"unknown layers_block_type entry {exc.args[0]!r}") from exc

    suffix = _mtp_suffix(llm_config)
    if not suffix and existing:
        return llm_config
    # Do not mutate the caller's config object: it is shared with the model
    # provider, which derives the same pattern for MCore and would then disagree
    # about where it came from.
    return _PatternOverride(llm_config, pattern + suffix)


class _PatternOverride:
    """A read-through view of a config with ``hybrid_override_pattern`` replaced.

    Not ``copy.copy`` plus an assignment: ``NemotronHConfig`` exposes
    ``hybrid_override_pattern`` as a read-only ``property`` derived from
    ``layers_block_type``, the property lives on the class, and a shallow copy
    keeps it -- so assigning raises ``AttributeError: property ... has no
    setter``. Job 19535775 died of exactly that.

    ``__getattr__`` runs only when normal lookup fails, so the instance
    attribute set here wins and everything else falls through to the real
    config.
    """

    def __init__(self, config, pattern: str):
        self._config = config
        self.hybrid_override_pattern = pattern

    def __getattr__(self, name):
        return getattr(self._config, name)


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
        return _vision_tensor(name[len("vision_model.") :], reader, hf_config)

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
