"""Nemotron-3 (HF ``model_type: nemotron_h``) as a ``--spec`` target.

Used from ``scripts/models/nemotron-3-super-120b-a12b.sh``::

    --spec "slime_plugins.models.nemotron_h" "get_nemotron_h_spec"

Two tokens, not a dotted path: ``--spec`` is ``nargs='*'`` and
``spec_utils.import_module()`` unpacks ``base_path, name = module_path``.

WHY THIS IS A MODEL PROVIDER AND NOT A LAYER SPEC
-------------------------------------------------
``slime/backends/megatron_utils/model_provider.py`` calls the ``--spec`` target
as ``spec(args, config, vp_stage)`` and then inspects the result: if it is itself
callable with a ``pre_process`` parameter, slime delegates model construction to
it and never reaches its ``GPTModel`` branch. ``qwen3_next`` cannot use that
path -- megatron.core has no ``Qwen3NextModel``, so it swaps a ``ModuleSpec``
*inside* ``GPTModel`` -- but Nemotron-3 is a hybrid Mamba-2 / attention /
latent-MoE stack for which megatron.core has a native model class, so returning
a provider is both shorter and more faithful.

There is no Nemotron model class in megatron.core either, and there does not
need to be: every Nemotron generation is the *generic* hybrid stack configured
by arguments. ``MambaStack`` allocates one of {``MambaLayer``,
``TransformerLayer``, ``MLPLayer``, ``MoETransformerLayer``} per character of
``--hybrid-override-pattern``, and megatron.core 0.16 is the first release where
``E`` (MoE) is a legal character
(``ssm/mamba_hybrid_layer_allocation.py::Symbols``). Everything Nemotron-3 Super
needs on top of that -- latent MoE, sigmoid routing with an expert bias, a
shared expert, squared-ReLU experts -- is ordinary ``TransformerConfig``, which
is why this file has no modules in it.

TWO MEGATRONS
-------------
This file is loaded by both backends this tree runs: the container image's
megatron.core 0.16.0rc0 (NVIDIA/Megatron-LM ``1dcf0daf``) and the workspace
checkout's 0.20 (``9e68a11b``). 0.19 renamed the whole hybrid stack --
``MambaModel`` -> ``HybridModel``, ``ssm/mamba_hybrid_layer_allocation`` ->
``models/hybrid/hybrid_layer_allocation``, ``--hybrid-override-pattern`` ->
``--hybrid-layer-pattern`` -- and kept re-export shims for every old name, so
the imports below resolve on both. The two places where the shim is not enough
are handled explicitly and marked ``0.16 / 0.20``:

* the layer pattern, which 0.20 reads from ``hybrid_layer_pattern``;
* ``loss_mask``, which 0.20's ``forward`` finally declares.
"""

from __future__ import annotations

import inspect

from megatron.core.models.mamba import MambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec

# 0.16 / 0.20. `hybrid_override_pattern` is deprecated but still accepted on
# 0.20 (it logs a warning and copies itself over), so this is about spelling the
# supported name rather than about working at all. Asking the class that owns
# the parameter, not `MambaModel`, because on 0.20 `MambaModel.__init__` is a
# `**kwargs` shim and its signature says nothing.
try:
    from megatron.core.models.hybrid.hybrid_model import HybridModel as _pattern_owner
except ImportError:  # megatron.core < 0.19
    _pattern_owner = MambaModel

_PATTERN_KWARG = (
    "hybrid_layer_pattern"
    if "hybrid_layer_pattern" in inspect.signature(_pattern_owner.__init__).parameters
    else "hybrid_override_pattern"
)

# 0.16 / 0.20. See NemotronHModel.forward.
_PARENT_FORWARD_TAKES_LOSS_MASK = "loss_mask" in inspect.signature(MambaModel.forward).parameters


class NemotronHModel(MambaModel):
    """``MambaModel`` that tolerates the one keyword slime always sends.

    slime builds its forward kwargs unconditionally
    (``slime/backends/megatron_utils/model.py``) and ``loss_mask`` is always
    among them. ``GPTModel.forward`` declares it keyword-only; ``MambaModel``
    did not until megatron.core 0.19, and on 0.16 the string ``loss_mask`` does
    not occur anywhere in ``mamba_model.py``. Without this subclass the first
    microbatch dies there with::

        TypeError: forward() got an unexpected keyword argument 'loss_mask'

    Dropping it loses nothing on that backend. In ``GPTModel`` the argument
    feeds exactly one consumer -- the multi-token-prediction block, entered
    alongside ``mtp_in_postprocess=self.mtp_process`` -- and a 0.16 Mamba stack
    has no MTP block at all. If MTP is ever configured there, silently
    discarding the mask would be wrong, so that case raises instead.

    0.20's ``HybridModel.forward`` declares ``loss_mask`` and does grow an MTP
    block (a ``/``-separated hybrid pattern builds one), so on that backend the
    mask is handed straight through and this subclass costs nothing but a frame.
    """

    def forward(self, *args, loss_mask=None, **kwargs):
        if _PARENT_FORWARD_TAKES_LOSS_MASK:
            return super().forward(*args, loss_mask=loss_mask, **kwargs)
        if loss_mask is not None and getattr(self.config, "mtp_num_layers", None):
            raise ValueError(
                "nemotron_h: loss_mask was passed with mtp_num_layers set, but a Mamba stack "
                "has no MTP block to consume it."
            )
        return super().forward(*args, **kwargs)


def get_nemotron_h_spec(args, config, vp_stage: int | None = None):
    """Return a model provider for the Nemotron-3 hybrid stack.

    ``config`` arrives already built by slime, so unlike a standalone provider
    this does not call ``core_transformer_config_from_args`` itself -- that is
    the call that turns ``--squared-relu`` into ``config.activation_func``,
    ``--num-experts`` into ``config.num_moe_experts`` and the four ``--mamba-*``
    flags into the fields ``MambaMixer`` reads, and slime has already made it.
    """
    _assert_supported(args, config, vp_stage)
    pattern = _hybrid_pattern(args)
    # 0.16 / 0.20. The ratios are the *other* way of describing the stack, for
    # runs that let Megatron place the layers rather than naming them. 0.20
    # deleted both CLI flags -- the pattern is the only supported spelling now --
    # while HybridModel still accepts the keywords, so they are forwarded only
    # when the parser this megatron.core ships actually produced them. Reading
    # them unconditionally is what job 18803796 died of, at model construction,
    # eight ranks in.
    ratios = {
        name: getattr(args, name) for name in ("hybrid_attention_ratio", "hybrid_mlp_ratio") if hasattr(args, name)
    }

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
        _assert_no_virtual_pipeline(vp_stage)
        return NemotronHModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            # An explicit pattern wins over these, and passing both is only legal
            # when they agree, so they stay at their argument defaults (0.0)
            # wherever they still exist and the pattern decides. See above.
            **ratios,
            # 0.16 / 0.20, see _PATTERN_KWARG. Passing the pattern under the
            # name this megatron.core retired would not raise -- it would build
            # an all-Mamba stack of the right depth and fail much later, on a
            # checkpoint key that does not exist.
            **{_PATTERN_KWARG: pattern},
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            # 'none' for Nemotron-3; the two rotary arguments are then unused and
            # are passed only so a RoPE variant of this file stays a flag change.
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
        )

    return model_provider


def _assert_no_virtual_pipeline(vp_stage: int | None) -> None:
    # 0.16's MambaStack takes no vp_stage and megatron's own mamba_builder never
    # passes one, so a virtual-pipeline schedule would build the wrong number of
    # layers per chunk and surface only as a shape mismatch deep in the
    # checkpoint load. 0.20's HybridStack does implement VPP, through `|`
    # segments in the pattern -- but slime derives its global parameter names
    # from get_transformer_layer_offset(), which knows nothing about those
    # segments, so the two would disagree about which layer is which. Refuse it
    # on both backends, where the message can still say why.
    if vp_stage is not None:
        raise ValueError(
            "nemotron_h: virtual pipeline parallelism is not supported here; drop "
            "--num-layers-per-virtual-pipeline-stage (megatron.core >= 0.19: drop the '|' "
            "pipeline separators from the hybrid layer pattern)."
        )


def _hybrid_pattern(args) -> str | None:
    """The layer pattern, under whichever flag this megatron.core spells it.

    0.16 / 0.20. 0.20's ``validate_args`` copies ``--hybrid-override-pattern``
    into ``args.hybrid_layer_pattern`` and leaves the old attribute in place, so
    both are set on that backend and either order works; the new name is
    preferred so that a launcher which passes ``--hybrid-layer-pattern``
    directly is not read as "no pattern at all".
    """
    return getattr(args, "hybrid_layer_pattern", None) or getattr(args, "hybrid_override_pattern", None)


def _pattern_layer_count(pattern: str) -> int:
    """Layers in *pattern*, which on 0.20 may carry ``|`` and ``/`` separators."""
    try:
        from megatron.core.models.hybrid.hybrid_layer_allocation import get_hybrid_total_layer_count
    except ImportError:  # megatron.core < 0.19: one character, one layer
        return len(pattern)
    return get_hybrid_total_layer_count(pattern)


def _assert_supported(args, config, vp_stage: int | None) -> None:
    _assert_no_virtual_pipeline(vp_stage)

    # `is_hybrid_model` gates the correct activation-memory and layer-offset
    # arithmetic in megatron.training. The pattern being non-empty and the right
    # length is what makes this a Nemotron rather than a pure Mamba stack, and
    # getting either wrong produces a model that builds and trains and is simply
    # not the model the checkpoint holds.
    if not config.is_hybrid_model:
        raise ValueError("nemotron_h: --is-hybrid-model is required")
    pattern = _hybrid_pattern(args)
    if not pattern:
        raise ValueError("nemotron_h: --hybrid-override-pattern (0.20: --hybrid-layer-pattern) is required")
    num_layers_in_pattern = _pattern_layer_count(pattern)
    if num_layers_in_pattern != args.num_layers:
        raise ValueError(
            f"nemotron_h: the hybrid layer pattern has {num_layers_in_pattern} "
            f"layers but --num-layers is {args.num_layers}"
        )
