"""Thin slime provider for the public MCore main HybridModel implementation."""

from __future__ import annotations

from megatron.core.models.hybrid.hybrid_layer_allocation import get_hybrid_total_layer_count
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel


def get_nemotron_h_spec(
    args,
    config,
    vp_stage: int | None = None,
    *,
    scatter_embedding_sequence_parallel: bool = True,
):
    """Bind slime's model arguments to MCore's native hybrid language model.

    The VL adapter disables embedding scatter until image features have been
    injected. MCore owns the hybrid layers, forward/loss-mask handling and
    checkpoint sharding; no legacy MambaModel compatibility class is needed.
    """
    _assert_no_virtual_pipeline(vp_stage)
    if not config.is_hybrid_model:
        raise ValueError("nemotron_h: --is-hybrid-model is required")
    pattern = args.hybrid_layer_pattern
    if not pattern:
        raise ValueError("nemotron_h: --hybrid-layer-pattern is required")
    num_layers = get_hybrid_total_layer_count(pattern)
    if num_layers != args.num_layers:
        raise ValueError(
            f"nemotron_h: the hybrid layer pattern has {num_layers} layers but --num-layers is {args.num_layers}"
        )

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
        _assert_no_virtual_pipeline(vp_stage)
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern=pattern,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
        )

    return model_provider


def _assert_no_virtual_pipeline(vp_stage: int | None) -> None:
    if vp_stage is not None:
        raise ValueError("nemotron_h: virtual pipeline parallelism is outside the current preview")
