#!/bin/bash
# Model arguments reused from shifangx/slime@328760de for the 120B/A12B VL checkpoint.
# Uses the native hybrid language provider and checkpoint-owned frozen RADIO.
# This preview trains the projector and LLM, without auxiliary MTP training.

N35VL_HYBRID_PATTERN="MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"

MODEL_ARGS=(
   --spec "slime_plugins.models.nemotron_35_super_vl" "get_nemotron_35_super_vl_model_provider"

   --trust-remote-code

   --is-hybrid-model
   --hybrid-layer-pattern "${N35VL_HYBRID_PATTERN}"
   --mamba-num-heads 128
   --mamba-head-dim 64
   --mamba-state-dim 128
   --mamba-num-groups 8

   --num-layers 88
   --hidden-size 4096
   --ffn-hidden-size 2688
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 2
   --kv-channels 128

   --normalization RMSNorm
   --norm-epsilon 1e-5
   --squared-relu
   --disable-bias-linear
   --untie-embeddings-and-output-weights
   --position-embedding-type none
   --init-method-std 0.014
   --vocab-size 131072
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash

   --num-experts 512
   --moe-latent-size 1024
   --moe-ffn-hidden-size 2688
   --moe-router-topk 22
   --moe-router-score-function sigmoid
   --moe-router-enable-expert-bias
   --moe-router-load-balancing-type seq_aux_loss
   --moe-router-topk-scaling-factor 5.0
   --moe-router-dtype fp32
   --moe-shared-expert-intermediate-size 5376
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
   --moe-permute-fusion
   --moe-aux-loss-coeff "${N35VL_AUX_LOSS_COEFF:-0}"
   --moe-router-bias-update-rate "${N35VL_BIAS_UPDATE_RATE:-0}"
)


if [[ "${USE_FUSION_ARGS:-0}" == "1" ]]; then
   MODEL_ARGS+=(
      --enable-experimental
      --use-fused-weighted-squared-relu
      --cross-entropy-loss-fusion
      --cross-entropy-fusion-impl native
   )
   echo "[model] fusion args enabled (Megatron-LM's conf defaults)"
fi
