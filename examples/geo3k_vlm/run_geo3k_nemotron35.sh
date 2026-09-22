#!/bin/bash

# NVIDIA-Nemotron-3.5-Super-EA-09112026 VL RL training on geo3k dataset
#
# run_geo3k_qwen35.sh with the model swapped. Every training hyperparameter is
# that file's, unchanged: GRPO, lr 1e-6, PP1, CP1, full recompute, rollout batch
# 64 x 8 samples = global batch 512, micro-batch 1, TP2. Five things differ, and
# all five follow from the model:
#
#   * 4 nodes, not 1. EP x ETP must divide the world, and this model's 512
#     routed experts want EP32 -> 32 GPUs. Multi-node therefore needs an
#     external Ray cluster: SLIME_SCRIPT_EXTERNAL_RAY=1 with the head already
#     up. SLIME_SCRIPT_NUM_NODES=1 + EXPERT_MODEL_PARALLEL_SIZE=8 runs the
#     single-node shape instead, but has not been shown to fit in 80 GB.
#   * mamba-ssm and causal-conv1d instead of `pip install -U transformers`.
#     megatron.core's MambaMixer hard-requires them for the 40 Mamba2 layers
#     (ssm/mamba_mixer.py:205) and the slime image ships neither.
#   * --offload-optimizer-states. Adam's fp32 master weights and moments do not
#     stay resident at this size; micro-batch and recompute are already at their
#     limits. This is mcore 0.16's spelling -- on 0.20 use
#     `--optimizer-cpu-offload --use-precision-aware-optimizer` instead.
#   * no MTP speculative decoding. The option is real (this checkpoint has an
#     MTP layer), but in the Qwen GRPO case SGLang returned a None inside
#     meta_info["output_token_logprobs"] during eval, and GRPO needs a real log
#     prob for every trainable response token. USE_MTP=1 turns it back on.
#   * scripts/models/nemotron3.5-super-vl.sh, which sources the Nemotron 3 text
#     config and overrides only --spec: the two models' language configs are
#     field-for-field identical, and the spec is what hangs the C-RADIO v4-H
#     tower and the projector off the same MCore HybridModel.
#
# The model repo is GATED -- `hf download` needs HF_TOKEN with access granted.
#
# TP2 depends on slime's shared-expert sharding fix (hf_to_megatron/common.py
# keys the gate/up split on whether the model is gated; Nemotron is squared-ReLU
# with no gate half). An older slime tree loads this model wrong at TP2 and says
# nothing.

pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm

# Configuration
TRAIN_BACKEND="megatron"
MODEL_NAME="NVIDIA-Nemotron-3.5-Super-EA-09112026"
DATASET_NAME=${SLIME_SCRIPT_DATASET_NAME:-"chenhegu/geo3k_imgurl"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
NUM_NODES=${SLIME_SCRIPT_NUM_NODES:-4}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# This script's own `ray start --head` forms a one-node cluster, so anything
# wider has to be pointed at a cluster that already spans the nodes.
if [ "$NUM_NODES" -gt 1 ] && [ "$USE_EXTERNAL_RAY" = "0" ]; then
   echo "NUM_NODES=$NUM_NODES requires SLIME_SCRIPT_EXTERNAL_RAY=1 and a head already running" >&2
   exit 1
fi

# Cleanup
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

set -ex

export PYTHONUNBUFFERED=1

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Download model and dataset
mkdir -p /root/models /root/datasets
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
   hf download nvidia/${MODEL_NAME} --local-dir /root/models/${MODEL_NAME}
fi
if [ ! -d "/root/datasets/${DATASET_LOCAL_NAME}" ]; then
   hf download --repo-type dataset ${DATASET_NAME} --local-dir /root/datasets/${DATASET_LOCAL_NAME}
fi

# Common args
CKPT_ARGS=(
   --hf-checkpoint /root/models/${MODEL_NAME}
   --load /root/models/${MODEL_NAME}
)

ROLLOUT_ARGS=(
   --prompt-data /root/datasets/${DATASET_LOCAL_NAME}/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 512
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data ${DATASET_LOCAL_NAME} /root/datasets/${DATASET_LOCAL_NAME}/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # 80 GB per GPU: Adam's fp32 master weights and moments do not stay resident.
   # mcore 0.16's spelling; 0.20 replaced it with --optimizer-cpu-offload
   # --use-precision-aware-optimizer. slime parses Megatron args with
   # ignore_unknown_args=True, so on a tree without this flag nothing fails --
   # the offload just stops happening.
   --offload-optimizer-states
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256

   --sglang-max-running-requests 512

   # Neither of the Qwen recipe's two engine pins carries over:
   #   --sglang-moe-runner-backend triton is pinned there because the weight
   #   sync pushes Megatron's layout straight into w13 and the flashinfer_trtllm
   #   runner keeps that parameter padded. Nemotron's experts are NOT gated
   #   (relu2, up_proj + down_proj, no gate_proj), so there is no cat(gate, up)
   #   and no w13 to mismatch.
   #   --sglang-mm-attention-backend sdpa is a Qwen vision-tower flag; this
   #   model's tower is C-RADIO v4-H, served by sglang's own radio.py.
)

# MTP speculative decoding, off by default -- see the header.
if [ "${USE_MTP:-0}" = "1" ]; then
   SGLANG_ARGS+=(
      --sglang-speculative-algorithm EAGLE
      --sglang-speculative-num-steps 2
      --sglang-speculative-eagle-topk 1
      --sglang-speculative-num-draft-tokens 3
      --sglang-speculative-moe-runner-backend triton
   )
fi

# Wandb args (only if WANDB_API_KEY is set)
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime-geo3k-vlm
      --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   --colocate
)

# Backend-specific args
# megatron backend
BACKEND_ARGS=(
   --train-backend megatron
   # Nemotron 3.5 Super has num_query_groups = 2, as Qwen3.5-35B-A3B does
   --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE:-2}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   # 512 routed experts: EP32 leaves 16 per rank, EP8 leaves 64. Both divide
   # evenly, so no rank gets a short shard.
   --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE:-32}
   --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE:-1}
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --micro-batch-size 1
)

SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
source "${SLIME_DIR}/scripts/models/nemotron3.5-super-vl.sh"

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# Every array expansion below is QUOTED. --hybrid-override-pattern's value is an
# 88-character string containing `*`, and an unquoted expansion is subject to
# pathname expansion against the working directory -- a file there matching the
# pattern would silently replace it, and the run would build a different
# architecture with nothing in the log naming the cause. The Qwen recipe can
# leave them unquoted because no Qwen value contains a glob character.
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
   "${BACKEND_ARGS[@]}" \
   "${MISC_ARGS[@]}"
