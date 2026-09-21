# Nemotron 3.5 Super VL: direct RL preview

This recipe uses public MCore's HybridModel, native checkpoint conversion and full
weight synchronization. It starts RL from an existing BF16 HF checkpoint; SFT is
not a prerequisite. The vision encoder is frozen, while the projector and
language model are trained.

This is a candidate recipe, not a GPU-qualified configuration. Validate the same
checkpoint on HF, Megatron and SGLang before interpreting rewards or loss.

## Model ownership and public reuse

- **Language:** directly construct MCore `HybridModel` with its public
  `hybrid_stack_spec` and `--hybrid-layer-pattern`. MCore implements the
  Mamba/attention/MoE stack, forward/loss-mask handling and parameter sharding.
  The slime provider only binds arguments; there is no legacy MambaModel subclass
  or old-MCore compatibility branch.
- **Vision:** retain Shifang's slime adapter around the checkpoint's HF
  `RadioModel` and Super VL `VisionProjector`. Slime owns image-feature injection,
  ragged-input handling and the frozen-encoder/trainable-projector boundary; it
  does not implement another RADIO or projector network.
- **Bridge:** use the [public Nemotron-H mappings](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ea5dded83078cf7e61ef91960273053cf564d5ac/src/megatron/bridge/models/nemotronh/nemotron_h_bridge.py)
  and [public Super VL implementation](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/5b5ef52d3a673d5f867794c27035171680b23a9f/src/megatron/bridge/models/nemotron_omni/nemotron_omni_bridge.py)
  as configuration/layout references. They are not runtime imports or a second
  training backend. Bridge's MCore-native vision layout and its two-depth MTP
  provider are not drop-in replacements for this image-only, no-MTP HF adapter.

The existing slime checkpoint/refit adapters bridge the concrete HF parameter
names and the already-gathered Megatron tensors. Full vision synchronization
retains RADIO LayerScale; the HF projector's final LayerNorm remains trainable
along with the rest of the projector. Do not silently change that freeze boundary
when comparing with a Bridge recipe.

## Environment and scope

- Use the companion **main-based** Megatron checkout, including the hybrid
  `IdentityOp` guards in the slime post-layernorm patch. Do not apply an old dev
  patch again on top of an already patched checkout.
- Use SGLang's native `NemotronH_Omni_Reasoning_V3` implementation with the companion
  Super processor and multimodal cache invalidation fixes. The old experimental
  SGLang registration alone is not sufficient.

- Record the three repository SHAs, container digest/package versions and exact
  checkpoint snapshot in the run manifest. The default slime Dockerfile's old
  Megatron pin is not this preview environment.
- Use image/text, BF16, PP=CP=1, ordinary next-token generation. Disable MTP
  training and speculative decoding, but preserve the source HF config and MTP
  tensors: the serving model also uses this config to construct its vision LN.
- Keep the full vision weight synchronization. The synthesized RADIO LayerScale
  tensors must also be restored after rollout memory offload/onload.

Companion drafts: [Megatron-LM #1](https://github.com/shifangx/Megatron-LM/pull/1)
and [SGLang #1](https://github.com/shifangx/sglang/pull/1). The code revisions for
this candidate are MCore `1f802f259b1291c4bc5cea175de644433dde2d53` and SGLang
`70bfd31e29081beab29546a0b447be6fef16ddc8`; GPU qualification remains pending.
Use those checkouts directly in the selected CUDA container; do not reapply the
slime MCore patch. Changing only the existing Docker build arguments does not
reproduce this combination or qualify the image.

The example uses trainer TP2/EP8 and rollout TP8/EP1. Choose the actor allocation
for the actual hardware and memory requirements; this is not a minimum-GPU claim.
Run inside an existing compatible container and Ray allocation. All workers
must have the same installed code, environment and shared paths.

## Data and reward

Use an independent training dataset, not the public VIALS benchmark. JSONL or
Parquet rows can use the existing slime multimodal schema:

```json
{"prompt":"<image>\n<image>\nCompare the two assays. Reason briefly and finish with ANSWER:","images":["/shared/train/a.png","/shared/train/b.png"],"label":"reference answer","metadata":{"id":"train-001","domain":"biology"}}
```

Match each `<image>` to an image in order. Keep the reference answer and any
numeric bounds in label/metadata, not the generated prompt. Use the same prompt
format as the intended evaluation.

Set `CUSTOM_RM_PATH` to an importable async function with the existing signature
`async def reward(args, sample, **kwargs)`. It can return a float; if returning a
dictionary, also pass `--reward-key <field>`. The sample provides response,
label, metadata and multimodal inputs. Reuse the customer's verifier rather than
the geo3k boxed-math reward. Keep the rollout and judge endpoints independent,
and surface judge service failures instead of silently producing all-zero scores.

## Import and run

Set `HF_CHECKPOINT`, `IMPORT_DIR`, `SAVE_DIR`, `TRAIN_DATA`, `CUSTOM_RM_PATH`,
`ACTOR_NODES` and `ACTOR_GPUS_PER_NODE` to existing model/data/allocation paths
and dedicated output directories. Set `RAY_ADDRESS` for the existing cluster.
The fixed model argument bundle must match the actual EA checkpoint config;
do not assume a midtrain or SourceOfTruth alias is identical.

From the slime repository, in Bash:

```bash
source scripts/models/nemotron-3.5-super-vl-120b-a12b.sh

# One-time conversion on an existing 8-GPU allocation; do not force auto-PP.
torchrun --standalone --nproc-per-node=8 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_CHECKPOINT" --save "$IMPORT_DIR" \
  --bf16 --no-auto-pipeline --tensor-model-parallel-size 1 \
  --expert-model-parallel-size 8 --expert-tensor-parallel-size 1 \
  --pipeline-model-parallel-size 1 --context-parallel-size 1

# Use the immutable original HF snapshot for initial rollout weights/processor.
# No prior SFT checkpoint is required.
export SGLANG_RETURN_ORIGINAL_LOGPROB=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
python train.py "${MODEL_ARGS[@]}" \
  --train-backend megatron --bf16 --hf-checkpoint "$HF_CHECKPOINT" \
  --load "$IMPORT_DIR" --finetune --save "$SAVE_DIR" --save-interval 1 \
  --ckpt-format torch_dist --actor-num-nodes "$ACTOR_NODES" \
  --actor-num-gpus-per-node "$ACTOR_GPUS_PER_NODE" --colocate \
  --tensor-model-parallel-size 2 --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --sequence-parallel --micro-batch-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --seq-length 8192 --max-position-embeddings 8192 \
  --prompt-data "$TRAIN_DATA" --input-key prompt --label-key label \
  --multimodal-keys '{"image":"images"}' --apply-chat-template \
  --custom-rm-path "$CUSTOM_RM_PATH" --advantage-estimator grpo \
  --rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16 \
  --num-rollout 2 --rollout-max-context-len 8192 --rollout-max-prompt-len 4096 \
  --rollout-max-response-len 4096 --rollout-temperature 1 --rollout-top-p 1 \
  --rollout-top-k -1 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --rollout-num-gpus-per-engine 8 --sglang-ep-size 1 \
  --sglang-context-length 8192 --sglang-moe-runner-backend triton \
  --sglang-moe-a2a-backend none --sglang-mm-attention-backend sdpa \
  --sglang-disable-cuda-graph --sglang-disable-overlap-schedule
```

These are short functional-run settings, not a tuned learning-rate or quality
recipe. Adapt batch sizes to DP divisibility. Ensure a group has meaningful
reward variation; all-equal rewards legitimately produce zero GRPO advantage.
GSPO is available through `--advantage-estimator gspo`, but must be validated if
used for the delivered run. Propagate the environment above to Ray workers, not
only the submission shell.

For **resume**, use the same command with `--load "$SAVE_DIR"`, remove
`--finetune`, increase the total `--num-rollout`, and add
`--use-checkpoint-opt-param-scheduler`. Do not add `--no-load-optim` or
`--no-load-rng`. The actor pushes the restored weights to rollout before use.

For **HF export**, set `DCP_ITERATION_DIR` to the actual saved iteration directory
(not its parent) and `EXPORT_DIR` to a new output path:

```bash
python tools/convert_torch_dist_to_hf.py --model-name nemotron_h_omni \
  --input-dir "$DCP_ITERATION_DIR" --output-dir "$EXPORT_DIR" \
  --origin-hf-dir "$HF_CHECKPOINT" --add-missing-from-origin-hf \
  --vocab-size 131072
```

Export needs sufficient host RAM for the existing full-state loader. Missing
trained language/projector weights must fail, not be replaced with stale source
weights. Preserved source MTP weights are not a newly trained speculative model.

## Before calling the preview validated

1. Check the same checkpoint's HF/Megatron teacher-forced logits/loss on text,
   one image and multiple images, with identical IDs, labels and loss masks.
   A first-step SFT loss value alone is not a diagnosis; this recipe does not
   claim to resolve an unexplained high-loss report.
2. Run two real rollout/update/refit cycles. Check frozen encoder state,
   projector/LLM updates, all-worker update success and invalidated caches.
3. Resume with optimizer/scheduler/RNG state, export, and reload the trained model.
4. Evaluate a held-out subset with identical before/after settings. The example's
   4K response budget is not VIALS's full 32K-completion evaluation protocol.

SFT can be used to isolate shared model/processor/loss issues, but a complete SFT
recipe or an SFT-trained initializer is not required for direct RL.
The existing VL SFT helper is retained for that check: use
`--rollout-function-path slime.rollout.sft_rollout_vl.generate_rollout`,
`--input-key messages`, `--loss-type sft_loss` and `--debug-train-only`, with the
checkpoint-appropriate `--loss-mask-type`. Do not use `--apply-chat-template`
for that helper, since it needs the original message list. Inspect its actual
labels/mask against HF; no change here establishes a cause or fix for high SFT loss.
