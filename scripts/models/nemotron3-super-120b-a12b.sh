# NVIDIA Nemotron 3 Super 120B-A12B (BF16), for MCore's HybridModel.
#
# Every value below is config.json from nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16,
# translated into the MCore flag that carries it. The HF field name is in the
# comment wherever the two differ.
#
# Unlike the other files in this directory this model is not a GPTModel: the 88
# layers are a hybrid of Mamba2, MoE and attention, chosen per layer by
# --hybrid-override-pattern. --spec therefore points at a *model provider*
# rather than a layer spec, and slime calls it directly
# (backends/megatron_utils/model_provider.py:146-153).
#
# Most of the flags below are not declared in megatron/training/arguments.py --
# they are generated from TransformerConfig dataclass fields by
# megatron/training/argument_utils.py, the same mechanism that produces
# --moe-flex-dispatcher-backend. Verified present on mcore 0.20 (f6c33bde4).

# 'M' Mamba2 x40, 'E' MoE x40, '*' attention x8, in this order.
NEMOTRON_H_PATTERN="MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"

MODEL_ARGS=(
   --spec "slime_plugins.models.nemotron_h" "get_nemotron_h_model_provider"
   --hybrid-override-pattern "${NEMOTRON_H_PATTERN}"

   # ---------------------------------------------------------------- shape --
   --num-layers 88                         # num_hidden_layers
   --hidden-size 4096                      # hidden_size
   --num-attention-heads 32                # num_attention_heads
   --num-query-groups 2                    # num_key_value_heads (GQA)
   --kv-channels 128                       # head_dim
   --group-query-attention
   --vocab-size 131072                     # vocab_size
   --untie-embeddings-and-output-weights   # tie_word_embeddings: false
   --disable-bias-linear                   # use_bias / attention_bias / mlp_bias: false
   --normalization RMSNorm
   --norm-epsilon 1e-5                     # norm_eps / layer_norm_epsilon
   --position-embedding-type rope
   --rotary-base 10000                     # rope_theta
   --rotary-percent 1.0                    # partial_rotary_factor

   # ---------------------------------------------------------------- mamba --
   # in_proj packs [z, x, B, C, dt]: 2*d_inner + 2*n_groups*d_state + nheads
   #   d_inner = expand(2) x hidden(4096)        = 8192
   #           = mamba_num_heads(128) x head_dim(64)
   #   conv_dim = d_inner + 2*n_groups*d_state   = 10240
   --mamba-num-heads 128                   # mamba_num_heads
   --mamba-head-dim 64                     # mamba_head_dim
   --mamba-state-dim 128                   # ssm_state_size
   --mamba-num-groups 8                    # n_groups

   # ------------------------------------------------------------------ moe --
   # The experts run inside a 1024-wide latent space, not on the 4096 residual:
   # fc1_latent_proj 4096->1024, expert 1024->2688->1024, fc2_latent_proj
   # 1024->4096. MoELayer builds those two projections under exactly the names
   # the HF checkpoint uses.
   --num-experts 512                       # n_routed_experts
   --moe-latent-size 1024                  # moe_latent_size
   --moe-ffn-hidden-size 2688              # moe_intermediate_size
   --moe-router-topk 22                    # num_experts_per_tok
   --moe-router-score-function softmax
   --moe-router-enable-expert-bias         # gate.e_score_correction_bias exists
   --moe-router-topk-scaling-factor 5.0    # routed_scaling_factor
   --moe-shared-expert-intermediate-size 5376   # moe_shared_expert_intermediate_size
   --moe-grouped-gemm
   --moe-token-dispatcher-type alltoall
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0

   # ------------------------------------------------------------- non-gated --
   # mlp_hidden_act: relu2. The checkpoint has up_proj and down_proj and no
   # gate_proj, so linear_fc1 is up_proj alone rather than cat(gate, up), and
   # the activation is squared ReLU. MCore asserts these two agree
   # (transformer_config.py:1856-1860).
   --squared-relu
   # (gated_linear_unit stays at its False default; --swiglu is what would set it)

   # --------------------------------------------------------------- context --
   # max_position_embeddings is 262144. That is the model's capability, not a
   # per-run cost here -- SFT sequences are far shorter -- but it sizes RoPE.
   --max-position-embeddings 262144
)
