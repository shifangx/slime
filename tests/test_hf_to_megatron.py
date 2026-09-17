import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# These mapping tests also run in the CPU CI image, which does not install
# Megatron. In that environment, mount the source package without executing
# megatron_utils' runtime patch initialization.
try:
    _has_megatron = importlib.util.find_spec("megatron.core") is not None
except ModuleNotFoundError:
    _has_megatron = False
if not _has_megatron:
    _megatron_utils = types.ModuleType("slime.backends.megatron_utils")
    _megatron_utils.__path__ = [str(Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils")]
    sys.modules["slime.backends.megatron_utils"] = _megatron_utils

from slime.backends.megatron_utils.hf_to_megatron import _LOADERS
from slime.backends.megatron_utils.hf_to_megatron import nemotron_h as nemotron_h_loader
from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from slime.backends.megatron_utils.hf_to_megatron.deepseek import deepseek_hf_tensor
from slime.backends.megatron_utils.hf_to_megatron.glm import glm4_hf_tensor, glm4_moe_hf_tensor
from slime.backends.megatron_utils.hf_to_megatron.qwen import (
    mimo_hf_tensor,
    minimax_m2_hf_tensor,
    qwen_hf_tensor,
    qwen_moe_hf_tensor,
)
from slime.backends.megatron_utils.hf_to_megatron.qwen3_next import qwen3_next_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
from slime.backends.megatron_utils.megatron_to_hf.deepseekv3 import convert_deepseekv3_to_hf
from slime.backends.megatron_utils.megatron_to_hf.glm4 import convert_glm4_to_hf
from slime.backends.megatron_utils.megatron_to_hf.glm4moe import convert_glm4moe_to_hf
from slime.backends.megatron_utils.megatron_to_hf.mimo import convert_mimo_to_hf
from slime.backends.megatron_utils.megatron_to_hf.minimax_m2 import convert_minimax_m2_to_hf
from slime.backends.megatron_utils.megatron_to_hf.nemotron_h import convert_nemotron_h_to_hf
from slime.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf
from slime.backends.megatron_utils.megatron_to_hf.qwen3_next import convert_qwen3_next_to_hf
from slime.backends.megatron_utils.megatron_to_hf.qwen3moe import convert_qwen3moe_to_hf

NUM_GPUS = 0


class Reader:
    def __init__(self, **tensors):
        self.tensors = tensors

    def __contains__(self, name):
        return name in self.tensors

    def get_tensor(self, name):
        return self.tensors[name]


def _config(model_type="qwen3"):
    return types.SimpleNamespace(
        model_type=model_type,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=2,
        tie_word_embeddings=False,
    )


_EXPORT_ARGS = types.SimpleNamespace(
    kv_channels=2,
    hidden_size=8,
    num_attention_heads=4,
    num_query_groups=2,
    num_layers=2,
    q_lora_rank=None,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("loader", "exporter", "model_type", "name", "shape"),
    [
        (
            qwen_hf_tensor,
            convert_qwen2_to_hf,
            "qwen3",
            "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
            (16, 8),
        ),
        (
            qwen_moe_hf_tensor,
            convert_qwen3moe_to_hf,
            "qwen3_moe",
            "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight3",
            (12, 8),
        ),
        (
            mimo_hf_tensor,
            convert_mimo_to_hf,
            "mimo",
            "module.module.mtp.layers.0.eh_proj.weight",
            (8, 8),
        ),
        (
            minimax_m2_hf_tensor,
            convert_minimax_m2_to_hf,
            "minimax_m2",
            "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight3",
            (12, 8),
        ),
        (
            deepseek_hf_tensor,
            convert_deepseekv3_to_hf,
            "deepseek_v32",
            "module.module.decoder.layers.0.self_attention.wq_b.weight",
            (256, 8),
        ),
        (
            deepseek_hf_tensor,
            convert_deepseekv3_to_hf,
            "deepseek_v3",
            "module.module.mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight3",
            (12, 8),
        ),
        (
            glm4_hf_tensor,
            convert_glm4_to_hf,
            "glm4",
            "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
            (16, 8),
        ),
        (
            glm4_moe_hf_tensor,
            convert_glm4moe_to_hf,
            "glm4_moe",
            "module.module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
            (12, 8),
        ),
        (
            qwen3_next_hf_tensor,
            convert_qwen3_next_to_hf,
            "qwen3_next",
            "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
            (24, 8),
        ),
    ],
)
def test_hf_and_megatron_mappings_round_trip(loader, exporter, model_type, name, shape):
    parameter = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    hf_tensors = dict(exporter(_EXPORT_ARGS, name, parameter))
    config = types.SimpleNamespace(
        model_type=model_type,
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=2,
        tie_word_embeddings=False,
    )

    loaded = loader(name, Reader(**hf_tensors), config)

    assert torch.equal(loaded, parameter)


@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["deepseekv32config", "kimik2config"])
def test_deepseek_family_parameter_updates_use_the_direct_exporter(model_name):
    parameter = torch.randn(8, 8)

    converted = _convert_to_hf_core(
        _EXPORT_ARGS,
        model_name,
        "module.module.decoder.layers.0.self_attention.linear_proj.weight",
        parameter,
    )

    assert len(converted) == 1
    assert converted[0][0] == "model.layers.0.self_attn.o_proj.weight"
    assert converted[0][1] is parameter


@pytest.mark.unit
def test_qwen2_moe_parameter_updates_use_the_moe_exporter():
    parameter = torch.randn(12, 8)

    converted = _convert_to_hf_core(
        _EXPORT_ARGS,
        "qwen2_moe",
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight3",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.3.up_proj.weight",
    ]
    assert torch.equal(converted[0][1], parameter[:6])
    assert torch.equal(converted[1][1], parameter[6:])


@pytest.mark.unit
def test_qwen_and_llama_share_the_basic_qkv_mapping():
    q = torch.arange(16).view(4, 4)
    k = torch.arange(8).view(2, 4) + 100
    v = torch.arange(8).view(2, 4) + 200
    reader = Reader(
        **{
            "model.layers.3.self_attn.q_proj.weight": q,
            "model.layers.3.self_attn.k_proj.weight": k,
            "model.layers.3.self_attn.v_proj.weight": v,
        }
    )

    loaded = qwen_hf_tensor(
        "module.module.decoder.layers.3.self_attention.linear_qkv.weight",
        reader,
        _config(),
    )

    assert torch.equal(loaded, torch.cat((q, k, v)))
    assert _LOADERS["qwen3"] is _LOADERS["llama"] is qwen_hf_tensor


@pytest.mark.unit
def test_qwen_moe_merges_one_global_expert():
    gate = torch.randn(4, 3)
    up = torch.randn(4, 3)
    reader = Reader(
        **{
            "model.layers.1.mlp.experts.9.gate_proj.weight": gate,
            "model.layers.1.mlp.experts.9.up_proj.weight": up,
        }
    )

    loaded = qwen_moe_hf_tensor(
        "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight9",
        reader,
        _config(),
    )

    assert torch.equal(loaded, torch.cat((gate, up)))


@pytest.mark.unit
def test_deepseek_mapping_handles_kimi_and_dsa_layouts():
    mla = torch.randn(8, 4)
    kimi = deepseek_hf_tensor(
        "module.module.decoder.layers.0.self_attention.linear_kv_down_proj.weight",
        Reader(**{"model.layers.0.self_attn.kv_a_proj_with_mqa.weight": mla}),
        _config("kimi_k2"),
    )
    assert kimi is mla

    dsa = torch.arange(128 * 2).view(128, 2)
    reordered = deepseek_hf_tensor(
        "module.module.decoder.layers.0.self_attention.wk.weight",
        Reader(**{"model.layers.0.self_attn.indexer.wk.weight": dsa}),
        _config("glm_moe_dsa"),
    )
    assert torch.equal(reordered, torch.cat((dsa[64:], dsa[:64])))


@pytest.mark.unit
def test_glm_dense_and_moe_mtp_use_native_mappings():
    fused = torch.randn(8, 4)
    dense = glm4_hf_tensor(
        "module.module.decoder.layers.1.mlp.linear_fc1.weight",
        Reader(**{"model.layers.1.mlp.gate_up_proj.weight": fused}),
        _config("glm4"),
    )
    assert dense is fused

    mtp = torch.randn(4, 4)
    moe = glm4_moe_hf_tensor(
        "module.module.mtp.layers.0.eh_proj.weight",
        Reader(**{"model.layers.2.eh_proj.weight": mtp}),
        _config("glm4_moe"),
    )
    assert moe is mtp

    gate = torch.randn(4, 3)
    up = torch.randn(4, 3)
    shared = glm4_moe_hf_tensor(
        "module.module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
        Reader(
            **{
                "model.layers.0.mlp.shared_experts.gate_proj.weight": gate,
                "model.layers.0.mlp.shared_experts.up_proj.weight": up,
            }
        ),
        _config("glm4_moe"),
    )
    assert torch.equal(shared, torch.cat((gate, up)))


@pytest.mark.unit
def test_minimax_and_mimo_keep_their_small_qwen_deltas():
    gate = torch.randn(4, 3)
    up = torch.randn(4, 3)
    minimax = minimax_m2_hf_tensor(
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight2",
        Reader(
            **{
                "model.layers.0.block_sparse_moe.experts.2.w1.weight": gate,
                "model.layers.0.block_sparse_moe.experts.2.w3.weight": up,
            }
        ),
        _config("minimax_m2"),
    )
    assert torch.equal(minimax, torch.cat((gate, up)))

    hf_eh = torch.arange(24).view(3, 8)
    mimo = mimo_hf_tensor(
        "module.module.mtp.layers.0.eh_proj.weight",
        Reader(**{"model.mtp_layers.0.input_proj.weight": hf_eh}),
        _config("mimo"),
    )
    assert torch.equal(mimo, torch.cat((hf_eh[:, 4:], hf_eh[:, :4]), dim=1))


@pytest.mark.unit
def test_loader_scope_stays_explicit():
    assert set(_LOADERS) == {
        "deepseek_v3",
        "deepseek_v32",
        "glm4",
        "glm4_moe",
        "glm4_moe_lite",
        "glm_moe_dsa",
        "kimi_k2",
        "llama",
        "mimo",
        "minimax_m2",
        "nemotron_h",
        # Nemotron 3.5 Super VL, and deliberately the *same* function object as
        # `nemotron_h` rather than a second loader: its language tower is
        # Nemotron-3's under a `language_model.` prefix, which that file derives
        # from the config it is handed. tests/test_nemotron_35_super_vl.py pins
        # the identity.
        "nemotron_h_omni",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_moe",
        "qwen3_next",
    }


# A toy Nemotron-3: hidden 8, 4 attention heads in 2 KV groups of head_dim 2,
# mamba d_inner 4 with 2 heads and 1 group of state 2, 4 routed experts in a
# latent space of 3. Same proportions as the 120B checkpoint, small enough to
# write shapes down.
_NEMOTRON_ARGS = types.SimpleNamespace(
    kv_channels=2,
    hidden_size=8,
    num_attention_heads=4,
    num_query_groups=2,
    num_layers=3,
    q_lora_rank=None,
)

_NEMOTRON_CONFIG = types.SimpleNamespace(
    model_type="nemotron_h",
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=2,
    num_hidden_layers=3,
    tie_word_embeddings=False,
)

# Every parameter the hybrid stack creates, one per name the model actually asks
# for: layer 0 Mamba-2, layer 1 attention, layer 2 latent MoE. The shapes are the
# real ones for the toy config -- in_proj packs [z(4) x(4) B(2) C(2) dt(2)] and
# conv1d packs [x(4) B(2) C(2)].
_NEMOTRON_PARAMETERS = [
    ("embedding.word_embeddings.weight", (16, 8)),
    ("decoder.final_norm.weight", (8,)),
    ("output_layer.weight", (16, 8)),
    ("decoder.layers.0.mixer.in_proj.layer_norm_weight", (8,)),
    ("decoder.layers.0.mixer.in_proj.weight", (14, 8)),
    ("decoder.layers.0.mixer.conv1d_weight", (8, 1, 4)),
    ("decoder.layers.0.mixer.conv1d_bias", (8,)),
    ("decoder.layers.0.mixer.A_log", (2,)),
    ("decoder.layers.0.mixer.D", (2,)),
    ("decoder.layers.0.mixer.dt_bias", (2,)),
    ("decoder.layers.0.mixer.norm.weight", (4,)),
    ("decoder.layers.0.mixer.out_proj.weight", (8, 4)),
    ("decoder.layers.1.self_attention.linear_qkv.layer_norm_weight", (8,)),
    ("decoder.layers.1.self_attention.linear_qkv.weight", (16, 8)),
    ("decoder.layers.1.self_attention.linear_proj.weight", (8, 8)),
    ("decoder.layers.2.pre_mlp_layernorm.weight", (8,)),
    ("decoder.layers.2.mlp.router.weight", (4, 8)),
    ("decoder.layers.2.mlp.router.expert_bias", (4,)),
    ("decoder.layers.2.mlp.fc1_latent_proj.weight", (3, 8)),
    ("decoder.layers.2.mlp.fc2_latent_proj.weight", (8, 3)),
    ("decoder.layers.2.mlp.shared_experts.linear_fc1.weight", (6, 8)),
    ("decoder.layers.2.mlp.shared_experts.linear_fc2.weight", (8, 6)),
    ("decoder.layers.2.mlp.experts.linear_fc1.weight3", (5, 3)),
    ("decoder.layers.2.mlp.experts.linear_fc2.weight3", (3, 5)),
]


@pytest.fixture
def nemotron_h_loader_without_tp_guard(monkeypatch):
    """The loader refuses TP > 1 by asking megatron's mpu, which CPU CI has no copy of."""
    monkeypatch.setattr(nemotron_h_loader, "_assert_no_tensor_parallel", lambda: None)
    return nemotron_h_loader.nemotron_h_hf_tensor


@pytest.mark.unit
@pytest.mark.parametrize(("suffix", "shape"), _NEMOTRON_PARAMETERS, ids=[name for name, _ in _NEMOTRON_PARAMETERS])
def test_nemotron_h_round_trips_every_parameter(nemotron_h_loader_without_tp_guard, suffix, shape):
    name = f"module.module.{suffix}"
    parameter = torch.arange(torch.tensor(shape).prod()).reshape(shape)

    hf_tensors = dict(convert_nemotron_h_to_hf(_NEMOTRON_ARGS, name, parameter))
    loaded = nemotron_h_loader_without_tp_guard(name, Reader(**hf_tensors), _NEMOTRON_CONFIG)

    assert torch.equal(loaded, parameter)


@pytest.mark.unit
def test_nemotron_h_exports_the_names_the_hf_checkpoint_uses():
    # The three that are not a rename of the Megatron name, spelled out so a
    # typo in the shared map is a failing test rather than a silent weight sync.
    qkv = dict(
        convert_nemotron_h_to_hf(
            _NEMOTRON_ARGS,
            "module.module.decoder.layers.1.self_attention.linear_qkv.weight",
            torch.arange(16 * 8).reshape(16, 8),
        )
    )
    assert list(qkv) == [
        "backbone.layers.1.mixer.q_proj.weight",
        "backbone.layers.1.mixer.k_proj.weight",
        "backbone.layers.1.mixer.v_proj.weight",
    ]
    # Group 0 is rows 0..7 of the fused tensor: 4 q rows, then k, then v.
    assert torch.equal(qkv["backbone.layers.1.mixer.q_proj.weight"][:4], torch.arange(32).reshape(4, 8))
    assert torch.equal(qkv["backbone.layers.1.mixer.k_proj.weight"][:2], torch.arange(32, 48).reshape(2, 8))

    # Squared-ReLU experts: linear_fc1 is up_proj alone, with no gate half to split.
    expert = convert_nemotron_h_to_hf(
        _NEMOTRON_ARGS,
        "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight3",
        torch.randn(5, 3),
    )
    assert [name for name, _ in expert] == ["backbone.layers.2.mixer.experts.3.up_proj.weight"]

    router = convert_nemotron_h_to_hf(
        _NEMOTRON_ARGS, "module.module.decoder.layers.2.mlp.router.expert_bias", torch.randn(4)
    )
    assert [name for name, _ in router] == ["backbone.layers.2.mixer.gate.e_score_correction_bias"]


@pytest.mark.unit
def test_nemotron_h_weight_sync_reaches_the_exporter():
    # What the RL path actually passes: model_name is type(hf_config).__name__.lower(),
    # i.e. "nemotronhconfig". Without this branch every rollout's update_weights
    # raises ValueError: Unsupported model.
    parameter = torch.randn(8, 8)

    converted = _convert_to_hf_core(
        _NEMOTRON_ARGS,
        "nemotronhconfig",
        "module.module.decoder.layers.1.self_attention.linear_proj.weight",
        parameter,
    )

    assert converted == [("backbone.layers.1.mixer.o_proj.weight", parameter)]


@pytest.mark.unit
def test_nemotron_h_refuses_to_export_an_mtp_block():
    # The released checkpoint has one and this stack does not build it; a name
    # from one that did would otherwise fall through to the layer regex.
    with pytest.raises(ValueError, match="no MTP block"):
        convert_nemotron_h_to_hf(_NEMOTRON_ARGS, "module.module.mtp.layers.0.enorm.weight", torch.randn(8))


def _update_weight_common():
    """``update_weight/common.py``, importable without megatron (CPU CI)."""
    name = "slime.backends.megatron_utils.update_weight.common"
    if name in sys.modules:
        return sys.modules[name]
    if not _has_megatron:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        core.mpu = types.ModuleType("megatron.core.mpu")
        transformer_layer = types.ModuleType("megatron.core.transformer.transformer_layer")
        transformer_layer.get_transformer_layer_offset = lambda *args, **kwargs: 0
        for module_name, module in {
            "megatron": megatron,
            "megatron.core": core,
            "megatron.core.mpu": core.mpu,
            "megatron.core.transformer": types.ModuleType("megatron.core.transformer"),
            "megatron.core.transformer.transformer_layer": transformer_layer,
        }.items():
            sys.modules.setdefault(module_name, module)
    return importlib.import_module(name)


@pytest.mark.unit
def test_ungated_linear_fc1_is_a_plain_concat_across_tp():
    # Nemotron's experts are squared-ReLU: linear_fc1 is up_proj alone. Splitting
    # each rank's shard in half and reordering -- what a gated model needs -- would
    # interleave garbage here.
    common = _update_weight_common()
    ungated = types.SimpleNamespace(swiglu=False, squared_relu=True)
    partitions = [torch.arange(8).reshape(4, 2), torch.arange(8, 16).reshape(4, 2)]

    merged = common.merge_tp_partitions(ungated, "mlp.experts.linear_fc1.weight0", partitions, 0)

    assert torch.equal(merged, torch.arange(16).reshape(8, 2))


@pytest.mark.unit
def test_gated_linear_fc1_still_degroups_gate_and_up():
    common = _update_weight_common()
    gated = types.SimpleNamespace(swiglu=True)
    # Each rank holds [gate_r; up_r]; the full tensor is [gate_0, gate_1, up_0, up_1].
    partitions = [torch.tensor([[0.0], [10.0]]), torch.tensor([[1.0], [11.0]])]

    merged = common.merge_tp_partitions(gated, "mlp.linear_fc1.weight", partitions, 0)

    assert torch.equal(merged, torch.tensor([[0.0], [1.0], [10.0], [11.0]]))


@pytest.mark.unit
def test_partition_sizes_reassemble_the_mamba_in_proj():
    # in_proj packs [z, x, B, C, dt] and every TP rank holds a slice of each, so
    # the gathered tensor has to be regrouped by component. Two ranks, one row of
    # z/x and two of dt each.
    common = _update_weight_common()
    args = types.SimpleNamespace(swiglu=False)
    partitions = [
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        torch.tensor([[4.0], [5.0], [6.0], [7.0]]),
    ]

    merged = common.merge_tp_partitions(args, "mixer.in_proj.weight", partitions, 0, [1, 1, 2])

    assert torch.equal(merged, torch.tensor([[0.0], [4.0], [1.0], [5.0], [2.0], [3.0], [6.0], [7.0]]))


@pytest.mark.unit
def test_reader_dequantizes_block_scaled_fp8(tmp_path):
    weight = torch.linspace(-2, 2, 128 * 128).view(128, 128).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0]])
    save_file(
        {"weight": weight, "weight_scale_inv": scale},
        tmp_path / "model.safetensors",
    )

    loaded = SafetensorReader(tmp_path).get_tensor("weight")

    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded, weight.to(torch.bfloat16) * 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# ---------------------------------------------------------------------------
# Nemotron 3.5 Super VL. Everything above is the text Nemotron-3; these three
# tests cover what job 18823425 found the hard way -- the first RL run to reach
# update_weights on the VL model died in the exporter, twice over.
# ---------------------------------------------------------------------------

_NEMOTRON_VL_CONFIG = types.SimpleNamespace(
    model_type="nemotron_h_omni",
    llm_config=_NEMOTRON_CONFIG,
    vision_config=types.SimpleNamespace(
        hidden_size=6,
        summary_idxs=[0, 1],
        layerscale_value=1.0,
    ),
)

# `_convert_to_hf_core` lowercases and strips _/- from the model name. Both the
# config class and an explicit --model-name normalise into the "nemotronh"
# branch, and only the VL one carries "omni".
_VL_MODEL_NAME = "nemotronhomnireasoningv3config"


@pytest.mark.unit
def test_nemotron_vl_prefixes_the_language_tower():
    """42,683 of the checkpoint's 43,078 tensors live under `language_model.`.

    SGLang's NemotronH_Nano_VL_V2.load_weights routes on four prefixes with no
    else branch, so an unprefixed name is dropped in silence -- the sync reports
    success and the engine keeps its startup weights forever. This is the bug
    that would have survived fixing the vision half alone.
    """
    embedding = convert_nemotron_h_to_hf(
        _NEMOTRON_ARGS, "module.module.embedding.word_embeddings.weight", torch.randn(4, 8), _VL_MODEL_NAME
    )
    assert [name for name, _ in embedding] == ["language_model.backbone.embeddings.weight"]

    layer = convert_nemotron_h_to_hf(
        _NEMOTRON_ARGS, "module.module.decoder.layers.2.mlp.router.expert_bias", torch.randn(4), _VL_MODEL_NAME
    )
    assert [name for name, _ in layer] == [
        "language_model.backbone.layers.2.mixer.gate.e_score_correction_bias"
    ]

    # The text model must not grow the prefix, and the default keeps it off so
    # every existing caller is unchanged.
    text = convert_nemotron_h_to_hf(
        _NEMOTRON_ARGS, "module.module.decoder.layers.2.mlp.router.expert_bias", torch.randn(4)
    )
    assert [name for name, _ in text] == ["backbone.layers.2.mixer.gate.e_score_correction_bias"]


@pytest.mark.unit
def test_nemotron_vl_vision_round_trips(nemotron_h_loader_without_tp_guard):
    """The vision half, through the same shared tables the loader reads."""
    cases = [
        ("vision_model.embeddings.cls_register_token", (1, 1, 6)),
        ("vision_model.embeddings.patch_projection.weight", (6, 3)),
        ("vision_projector.mlp1.norm.weight", (6,)),
        ("vision_projector.mlp1.linear1.weight", (4, 6)),
        ("vision_model.encoder.layer.0.norm1.weight", (6,)),
        ("vision_model.encoder.layer.0.mlp.fc1.weight", (12, 6)),
        ("vision_model.encoder.layer.0.attention.output.dense.weight", (6, 6)),
    ]
    for name, shape in cases:
        parameter = torch.arange(torch.tensor(shape).prod()).reshape(shape)
        hf_tensors = dict(convert_nemotron_h_to_hf(_NEMOTRON_ARGS, name, parameter, _VL_MODEL_NAME))
        assert hf_tensors, f"{name} exported nothing"
        loaded = nemotron_h_loader_without_tp_guard(name, Reader(**hf_tensors), _NEMOTRON_VL_CONFIG)
        assert torch.equal(loaded, parameter), name


@pytest.mark.unit
def test_nemotron_vl_fuses_vision_qkv_and_skips_what_the_checkpoint_lacks():
    """q/k/v are three Megatron parameters and one HF tensor.

    `convert_to_hf` sees one parameter at a time, so the first two return
    nothing and the third emits the fused tensor. RADIO is plain MHA, so the
    fusion is a plain cat in q, k, v order -- the inverse of the loader's
    `fused.chunk(3, dim=0)`.
    """
    q = torch.full((6, 6), 1.0)
    k = torch.full((6, 6), 2.0)
    v = torch.full((6, 6), 3.0)

    base = "vision_model.encoder.layer.1.attention.attention"
    assert convert_nemotron_h_to_hf(_NEMOTRON_ARGS, f"{base}.query.weight", q, _VL_MODEL_NAME) == []
    assert convert_nemotron_h_to_hf(_NEMOTRON_ARGS, f"{base}.key.weight", k, _VL_MODEL_NAME) == []
    fused = convert_nemotron_h_to_hf(_NEMOTRON_ARGS, f"{base}.value.weight", v, _VL_MODEL_NAME)

    assert [name for name, _ in fused] == ["vision_model.radio_model.model.blocks.1.attn.qkv.weight"]
    assert torch.equal(fused[0][1], torch.cat([q, k, v], dim=0))

    # Two parameters exist in the module tree and in no checkpoint: a
    # config-derived buffer and an identity LayerScale. Both are frozen
    # (nemotron_35_super_vl.py calls vision_model.requires_grad_(False)), so
    # exporting nothing is exact rather than approximate.
    assert convert_nemotron_h_to_hf(_NEMOTRON_ARGS, "vision_model.summary_idxs", torch.tensor([0, 1]), _VL_MODEL_NAME) == []
    assert (
        convert_nemotron_h_to_hf(
            _NEMOTRON_ARGS, "vision_model.encoder.layer.1.layer_scale1.lambda1", torch.ones(6), _VL_MODEL_NAME
        )
        == []
    )
