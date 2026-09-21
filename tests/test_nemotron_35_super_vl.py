"""Small mapping regressions for the native Nemotron VL preview.

Model/mapping cases reuse Shifang's implementation; no real checkpoint is loaded.
The plugin execution test additionally requires an importable MCore environment.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.hf_to_megatron import _LOADERS
from slime.backends.megatron_utils.hf_to_megatron import nemotron_h as loader
from slime.backends.megatron_utils.hf_to_megatron.common import restore_fp32_router_buffers
from slime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core, assert_conversion_buffers_drained
from slime.backends.megatron_utils.megatron_to_hf import nemotron_h as exporter

NUM_GPUS = 0

ARGS = SimpleNamespace(
    hidden_size=8, num_attention_heads=4, num_query_groups=2, kv_channels=2, vocab_size=16, q_lora_rank=None
)
CONFIG = SimpleNamespace(
    model_type="nemotron_h_omni",
    llm_config=SimpleNamespace(hidden_size=8, num_attention_heads=4, num_key_value_heads=2, head_dim=2),
    vision_config=SimpleNamespace(hidden_size=6, layerscale_value=1.0, summary_idxs=[0, 1]),
)


class Reader(dict):
    def get_tensor(self, name):
        return self[name]


def test_provider_uses_public_hybrid_model_directly(monkeypatch):
    """Check the adapter's constructor contract without initializing a GPU model."""
    stack_spec = object()

    class HybridModelSpy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    for module_name, module in {
        "hybrid_layer_allocation": SimpleNamespace(get_hybrid_total_layer_count=len),
        "hybrid_layer_specs": SimpleNamespace(hybrid_stack_spec=stack_spec),
        "hybrid_model": SimpleNamespace(HybridModel=HybridModelSpy),
    }.items():
        monkeypatch.setitem(sys.modules, f"megatron.core.models.hybrid.{module_name}", module)

    path = Path(__file__).resolve().parents[1] / "slime_plugins/models/nemotron_h.py"
    spec = importlib.util.spec_from_file_location("nemotron_main_provider_probe", path)
    provider_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider_module)
    args = SimpleNamespace(
        hybrid_layer_pattern="M*E",
        num_layers=3,
        padded_vocab_size=128,
        max_position_embeddings=512,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=True,
        position_embedding_type="none",
        rotary_percent=1.0,
        rotary_base=10000,
    )
    config = SimpleNamespace(is_hybrid_model=True)
    build = provider_module.get_nemotron_h_spec(args, config, scatter_embedding_sequence_parallel=False)
    model = build(pre_process=True, post_process=False)

    assert type(model) is HybridModelSpy
    assert model.kwargs["config"] is config
    assert model.kwargs["hybrid_stack_spec"] is stack_spec
    assert model.kwargs["hybrid_layer_pattern"] == "M*E"
    assert model.kwargs["scatter_embedding_sequence_parallel"] is False
    assert model.kwargs["post_process"] is False
    assert "mamba_stack_spec" not in model.kwargs
    with pytest.raises(ValueError, match="virtual pipeline"):
        build(vp_stage=1)
    args.num_layers = 2
    with pytest.raises(ValueError, match="pattern has 3 layers"):
        provider_module.get_nemotron_h_spec(args, config)


@pytest.mark.parametrize(
    ("suffix", "shape"),
    [
        ("embedding.word_embeddings.weight", (16, 8)),
        ("decoder.final_norm.weight", (8,)),
        ("decoder.layers.0.mixer.in_proj.weight", (14, 8)),
        ("decoder.layers.0.mixer.conv1d_weight", (8, 1, 4)),
        ("decoder.layers.0.mixer.A_log", (2,)),
        ("decoder.layers.1.self_attention.linear_qkv.weight", (16, 8)),
        ("decoder.layers.1.self_attention.linear_qkv.layer_norm_weight", (8,)),
        ("decoder.layers.2.mlp.router.expert_bias", (4,)),
        ("decoder.layers.2.mlp.fc1_latent_proj.weight", (3, 8)),
        ("decoder.layers.2.mlp.shared_experts.linear_fc1.weight", (6, 8)),
        ("decoder.layers.2.mlp.experts.linear_fc1.weight3", (5, 3)),
        ("decoder.layers.2.mlp.experts.linear_fc2.weight3", (3, 5)),
    ],
)
def test_vl_language_mapping_round_trip(monkeypatch, suffix, shape):
    monkeypatch.setattr(loader, "_assert_no_tensor_parallel", lambda: None)
    assert _LOADERS["nemotron_h"] is _LOADERS["nemotron_h_omni"]
    parameter = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    name = f"module.module.language_model.{suffix}"

    converted = _convert_to_hf_core(ARGS, "NemotronH_Omni_Reasoning_V3_Config", name, parameter)
    assert all(key.startswith("language_model.") for key, _ in converted)
    assert torch.equal(loader.nemotron_h_hf_tensor(name, Reader(converted), CONFIG), parameter)


def test_vl_vision_qkv_and_projector_mapping():
    parts = {name: torch.full((6, 6), float(i)) for i, name in enumerate(("query", "key", "value"))}
    prefix = "vision_model.encoder.layer.0.attention.attention"
    assert exporter.convert_nemotron_h_to_hf(ARGS, f"{prefix}.query.weight", parts["query"]) == []
    with pytest.raises(ValueError, match="Incomplete Nemotron vision QKV"):
        assert_conversion_buffers_drained()
    assert exporter.convert_nemotron_h_to_hf(ARGS, f"{prefix}.key.weight", parts["key"]) == []
    converted = exporter.convert_nemotron_h_to_hf(ARGS, f"{prefix}.value.weight", parts["value"])
    assert_conversion_buffers_drained()
    for name, parameter in parts.items():
        assert torch.equal(loader._vision_hf_tensor(f"{prefix}.{name}.weight", Reader(converted), CONFIG), parameter)

    for name in (
        "vision_projector.mlp1.norm.weight",
        "vision_projector.vision_final_layernorm.weight",
        "vision_projector.vision_final_layernorm.bias",
    ):
        parameter = torch.arange(6)
        converted = exporter.convert_nemotron_h_to_hf(ARGS, name, parameter)
        assert torch.equal(loader._vision_hf_tensor(name, Reader(converted), CONFIG), parameter)


def test_full_vision_refit_includes_synthesized_layer_scales():
    for index in (1, 2):
        name = f"vision_model.encoder.layer.0.layer_scale{index}.lambda1"
        parameter = loader._vision_hf_tensor(name, Reader(), CONFIG)
        assert torch.equal(parameter, torch.ones(6))
        converted = exporter.convert_nemotron_h_to_hf(ARGS, name, parameter)
        assert converted[0][0] == f"vision_model.radio_model.model.blocks.0.ls{index}"
        assert torch.equal(loader._vision_hf_tensor(name, Reader(converted), CONFIG), parameter)


def test_router_buffer_is_fp32_before_hf_copy():
    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(2, dtype=torch.bfloat16))

        def _maintain_float32_expert_bias(self):
            self.expert_bias = self.expert_bias.float()

    router = Router()
    source = torch.tensor([0.001001, 0.001003], dtype=torch.float32)
    assert not torch.equal(source.bfloat16().float(), source)
    assert restore_fp32_router_buffers([router]) == 1
    router.expert_bias.copy_(source)
    assert torch.equal(router.expert_bias, source)


def test_cp1_odd_length_and_frozen_encoder_keep_projector_gradients(monkeypatch):
    # Importing core alone does not establish that its model dependencies
    # (e.g. Triton in the training environment) are available on a CPU host.
    pytest.importorskip("megatron.core.models.hybrid.hybrid_model", exc_type=ImportError)
    from slime_plugins.models import nemotron_35_super_vl as plugin

    def reject_cp_indices(*args):
        raise AssertionError("CP1 must not invoke two-chunk CP indexing")

    monkeypatch.setattr(plugin, "get_packed_cp_local_indices", reject_cp_indices)
    model = plugin.NemotronOmniVLModel.__new__(plugin.NemotronOmniVLModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False)
    model.image_token_id = 18
    model.language_model = SimpleNamespace(
        embedding=lambda input_ids, position_ids: torch.zeros(input_ids.shape[1], 1, 2, dtype=torch.bfloat16)
    )
    model.vision_model = torch.nn.Conv2d(3, 2, 1, dtype=torch.bfloat16).requires_grad_(False)

    class Projector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

        def forward(self, pixels, vision):
            return vision(pixels).flatten(2).transpose(1, 2) * self.weight

    model.vision_projector = Projector()
    model.train()
    tokens = torch.tensor([[1, 18, 2, 3, 4]])
    output = model._inject_vision_embeddings(tokens, tokens, torch.tensor([0, 5]), None, torch.ones(3, 1, 1))
    output.float().sum().backward()
    assert output.shape == (5, 1, 2)
    assert not model.vision_model.training
    assert model.vision_projector.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.vision_model.parameters())
