"""Small checkpoint/processor regressions; no 120B weights or serving engine."""

import argparse
import io
import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dist_cp
from PIL import Image
from safetensors.torch import save_file
from tools.convert_torch_dist_to_hf import load_conversion_args, merge_mamba_components, save_tensors

from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from slime.backends.megatron_utils.megatron_to_hf.nemotron_h import convert_nemotron_h_to_hf
from slime.utils.data import filter_long_prompt
from slime.utils.processing_utils import process_vision_info

NUM_GPUS = 0


def test_nemotron_images_keep_original_geometry():
    processor = SimpleNamespace(_slime_model_type="nemotron_h_omni")
    images = [Image.new("RGBA", (41, 63)), Image.new("RGB", (97, 31))]
    messages = [{"role": "user", "content": [{"type": "image", "image": image} for image in images]}]

    result = process_vision_info(messages, processor)

    assert [image.size for image in result["images"]] == [(41, 63), (97, 31)]
    assert all(image.mode == "RGB" for image in result["images"])


def test_nemotron_length_filter_reuses_images_after_rendering():
    images = [Image.new("RGB", (41, 63)), Image.new("RGB", (97, 31))]

    class Processor:
        _slime_model_type = "nemotron_h_omni"

        def __call__(self, *, text, images, return_tensors):
            assert text == "<image> compare <image>"
            assert images is sample.multimodal_inputs["images"]
            assert return_tensors == "pt"
            return {"input_ids": [[1, 2, 3, 4]]}

    sample = SimpleNamespace(prompt="<image> compare <image>", multimodal_inputs={"images": images})
    assert filter_long_prompt([sample], None, Processor(), 4) == [sample]
    assert filter_long_prompt([sample], None, Processor(), 3) == []


@pytest.mark.parametrize("legacy", [True, False])
def test_export_reads_real_dcp_common_metadata(tmp_path, legacy):
    common = {"args": argparse.Namespace(hidden_size=8), "checkpoint_version": 3.0, "iteration": 2}
    if legacy:
        torch.save(common, tmp_path / "common.pt")
    else:
        # Match MCore's ShardedObject representation, rather than letting DCP
        # flatten the Python dictionary into separate metadata keys.
        serialized = io.BytesIO()
        torch.save([common], serialized)
        dist_cp.save({"common_state/shard_0_1": serialized}, checkpoint_id=tmp_path)

    assert load_conversion_args(tmp_path).hidden_size == 8


def test_vl_export_trims_padding_after_language_prefix():
    parameter = torch.arange(12).reshape(6, 2)
    result = convert_nemotron_h_to_hf(
        SimpleNamespace(vocab_size=4), "module.module.language_model.output_layer.weight", parameter
    )

    assert result[0][0] == "language_model.lm_head.weight"
    assert torch.equal(result[0][1], parameter[:4])


def test_mamba_export_rejects_unknown_and_duplicate_components():
    prefix = "language_model.decoder.layers.0.mixer.conv1d.weight"
    state = {f"{prefix}.{component}": torch.ones(2, 2) for component in ("x", "B", "C")}
    with pytest.raises(ValueError, match="unexpected"):
        merge_mamba_components({**state, f"{prefix}.dt": torch.ones(2, 2)})
    with pytest.raises(ValueError, match="Both packed and component"):
        merge_mamba_components({**state, prefix: torch.ones(6, 2)})


def test_complete_export_preserves_fp32_bias_and_synthesized_layer_scale(tmp_path):
    origin, output = tmp_path / "origin", tmp_path / "export"
    origin.mkdir()
    bias = torch.tensor([0.001001, 0.001003], dtype=torch.float32)
    source = {
        "language_model.lm_head.weight": torch.zeros(4, 2, dtype=torch.bfloat16),
        "language_model.backbone.layers.0.mixer.gate.e_score_correction_bias": bias,
        "mlp1.0.weight": torch.zeros(2, dtype=torch.bfloat16),
        "language_model.mtp.layers.0.norm.weight": torch.ones(2, dtype=torch.bfloat16),
        "vision_model.radio_model.input_conditioner.norm_mean": torch.zeros(3),
    }
    save_file(source, origin / "model.safetensors")
    state = {
        "language_model.output_layer.weight": torch.ones(4, 2),
        "language_model.decoder.layers.0.mlp.router.expert_bias": bias.clone(),
        "vision_projector.mlp1.norm.weight": torch.ones(2),
        "vision_model.encoder.layer.0.layer_scale1.lambda1": torch.ones(2),
    }
    args = argparse.Namespace(vocab_size=4, num_experts=0, num_layers=1, q_lora_rank=None)

    save_tensors(args, "nemotron_h_omni", state, output, 1024, origin_hf_dir=origin)

    exported = SafetensorReader(output)
    assert torch.equal(exported.get_tensor("language_model.lm_head.weight"), torch.ones(4, 2, dtype=torch.bfloat16))
    assert (
        exported.get_tensor("language_model.backbone.layers.0.mixer.gate.e_score_correction_bias").dtype
        == torch.float32
    )
    assert torch.equal(
        exported.get_tensor("language_model.backbone.layers.0.mixer.gate.e_score_correction_bias"), bias
    )
    assert torch.equal(exported.get_tensor("vision_model.radio_model.model.blocks.0.ls1"), torch.ones(2))
    assert torch.equal(
        exported.get_tensor("language_model.mtp.layers.0.norm.weight"),
        source["language_model.mtp.layers.0.norm.weight"],
    )
    metadata = json.loads((output / "model.safetensors.index.json").read_text())
    tensors = [exported.get_tensor(name) for name in exported.weight_map]
    assert metadata["metadata"]["total_size"] == sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    del state["vision_projector.mlp1.norm.weight"]
    with pytest.raises(ValueError, match="Missing trained Nemotron tensors"):
        save_tensors(args, "nemotron_h_omni", state, tmp_path / "incomplete", 1024, origin_hf_dir=origin)
