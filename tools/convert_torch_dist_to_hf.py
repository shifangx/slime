import argparse
import io
import json
import os
import pickle
import re
import shutil
import time


import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig
from typing_extensions import override

from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from slime.backends.megatron_utils.megatron_to_hf import (
    assert_conversion_buffers_drained,
    convert_to_hf,
    remove_padding,
)


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def load_conversion_args(input_dir):
    """Read both legacy common.pt and current MCore embedded common state."""
    legacy = os.path.join(input_dir, "common.pt")
    if os.path.isfile(legacy):
        common = torch.load(legacy, weights_only=False)
    else:
        key = "common_state/shard_0_1"
        state = {key: io.BytesIO()}
        dist_cp.load(state, storage_reader=WrappedStorageReader(input_dir), no_dist=True)
        common = state[key]
        if isinstance(common, io.BytesIO):
            common.seek(0)
            common = torch.load(common, weights_only=False)
        if isinstance(common, list):
            if len(common) != 1:
                raise ValueError("Expected one global MCore common_state shard")
            common = common[0]
    return common["args"]


def get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == num_experts
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            expert_param = param[expert_id]
            yield expert_name, expert_param
    else:
        yield name, param


def get_layer_param(args, name, param):
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        assert param.shape[0] == num_layers
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            layer_param = param[layer_id]
            yield from get_expert_param(args, layer_name, layer_param)
    else:
        yield from get_expert_param(args, name, param)


# Megatron's dist-checkpoint stores the Mamba mixers' packed tensors as one
# entry per *component*, named by it -- `mixer.in_proj.weight.z`,
# `...weight.x`, `.B`, `.C`, `.dt` -- where the model itself holds a single
# packed tensor. Nothing downstream knows those names: `convert_to_hf` is
# written against the packed spelling, so without this every Mamba parameter of
# a torch_dist checkpoint raises `Unknown parameter name`. On the RL weight-sync
# path the question never arises, because there the tensors come from a live
# model and `update_weight/common.py::merge_tp_partitions` has already
# reassembled them.
#
# The order is the one that function documents and that
# `megatron_to_hf/nemotron_h.py` repeats: `in_proj` is `[z, x, B, C, dt]` and
# `conv1d` is `[x, B, C]`. Getting it wrong is the dangerous failure here --
# the tensor keeps its shape and changes its contents, so nothing raises. Two
# independent statements of the order in this codebase agree, and
# `tests/test_mamba_component_merge.py` pins it, but neither is a measurement
# against the weights: an exported checkpoint that serves like the original is.
MAMBA_PACKED_COMPONENTS = {
    "in_proj.weight": ("z", "x", "B", "C", "dt"),
    "conv1d.weight": ("x", "B", "C"),
    "conv1d.bias": ("x", "B", "C"),
}


def merge_mamba_components(state_dict):
    """Concatenate per-component Mamba entries back into their packed tensors."""
    merged = {}
    groups = {}
    for name, param in state_dict.items():
        base, _, component = name.rpartition(".")
        packed = next((p for p in MAMBA_PACKED_COMPONENTS if base.endswith(p)), None)
        if packed is None:
            merged[name] = param
        else:
            groups.setdefault((base, packed), {})[component] = param

    for (base, packed), parts in groups.items():
        order = MAMBA_PACKED_COMPONENTS[packed]
        missing = [component for component in order if component not in parts]
        extra = [component for component in parts if component not in order]
        if missing or extra:
            raise ValueError(
                f"{base}: cannot reassemble the packed Mamba tensor -- "
                f"missing {missing}, unexpected {extra}; expected exactly {list(order)}"
            )
        if base in merged:
            raise ValueError(f"Both packed and component forms exist for {base}")
        merged[base] = torch.cat([parts[component] for component in order], dim=0)

    if groups:
        print(f"reassembled {len(groups)} packed Mamba tensors from {sum(len(p) for p in groups.values())} components")
    return merged


def get_named_params(args, state_dict):
    for name, param in merge_mamba_components(state_dict).items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)


def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None, origin_hf_dir=None):
    nemotron = "nemotronh" in model_name.lower().replace("_", "").replace("-", "")
    if nemotron and origin_hf_dir is None:
        raise ValueError("Nemotron export requires --origin-hf-dir to retain frozen vision and untrained MTP tensors")
    if origin_hf_dir is not None and os.path.realpath(output_dir) == os.path.realpath(origin_hf_dir):
        raise ValueError("Output must not overwrite the original HF checkpoint")
    reference = SafetensorReader(origin_hf_dir) if nemotron else None
    print(f"start saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # 2GB
    current_size = 0
    total_size = 0
    modeltensors = [{}]
    converted_names = set()
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        converted_named_tensors = convert_to_hf(args, model_name, name, param)
        for converted_name, converted_param in converted_named_tensors:
            if converted_name in converted_names:
                raise ValueError(f"Duplicate converted tensor: {converted_name}")
            if reference is not None:
                if converted_name in reference:
                    source = reference.get_slice(converted_name)
                    if tuple(source.get_shape()) != tuple(converted_param.shape):
                        raise ValueError(f"Export shape differs from the original checkpoint: {converted_name}")
                    # MCore keeps router bias/norms in FP32. Preserve the actual
                    # source storage dtype instead of casting the whole model.
                    source_dtype = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
                    converted_param = converted_param.to(dtype=source_dtype[source.get_dtype()])
                elif not re.fullmatch(r"vision_model\.radio_model\.model\.blocks\.\d+\.ls[12]", converted_name):
                    # Shifang's refit includes synthesized RADIO LayerScale
                    # tensors even when the original HF snapshot omits them.
                    raise ValueError(f"Unexpected Nemotron export tensor: {converted_name}")
            converted_names.add(converted_name)
            tensor_size = converted_param.numel() * converted_param.element_size()
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param
            current_size += tensor_size
            total_size += tensor_size

    if nemotron:
        assert_conversion_buffers_drained("end of HF export")
        missing = set(reference.weight_map) - converted_names
        unsupported = sorted(
            name for name in missing if not name.startswith(("vision_model.", "language_model.mtp.", "mtp."))
        )
        if unsupported:
            raise ValueError(f"Missing trained Nemotron tensors; refusing stale source fallback: {unsupported}")

    if origin_hf_dir is not None:
        safetensors_files = [f for f in os.listdir(origin_hf_dir) if f.endswith(".safetensors")]
        for filename in safetensors_files:
            with safetensors.safe_open(os.path.join(origin_hf_dir, filename), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k not in converted_names:
                        converted_name = k
                        print(f"add {k} from origin hf checkpoint")
                        converted_param = f.get_tensor(k)
                        converted_names.add(k)
                        tensor_size = converted_param.numel() * converted_param.element_size()
                        if tensor_size + current_size > chunk_size:
                            modeltensors.append({})
                            current_size = 0
                        modeltensors[-1][converted_name] = converted_param
                        current_size += tensor_size
                        total_size += tensor_size

    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}

    num_files = len(modeltensors)
    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        for key in tensors.keys():
            metadata["weight_map"][key] = filename
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    json.dump(metadata, open(index_filepath, "w"), indent=2)
    print(f"{index_filepath} saved.")

    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        t = time.time()
        filepath = os.path.join(output_dir, filename)
        safetensors.torch.save_file(tensors, filepath)
        print(f"{filename} saved in {time.time() - t:.2f} sec.")


def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        default=None,
        help="use the origin hf dir to copy files like tokenizer, config.json, etc.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "-a", "--add-missing-from-origin-hf", action="store_true", help="Add missing weights from origin hf checkpoint"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5 * 1024**3,
        help="Chunk size for saving tensors, default is 2GB.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocab size for removing padding, if applicable. If not provided, no padding will be removed.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError(
            "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
        )

    if args.model_name is None:
        hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
        args.model_name = type(hf_config).__name__.lower()

    state_dict = {}
    print(f"loading model from {args.input_dir}")
    t = time.time()
    megatron_args = load_conversion_args(args.input_dir)
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(args.input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"model loaded in {time.time()-t:.2f} sec.")

    save_tensors(
        megatron_args,
        args.model_name,
        state_dict,
        args.output_dir,
        args.chunk_size,
        args.vocab_size,
        (
            args.origin_hf_dir
            if args.add_missing_from_origin_hf
            or "nemotronh" in args.model_name.lower().replace("_", "").replace("-", "")
            else None
        ),
    )

    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)
