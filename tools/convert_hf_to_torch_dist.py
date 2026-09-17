import argparse
import gc
import os
import pickle
import shutil

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

from slime.backends.megatron_utils.arguments import set_default_megatron_args
from slime.backends.megatron_utils.hf_to_megatron import load_hf_weights
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func
from slime.observability.logging_utils import configure_logger
from slime.utils import accelerator
from slime.utils.memory_utils import print_memory


def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--custom-model-provider-path",
        type=str,
        default=None,
        help="Path to a custom model provider function.",
    )
    parser.add_argument("--allgather-cp", action="store_true", default=False)
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser


def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using too many GPUs for this conversion."
    )

    def ceildiv(a, b):
        return -(a // -b)

    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            if args.decoder_last_pipeline_num_layers > 0:
                break

            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    return args


def save_common_state(args, checkpoint_dir):
    """Write `common.pt` next to the shards.

    megatron's `torch_dist` save does not produce one -- no checkpoint in this
    tree has it, converted or trained, only `.metadata` / `metadata.json` /
    `__N_M.distcp` -- but slime's own `tools/convert_torch_dist_to_hf.py` opens
    `<iteration>/common.pt` and reads `["args"]` from it unconditionally. So
    without this file the exporter cannot read the checkpoint the importer just
    wrote, which is how a 227 GB artifact ends up being unreadable by the tool
    written to consume it.

    What the readers actually want out of it is small -- `num_layers`,
    `num_experts`, `vocab_size`, `hidden_size`, `kv_channels`,
    `num_attention_heads`, `num_query_groups`, `q_lora_rank` -- but the whole
    namespace is stored, because the next consumer will want a different field
    and guessing which is how this file came to be missing in the first place.

    Attributes that do not pickle are dropped and named rather than failing the
    job: the checkpoint is already on disk by the time this runs, and an hour of
    conversion should not be lost to one un-serializable flag.
    """
    keep, dropped = {}, []
    for key, value in vars(args).items():
        try:
            pickle.dumps(value)
        except Exception:  # noqa: BLE001 -- the reason does not change what we do
            dropped.append(key)
        else:
            keep[key] = value

    path = os.path.join(checkpoint_dir, "common.pt")
    torch.save({"args": argparse.Namespace(**keep)}, path)
    print(f"wrote {path}" + (f" (dropped {len(dropped)} unpicklable: {sorted(dropped)})" if dropped else ""))


def main():
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    configure_logger()

    # Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    accelerator.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend=accelerator.process_group_backend(),
        world_size=world_size,
        rank=global_rank,
        device_id=accelerator.distributed_device_id(local_rank),
    )
    args = get_args()
    init(args)

    # if using AMD gpus, we have to do the conversion in cpu
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load model
    hf_model_path = args.hf_checkpoint
    load_hf_weights(args, model, hf_model_path)
    print(f"Model loaded: {hf_model_path}")

    if args.use_cpu_initialization:
        model[0] = model[0].cpu()

    print_memory("after loading model")
    accelerator.synchronize()
    gc.collect()
    accelerator.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        # change to release ckpt
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write("release")
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)
        save_common_state(args, target_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
