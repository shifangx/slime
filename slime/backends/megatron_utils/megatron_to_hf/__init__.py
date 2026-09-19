from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .llama import convert_llama_to_hf
from .mimo import convert_mimo_to_hf
from .minimax_m2 import convert_minimax_m2_to_hf
from .nemotron_h import convert_nemotron_h_to_hf, pending_vision_qkv
from .processors import quantize_params, remove_padding
from .qwen2 import convert_qwen2_to_hf
from .qwen3_5 import convert_qwen3_5_to_hf
from .qwen3_next import convert_qwen3_next_to_hf
from .qwen3_vl import convert_qwen3vl_to_hf
from .qwen3moe import convert_qwen3moe_to_hf


# TODO optimize code details
def convert_to_hf(args, model_name, name, param, quantization_config=None, transform_ue8m0=True):
    hf_name = name
    while hf_name.startswith("module."):
        hf_name = hf_name.removeprefix("module.")
    if hf_name.startswith("model.visual."):
        return [(hf_name, param)]

    param = remove_padding(name, param, args.vocab_size)
    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)

    return quantize_params(args, name, converted_named_tensors, quantization_config, transform_ue8m0)


def assert_conversion_buffers_drained(context: str = "end of weight sync") -> None:
    """Fail loudly if a many-to-one converter is still holding half a tensor.

    Some conversions are many-to-one -- three Megatron parameters becoming one
    HF tensor -- and `convert_to_hf` sees one parameter at a time, so they buffer
    the early arrivals and emit on the last one. Every early arrival is returned
    as ``[]``, which is indistinguishable from "this parameter is deliberately
    not exported".

    That makes an incomplete buffer silent in the worst way. The tensors that did
    arrive are never sent, the engine keeps whatever it loaded at startup for
    that slice of the model, every other slice is updated, and no log line
    anywhere says a partial model is now being served. `load_weights` cannot
    catch it either -- it only sees what did arrive.

    A sync ends with every buffer consumed or it did not do what it claimed.
    Call this once per sync, after the last parameter has gone through.

    Only the Nemotron VL vision buffer is checked. `_cached_tensors` below is the
    same shape of thing for DeepSeek's q_a_proj/kv_a_proj pair, and is
    deliberately left out until someone establishes whether it is meant to
    survive a sync -- its name says cache, and a false alarm here kills a
    training job.
    """
    pending = pending_vision_qkv()
    if not pending:
        return
    raise ValueError(
        f"nemotron_h: {len(pending)} vision qkv slot(s) still partially filled at {context}: "
        f"{pending}. Each of these is a RADIO block whose q/k/v did not all arrive in one "
        "pass, so its fused attention tensor was never emitted and the engine is still "
        "serving the weights it loaded at startup for that block while the rest of the "
        "tower was updated. Do not read a reward or an eval score from this run."
    )


# TODO optimize
_cached_tensors = {}


# TODO optimize code details
def _convert_to_hf_core(args, model_name, name, param):
    model_name = model_name.lower().replace("_", "").replace("-", "")
    if "minimaxm2" in model_name:
        converted_named_tensors = convert_minimax_m2_to_hf(args, name, param)
    # `NemotronHConfig` normalises to "nemotronhconfig"; the HF model_type spelling
    # `nemotron_h` normalises to the same "nemotronh" for an explicit --model-name.
    elif "nemotronh" in model_name:
        # model_name is forwarded because Nemotron-3 and Nemotron 3.5 Super VL
        # normalise into this same branch, and the VL one has to prefix its
        # language tensors with `language_model.` -- see that module.
        converted_named_tensors = convert_nemotron_h_to_hf(args, name, param, model_name)
    elif any(family in model_name for family in ("glm4moelite", "deepseekv3", "deepseekv32", "glmmoedsa", "kimi")):
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
    elif "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3next" in model_name:
        converted_named_tensors = convert_qwen3_next_to_hf(args, name, param)
    elif "qwen35" in model_name:
        converted_named_tensors = convert_qwen3_5_to_hf(args, name, param)
    elif "qwen3vl" in model_name:
        converted_named_tensors = convert_qwen3vl_to_hf(args, name, param)
    elif "qwen2moe" in model_name or "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    elif "mimo" in model_name:
        converted_named_tensors = convert_mimo_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # to compatible with sglang implementation
    if args.q_lora_rank is not None:
        old_converted_named_tensors = converted_named_tensors
        converted_named_tensors = []
        for converted_name, converted_param in old_converted_named_tensors:
            if "q_a_proj" in converted_name:
                pair_name = converted_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            elif "kv_a_proj_with_mqa" in converted_name:
                pair_name = converted_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            else:
                converted_named_tensors.append((converted_name, converted_param))
    return converted_named_tensors
