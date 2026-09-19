"""`--rollout-top-logprobs-num` has to reach SGLang as a request field.

It was first wired into `sampling_params`, where SGLang's kw_only msgspec
`SamplingParams` rejects it outright: every generate request failed with
`TypeError: Unexpected keyword argument 'top_logprobs_num'`, a whole 4-node
rollout produced zero samples, and nothing surfaced until the job had died.
The payload-shape test below pins where the field goes; the contract test
below it checks that against the real SGLang structs, which is what would
have caught the original placement in the first place.
"""

import asyncio
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import sglang_router  # noqa: F401
except ImportError:
    sys.modules["sglang_router"] = SimpleNamespace(__version__="0.3.0")

try:
    import transformers  # noqa: F401
except ImportError:
    sys.modules["transformers"] = SimpleNamespace(
        AutoProcessor=object,
        AutoTokenizer=object,
        PreTrainedTokenizerBase=object,
        ProcessorMixin=object,
    )

from slime.rollout import sglang_rollout
from slime.utils.types import Sample

NUM_GPUS = 0


def _args(**overrides):
    args = SimpleNamespace(
        ci_test=False,
        sglang_router_ip="frontend",
        sglang_router_port=8000,
        use_rollout_routing_replay=False,
        router_policy=None,
        rollout_top_logprobs_num=20,
        rollout_top_logprobs_positions=2,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _top_logprobs(num_positions, k=3):
    # SGLang shape: one list per generated position, each (logprob, token_id, text).
    return [[(-0.1 * i, 100 + i * k + j, None) for j in range(k)] for i in range(num_positions)]


def _run_generate(monkeypatch, args, meta_info):
    """Drive `sglang_rollout.generate` against a stub server, return its payload."""
    captured = {}

    async def post(url, payload, headers=None):
        captured["url"] = url
        captured["payload"] = payload
        return {"text": "ab", "meta_info": meta_info}

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: SimpleNamespace(tokenizer=None, processor=None))
    monkeypatch.setattr(sglang_rollout, "_prepare_prompt_ids", lambda *_args: [1, 2])
    monkeypatch.setattr(sglang_rollout, "post", post)
    monkeypatch.setattr(
        sglang_rollout,
        "trace_span",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace(update=lambda *_a, **_kw: None)),
    )

    sample = Sample(prompt="p")
    sampling_params = {"max_new_tokens": 8, "temperature": 0.8}
    sample = asyncio.run(sglang_rollout.generate(args, sample, sampling_params))
    return captured["payload"], sample


def test_top_logprobs_num_is_a_request_field_not_a_sampling_param(monkeypatch):
    payload, _ = _run_generate(
        monkeypatch,
        _args(),
        {"output_token_logprobs": [[-0.1, 11, None], [-0.2, 12, None]]},
    )

    assert payload["top_logprobs_num"] == 20
    assert payload["return_logprob"] is True
    assert "top_logprobs_num" not in payload["sampling_params"]


def test_top_logprobs_num_omitted_when_disabled(monkeypatch):
    payload, _ = _run_generate(
        monkeypatch,
        _args(rollout_top_logprobs_num=0),
        {"output_token_logprobs": [[-0.1, 11, None], [-0.2, 12, None]]},
    )

    assert "top_logprobs_num" not in payload
    assert "top_logprobs_num" not in payload["sampling_params"]


def test_top_logprobs_kept_on_metadata_truncated_to_positions(monkeypatch):
    _, sample = _run_generate(
        monkeypatch,
        _args(rollout_top_logprobs_positions=2),
        {
            "output_token_logprobs": [[-0.1, 11, None], [-0.2, 12, None]],
            "output_top_logprobs": _top_logprobs(5),
        },
    )

    assert len(sample.metadata["output_top_logprobs"]) == 2


def test_top_logprobs_positions_zero_keeps_every_position(monkeypatch):
    _, sample = _run_generate(
        monkeypatch,
        _args(rollout_top_logprobs_positions=0),
        {
            "output_token_logprobs": [[-0.1, 11, None], [-0.2, 12, None]],
            "output_top_logprobs": _top_logprobs(5),
        },
    )

    assert len(sample.metadata["output_top_logprobs"]) == 5


def test_payload_is_accepted_by_sglang_request_structs(monkeypatch):
    """The check that actually knows where the field lives: SGLang's own structs.

    Skipped where SGLang is not installed (it is not importable on a login
    node); it runs in the training container, which is where the mismatch
    this file exists for would have shown up.
    """
    io_struct = pytest.importorskip("sglang.srt.managers.io_struct")
    sampling_params_mod = pytest.importorskip("sglang.srt.sampling.sampling_params")

    payload, _ = _run_generate(
        monkeypatch,
        _args(),
        {"output_token_logprobs": [[-0.1, 11, None], [-0.2, 12, None]]},
    )

    # Mirrors tokenizer_manager._create_tokenized_object: sampling_params is
    # splatted into SamplingParams, everything else is a GenerateReqInput field.
    sampling_params_mod.SamplingParams(**payload["sampling_params"])
    io_struct.GenerateReqInput(**{k: v for k, v in payload.items() if k != "sampling_params"})
