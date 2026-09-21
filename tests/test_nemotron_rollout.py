"""Nemotron rollout must acknowledge the trainer's exact processed prompt IDs."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from slime.rollout import sglang_rollout
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.mark.parametrize("returned_ids", [[1, 2], [1, 3], None])
def test_rollout_checks_processed_prompt_ids(monkeypatch, returned_ids):
    args = SimpleNamespace(
        ci_test=False,
        sglang_router_ip="frontend",
        sglang_router_port=8000,
        use_rollout_routing_replay=False,
        router_policy=None,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
    )
    captured = {}

    async def post(url, payload, headers=None):
        captured.update(payload)
        return {
            "text": "answer",
            "prompt_token_ids": returned_ids,
            "meta_info": {"prompt_tokens": 2, "output_token_logprobs": [[-0.1, 11, None]]},
        }

    monkeypatch.setattr(
        sglang_rollout,
        "GenerateState",
        lambda _: SimpleNamespace(tokenizer=None, processor=SimpleNamespace(_slime_model_type="nemotron_h_omni")),
    )
    monkeypatch.setattr(sglang_rollout, "_prepare_prompt_ids", lambda *_: [1, 2])
    monkeypatch.setattr(sglang_rollout, "post", post)
    monkeypatch.setattr(
        sglang_rollout,
        "trace_span",
        lambda *_, **__: nullcontext(SimpleNamespace(update=lambda *_, **__: None)),
    )
    request = sglang_rollout.generate(args, Sample(prompt="prompt"), {"max_new_tokens": 8, "temperature": 1})
    if returned_ids == [1, 2]:
        asyncio.run(request)
    else:
        with pytest.raises(ValueError, match="different prompt tokens"):
            asyncio.run(request)
    assert captured["return_prompt_token_ids"] is True
