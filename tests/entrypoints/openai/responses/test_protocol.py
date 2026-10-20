# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

import pytest
from openai_harmony import (
    Message,
)

from vllm.config import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    serialize_message,
    serialize_messages,
)
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize("budget", [None, -1, 0, 32, 16384])
def test_responses_native_thinking_budget_preserves_total_output(budget):
    request = ResponsesRequest(
        input="Explain KV cache",
        max_output_tokens=32768,
        thinking_token_budget=budget,
    )
    params = request.to_sampling_params(default_max_tokens=32768)
    assert params.thinking_token_budget == (None if budget == -1 else budget)
    assert params.max_tokens == 32768


@pytest.mark.parametrize("budget", [-2, True, 1.5, "32"])
def test_responses_rejects_invalid_native_thinking_budget(budget):
    with pytest.raises(VLLMValidationError):
        ResponsesRequest(input="Explain KV cache", thinking_token_budget=budget)


@pytest.mark.parametrize(
    "explicit, expected", [("unset", 128), (None, None), (-1, None), (0, 0), (64, 64)]
)
def test_native_model_budget_defaults_and_request_precedence(explicit, expected):
    config = cast(
        ModelConfig,
        SimpleNamespace(
            generation_config="vllm",
            override_generation_config={"thinking_token_budget": 128},
        ),
    )
    defaults = ModelConfig.get_diff_sampling_param(config)
    assert defaults == {"thinking_token_budget": 128}
    kwargs = {} if explicit == "unset" else {"thinking_token_budget": explicit}
    responses = ResponsesRequest(
        input="Explain KV cache",
        max_output_tokens=256,
        **kwargs,
    ).to_sampling_params(default_max_tokens=256, default_sampling_params=defaults)
    chat = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Explain KV cache"}],
        **kwargs,
    ).to_sampling_params(max_tokens=256, default_sampling_params=defaults)
    for params in (responses, chat):
        assert params.thinking_token_budget == expected
        assert params.max_tokens == 256


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value

    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 1"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_message(msg) == msg_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 2"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_messages([msg, dict_value]) == [msg_value, dict_value]
