# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

import pytest
from openai_harmony import (
    Message,
)
from pydantic import ValidationError

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


@pytest.mark.parametrize(
    "explicit, presence, frequency",
    [("unset", 1.5, 0.5), (None, 1.5, 0.5), (0.0, 0.0, 0.0), (0.7, 0.7, 0.7)],
)
def test_model_penalties_reach_both_protocols_and_explicit_zero_wins(
    explicit, presence, frequency
):
    config = cast(
        ModelConfig,
        SimpleNamespace(
            generation_config="vllm",
            override_generation_config={
                "presence_penalty": 1.5,
                "frequency_penalty": 0.5,
            },
        ),
    )
    defaults = ModelConfig.get_diff_sampling_param(config)
    kwargs = (
        {}
        if explicit == "unset"
        else {"presence_penalty": explicit, "frequency_penalty": explicit}
    )
    responses = ResponsesRequest(input="Explain KV cache", **kwargs).to_sampling_params(
        default_max_tokens=32768, default_sampling_params=defaults
    )
    chat = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Explain KV cache"}],
        **kwargs,
    ).to_sampling_params(max_tokens=32768, default_sampling_params=defaults)
    for params in (responses, chat):
        assert params.presence_penalty == presence
        assert params.frequency_penalty == frequency
        assert params.max_tokens == 32768


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


def test_custom_tool_grammar_format_rejected() -> None:
    grammar = {"type": "grammar", "syntax": "lark", "definition": 'start: "pwd"'}
    with pytest.raises(VLLMValidationError) as exc_info:
        ResponsesRequest(
            input="hi", tools=[{"type": "custom", "name": "emit", "format": grammar}]
        )
    assert exc_info.value.parameter == "tools"

    request = ResponsesRequest(
        input="hi",
        tools=[{"type": "custom", "name": "emit", "format": {"type": "text"}}],
    )
    assert request.tools is not None and request.tools[0].type == "custom"


@pytest.mark.parametrize(
    "tools",
    [
        "abc",
        {"type": "custom"},
        [None],
        [{"type": "custom", "name": "x", "format": "text"}],
    ],
)
def test_malformed_tools_rejected_by_field_validation(tools) -> None:
    """Malformed tools bodies fail pydantic field validation (4xx), not an
    AttributeError inside the custom-format validator (500)."""
    with pytest.raises(ValidationError):
        ResponsesRequest(input="hi", tools=tools)
