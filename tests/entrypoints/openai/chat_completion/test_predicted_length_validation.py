# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI protocol coverage for the ``predicted_length`` extension field.

The field is a vLLM extension (not part of the OpenAI schema). It must be
validated strictly at the protocol boundary - bools, strings, floats and
negative values are rejected here - and, once accepted, it must travel to the
engine through ``SamplingParams.extra_args`` unchanged. The scheduler-side
clamping of oversized values is covered by
``tests/v1/core/test_prediction_admission.py``.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize("raw_value", [True, False, "64", 64.0, 64.5, -1, "abc"])
def test_chat_completion_request_rejects_malformed_predicted_length(raw_value):
    with pytest.raises(VLLMValidationError, match="predicted_length"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "predicted_length": raw_value,
            }
        )


def test_chat_completion_request_accepts_non_negative_integer():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "predicted_length": 64,
        }
    )
    assert request.predicted_length == 64


def test_chat_completion_request_accepts_oversized_value():
    """The protocol accepts any non-negative int; the scheduler clamps later."""
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "predicted_length": 10**9,
        }
    )
    assert request.predicted_length == 10**9


def test_chat_completion_request_forwards_predicted_length_via_extra_args():
    request = ChatCompletionRequest(
        model="qwen",
        messages=[{"role": "user", "content": "hello"}],
        predicted_length=64,
        max_tokens=128,
    )
    sampling_params = request.to_sampling_params(
        max_tokens=128, default_sampling_params={}
    )
    assert sampling_params.extra_args["predicted_length"] == 64


def test_chat_completion_request_without_prediction_leaves_extra_args_clean():
    request = ChatCompletionRequest(
        model="qwen",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=128,
    )
    sampling_params = request.to_sampling_params(
        max_tokens=128, default_sampling_params={}
    )
    assert "predicted_length" not in (sampling_params.extra_args or {})


@pytest.mark.parametrize("raw_value", [True, "64", 64.5, -1])
def test_completion_request_rejects_malformed_predicted_length(raw_value):
    with pytest.raises(VLLMValidationError, match="predicted_length"):
        CompletionRequest.model_validate(
            {
                "model": "qwen",
                "prompt": "hello",
                "predicted_length": raw_value,
            }
        )


def test_completion_request_accepts_and_forwards_predicted_length():
    request = CompletionRequest(
        model="qwen",
        prompt="hello",
        predicted_length=32,
        max_tokens=128,
    )
    assert request.predicted_length == 32
    sampling_params = request.to_sampling_params(
        max_tokens=128, default_sampling_params={}
    )
    assert sampling_params.extra_args["predicted_length"] == 32
