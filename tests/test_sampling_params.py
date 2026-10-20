# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.input_processor import InputProcessor


@pytest.mark.parametrize(
    "field,limit",
    [("logit_bias", 1024), ("allowed_token_ids", 1024), ("stop_token_ids", 128)],
)
def test_v2_rejects_oversized_sampling_buffers_before_worker_writes(field, limit):
    from vllm.v1.worker.gpu.sample.logit_bias import (
        LogitBiasState,
        validate_logit_bias_params,
    )

    values = (
        dict.fromkeys(range(limit + 1), 0.5)
        if field == "logit_bias"
        else list(range(limit + 1))
    )
    params = SamplingParams(**{field: values}, min_tokens=1)
    with pytest.raises(VLLMValidationError) as error:
        validate_logit_bias_params(params)
    assert error.value.parameter == field
    assert error.value.value == limit + 1
    # No CUDA allocation or partially staged request state is needed to reject it.
    with pytest.raises(VLLMValidationError):
        LogitBiasState.add_request(SimpleNamespace(), 0, 0, params)


@pytest.mark.parametrize(
    "field,limit",
    [("logit_bias", 1024), ("allowed_token_ids", 1024), ("stop_token_ids", 128)],
)
def test_v2_accepts_sampling_buffers_at_capacity(field, limit):
    from vllm.v1.worker.gpu.sample.logit_bias import validate_logit_bias_params

    values = (
        dict.fromkeys(range(limit), 0.5)
        if field == "logit_bias"
        else list(range(limit))
    )
    validate_logit_bias_params(SamplingParams(**{field: values}, min_tokens=1))


def test_v2_stop_limit_uses_effective_eos_and_only_min_tokens_mask():
    from vllm.v1.worker.gpu.sample.logit_bias import validate_logit_bias_params

    params = SamplingParams(stop_token_ids=list(range(128)), min_tokens=1)
    params.update_from_generation_config({}, eos_token_id=200)
    with pytest.raises(VLLMValidationError, match="129 tokens"):
        validate_logit_bias_params(params)
    params.min_tokens = 0
    validate_logit_bias_params(params)


@pytest.mark.parametrize("field", ["logit_bias", "allowed_token_ids", "stop_token_ids"])
def test_v2_admission_rejects_effective_params_without_scheduling(field):
    params = SamplingParams(
        **{
            field: (
                dict.fromkeys(range(1025), 0.5)
                if field == "logit_bias"
                else list(range(1025 if field == "allowed_token_ids" else 128))
            )
        },
        min_tokens=1,
    )
    processor = SimpleNamespace(
        _validate_params=lambda *args: False,
        _validate_lora=lambda *args: None,
        _validate_model_inputs=lambda *args: None,
        apply_watermarking=lambda *args: None,
        model_config=SimpleNamespace(max_model_len=4096),
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_size_local=1,
                local_engines_only=False,
            ),
        ),
        generation_config_fields={},
        renderer=SimpleNamespace(get_eos_token_id=lambda: 200),
        tokenizer=None,
    )
    with pytest.raises(VLLMValidationError) as error:
        InputProcessor.process_inputs(
            processor,
            "test-admission",
            {"type": "tokens", "prompt_token_ids": [1, 2, 3]},
            params,
            ("generate",),
        )
    assert error.value.parameter == field


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(VLLMValidationError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)
