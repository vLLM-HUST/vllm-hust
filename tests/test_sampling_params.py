# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm import SamplingParams
from vllm.config.diffusion import DiffusionConfig
from vllm.exceptions import VLLMValidationError
from vllm.utils.diffusion import validate_diffusion_sampling_params
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
        LogitBiasState.add_request(SimpleNamespace(), 0, params)


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
    return_sampling_mask: bool = False

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


def test_verify_leaves_logits_processors_to_admission():
    """verify() is runner-agnostic; LP validation lives in the admission
    layer, so an unimportable FQCN must not fail verify()."""
    SamplingParams().verify(
        MockModelConfig(logits_processors=["no.such:Cls"]), None, None, None
    )


def _verify_diffusion(params: SamplingParams, canvas_length: int | None = None):
    model_config = MockModelConfig(is_diffusion=True)
    params.verify(model_config, None, None, None)
    validate_diffusion_sampling_params(
        params,
        canvas_length=canvas_length,
        vocab_size=model_config.get_vocab_size(),
    )


def test_diffusion_extra_args_are_validated_without_a_served_canvas():
    # No --diffusion-config: the canvas is unknown, the ids are still checked
    # and a read-only request is still normalised.
    params = SamplingParams(
        max_tokens=64,
        extra_args={"diffusion_seed_canvas": [0, 1], "diffusion_read_only": True},
    )
    _verify_diffusion(params, canvas_length=None)
    assert params.ignore_eos is True

    bad = SamplingParams(extra_args={"diffusion_seed_canvas": [0, 10**9]})
    with pytest.raises(VLLMValidationError, match="ids must be in"):
        _verify_diffusion(bad, canvas_length=None)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    "extra_args",
    [
        {},
        {"diffusion_canvas_length": None},
        {"diffusion_canvas_length": 4},
        {"diffusion_canvas_length": 8},
    ],
)
def test_diffusion_canvas_width_is_accepted_with_either_scheduler(
    async_scheduling, extra_args
):
    processor = SimpleNamespace(
        model_config=MockModelConfig(is_diffusion=True),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=async_scheduling)
        ),
        speculative_config=None,
        structured_outputs_config=None,
        diffusion_config=DiffusionConfig(canvas_length=8),
        tokenizer=None,
        validate_logits_processors_params=lambda params: None,
        resolve_watermarking=lambda params: False,
    )
    params = SamplingParams(extra_args=extra_args)
    InputProcessor._validate_params(processor, params, ("generate",))


@pytest.mark.parametrize(
    "extra_args, match",
    [
        ({"diffusion_seed_canvas": "abc"}, "list of token ids"),
        ({"diffusion_seed_canvas": [1, 2.5]}, "list of token ids"),
        ({"diffusion_seed_canvas": [1, True]}, "list of token ids"),
        ({"diffusion_seed_canvas": [0, 1024]}, r"in \[0, 1024\)"),
        ({"diffusion_seed_canvas": [0, -1]}, r"in \[0, 1024\)"),
        ({"diffusion_max_steps": 0}, "positive integer"),
        ({"diffusion_max_steps": "1"}, "positive integer"),
        ({"diffusion_max_steps": True}, "positive integer"),
        ({"diffusion_read_only": "yes"}, "boolean"),
        ({"diffusion_read_only": 2}, "boolean"),
        ({"diffusion_pinned": "0,1"}, "list of canvas positions"),
        ({"diffusion_pinned": [0, True]}, "list of canvas positions"),
        ({"diffusion_pinned": [0]}, "needs a diffusion_seed_canvas"),
    ],
)
def test_diffusion_rejects_bad_extra_args(extra_args: dict, match: str):
    with pytest.raises(VLLMValidationError, match=match):
        _verify_diffusion(SamplingParams(extra_args=extra_args))


@pytest.mark.parametrize("flag", [True, 1])
def test_diffusion_constrained_needs_logprob_token_ids(flag):
    bad = SamplingParams(extra_args={"diffusion_constrained": flag})
    with pytest.raises(VLLMValidationError, match="needs logprob_token_ids"):
        _verify_diffusion(bad)

    ok = SamplingParams(
        logprob_token_ids=[3, 5], extra_args={"diffusion_constrained": flag}
    )
    _verify_diffusion(ok)

    # An unset or false flag needs no ids.
    _verify_diffusion(SamplingParams(extra_args={"diffusion_constrained": False}))
    _verify_diffusion(SamplingParams(extra_args={"diffusion_constrained": 0}))


@pytest.mark.parametrize("value", ["yes", 2, [1]])
def test_diffusion_constrained_must_be_a_bool(value):
    params = SamplingParams(extra_args={"diffusion_constrained": value})
    with pytest.raises(VLLMValidationError, match="must be a boolean"):
        _verify_diffusion(params)


def test_diffusion_seed_canvas_must_fill_the_canvas():
    params = SamplingParams(extra_args={"diffusion_seed_canvas": [0] * 7})
    with pytest.raises(VLLMValidationError, match="exactly 8 ids, got 7"):
        _verify_diffusion(params, canvas_length=8)
    # Without the served diffusion config the canvas length is unknown.
    _verify_diffusion(params)


@pytest.mark.parametrize("max_tokens, expected", [(100, 8), (5, 5), (None, 8)])
@pytest.mark.parametrize("flag", [True, 1])
def test_diffusion_read_only_ends_after_one_canvas(max_tokens, expected, flag):
    params = SamplingParams(
        max_tokens=max_tokens, extra_args={"diffusion_read_only": flag}
    )
    _verify_diffusion(params, canvas_length=8)
    assert params.max_tokens == expected
    assert params.ignore_eos


@pytest.mark.parametrize(
    "width, match",
    [
        (0, "positive integer"),
        ("4", "positive integer"),
        (True, "positive integer"),
        (9, "no larger"),
    ],
)
def test_diffusion_rejects_bad_canvas_length(width, match):
    with pytest.raises(VLLMValidationError, match=match):
        _verify_diffusion(
            SamplingParams(extra_args={"diffusion_canvas_length": width}),
            canvas_length=8,
        )


def test_diffusion_canvas_length_sizes_the_seed_and_the_read():
    params = SamplingParams(
        max_tokens=100,
        extra_args={
            "diffusion_canvas_length": 4,
            "diffusion_seed_canvas": [1, 2, 3, 4],
            "diffusion_read_only": True,
        },
    )
    _verify_diffusion(params, canvas_length=8)
    assert params.max_tokens == 4

    params = SamplingParams(
        extra_args={"diffusion_canvas_length": 4, "diffusion_seed_canvas": [0] * 8}
    )
    with pytest.raises(VLLMValidationError, match="exactly 4 ids, got 8"):
        _verify_diffusion(params, canvas_length=8)


def test_diffusion_pinned_positions_stay_inside_the_canvas():
    seed = [0] * 8
    with pytest.raises(VLLMValidationError, match="inside the canvas"):
        _verify_diffusion(
            SamplingParams(
                extra_args={"diffusion_seed_canvas": seed, "diffusion_pinned": [7, 8]}
            ),
            canvas_length=8,
        )
    with pytest.raises(VLLMValidationError, match="inside the canvas"):
        _verify_diffusion(
            SamplingParams(
                extra_args={"diffusion_seed_canvas": seed, "diffusion_pinned": [-1]}
            )
        )
    # A narrower request canvas bounds the positions.
    with pytest.raises(VLLMValidationError, match="inside the canvas"):
        _verify_diffusion(
            SamplingParams(
                extra_args={
                    "diffusion_canvas_length": 4,
                    "diffusion_seed_canvas": seed[:4],
                    "diffusion_pinned": [4],
                }
            ),
            canvas_length=8,
        )


def test_diffusion_accepts_extra_args():
    params = SamplingParams(
        extra_args={
            "diffusion_seed_canvas": list(range(8)),
            "diffusion_pinned": [0, 1, 7],
            "diffusion_max_steps": 4,
            "diffusion_read_only": True,
        }
    )
    _verify_diffusion(params, canvas_length=8)


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
def test_extra_args_rejects_nested_integer_overflow(value):
    """Reject extension values before they reach the engine transport."""
    with pytest.raises(VLLMValidationError, match="extra_args integers"):
        SamplingParams(extra_args={"ec_transfer_params": {"nested": [{"x": value}]}})


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**63, 2**64 - 1, True])
def test_extra_args_accepts_messagepack_integer_boundaries(value):
    extra_args = {"kv_transfer_params": {"nested": [{"x": value}]}}
    assert SamplingParams(extra_args=extra_args).extra_args == extra_args


def test_extra_args_preserves_custom_objects_and_shared_containers():
    custom = object()
    shared = [custom, (None, "value", 1.5)]
    extra_args = {"first": shared, "second": shared}
    params = SamplingParams(extra_args=extra_args)
    assert params.extra_args["first"][0] is custom
    assert params.extra_args["first"] is params.extra_args["second"]
