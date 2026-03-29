# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.utils import (
    _format_ascend_torch_preflight_failure,
    _maybe_run_ascend_torch_preflight,
    _should_run_ascend_torch_preflight,
    get_max_tokens,
    sanitize_message,
    should_include_usage,
)


def test_sanitize_message():
    assert (
        sanitize_message("<_io.BytesIO object at 0x7a95e299e750>")
        == "<_io.BytesIO object>"
    )


@pytest.mark.parametrize(
    ("stream_options", "expected"),
    [
        (None, (True, True)),
        (StreamOptions(include_usage=False), (True, True)),
        (
            StreamOptions(include_usage=False, continuous_usage_stats=False),
            (True, True),
        ),
        (
            StreamOptions(include_usage=True, continuous_usage_stats=False),
            (True, True),
        ),
    ],
)
def test_should_include_usage_force_enables_continuous_usage(stream_options, expected):
    assert should_include_usage(stream_options, True) == expected


def test_should_run_ascend_torch_preflight_for_serve_on_ascend(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_TORCH_PREFLIGHT", raising=False)
    monkeypatch.setattr(
        "vllm.entrypoints.utils._has_ascend_runtime_hints",
        lambda: True,
    )

    assert _should_run_ascend_torch_preflight(["vllm-hust", "serve", "foo"])


def test_should_skip_ascend_torch_preflight_for_cpu_backend(monkeypatch):
    monkeypatch.setattr(
        "vllm.entrypoints.utils._has_ascend_runtime_hints",
        lambda: True,
    )

    assert not _should_run_ascend_torch_preflight(
        ["vllm-hust", "serve", "--backend", "cpu", "foo"]
    )


def test_should_skip_ascend_torch_preflight_when_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_TORCH_PREFLIGHT", "0")
    monkeypatch.setattr(
        "vllm.entrypoints.utils._has_ascend_runtime_hints",
        lambda: True,
    )

    assert not _should_run_ascend_torch_preflight(["vllm-hust", "serve", "foo"])


def test_format_ascend_torch_preflight_failure_mentions_runtime_layer():
    result = SimpleNamespace(
        stdout="torch.npu.set_device ok",
        stderr="RuntimeError: Parse dynamic kernel config fail",
        returncode=1,
    )

    message = _format_ascend_torch_preflight_failure(result)

    assert "before vLLM engine startup" in message
    assert "below vLLM" in message
    assert "Parse dynamic kernel config fail" in message
    assert "VLLM_ASCEND_TORCH_PREFLIGHT=0" in message


def test_maybe_run_ascend_torch_preflight_raises_system_exit(monkeypatch):
    monkeypatch.setattr(
        "vllm.entrypoints.utils._should_run_ascend_torch_preflight",
        lambda argv=None: True,
    )
    monkeypatch.setattr(
        "vllm.entrypoints.utils._run_ascend_torch_preflight",
        lambda: (_ for _ in ()).throw(SystemExit("torch preflight failed")),
    )

    with pytest.raises(SystemExit, match="torch preflight failed"):
        _maybe_run_ascend_torch_preflight(["vllm-hust", "serve", "foo"])


class TestGetMaxTokens:
    """Tests for get_max_tokens() to ensure generation_config's max_tokens
    acts as a default when from model author, and as a ceiling when
    explicitly set by the user."""

    def test_default_sampling_params_used_when_no_request_max_tokens(self):
        """When user doesn't specify max_tokens, generation_config default
        should apply."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2048

    def test_request_max_tokens_not_capped_by_default_sampling_params(self):
        """When user specifies max_tokens in request, model author's
        generation_config max_tokens must NOT cap it (fixes #34005)."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 5000

    def test_override_max_tokens_caps_request(self):
        """When user explicitly sets max_tokens, it acts as a ceiling."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_override_max_tokens_used_as_default(self):
        """When no request max_tokens, override still applies as default."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_max_model_len_still_caps_output(self):
        """max_model_len - input_length is always the hard ceiling."""
        result = get_max_tokens(
            max_model_len=3000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2900  # 3000 - 100

    def test_request_max_tokens_smaller_than_default(self):
        """When user explicitly requests fewer tokens than gen_config default,
        that should be respected."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=512,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 512

    def test_input_length_exceeds_max_model_len(self):
        with pytest.raises(
            ValueError,
            match="Input length .* exceeds model's maximum context length .*",
        ):
            get_max_tokens(
                max_model_len=100,
                max_tokens=50,
                input_length=150,
                default_sampling_params={"max_tokens": 2048},
            )
