"""Unit tests for vllm.v1.core.length_prediction (runs WITHOUT vllm installed)."""
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


def _make_module(name, path=None, **attrs):
    m = types.ModuleType(name)
    m.__path__ = path if path is not None else []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


@dataclass
class _StubRequest:
    prompt_token_ids: list
    num_computed_tokens: int
    # emulate Request: _all_token_ids = prompt + generated
    _all_token_ids: list

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)


def _install_stubs():
    repo = Path(__file__).resolve().parent.parent.parent.parent  # tests/v1/core -> repo
    vllm_root = repo / "vllm"
    for name, sub in [("vllm", vllm_root), ("vllm.v1", vllm_root / "v1"),
                      ("vllm.v1.core", vllm_root / "v1" / "core")]:
        if name not in sys.modules:
            _make_module(name, path=[str(sub)])
    _make_module("vllm.v1.request", Request=_StubRequest)


_install_stubs()

from vllm.v1.core.length_prediction import predicted_full_sequence_tokens  # noqa: E402


def _req(prompt_len, computed, generated):
    return _StubRequest(
        prompt_token_ids=[0] * prompt_len,
        num_computed_tokens=computed,
        _all_token_ids=[0] * (prompt_len + generated),
    )


class TestPredictedFullSequenceTokens:
    def test_no_prediction_returns_current(self):
        r = _req(prompt_len=100, computed=0, generated=0)
        assert predicted_full_sequence_tokens(r, None, 4096) == 100
        r2 = _req(prompt_len=100, computed=100, generated=50)
        assert predicted_full_sequence_tokens(r2, None, 4096) == 150

    def test_prefill_adds_predicted_output(self):
        # waiting/prefill: prompt=100, prediction=200 -> reserve 300
        r = _req(prompt_len=100, computed=0, generated=0)
        assert predicted_full_sequence_tokens(r, 200, 4096) == 300

    def test_decode_ignores_prediction(self):
        # prefill finished (computed >= prompt): keep upstream reactive behaviour
        r = _req(prompt_len=100, computed=100, generated=40)
        assert predicted_full_sequence_tokens(r, 200, 4096) == 140

    def test_partial_prefill_chunk_still_reserves(self):
        r = _req(prompt_len=100, computed=40, generated=0)
        assert predicted_full_sequence_tokens(r, 200, 4096) == 300

    def test_caps_at_max_model_len(self):
        r = _req(prompt_len=100, computed=0, generated=0)
        assert predicted_full_sequence_tokens(r, 100_000, 4096) == 4096

    def test_zero_or_negative_prediction(self):
        r = _req(prompt_len=50, computed=0, generated=0)
        assert predicted_full_sequence_tokens(r, 0, 4096) == 50
        r2 = _req(prompt_len=50, computed=50, generated=5)
        assert predicted_full_sequence_tokens(r2, 0, 4096) == 55
