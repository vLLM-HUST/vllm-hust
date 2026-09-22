# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the prediction-aware admission helper.

``predicted_full_sequence_tokens`` is the single place where the scheduler
decides how much room a predicted output length may reserve. It is pure, so it
is tested directly rather than through a Scheduler: the admission path itself
is covered by ``test_scheduler.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.core.length_prediction import predicted_full_sequence_tokens

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def make_request(
    *,
    prompt_tokens: int = 100,
    output_tokens: int = 0,
    computed_tokens: int | None = None,
    max_tokens: int = 256,
) -> SimpleNamespace:
    """Minimal stand-in for the Request attributes the helper reads."""
    return SimpleNamespace(
        num_prompt_tokens=prompt_tokens,
        num_tokens=prompt_tokens + output_tokens,
        num_computed_tokens=(
            computed_tokens if computed_tokens is not None else prompt_tokens
        ),
        max_tokens=max_tokens,
    )


def test_no_prediction_reproduces_upstream_behaviour():
    """None must plan for exactly the tokens the request has."""
    request = make_request(prompt_tokens=100)
    assert predicted_full_sequence_tokens(request, None, 2048) == 100


def test_no_prediction_is_clamped_to_max_model_len():
    request = make_request(prompt_tokens=100)
    assert predicted_full_sequence_tokens(request, None, 64) == 64


def test_prefill_plans_for_prompt_plus_predicted_output():
    """The reserve covers prompt + the predicted output, not just the prompt."""
    request = make_request(prompt_tokens=100, computed_tokens=0, max_tokens=256)
    assert predicted_full_sequence_tokens(request, 128, 2048) == 228


def test_prediction_is_clamped_to_the_request_output_budget():
    """A client cannot reserve more output than the request may generate.

    This is the bound that stops an oversized prediction from turning into an
    over-reserve, head-of-line blocking or starvation.
    """
    request = make_request(prompt_tokens=100, computed_tokens=0, max_tokens=64)
    # 10_000 predicted, but this request may only ever produce 64.
    assert predicted_full_sequence_tokens(request, 10_000, 2048) == 164


def test_result_never_exceeds_max_model_len():
    request = make_request(prompt_tokens=100, computed_tokens=0, max_tokens=10_000)
    assert predicted_full_sequence_tokens(request, 10_000, 512) == 512


def test_negative_prediction_is_treated_as_no_extra_reserve():
    """Defensive: a nonsense value degrades to 'no reserve', not a failure."""
    request = make_request(prompt_tokens=100, computed_tokens=0, max_tokens=256)
    assert predicted_full_sequence_tokens(request, -5, 2048) == 100


def test_prediction_is_ignored_once_decoding_started():
    """After prefill the real length is known, so a prediction must not inflate it."""
    request = make_request(
        prompt_tokens=100,
        output_tokens=20,
        computed_tokens=120,
        max_tokens=256,
    )
    # num_computed_tokens >= num_prompt_tokens means decoding started.
    assert predicted_full_sequence_tokens(request, 200, 2048) == 120


def test_prediction_only_reserves_the_remaining_output():
    """A partially prefilled sequence plans for its full predicted output."""
    request = make_request(
        prompt_tokens=100,
        computed_tokens=50,  # still prefilling: 50 < 100
        max_tokens=256,
    )
    # predicted total output 100, nothing generated yet -> 100 + 100.
    assert predicted_full_sequence_tokens(request, 100, 2048) == 200


def test_partial_output_reduces_the_remaining_reserve():
    """Output already produced must not be reserved a second time."""
    request = make_request(
        prompt_tokens=100,
        output_tokens=30,
        computed_tokens=50,  # still prefilling
        max_tokens=256,
    )
    # 30 of the predicted 100 are already generated -> reserve 70 more.
    assert predicted_full_sequence_tokens(request, 100, 2048) == 130 + 70


def test_zero_prediction_adds_nothing_beyond_the_prompt():
    request = make_request(prompt_tokens=100, computed_tokens=0, max_tokens=256)
    assert predicted_full_sequence_tokens(request, 0, 2048) == 100
