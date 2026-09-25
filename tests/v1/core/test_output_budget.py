# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Known-length admission boundaries; no accelerator or runtime initialization."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

admission_sequence_tokens = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "vllm/v1/core/output_budget.py")
)["admission_sequence_tokens"]


def request(*, prompt=100, output=0, computed=0, budget=50, ignore_eos=True):
    return SimpleNamespace(
        num_tokens=prompt + output,
        num_prompt_tokens=prompt,
        num_output_tokens=output,
        num_computed_tokens=computed,
        max_tokens=budget,
        sampling_params=SimpleNamespace(ignore_eos=ignore_eos),
    )


def test_default_keeps_native_admission():
    assert admission_sequence_tokens(request(budget=10000), 1000) == 100


def test_reserves_declared_future_output_during_prefill():
    assert admission_sequence_tokens(request(), 1000, reserve_output_budget=True) == 150


def test_preempted_request_does_not_double_count_generated_tokens():
    assert (
        admission_sequence_tokens(
            request(output=40, computed=0), 1000, reserve_output_budget=True
        )
        == 150
    )


def test_completed_prefill_preserves_reactive_decode_allocation():
    assert (
        admission_sequence_tokens(
            request(output=40, computed=140), 1000, reserve_output_budget=True
        )
        == 140
    )


@pytest.mark.parametrize("ignore_eos", [False, None])
def test_natural_stopping_does_not_pretend_budget_is_prediction(ignore_eos):
    assert (
        admission_sequence_tokens(
            request(ignore_eos=ignore_eos), 1000, reserve_output_budget=True
        )
        == 100
    )


def test_caps_at_context_capacity_and_handles_exhausted_budget():
    assert (
        admission_sequence_tokens(
            request(budget=10000), 256, reserve_output_budget=True
        )
        == 256
    )
    assert (
        admission_sequence_tokens(request(output=50), 1000, reserve_output_budget=True)
        == 150
    )
