# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.preemption import (
    PreemptionCandidate,
    PreemptionContext,
    PreemptionPolicyController,
)

pytestmark = pytest.mark.cpu_test


def make_context(policy: str = "fcfs") -> PreemptionContext:
    return PreemptionContext(
        candidates=(
            PreemptionCandidate("first", 1, 1.0, 10, 2, 12, 0, 20),
            PreemptionCandidate("last", 3, 2.0, 8, 4, 12, 1, 20),
        ),
        scheduling_policy=policy,
        requesting_request_id="first",
        kv_cache_usage=0.95,
        now=3.0,
        builtin_victim_id="last",
    )


def make_config(policy=None):
    return SimpleNamespace(scheduler_config=SimpleNamespace(preemption_policy=policy))


class SelectFirstPolicy:
    def select_victim(self, context: PreemptionContext) -> str:
        return context.candidates[0].request_id


class AbstainingPolicy:
    def select_victim(self, context: PreemptionContext) -> None:
        return None


class InvalidPolicy:
    def select_victim(self, context: PreemptionContext) -> str:
        return "missing"


class FailingPolicy:
    calls = 0

    def select_victim(self, context: PreemptionContext) -> str:
        type(self).calls += 1
        raise RuntimeError("policy failure")


def test_builtin_policy_preserves_fcfs_and_priority() -> None:
    controller = PreemptionPolicyController(make_config())

    assert controller.select_victim(make_context("fcfs")) == "last"
    assert controller.select_victim(make_context("priority")) == "last"
    assert controller.export_stats() == {
        "policy_name": "builtin",
        "enabled": False,
        "calls": 0,
        "selections": 0,
        "abstentions": 0,
        "failures": 0,
        "invalid_selections": 0,
    }


def test_policy_receives_immutable_snapshots() -> None:
    context = make_context()

    with pytest.raises(FrozenInstanceError):
        context.kv_cache_usage = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.candidates[0].num_computed_tokens = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.builtin_victim_id = "first"  # type: ignore[misc]


def test_context_exposes_requester_and_builtin_victim() -> None:
    context = make_context()

    assert context.requesting_request_id == "first"
    assert context.builtin_victim_id == "last"


def test_custom_policy_selection_and_abstention() -> None:
    controller = PreemptionPolicyController(make_config(SelectFirstPolicy))
    assert controller.select_victim(make_context()) == "first"
    assert controller.export_stats()["selections"] == 1

    controller = PreemptionPolicyController(make_config(AbstainingPolicy))
    assert controller.select_victim(make_context()) == "last"
    assert controller.export_stats()["abstentions"] == 1


def test_invalid_selection_disables_policy_and_falls_back() -> None:
    controller = PreemptionPolicyController(make_config(InvalidPolicy))

    assert controller.select_victim(make_context()) == "last"
    assert controller.select_victim(make_context()) == "last"
    assert controller.export_stats() == {
        "policy_name": f"{__name__}.InvalidPolicy",
        "enabled": False,
        "calls": 1,
        "selections": 0,
        "abstentions": 0,
        "failures": 1,
        "invalid_selections": 1,
    }


def test_policy_exception_disables_policy_and_falls_back() -> None:
    FailingPolicy.calls = 0
    controller = PreemptionPolicyController(make_config(FailingPolicy))

    assert controller.select_victim(make_context()) == "last"
    assert controller.select_victim(make_context()) == "last"
    assert FailingPolicy.calls == 1
    assert controller.export_stats()["failures"] == 1
