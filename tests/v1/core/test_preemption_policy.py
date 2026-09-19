# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from vllm.plugins import evidence
from vllm.v1.core.sched.preemption import (
    PreemptionCandidate,
    PreemptionContext,
    PreemptionPolicyController,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def reset_evidence(monkeypatch):
    evidence.reset_for_tests()
    monkeypatch.delenv("VLLM_ECPA_EVIDENCE_SINK", raising=False)
    monkeypatch.delenv("VLLM_ECPA_EVIDENCE_STRICT", raising=False)
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ROLE", "engine-core-scheduler")
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ORDINAL", "0")
    monkeypatch.setenv("VLLM_ECPA_PROCESS_EPOCH", "7")


def capture_evidence(events):
    evidence._sink = events.append
    evidence._sink_state = "ready"


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


def test_loaded_policy_without_pressure_emits_only_resolved() -> None:
    events = []
    capture_evidence(events)

    PreemptionPolicyController(make_config(SelectFirstPolicy))

    assert [(event["event"], event["detail"]) for event in events] == [
        ("resolved", "engine-core.scheduler:protocol-validated")
    ]
    assert events[0]["process"]["role"] == "engine-core-scheduler"
    assert events[0]["plugin_id"] is None
    assert events[0]["artifact_digest"] is None


def test_protocol_rejection_never_emits_resolved() -> None:
    events = []
    capture_evidence(events)

    with pytest.raises(TypeError, match="implementing PreemptionPolicy"):
        PreemptionPolicyController(make_config(object()))

    assert events == []


@pytest.mark.parametrize(
    ("policy", "expected_victim", "outcome"),
    [
        (SelectFirstPolicy, "first", "selected"),
        (AbstainingPolicy, "last", "abstained"),
        (InvalidPolicy, "last", "invalid"),
        (FailingPolicy, "last", "exception"),
    ],
)
def test_native_dispatch_emits_process_owned_outcome(
    policy, expected_victim, outcome
) -> None:
    events = []
    capture_evidence(events)
    controller = PreemptionPolicyController(make_config(policy))

    assert controller.select_victim(make_context()) == expected_victim

    invoked = [event for event in events if event["event"] == "invoked"]
    assert len(invoked) == 1
    assert invoked[0]["detail"] == f"engine-core.scheduler:{outcome}"
    assert invoked[0]["process"]["role"] == "engine-core-scheduler"


def test_evidence_failure_never_changes_selection_or_fallback(monkeypatch) -> None:
    def broken(_event):
        raise RuntimeError("sink unavailable")

    evidence._sink, evidence._sink_state = broken, "ready"
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_STRICT", "1")
    selected = PreemptionPolicyController(make_config(SelectFirstPolicy))
    invalid = PreemptionPolicyController(make_config(InvalidPolicy))

    assert selected.select_victim(make_context()) == "first"
    assert invalid.select_victim(make_context()) == "last"
    assert invalid.export_stats()["enabled"] is False


def test_process_epoch_change_is_reflected_in_dispatch_evidence(monkeypatch) -> None:
    events = []
    capture_evidence(events)
    controller = PreemptionPolicyController(make_config(SelectFirstPolicy))
    controller.select_victim(make_context())
    monkeypatch.setenv("VLLM_ECPA_PROCESS_EPOCH", "8")
    controller.select_victim(make_context())

    invoked = [event for event in events if event["event"] == "invoked"]
    assert [event["process"]["process_epoch"] for event in invoked] == [7, 8]
