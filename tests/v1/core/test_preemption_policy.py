# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from vllm.plugins import evidence
from vllm.v1.core.sched.preemption import (
    PreemptionCandidate,
    PreemptionContext,
    PreemptionPolicyController,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture(autouse=True)
def reset_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence.reset_for_tests()
    monkeypatch.delenv("VLLM_ECPA_EVIDENCE_SINK", raising=False)
    monkeypatch.delenv("VLLM_ECPA_EVIDENCE_STRICT", raising=False)
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ROLE", "engine-core-scheduler")
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ORDINAL", "0")
    monkeypatch.setenv("VLLM_ECPA_PROCESS_EPOCH", "7")
    monkeypatch.setenv("VLLM_ECPA_PLAN_ID", "plan-1")
    monkeypatch.setenv("VLLM_ECPA_LAUNCH_ID", "launch-1")


def capture_evidence(events: list[dict[str, Any]]) -> None:
    evidence._sink = events.append
    evidence._sink_state = "ready"


def make_context(policy: Literal["fcfs", "priority"] = "fcfs") -> PreemptionContext:
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


def make_config(policy: Any = None) -> SimpleNamespace:
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
    events: list[dict[str, Any]] = []
    capture_evidence(events)

    PreemptionPolicyController(make_config(SelectFirstPolicy))

    assert [(event["event"], event["detail"]) for event in events] == [
        ("resolved", "engine-core.scheduler:protocol-validated")
    ]
    assert events[0]["process"]["role"] == "engine-core-scheduler"
    assert events[0]["observation_kind"] == "scheduler_resolution"
    assert events[0]["binding_status"] == "bound"
    assert events[0]["plan_id"] == "plan-1"
    assert events[0]["launch_id"] == "launch-1"
    assert events[0]["plugin_id"] is None
    assert events[0]["artifact_digest"] is None


def test_protocol_rejection_never_emits_resolved() -> None:
    events: list[dict[str, Any]] = []
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
    policy: type[Any], expected_victim: str, outcome: str
) -> None:
    events: list[dict[str, Any]] = []
    capture_evidence(events)
    controller = PreemptionPolicyController(make_config(policy))

    assert controller.select_victim(make_context()) == expected_victim

    invoked = [event for event in events if event["event"] == "invoked"]
    assert len(invoked) == 1
    assert invoked[0]["detail"] == f"engine-core.scheduler:{outcome}"
    assert invoked[0]["observation_kind"] == "scheduler_dispatch"
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


def test_broken_sink_storm_calls_sink_once_and_preserves_dispatches(caplog) -> None:
    sink_calls = 0

    def broken(_event):
        nonlocal sink_calls
        sink_calls += 1
        raise RuntimeError("sink unavailable")

    evidence._sink, evidence._sink_state = broken, "ready"
    controller = PreemptionPolicyController(make_config(SelectFirstPolicy))

    for _ in range(100):
        assert controller.select_victim(make_context()) == "first"

    assert sink_calls == 1
    assert controller.export_stats()["calls"] == 100
    assert controller.export_stats()["selections"] == 100
    assert caplog.text.count("ECPA_EVIDENCE_SINK_WRITE_FAILED") == 1


def test_dispatch_scope_is_frozen_across_environment_mutation(monkeypatch) -> None:
    events: list[dict[str, Any]] = []
    capture_evidence(events)
    controller = PreemptionPolicyController(make_config(SelectFirstPolicy))
    controller.select_victim(make_context())
    monkeypatch.setenv("VLLM_ECPA_PROCESS_EPOCH", "8")
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ROLE", "worker")
    monkeypatch.setenv("VLLM_ECPA_PLAN_ID", "plan-2")
    monkeypatch.setenv("VLLM_ECPA_LAUNCH_ID", "launch-2")
    controller.select_victim(make_context())

    invoked = [event for event in events if event["event"] == "invoked"]
    resolved = next(event for event in events if event["event"] == "resolved")
    assert [event["process"]["process_epoch"] for event in invoked] == [7, 7]
    assert [event["process"]["role"] for event in invoked] == [
        "engine-core-scheduler",
        "engine-core-scheduler",
    ]
    assert [event["plan_id"] for event in invoked] == ["plan-1", "plan-1"]
    assert [event["launch_id"] for event in invoked] == ["launch-1", "launch-1"]
    assert all(event["process"] == resolved["process"] for event in invoked)
    assert all(event["plan_id"] == resolved["plan_id"] for event in invoked)
    assert all(event["launch_id"] == resolved["launch_id"] for event in invoked)


@pytest.mark.parametrize(
    "policy,outcome", [(SelectFirstPolicy, "selected"), (AbstainingPolicy, "abstained")]
)
def test_repeated_dispatches_are_delivered_and_align_with_calls(
    policy: type[Any], outcome: str
) -> None:
    events: list[dict[str, Any]] = []
    capture_evidence(events)
    controller = PreemptionPolicyController(make_config(policy))

    for _ in range(3):
        controller.select_victim(make_context())

    invoked = [event for event in events if event["event"] == "invoked"]
    assert len(invoked) == controller.export_stats()["calls"] == 3
    assert [event["occurrence_id"] for event in invoked] == [1, 2, 3]
    assert [event["invocation_seq"] for event in invoked] == [1, 2, 3]
    assert len({event["dispatch_id"] for event in invoked}) == 3
    assert {event["detail"] for event in invoked} == {
        f"engine-core.scheduler:{outcome}"
    }


def test_worker_role_rejects_scheduler_evidence_without_changing_policy(
    monkeypatch,
) -> None:
    events: list[dict[str, Any]] = []
    capture_evidence(events)
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ROLE", "worker")

    controller = PreemptionPolicyController(make_config(SelectFirstPolicy))

    assert controller.select_victim(make_context()) == "first"
    assert events == []
    assert controller.export_stats()["selections"] == 1


def test_missing_plan_and_launch_is_explicitly_unbound(monkeypatch) -> None:
    events: list[dict[str, Any]] = []
    capture_evidence(events)
    monkeypatch.delenv("VLLM_ECPA_PLAN_ID")
    monkeypatch.delenv("VLLM_ECPA_LAUNCH_ID")

    controller = PreemptionPolicyController(make_config(SelectFirstPolicy))
    controller.select_victim(make_context())

    assert {event["binding_status"] for event in events} == {"unbound"}
    assert all(event["plan_id"] is None for event in events)
    assert all(event["launch_id"] is None for event in events)


def test_new_launch_in_same_pid_is_not_hidden_by_resolved_dedupe(monkeypatch) -> None:
    events: list[dict[str, Any]] = []
    capture_evidence(events)
    PreemptionPolicyController(make_config(SelectFirstPolicy))
    monkeypatch.setenv("VLLM_ECPA_PLAN_ID", "plan-2")
    monkeypatch.setenv("VLLM_ECPA_LAUNCH_ID", "launch-2")
    PreemptionPolicyController(make_config(SelectFirstPolicy))

    resolved = [event for event in events if event["event"] == "resolved"]
    assert [(event["plan_id"], event["launch_id"]) for event in resolved] == [
        ("plan-1", "launch-1"),
        ("plan-2", "launch-2"),
    ]


def test_same_launch_controller_recreation_has_unique_dispatches() -> None:
    events: list[dict[str, Any]] = []
    capture_evidence(events)
    first = PreemptionPolicyController(make_config(SelectFirstPolicy))
    second = PreemptionPolicyController(make_config(SelectFirstPolicy))

    assert first.select_victim(make_context()) == "first"
    assert second.select_victim(make_context()) == "first"

    resolved = [event for event in events if event["event"] == "resolved"]
    invoked = [event for event in events if event["event"] == "invoked"]
    assert len(resolved) == 2
    assert len({event["controller_instance_id"] for event in resolved}) == 2
    assert [event["invocation_seq"] for event in invoked] == [1, 2]
    assert len({event["dispatch_id"] for event in invoked}) == 2


def test_cross_launch_dispatch_ids_do_not_collide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = []
    capture_evidence(events)
    first = PreemptionPolicyController(make_config(SelectFirstPolicy))
    assert first.select_victim(make_context()) == "first"
    monkeypatch.setenv("VLLM_ECPA_PLAN_ID", "plan-2")
    monkeypatch.setenv("VLLM_ECPA_LAUNCH_ID", "launch-2")
    second = PreemptionPolicyController(make_config(SelectFirstPolicy))
    assert second.select_victim(make_context()) == "first"

    invoked = [event for event in events if event["event"] == "invoked"]
    assert len({event["dispatch_id"] for event in invoked}) == 2
    assert [(event["plan_id"], event["launch_id"]) for event in invoked] == [
        ("plan-1", "launch-1"),
        ("plan-2", "launch-2"),
    ]


def test_inherited_controller_recaptures_child_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_events: list[dict[str, Any]] = []
    child_events: list[dict[str, Any]] = []
    capture_evidence(parent_events)
    controller = PreemptionPolicyController(make_config(SelectFirstPolicy))
    parent_pid = evidence._owner_pid
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_SINK", "tests.child:sink")
    monkeypatch.setattr(evidence.os, "getpid", lambda: parent_pid + 1)
    monkeypatch.setattr(
        evidence, "_start_identity", lambda: f"pid:{parent_pid + 1}:start_ticks:2"
    )
    monkeypatch.setattr(
        evidence.importlib,
        "import_module",
        lambda _name: type("ChildSink", (), {"sink": child_events.append}),
    )

    assert controller.select_victim(make_context()) == "first"

    assert [event["event"] for event in child_events] == ["resolved", "invoked"]
    assert all(event["process"]["pid"] == parent_pid + 1 for event in child_events)
    assert all(event["process"]["pid"] != parent_pid for event in child_events)


def test_controller_rejects_cross_thread_scheduler_use() -> None:
    controller = PreemptionPolicyController(make_config(SelectFirstPolicy))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(controller.select_victim, make_context())
        with pytest.raises(RuntimeError, match="scheduler thread"):
            future.result()

    assert controller.export_stats()["calls"] == 0
