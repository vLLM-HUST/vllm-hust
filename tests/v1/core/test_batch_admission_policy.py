# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.batch_admission import (
    BatchAdmission,
    BatchAdmissionContext,
    BatchAdmissionPolicyController,
    BatchRequest,
)

pytestmark = pytest.mark.cpu_test


def make_context(in_flight: frozenset[str] = frozenset()) -> BatchAdmissionContext:
    return BatchAdmissionContext(
        requests=(
            BatchRequest("first", "running", 0, 1.0, 10, 2, 11, 1, 20),
            BatchRequest("second", "waiting", 0, 2.0, 8, 0, 0, 0, 20),
        ),
        in_flight_batch_ids=in_flight,
        max_concurrent_batches=2,
        pipeline_parallel_size=2,
        now=3.0,
    )


def make_config(policy=None):
    scheduler_config = SimpleNamespace(batch_admission_policy=policy)
    return SimpleNamespace(scheduler_config=scheduler_config)


class FirstPolicy:
    completed: list[str] = []
    aborted: list[str] = []

    def admit_batch(self, context: BatchAdmissionContext) -> BatchAdmission:
        return BatchAdmission("batch-0", (context.requests[0].request_id,))

    def on_batch_complete(self, batch_id: str) -> None:
        type(self).completed.append(batch_id)

    def on_batch_abort(self, batch_id: str) -> None:
        type(self).aborted.append(batch_id)


class WaitingPolicy:
    def admit_batch(self, context: BatchAdmissionContext) -> None:
        return None


class UnknownRequestPolicy:
    def admit_batch(self, context: BatchAdmissionContext) -> BatchAdmission:
        return BatchAdmission("batch-0", ("missing",))


class InFlightBatchPolicy:
    def admit_batch(self, context: BatchAdmissionContext) -> BatchAdmission:
        return BatchAdmission("batch-0", ("first",))


class FailingPolicy:
    def admit_batch(self, context: BatchAdmissionContext) -> BatchAdmission:
        raise RuntimeError("policy failure")


def test_policy_receives_immutable_snapshots() -> None:
    context = make_context()

    with pytest.raises(FrozenInstanceError):
        context.now = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.requests[0].num_computed_tokens = 0  # type: ignore[misc]


def test_admission_and_completion_are_counted() -> None:
    FirstPolicy.completed = []
    FirstPolicy.aborted = []
    controller = BatchAdmissionPolicyController(make_config(FirstPolicy))

    assert controller.admit_batch(make_context()) == BatchAdmission(
        "batch-0", ("first",)
    )
    controller.on_batch_complete("batch-0")
    controller.on_batch_abort("batch-0")

    assert FirstPolicy.completed == ["batch-0"]
    assert FirstPolicy.aborted == ["batch-0"]
    assert controller.export_stats()["admissions"] == 1
    assert controller.export_stats()["completions"] == 1
    assert controller.export_stats()["aborts"] == 1


@pytest.mark.parametrize(
    ("policy", "context"),
    [
        (UnknownRequestPolicy, make_context()),
        (InFlightBatchPolicy, make_context(frozenset({"batch-0"}))),
    ],
)
def test_invalid_admission_disables_policy(policy, context) -> None:
    controller = BatchAdmissionPolicyController(make_config(policy))

    assert controller.admit_batch(context) is None
    assert not controller.enabled
    assert controller.export_stats()["invalid_admissions"] == 1


def test_exception_disables_policy_and_restores_builtin() -> None:
    controller = BatchAdmissionPolicyController(make_config(FailingPolicy))

    assert controller.admit_batch(make_context()) is None
    assert not controller.enabled
    assert controller.export_stats()["failures"] == 1


def test_abstention_is_valid_only_while_a_batch_is_in_flight() -> None:
    controller = BatchAdmissionPolicyController(make_config(WaitingPolicy))
    assert controller.admit_batch(make_context(frozenset({"batch-0"}))) is None
    assert controller.enabled

    controller = BatchAdmissionPolicyController(make_config(WaitingPolicy))
    assert controller.admit_batch(make_context()) is None
    assert not controller.enabled
