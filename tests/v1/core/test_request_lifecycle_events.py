# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-level integration tests for the default-off lifecycle events.

These tests drive a real ``Scheduler`` (real ``KVCacheManager``, real request
queues) and assert the *timing contract* documented in ``vllm.v1.events``:
which event is emitted, at which point of the request lifecycle, and with
which payload. The whole point of the outlet is that a consumer can trust
those three things, so the tests cover the finished / aborted / preempted /
deferred-release paths plus the two EventBus guarantees (disabled without
sinks, a failing sink is dropped instead of the scheduler).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.events import (
    EventBus,
    RequestFinished,
    RequestKvReclaimed,
    RequestPreempted,
    SchedulerEvent,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@dataclass
class CollectingSink:
    """Sink that records every event it receives."""

    events: list[SchedulerEvent] = field(default_factory=list)

    def emit(self, event: SchedulerEvent) -> None:
        self.events.append(event)

    def of_type(self, kind: type) -> list:
        return [event for event in self.events if isinstance(event, kind)]


class FailingSink:
    """Sink that always raises; the bus must drop it, not the scheduler."""

    calls = 0

    def emit(self, event: SchedulerEvent) -> None:
        type(self).calls += 1
        raise RuntimeError("sink failure")


@pytest.fixture(autouse=True)
def clean_bus():
    """The bus is process-global, so keep every test hermetic."""
    EventBus._sinks = []
    EventBus.enabled = False
    FailingSink.calls = 0
    yield
    EventBus._sinks = []
    EventBus.enabled = False


@pytest.fixture
def sink() -> CollectingSink:
    collector = CollectingSink()
    EventBus.register_sink(collector)
    return collector


def make_model_output(scheduler_output, req_id: str) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_bus_is_disabled_until_a_sink_registers():
    """With no sink the outlet is a no-op guard, not a dispatch."""
    assert EventBus.enabled is False
    EventBus.emit(RequestKvReclaimed("r0", None, 3, "immediate"))

    collector = CollectingSink()
    EventBus.register_sink(collector)
    assert EventBus.enabled is True
    EventBus.emit(RequestKvReclaimed("r0", None, 3, "immediate"))
    assert len(collector.events) == 1


def test_abort_reports_real_blocks_three_token_views_and_reclaim_order(sink):
    """Abort path: the request really owned blocks, so kv_blocks must not be 0.

    This is the regression guard for the previous implementation, which read a
    non-existent ``Request.num_kv_blocks`` and therefore always reported 0.
    """
    scheduler = create_scheduler(block_size=16)
    requests = create_requests(num_requests=1, num_tokens=80, block_size=16)
    request = requests[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    assert len(scheduler_output.num_scheduled_tokens) == 1
    sink.events.clear()

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    finished = sink.of_type(RequestFinished)
    reclaimed = sink.of_type(RequestKvReclaimed)
    assert len(finished) == 1
    assert len(reclaimed) == 1

    event = finished[0]
    assert event.request_id == request.request_id
    # Real block count, not a placeholder zero.
    assert event.kv_blocks > 0
    assert event.kv_reclaim_deferred is False
    # The three token views are non-overlapping and self-consistent.
    assert event.prompt_tokens == request.num_prompt_tokens
    assert event.output_tokens == request.num_output_tokens
    assert event.sequence_tokens == event.prompt_tokens + event.output_tokens
    # num_computed_tokens must not leak into the sequence count.
    assert event.sequence_tokens == request.num_prompt_tokens
    assert event.finished_reason

    # The blocks were returned before the finish event was emitted.
    assert sink.events.index(reclaimed[0]) < sink.events.index(event)
    assert reclaimed[0].path == "immediate"
    assert reclaimed[0].kv_blocks == event.kv_blocks


def test_finish_after_decoding_counts_prompt_and_output_separately(sink):
    """A request that generated tokens reports prompt and output distinctly."""
    scheduler = create_scheduler(block_size=16)
    requests = create_requests(num_requests=1, num_tokens=80, block_size=16)
    request = requests[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output, make_model_output(scheduler_output, request.request_id)
    )
    generated = len(request.output_token_ids)
    assert generated > 0
    sink.events.clear()

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)

    event = sink.of_type(RequestFinished)[0]
    assert event.prompt_tokens == request.num_prompt_tokens
    assert event.output_tokens == generated
    assert event.sequence_tokens == event.prompt_tokens + generated
    assert event.kv_blocks > 0


def test_preemption_emits_preempted_after_the_kv_was_released(sink):
    """Preempt path: the decision event follows the block release it caused."""
    # 11 blocks minus the reserved null block leaves room for exactly one
    # 80-token/16-block-size request at a time, which forces a preemption.
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=11,
        enable_prefix_caching=False,
    )
    requests = create_requests(num_requests=2, num_tokens=80, block_size=16)

    scheduler.add_request(requests[0])
    first_output = scheduler.schedule()
    scheduler.add_request(requests[1])
    scheduler.schedule()
    scheduler.update_from_output(
        first_output, make_model_output(first_output, requests[0].request_id)
    )
    sink.events.clear()

    # KV pressure now forces the second request to be preempted.
    scheduler.schedule()
    assert requests[1].status == RequestStatus.PREEMPTED

    preempted = sink.of_type(RequestPreempted)
    reclaimed = sink.of_type(RequestKvReclaimed)
    assert len(preempted) == 1
    event = preempted[0]
    assert event.request_id == requests[1].request_id
    assert event.num_preemptions == 1
    assert event.kv_blocks > 0
    assert event.kv_reclaim_deferred is False

    # The reclaim happened first; the event describes the decision that
    # followed it, so consumers never read it as "KV released now".
    assert reclaimed, "preemption should have released KV blocks"
    assert sink.events.index(reclaimed[0]) < sink.events.index(event)
    assert reclaimed[0].request_id == requests[1].request_id


def test_deferred_release_reports_the_flag_then_the_real_reclaim(sink, monkeypatch):
    """A deferred release is not a reclaim until the fence step completes."""
    scheduler = create_scheduler(block_size=16)
    requests = create_requests(num_requests=1, num_tokens=80, block_size=16)
    request = requests[0]
    scheduler.add_request(request)
    scheduler.schedule()
    sink.events.clear()

    # Simulate a connector that must delay the block free.
    monkeypatch.setattr(
        scheduler, "_request_blocks_can_be_freed", lambda _request: False
    )
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    finished = sink.of_type(RequestFinished)
    assert len(finished) == 1
    assert finished[0].kv_reclaim_deferred is True
    assert finished[0].kv_blocks > 0
    # No reclaim yet: the blocks are still owned by the deferred FIFO.
    assert sink.of_type(RequestKvReclaimed) == []

    # Let the fence step complete, then drain.
    scheduler.processed_step_seq = scheduler.deferred_frees[-1][0]
    scheduler._drain_deferred_frees()

    reclaimed = sink.of_type(RequestKvReclaimed)
    assert len(reclaimed) == 1
    assert reclaimed[0].path == "deferred"
    assert reclaimed[0].request_id == request.request_id
    assert reclaimed[0].kv_blocks == finished[0].kv_blocks


def test_failing_sink_is_dropped_and_scheduling_continues():
    """A broken consumer loses its subscription; the scheduler is unharmed."""
    good = CollectingSink()
    bad = FailingSink()
    EventBus.register_sink(bad)
    EventBus.register_sink(good)

    scheduler = create_scheduler(block_size=16)
    requests = create_requests(num_requests=1, num_tokens=80, block_size=16)
    request = requests[0]
    scheduler.add_request(request)
    scheduler.schedule()
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    assert bad.calls > 0
    assert bad not in EventBus._sinks
    assert good in EventBus._sinks
    # The healthy sink still saw the whole lifecycle.
    assert sink_types(good) == {RequestKvReclaimed, RequestFinished}


def sink_types(collector: CollectingSink) -> set:
    return {type(event) for event in collector.events}


def test_waiting_abort_without_blocks_reports_zero_not_an_error(sink):
    """Aborting a never-scheduled request must not raise or invent blocks."""
    scheduler = create_scheduler(block_size=16)
    requests = create_requests(num_requests=1, num_tokens=80, block_size=16)
    request = requests[0]
    scheduler.add_request(request)

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    finished = sink.of_type(RequestFinished)
    assert len(finished) == 1
    assert finished[0].kv_blocks == 0
    assert finished[0].kv_reclaim_deferred is False


def test_scheduler_import_is_not_required_by_events_module():
    """The outlet stays a leaf module: no scheduler/KV-manager imports."""
    import vllm.v1.events as events_module

    source = pathlib.Path(events_module.__file__).read_text(encoding="utf-8")
    for banned in ("kv_cache_manager", "core.sched.scheduler", "core.sched.preemption"):
        assert banned not in source
    assert isinstance(Scheduler, type)
