# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for default-off request lifecycle events (vllm.v1.events).

Pure-Python: no NPU/torch required. Verifies:
  1. EventBus is disabled with no sinks; emit() is a no-op (zero overhead path).
  2. register_sink + emit dispatch typed RequestFinished / RequestPreempted.
  3. A failing sink is removed after its first failure; healthy sinks still
     receive subsequent events.
  4. unregister_sink disables the bus again when the last sink leaves.
  5. Events carry the documented payload fields.
"""

from __future__ import annotations

import pytest

from vllm.v1.events import (
    EventBus,
    RequestFinished,
    RequestPreempted,
    SchedulerEvent,
)


class RecordingSink:
    """Collects events; optionally raises on the Nth emit."""

    def __init__(self, fail_on: int | None = None):
        self.events: list[SchedulerEvent] = []
        self.fail_on = fail_on
        self._emits = 0

    def emit(self, event: SchedulerEvent) -> None:
        self._emits += 1
        if self.fail_on is not None and self._emits == self.fail_on:
            raise RuntimeError("sink failure")
        self.events.append(event)


@pytest.fixture(autouse=True)
def _clean_bus():
    """Ensure the bus starts/ends empty between tests."""
    assert not EventBus.enabled
    yield
    # Defensive: remove any sink a test may have leaked.
    EventBus._sinks.clear()
    EventBus.enabled = False


def test_disabled_by_default():
    # No sinks registered: emit must not raise and must do nothing.
    EventBus.emit(RequestFinished(request_id="r1", session_id=None,
                                  total_tokens=100,
                                  kv_blocks=5, finished_reason="stop"))
    assert not EventBus.enabled


def test_register_enables_and_dispatch():
    sink = RecordingSink()
    EventBus.register_sink(sink)
    assert EventBus.enabled

    EventBus.emit(RequestFinished(request_id="r1", session_id="sess-a",
                                  total_tokens=100,
                                  kv_blocks=5, finished_reason="stop"))
    EventBus.emit(RequestPreempted(request_id="r2", session_id="sess-a",
                                   freed_blocks=3,
                                   reason="preempt"))

    assert len(sink.events) == 2
    fin = sink.events[0]
    assert isinstance(fin, RequestFinished)
    assert fin.request_id == "r1"
    assert fin.session_id == "sess-a"
    assert fin.total_tokens == 100
    assert fin.kv_blocks == 5
    assert fin.finished_reason == "stop"
    # Timestamp is auto-filled at construction.
    assert fin.ts_monotonic_ns > 0

    pre = sink.events[1]
    assert isinstance(pre, RequestPreempted)
    assert pre.request_id == "r2"
    assert pre.session_id == "sess-a"
    assert pre.freed_blocks == 3
    assert pre.reason == "preempt"


def test_failing_sink_removed_healthy_sink_kept():
    bad = RecordingSink(fail_on=1)
    good = RecordingSink()
    EventBus.register_sink(bad)
    EventBus.register_sink(good)

    EventBus.emit(RequestFinished(request_id="r1", session_id=None,
                                  total_tokens=10,
                                  kv_blocks=1, finished_reason="stop"))
    # bad raised and was removed; good still received the event.
    assert bad.events == []
    assert len(good.events) == 1

    # Second emit: bad is gone, good still works, bus stays enabled.
    EventBus.emit(RequestFinished(request_id="r2", session_id=None,
                                  total_tokens=20,
                                  kv_blocks=2, finished_reason="stop"))
    assert len(good.events) == 2


def test_unregister_disables_when_last_sink():
    sink = RecordingSink()
    EventBus.register_sink(sink)
    assert EventBus.enabled
    EventBus.unregister_sink(sink)
    assert not EventBus.enabled
    # Emit after unregister is a no-op.
    EventBus.emit(RequestFinished(request_id="r1", session_id=None,
                                  total_tokens=1,
                                  kv_blocks=1, finished_reason="stop"))
    assert sink.events == []


def test_payload_field_types():
    import dataclasses
    fin = RequestFinished(request_id="abc", session_id="s",
                          total_tokens=123,
                          kv_blocks=7, finished_reason="length")
    fields = {f.name: f.type for f in dataclasses.fields(fin)}
    assert fields["request_id"] == "str"
    assert fields["session_id"] == "str | None"
    assert fields["total_tokens"] == "int"
    assert fields["kv_blocks"] == "int"
    assert fields["finished_reason"] == "str"
