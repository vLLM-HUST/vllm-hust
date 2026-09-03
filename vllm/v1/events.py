# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off typed events for request lifecycle instrumentation.

This module provides a process-local typed event outlet for the vLLM
scheduler. It is intentionally generic: it exposes request-level lifecycle
events (a request finished, KV state was reclaimed) with no experiment- or
product-specific semantics. Out-of-tree consumers (e.g. context-lifecycle
management plugins) register a sink via :meth:`EventBus.register_sink` and
receive typed events; with no sink registered the bus is disabled and every
:meth:`EventBus.emit` is a single no-op guard, so serving overhead is zero.

This is the request/session-level counterpart of the worker-level
``AsyncOutput*`` lifecycle events. The two layers are orthogonal: worker
events describe output-token movement on the GPU path; these events describe
request admission/finish and KV reclamation at the scheduler.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SchedulerEvent:
    """Base class for process-local scheduler events."""

    ts_monotonic_ns: int = field(init=False, default_factory=time.monotonic_ns)


@dataclass(frozen=True)
class RequestFinished(SchedulerEvent):
    """A request finished decoding (stopped or aborted).

    payload: request id, optional session id (vllm-hust session-scoped
    requests), cumulative input+output tokens, and the number of KV blocks
    the request occupied at finish. Consumers use this to learn that one
    turn of a long conversation completed and how large its KV footprint
    was.
    """

    request_id: str
    session_id: str | None
    total_tokens: int
    kv_blocks: int
    finished_reason: str


@dataclass(frozen=True)
class RequestPreempted(SchedulerEvent):
    """KV state of a running request was reclaimed (preempted/evicted).

    payload: request id, optional session id, and the number of KV blocks
    freed. Consumers use this to observe KV pressure and react (e.g.
    trigger context compaction) before latency degrades.
    """

    request_id: str
    session_id: str | None
    freed_blocks: int
    reason: str


class EventSink(Protocol):
    """Consumer interface for scheduler events."""

    def emit(self, event: SchedulerEvent) -> None: ...


class EventBus:
    """Process-local typed event outlet that is disabled without sinks."""

    _sinks: list[EventSink] = []
    _lock = threading.Lock()
    enabled = False

    @classmethod
    def register_sink(cls, sink: EventSink) -> None:
        """Register one sink instance."""
        with cls._lock:
            if sink not in cls._sinks:
                cls._sinks.append(sink)
            cls.enabled = bool(cls._sinks)

    @classmethod
    def unregister_sink(cls, sink: EventSink) -> None:
        """Remove one sink instance."""
        with cls._lock:
            if sink in cls._sinks:
                cls._sinks.remove(sink)
            cls.enabled = bool(cls._sinks)

    @classmethod
    def emit(cls, event: SchedulerEvent) -> None:
        """Dispatch an event, disabling sinks that fail."""
        if not cls.enabled:
            return
        with cls._lock:
            sinks = tuple(cls._sinks)
        for sink in sinks:
            try:
                sink.emit(event)
            except Exception:
                with contextlib.suppress(Exception):
                    cls.unregister_sink(sink)
