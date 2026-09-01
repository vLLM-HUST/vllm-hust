# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off typed events for worker lifecycle instrumentation."""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class WorkerEvent:
    """Base class for process-local worker events."""

    ts_monotonic_ns: int = field(init=False, default_factory=time.monotonic_ns)


@dataclass(frozen=True)
class AsyncOutputCreated(WorkerEvent):
    """An asynchronous model-runner output lifecycle began."""

    lifecycle_id: int
    request_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class AsyncOutputCopyIssued(WorkerEvent):
    """A non-blocking device-to-host output copy was issued."""

    lifecycle_id: int
    storage_id: int
    event_id: int
    nbytes: int
    dispatch_ns: int


@dataclass(frozen=True)
class AsyncOutputCopyFailed(WorkerEvent):
    """An asynchronous output copy failed before completion."""

    lifecycle_id: int
    phase: str


@dataclass(frozen=True)
class AsyncOutputWaitComplete(WorkerEvent):
    """The readiness wait for an asynchronous output completed."""

    lifecycle_id: int
    wait_ns: int


@dataclass(frozen=True)
class AsyncOutputMaterialized(WorkerEvent):
    """An asynchronous output was materialized as host data."""

    lifecycle_id: int
    materialization_ns: int


@dataclass(frozen=True)
class AsyncOutputRetained(WorkerEvent):
    """An output host buffer was retained by the next input batch."""

    storage_id: int


@dataclass(frozen=True)
class AsyncOutputConsumed(WorkerEvent):
    """A retained output host buffer was consumed by an input batch."""

    storage_id: int
    wait_ns: int
    materialization_ns: int


class EventSink(Protocol):
    """Consumer interface for worker events."""

    def emit(self, event: WorkerEvent) -> None: ...


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
    def emit(cls, event: WorkerEvent) -> None:
        """Dispatch an event, disabling sinks that fail."""
        with cls._lock:
            sinks = tuple(cls._sinks)
        for sink in sinks:
            try:
                sink.emit(event)
            except Exception:
                with contextlib.suppress(Exception):
                    cls.unregister_sink(sink)
