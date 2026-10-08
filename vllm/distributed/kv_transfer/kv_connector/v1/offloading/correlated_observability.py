# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off, identity-bearing recovery observations (contract v2)."""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final

from vllm.logger import init_logger

logger = init_logger(__name__)

KV_TRANSFER_CORRELATED_OBSERVER_CONTRACT: Final = "vllm.kv-transfer.observer.v2"
KV_TRANSFER_CORRELATED_OBSERVABILITY_API_VERSION: Final = "2.0"
MAX_CORRELATED_ROSTER: Final = 4096
_UUID = re.compile(r"[0-9a-f]{32}")
_MAX_UINT64 = 2**64 - 1
_MAX_UINT32 = 2**32 - 1


def _uuid(value: str) -> None:
    if type(value) is not str or not _UUID.fullmatch(value):
        raise ValueError("generation must be a 32-character lowercase hex token")


def _uint(value: int, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("observation identity must be an unsigned integer")


@dataclass(frozen=True, slots=True, order=True)
class WorkerReceipt:
    rank: int
    worker_generation: str

    def __post_init__(self) -> None:
        _uint(self.rank, _MAX_UINT32)
        _uuid(self.worker_generation)


@dataclass(frozen=True, slots=True, order=True)
class JobReceipt:
    job_id: int
    workers: tuple[WorkerReceipt, ...]

    def __post_init__(self) -> None:
        _uint(self.job_id, _MAX_UINT64)
        if (
            type(self.workers) is not tuple
            or not 0 < len(self.workers) <= MAX_CORRELATED_ROSTER
            or any(type(item) is not WorkerReceipt for item in self.workers)
            or tuple(sorted(set(self.workers))) != self.workers
            or len({item.rank for item in self.workers}) != len(self.workers)
        ):
            raise ValueError("workers must be a bounded exact rank/generation roster")


class CorrelatedEvent(str, Enum):
    RECOVERY_REQUEUED = "recovery_requeued"
    RESTORE_SUBMITTED = "restore_submitted"
    RESTORE_COMPLETED = "restore_completed"
    TRANSFER_RECEIPT = "transfer_receipt"
    RECOVERY_ADMITTED = "recovery_admitted"
    FIRST_COMPUTE = "first_compute"


@dataclass(frozen=True, slots=True)
class CorrelatedObservation:
    event: CorrelatedEvent
    scheduler_generation: str
    request_id: str
    recovery_epoch: int
    observed_at_ns: int
    job_id: int | None = None
    rank: int | None = None
    worker_generation: str | None = None
    block_count: int | None = None
    workers: tuple[WorkerReceipt, ...] = ()
    roster: tuple[JobReceipt, ...] = ()
    compute_kind: str | None = None

    def __post_init__(self) -> None:
        if type(self.event) is not CorrelatedEvent:
            raise ValueError("event must be a CorrelatedEvent")
        _uuid(self.scheduler_generation)
        if (
            type(self.request_id) is not str
            or not self.request_id.isascii()
            or not self.request_id.isprintable()
            or not 0 < len(self.request_id) <= 128
        ):
            raise ValueError("request_id must be bounded printable ASCII")
        _uint(self.recovery_epoch, _MAX_UINT64)
        if self.recovery_epoch == 0:
            raise ValueError("recovery_epoch must be positive")
        _uint(self.observed_at_ns, _MAX_UINT64)
        if self.job_id is not None:
            _uint(self.job_id, _MAX_UINT64)
        if self.rank is not None:
            _uint(self.rank, _MAX_UINT32)
        if self.worker_generation is not None:
            _uuid(self.worker_generation)
        if self.block_count is not None:
            _uint(self.block_count, _MAX_UINT32)
            if self.block_count == 0:
                raise ValueError("block_count must be positive")
        if (
            type(self.workers) is not tuple
            or len(self.workers) > MAX_CORRELATED_ROSTER
            or any(type(item) is not WorkerReceipt for item in self.workers)
            or (self.workers and tuple(sorted(set(self.workers))) != self.workers)
            or len({item.rank for item in self.workers}) != len(self.workers)
        ):
            raise ValueError("workers must be a bounded sorted roster")
        if (
            type(self.roster) is not tuple
            or len(self.roster) > MAX_CORRELATED_ROSTER
            or any(type(item) is not JobReceipt for item in self.roster)
            or (self.roster and tuple(sorted(set(self.roster))) != self.roster)
            or len({item.job_id for item in self.roster}) != len(self.roster)
            or sum(len(item.workers) for item in self.roster) > MAX_CORRELATED_ROSTER
        ):
            raise ValueError("roster must be a bounded sorted job roster")
        fields = {
            name
            for name in (
                "job_id",
                "rank",
                "worker_generation",
                "block_count",
                "workers",
                "roster",
                "compute_kind",
            )
            if getattr(self, name) not in (None, ())
        }
        required = {
            CorrelatedEvent.RECOVERY_REQUEUED: set(),
            CorrelatedEvent.RESTORE_SUBMITTED: {
                "job_id",
                "rank",
                "worker_generation",
                "block_count",
            },
            CorrelatedEvent.RESTORE_COMPLETED: {"job_id", "rank", "worker_generation"},
            CorrelatedEvent.TRANSFER_RECEIPT: {"job_id", "workers"},
            CorrelatedEvent.RECOVERY_ADMITTED: {"roster"},
            CorrelatedEvent.FIRST_COMPUTE: {
                "rank",
                "worker_generation",
                "roster",
                "compute_kind",
            },
        }
        if fields != required[self.event]:
            raise ValueError("fields do not match correlated event shape")
        if self.compute_kind is not None and self.compute_kind not in {
            "prefill",
            "decode",
        }:
            raise ValueError("compute_kind must be prefill or decode")


CorrelatedObserver = Callable[[CorrelatedObservation], bool | None]


@dataclass(frozen=True, slots=True)
class CorrelatedObserverHandle:
    contract: str
    token: int


_observers: dict[int, CorrelatedObserver] = {}
_lock = threading.Lock()
_next_token = 0


def _reset_after_fork() -> None:
    """Do not inherit parent observers or a lock held by a vanished thread."""
    global _observers, _lock
    _observers = {}
    # Keep the counter so a copied parent handle cannot remove a child entry.
    _lock = threading.Lock()


os.register_at_fork(after_in_child=_reset_after_fork)


def register_correlated_observer(
    observer: CorrelatedObserver,
) -> CorrelatedObserverHandle:
    if not callable(observer):
        raise ValueError("observer must be callable")
    global _next_token
    with _lock:
        _next_token += 1
        token = _next_token
        _observers[token] = observer
    return CorrelatedObserverHandle(KV_TRANSFER_CORRELATED_OBSERVER_CONTRACT, token)


def unregister_correlated_observer(handle: CorrelatedObserverHandle) -> None:
    if handle.contract != KV_TRANSFER_CORRELATED_OBSERVER_CONTRACT:
        raise ValueError("correlated observer handle has the wrong contract")
    with _lock:
        _observers.pop(handle.token, None)


def correlated_observers_configured() -> bool:
    return bool(_observers)


def reset_correlated_observers() -> None:
    with _lock:
        _observers.clear()


def emit_correlated_observation(
    event: CorrelatedEvent,
    *,
    scheduler_generation: str,
    request_id: str,
    recovery_epoch: int,
    job_id: int | None = None,
    rank: int | None = None,
    worker_generation: str | None = None,
    block_count: int | None = None,
    workers: tuple[WorkerReceipt, ...] = (),
    roster: tuple[JobReceipt, ...] = (),
    compute_kind: str | None = None,
) -> None:
    if not _observers:
        return
    try:
        record = CorrelatedObservation(
            event=event,
            scheduler_generation=scheduler_generation,
            request_id=request_id,
            recovery_epoch=recovery_epoch,
            observed_at_ns=time.monotonic_ns(),
            job_id=job_id,
            rank=rank,
            worker_generation=worker_generation,
            block_count=block_count,
            workers=workers,
            roster=roster,
            compute_kind=compute_kind,
        )
    except (TypeError, ValueError):
        logger.exception("Invalid correlated KV observation; leaving serving unchanged")
        return
    with _lock:
        observers = tuple(_observers.items())
    for token, observer in observers:
        try:
            observer(record)
        except Exception:
            logger.exception("Correlated KV observer %d failed; removing it", token)
            with _lock:
                _observers.pop(token, None)
