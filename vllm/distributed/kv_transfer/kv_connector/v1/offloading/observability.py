# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off observation seam for offloading KV transfers.

The seam publishes bounded, address-free lifecycle records for the
``OffloadingConnector``'s CPU<->device transfers. Out-of-tree observers own
serialization, queueing, destinations, and failure handling; the host only
hands them immutable primitive records. A record never carries tensors,
process or device addresses, block hashes, token IDs, or KV payloads: a
transfer is identified by the connector job id plus bounded request/rank
facts.

With no observer registered every publish is a single dictionary guard: the
disabled path reads no clock, builds no record, and schedules no work.
Registration is explicit and process-local and returns a removable handle;
unregistration is idempotent. Observer exceptions are logged and the failing
observer is removed without affecting serving.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final

from vllm.logger import init_logger

logger = init_logger(__name__)

# Contract identity published by this seam. Out-of-tree observers use it to
# confirm that they attached to the expected host API.
KV_TRANSFER_OBSERVER_CONTRACT: Final = "vllm.kv-transfer.observer.v1"
KV_TRANSFER_OBSERVABILITY_API_VERSION: Final = "1.0"


class KVTransferEvent(str, Enum):
    """Closed lifecycle vocabulary published by the transfer seam."""

    TRANSFER_SUBMITTED = "transfer_submitted"
    TRANSFER_COMPLETED = "transfer_completed"
    TRANSFER_CANCELLED = "transfer_cancelled"
    TRANSFER_RECEIPT = "transfer_receipt"
    TRANSFER_DESCRIPTORS = "transfer_descriptors"
    RECOVERY_REQUEUED = "recovery_requeued"
    RECOVERY_ADMITTED = "recovery_admitted"
    FIRST_COMPUTE = "first_compute"


class TransferOperation(str, Enum):
    """Offloaded transfer operation, named after the KV lifecycle it serves.

    ``D2H_PRESERVE`` keeps KV blocks by moving them off the device;
    ``H2D_RESTORE`` brings previously offloaded blocks back to the device.
    """

    D2H_PRESERVE = "d2h_preserve"
    H2D_RESTORE = "h2d_restore"

    @property
    def direction(self) -> str:
        """Direction token: ``d2h`` for preserves, ``h2d`` for restores."""
        if self is TransferOperation.D2H_PRESERVE:
            return "d2h"
        return "h2d"


class TransferCancellationReason(str, Enum):
    """Closed cancellation reasons this seam can attest."""

    HOST_SHUTDOWN = "host_shutdown"


class RecoveryRequeueReason(str, Enum):
    """Closed requeue reasons this seam can attest.

    The connector only learns that a request was preempted, not the
    scheduler-internal reason; ``unclassified`` is the honest value until a
    finer host fact exists.
    """

    UNCLASSIFIED = "unclassified"


class ComputeKind(str, Enum):
    """Shape of the first real forward of a recovery episode.

    Classified by the step's scheduled tokens: a one-token step is a decode
    step, anything wider is prefill work.
    """

    PREFILL = "prefill"
    DECODE = "decode"


# Bounds one transfer's region-descriptor inventory, matching the bounded
# inventory limit the consumer enforces.
MAX_DESCRIPTOR_REGIONS: Final = 4096


@dataclass(frozen=True, slots=True)
class KVRegionDescriptor:
    """One region-relative copy descriptor.

    Region ids name the data-holding regions of each side (one id space per
    side, stable within a process); offsets are relative to the region base
    and sizes are in bytes. No process or device address, tensor object,
    block hash, token id, or payload can be recovered from a descriptor.
    """

    src_region_id: int
    dst_region_id: int
    src_offset: int
    dst_offset: int
    size: int


@dataclass(frozen=True, slots=True)
class KVTransferObservation:
    """One bounded, address-free KV transfer lifecycle observation.

    Attributes:
        event: Which lifecycle transition was observed.
        observed_at_ns: Monotonic timestamp taken at observation.
        operation: Transfer operation that produced the observation.
        job_id: Connector-assigned transfer job id.
        rank: Rank of the reporting worker process.
        request_id: Request owning the transfer, when the connector knows it.
        block_count: Number of KV blocks in the transfer.
        success: Terminal success flag (completions only).
        bytes_moved: Backend-reported transferred bytes (completions only).
        duration_ns: Backend-reported transfer duration (completions only).
        reason: Cancellation reason (cancellations only).
        ranks: Sorted ranks that reported one transfer (receipts only).
        recovery_epoch: Positive preemption generation, taken from the
            request's preemption counter (recovery records only).
        job_ids: Sorted connector job ids of the restored transfers
            (admissions only).
        requeue_reason: Requeue reason (requeues only).
        descriptors: Region-relative copy descriptors (layouts only).
        dropped_descriptors: Descriptors dropped by the bounded inventory
            (layouts only).
        compute_kind: Shape of the reported forward (first compute only).

    """

    event: KVTransferEvent
    observed_at_ns: int
    operation: TransferOperation | None = None
    job_id: int | None = None
    rank: int | None = None
    request_id: str | None = None
    block_count: int | None = None
    success: bool | None = None
    bytes_moved: int | None = None
    duration_ns: int | None = None
    reason: TransferCancellationReason | None = None
    ranks: tuple[int, ...] = ()
    recovery_epoch: int | None = None
    job_ids: tuple[int, ...] = ()
    requeue_reason: RecoveryRequeueReason | None = None
    descriptors: tuple[KVRegionDescriptor, ...] = ()
    dropped_descriptors: int = 0
    compute_kind: ComputeKind | None = None


KVTransferObserver = Callable[[KVTransferObservation], bool | None]


@dataclass(frozen=True, slots=True)
class KVTransferObserverHandle:
    """Removable registration handle for one observer."""

    contract: str
    token: int


_observers: dict[int, tuple[str, KVTransferObserver]] = {}
_next_token = 0
_registry_lock = threading.Lock()


def _reset_after_fork() -> None:
    """Do not inherit parent observers or a lock held by a vanished thread."""
    global _observers, _registry_lock
    _observers = {}
    # Keep the counter so a copied parent handle cannot remove a child entry.
    _registry_lock = threading.Lock()


os.register_at_fork(after_in_child=_reset_after_fork)


def register_kv_transfer_observer(
    name: str, observer: KVTransferObserver
) -> KVTransferObserverHandle:
    """Register one process-local observer.

    Args:
        name: Non-empty label used when reporting observer failures.
        observer: Callable receiving immutable ``KVTransferObservation``
            records; it must not block or retain unbounded state. Returning
            exactly ``True`` attests that the record was accepted and lets the
            host emit process-owned runtime-effective evidence.

    Returns:
        A handle accepted by :func:`unregister_kv_transfer_observer`.

    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("observer name must be a non-empty string")
    if not callable(observer):
        raise ValueError("observer must be callable")
    global _next_token
    with _registry_lock:
        _next_token += 1
        token = _next_token
        _observers[token] = (name, observer)
    logger.debug("KV transfer observer %r registered (%d active)", name, token)
    return KVTransferObserverHandle(contract=KV_TRANSFER_OBSERVER_CONTRACT, token=token)


def unregister_kv_transfer_observer(handle: KVTransferObserverHandle) -> None:
    """Remove a registered observer; safe to call more than once."""
    if handle.contract != KV_TRANSFER_OBSERVER_CONTRACT:
        raise ValueError("handle was not issued by this observation contract")
    with _registry_lock:
        _observers.pop(handle.token, None)


def kv_transfer_observers_configured() -> bool:
    """Whether any observer is registered in this process."""
    return bool(_observers)


def reset_kv_transfer_observers() -> None:
    """Drop every observer; used by tests and process shutdown paths."""
    with _registry_lock:
        _observers.clear()


def _publish(observation: KVTransferObservation) -> None:
    with _registry_lock:
        snapshot = tuple(_observers.items())
    for token, (name, observer) in snapshot:
        try:
            accepted = observer(observation)
        except Exception:
            logger.exception("KV transfer observer %d failed; removing it", token)
            with _registry_lock:
                _observers.pop(token, None)
            continue
        if accepted is not True:
            continue
        try:
            from vllm.plugins import DEFAULT_PLUGINS_GROUP
            from vllm.plugins.evidence import _emit_host_event

            _emit_host_event(
                "effective",
                DEFAULT_PLUGINS_GROUP,
                name,
                "vllm.kv-transfer.observer.v1",
                detail=observation.event.value,
                occurrence_id=observation.observed_at_ns,
                observation_kind="runtime_effective",
            )
        except Exception:
            logger.exception("KV transfer observer %d evidence delivery failed", token)


def emit_kv_transfer_submitted(
    *,
    operation: TransferOperation,
    job_id: int,
    rank: int,
    request_id: str | None,
    block_count: int | None,
) -> None:
    """Publish a transfer whose backend submission succeeded.

    Called only after the offloading backend accepted the job, so the record
    is a submission receipt, not a queued request.
    """
    if not _observers:
        return
    _publish(
        KVTransferObservation(
            event=KVTransferEvent.TRANSFER_SUBMITTED,
            operation=operation,
            job_id=job_id,
            rank=rank,
            observed_at_ns=time.monotonic_ns(),
            request_id=request_id,
            block_count=block_count,
        )
    )


def emit_kv_transfer_completed(
    *,
    operation: TransferOperation,
    job_id: int,
    rank: int,
    request_id: str | None,
    success: bool,
    bytes_moved: int | None,
    duration_ns: int | None,
) -> None:
    """Publish a terminal transfer result reported by the backend."""
    if not _observers:
        return
    _publish(
        KVTransferObservation(
            event=KVTransferEvent.TRANSFER_COMPLETED,
            operation=operation,
            job_id=job_id,
            rank=rank,
            observed_at_ns=time.monotonic_ns(),
            request_id=request_id,
            success=success,
            bytes_moved=bytes_moved,
            duration_ns=duration_ns,
        )
    )


def emit_kv_transfer_cancelled(
    *,
    operation: TransferOperation,
    job_id: int,
    rank: int,
    request_id: str | None,
    reason: TransferCancellationReason,
) -> None:
    """Publish a still-open transfer that will not complete.

    Only reasons the host can attest at the observation point are accepted;
    a missing completion is never reported as a successful completion.
    """
    if not _observers:
        return
    _publish(
        KVTransferObservation(
            event=KVTransferEvent.TRANSFER_CANCELLED,
            operation=operation,
            job_id=job_id,
            rank=rank,
            observed_at_ns=time.monotonic_ns(),
            request_id=request_id,
            reason=reason,
        )
    )


def emit_kv_transfer_receipt(
    *,
    job_id: int,
    rank: int | None,
    request_id: str | None,
    ranks: tuple[int, ...],
) -> None:
    """Publish the aggregate completion receipt of one load transfer.

    The scheduler emits this once every worker that owes a completion for the
    job reported it, so ``ranks`` is the exact set of ranks that finished the
    restore on their own device.
    """
    if not _observers:
        return
    _publish(
        KVTransferObservation(
            event=KVTransferEvent.TRANSFER_RECEIPT,
            operation=TransferOperation.H2D_RESTORE,
            job_id=job_id,
            rank=rank,
            observed_at_ns=time.monotonic_ns(),
            request_id=request_id,
            success=True,
            ranks=ranks,
        )
    )


def emit_kv_recovery_requeued(
    *,
    request_id: str,
    recovery_epoch: int,
    reason: RecoveryRequeueReason,
) -> None:
    """Publish a request whose KV was just preempted and requeued.

    The request needs recovery before it can run again; this is not an
    admission and must not be reported as one.
    """
    if not _observers:
        return
    _publish(
        KVTransferObservation(
            event=KVTransferEvent.RECOVERY_REQUEUED,
            observed_at_ns=time.monotonic_ns(),
            request_id=request_id,
            recovery_epoch=recovery_epoch,
            requeue_reason=reason,
        )
    )


def emit_kv_recovery_admitted(
    *,
    request_id: str,
    recovery_epoch: int,
    job_ids: tuple[int, ...],
) -> None:
    """Publish a recovered request that is actually scheduled again.

    Emitted only when the exact roster of successfully restored transfers is
    known, and only at the point where the scheduler resumes the request; a
    completed transfer by itself is never reported as an admission.
    """
    if not _observers:
        return
    _publish(
        KVTransferObservation(
            event=KVTransferEvent.RECOVERY_ADMITTED,
            observed_at_ns=time.monotonic_ns(),
            request_id=request_id,
            recovery_epoch=recovery_epoch,
            job_ids=job_ids,
        )
    )


def emit_kv_transfer_descriptors(
    *,
    job_id: int,
    rank: int | None,
    operation: TransferOperation,
    descriptors: tuple[KVRegionDescriptor, ...],
    dropped_descriptors: int = 0,
) -> None:
    """Publish the bounded region-relative layout of one transfer.

    Built at the copy site from the filled copy ops: offsets are taken
    relative to each region's base, so implementation addresses never cross
    the callback boundary and no post-hoc filtering is required.
    """
    if not _observers:
        return
    _publish(
        KVTransferObservation(
            event=KVTransferEvent.TRANSFER_DESCRIPTORS,
            observed_at_ns=time.monotonic_ns(),
            operation=operation,
            job_id=job_id,
            rank=rank,
            descriptors=descriptors,
            dropped_descriptors=dropped_descriptors,
        )
    )


def emit_kv_first_compute(
    *,
    request_id: str,
    recovery_epoch: int,
    job_ids: tuple[int, ...],
    compute_kind: ComputeKind | None,
    rank: int | None,
) -> None:
    """Publish the first real forward of an admitted recovery episode.

    The caller owns consume-once: the pending marker is dropped before this
    is called, so a repeated batch can never report the same episode twice.
    """
    if not _observers:
        return
    _publish(
        KVTransferObservation(
            event=KVTransferEvent.FIRST_COMPUTE,
            observed_at_ns=time.monotonic_ns(),
            request_id=request_id,
            recovery_epoch=recovery_epoch,
            job_ids=job_ids,
            compute_kind=compute_kind,
            rank=rank,
        )
    )


__all__ = [
    "KV_TRANSFER_OBSERVABILITY_API_VERSION",
    "KV_TRANSFER_OBSERVER_CONTRACT",
    "MAX_DESCRIPTOR_REGIONS",
    "ComputeKind",
    "KVRegionDescriptor",
    "KVTransferEvent",
    "KVTransferObservation",
    "KVTransferObserver",
    "KVTransferObserverHandle",
    "RecoveryRequeueReason",
    "TransferCancellationReason",
    "TransferOperation",
    "emit_kv_first_compute",
    "emit_kv_recovery_admitted",
    "emit_kv_recovery_requeued",
    "emit_kv_transfer_cancelled",
    "emit_kv_transfer_completed",
    "emit_kv_transfer_descriptors",
    "emit_kv_transfer_receipt",
    "emit_kv_transfer_submitted",
    "kv_transfer_observers_configured",
    "register_kv_transfer_observer",
    "reset_kv_transfer_observers",
    "unregister_kv_transfer_observer",
]
