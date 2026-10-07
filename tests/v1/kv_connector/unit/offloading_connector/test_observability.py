# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the default-off offloading transfer observation seam."""

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading import observability
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    MAX_LOAD_RECEIPT_JOBS,
    MAX_PENDING_FIRST_COMPUTE,
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.observability import (
    KV_TRANSFER_OBSERVER_CONTRACT,
    ComputeKind,
    KVRegionDescriptor,
    KVTransferEvent,
    RecoveryRequeueReason,
    TransferCancellationReason,
    TransferOperation,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    TransferResult,
)

# The record is a closed, address-free primitive set; adding a field is a
# contract change and must update this test deliberately.
_CLOSED_FIELDS = {
    "event",
    "operation",
    "job_id",
    "rank",
    "observed_at_ns",
    "request_id",
    "block_count",
    "success",
    "bytes_moved",
    "duration_ns",
    "reason",
    "ranks",
    "recovery_epoch",
    "job_ids",
    "requeue_reason",
    "descriptors",
    "dropped_descriptors",
    "compute_kind",
}


class _ExplodingClock:
    """Fails the test if the disabled seam reads a timestamp."""

    def monotonic_ns(self) -> int:
        raise AssertionError("disabled observation seam must not read the clock")


@pytest.fixture(autouse=True)
def _clean_registry():
    observability.reset_kv_transfer_observers()
    yield
    observability.reset_kv_transfer_observers()


def _make_worker(rank: int = 0, replicated_layout: bool = False):
    spec = MagicMock(spec=OffloadingSpec)
    spec.replicated_layout = replicated_layout
    spec.config = MagicMock()
    spec.config.canonical_layout = False
    spec.config.parallel.rank = rank
    spec.get_worker.return_value = MagicMock()
    worker = OffloadingConnectorWorker(
        spec=spec,
        vllm_config=MagicMock(),
        kv_cache_config=KVCacheConfig(
            num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
        ),
    )
    worker.worker = MagicMock()
    return worker


def _store_metadata(job_id: int, req_id: str) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={
            job_id: TransferJob(
                req_id=req_id,
                src_spec=GPULoadStoreSpec([0, 1], group_sizes=(2,), block_indices=(0,)),
                dst_spec=LoadStoreSpec(),
            )
        },
    )


def _load_metadata(job_id: int, req_id: str) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={
            job_id: TransferJob(
                req_id=req_id,
                src_spec=LoadStoreSpec(),
                dst_spec=GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,)),
            )
        },
        store_jobs={},
    )


def _empty_metadata() -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(load_jobs={}, store_jobs={})


def _admission_metadata(
    req_id: str,
    epoch: int,
    roster: tuple[int, ...],
    compute_kind: str,
) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={},
        recovery_admissions={req_id: (epoch, roster, compute_kind)},
    )


def _observe() -> list:
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)
    return seen


# ---------------------------------------------------------------------------
# Registry semantics
# ---------------------------------------------------------------------------


def test_disabled_seam_reads_no_clock(monkeypatch):
    monkeypatch.setattr(observability, "time", _ExplodingClock())
    assert not observability.kv_transfer_observers_configured()

    observability.emit_kv_transfer_submitted(
        operation=TransferOperation.D2H_PRESERVE,
        job_id=1,
        rank=0,
        request_id="req",
        block_count=2,
    )
    observability.emit_kv_transfer_completed(
        operation=TransferOperation.H2D_RESTORE,
        job_id=1,
        rank=0,
        request_id="req",
        success=True,
        bytes_moved=1,
        duration_ns=1,
    )
    observability.emit_kv_transfer_cancelled(
        operation=TransferOperation.H2D_RESTORE,
        job_id=1,
        rank=0,
        request_id="req",
        reason=TransferCancellationReason.HOST_SHUTDOWN,
    )
    observability.emit_kv_transfer_receipt(
        job_id=1,
        rank=None,
        request_id="req",
        ranks=(0, 1),
    )
    observability.emit_kv_recovery_requeued(
        request_id="req",
        recovery_epoch=1,
        reason=RecoveryRequeueReason.UNCLASSIFIED,
    )
    observability.emit_kv_recovery_admitted(
        request_id="req",
        recovery_epoch=1,
        job_ids=(1,),
    )
    observability.emit_kv_transfer_descriptors(
        job_id=1,
        rank=None,
        operation=TransferOperation.D2H_PRESERVE,
        descriptors=(
            KVRegionDescriptor(
                src_region_id=0,
                dst_region_id=0,
                src_offset=4096,
                dst_offset=0,
                size=512,
            ),
        ),
    )


def test_registered_observer_receives_a_closed_primitive_record():
    seen: list = []
    handle = observability.register_kv_transfer_observer("test", seen.append)
    assert handle.contract == KV_TRANSFER_OBSERVER_CONTRACT
    assert observability.kv_transfer_observers_configured()

    observability.emit_kv_transfer_submitted(
        operation=TransferOperation.H2D_RESTORE,
        job_id=7,
        rank=1,
        request_id="req-7",
        block_count=5,
    )

    (record,) = seen
    assert {field.name for field in fields(record)} == _CLOSED_FIELDS
    assert record.event is KVTransferEvent.TRANSFER_SUBMITTED
    assert record.operation is TransferOperation.H2D_RESTORE
    assert record.operation.direction == "h2d"
    assert TransferOperation.D2H_PRESERVE.direction == "d2h"
    assert record.job_id == 7
    assert record.rank == 1
    assert record.request_id == "req-7"
    assert record.block_count == 5
    assert record.success is None
    assert record.bytes_moved is None
    assert record.reason is None
    assert record.observed_at_ns > 0


def test_registered_observer_receives_completion_measurements():
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)

    observability.emit_kv_transfer_completed(
        operation=TransferOperation.D2H_PRESERVE,
        job_id=8,
        rank=0,
        request_id=None,
        success=True,
        bytes_moved=4096,
        duration_ns=2_000_000,
    )

    (record,) = seen
    assert record.event is KVTransferEvent.TRANSFER_COMPLETED
    assert record.bytes_moved == 4096
    assert record.duration_ns == 2_000_000


def test_unregister_is_idempotent_and_handle_is_contract_bound():
    seen: list = []
    handle = observability.register_kv_transfer_observer("test", seen.append)
    observability.unregister_kv_transfer_observer(handle)
    observability.unregister_kv_transfer_observer(handle)
    assert not observability.kv_transfer_observers_configured()

    observability.emit_kv_transfer_submitted(
        operation=TransferOperation.D2H_PRESERVE,
        job_id=1,
        rank=0,
        request_id="req",
        block_count=1,
    )
    assert seen == []

    foreign = observability.KVTransferObserverHandle("other.contract", 1)
    with pytest.raises(ValueError):
        observability.unregister_kv_transfer_observer(foreign)


def test_registration_rejects_invalid_input():
    with pytest.raises(ValueError):
        observability.register_kv_transfer_observer("  ", lambda record: None)
    with pytest.raises(ValueError):
        observability.register_kv_transfer_observer("test", None)  # type: ignore[arg-type]


def test_failing_observer_is_removed_without_blocking_peers():
    failures: list = []

    def failing(record) -> None:
        failures.append(record)
        raise RuntimeError("boom")

    healthy: list = []
    observability.register_kv_transfer_observer("failing", failing)
    observability.register_kv_transfer_observer("healthy", healthy.append)

    observability.emit_kv_transfer_submitted(
        operation=TransferOperation.D2H_PRESERVE,
        job_id=1,
        rank=0,
        request_id="req",
        block_count=1,
    )
    assert len(failures) == 1
    assert len(healthy) == 1

    observability.emit_kv_transfer_submitted(
        operation=TransferOperation.D2H_PRESERVE,
        job_id=2,
        rank=0,
        request_id="req",
        block_count=1,
    )
    # The failing observer was removed; the healthy one keeps receiving.
    assert len(failures) == 1
    assert len(healthy) == 2


def test_dispatch_uses_a_snapshot_of_registered_observers():
    late: list = []
    seen: list = []
    registered: list = []

    def late_registering_observer(record) -> None:
        seen.append(record)
        if not registered:
            registered.append(1)
            observability.register_kv_transfer_observer("late", late.append)

    observability.register_kv_transfer_observer("first", late_registering_observer)

    observability.emit_kv_transfer_submitted(
        operation=TransferOperation.D2H_PRESERVE,
        job_id=1,
        rank=0,
        request_id="req",
        block_count=1,
    )
    # The in-flight dispatch is a snapshot: it did not pick up the new observer.
    assert len(seen) == 1
    assert late == []

    observability.emit_kv_transfer_submitted(
        operation=TransferOperation.D2H_PRESERVE,
        job_id=2,
        rank=0,
        request_id="req",
        block_count=1,
    )
    # The observer registered during the previous dispatch sees the next one.
    assert len(seen) == 2
    assert len(late) == 1


def test_receipt_record_reports_worker_ranks():
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)

    observability.emit_kv_transfer_receipt(
        job_id=9,
        rank=None,
        request_id="req-9",
        ranks=(0, 3),
    )

    (record,) = seen
    assert record.event is KVTransferEvent.TRANSFER_RECEIPT
    assert record.operation is TransferOperation.H2D_RESTORE
    assert record.job_id == 9
    assert record.rank is None
    assert record.ranks == (0, 3)
    assert record.success is True


def test_receipt_aggregation_is_bounded_and_reports_overflow():
    left = OffloadingWorkerMetadata(
        load_receipts={job_id: (0,) for job_id in range(MAX_LOAD_RECEIPT_JOBS)}
    )
    right = OffloadingWorkerMetadata(
        load_receipts={
            job_id: (1,)
            for job_id in range(MAX_LOAD_RECEIPT_JOBS, MAX_LOAD_RECEIPT_JOBS * 2)
        }
    )

    merged = left.aggregate(right)

    assert isinstance(merged, OffloadingWorkerMetadata)
    assert len(merged.load_receipts) == MAX_LOAD_RECEIPT_JOBS
    assert tuple(merged.load_receipts) == tuple(range(MAX_LOAD_RECEIPT_JOBS))
    assert merged.dropped_load_receipts == MAX_LOAD_RECEIPT_JOBS


def test_descriptor_record_carries_only_relative_layout():
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)

    observability.emit_kv_transfer_descriptors(
        job_id=13,
        rank=2,
        operation=TransferOperation.H2D_RESTORE,
        descriptors=(
            KVRegionDescriptor(
                src_region_id=1,
                dst_region_id=1,
                src_offset=8192,
                dst_offset=0,
                size=512,
            ),
        ),
    )

    (record,) = seen
    assert record.event is KVTransferEvent.TRANSFER_DESCRIPTORS
    assert record.operation is TransferOperation.H2D_RESTORE
    assert record.job_id == 13
    assert record.rank == 2
    assert record.dropped_descriptors == 0
    assert record.descriptors == (
        KVRegionDescriptor(
            src_region_id=1,
            dst_region_id=1,
            src_offset=8192,
            dst_offset=0,
            size=512,
        ),
    )


def test_recovery_records_carry_epoch_and_roster():
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)

    observability.emit_kv_recovery_requeued(
        request_id="req-1",
        recovery_epoch=2,
        reason=RecoveryRequeueReason.UNCLASSIFIED,
    )
    observability.emit_kv_recovery_admitted(
        request_id="req-1",
        recovery_epoch=2,
        job_ids=(4, 5),
    )

    requeue, admitted = seen
    assert requeue.event is KVTransferEvent.RECOVERY_REQUEUED
    assert requeue.requeue_reason is RecoveryRequeueReason.UNCLASSIFIED
    assert requeue.recovery_epoch == 2
    assert requeue.job_id is None
    assert requeue.request_id == "req-1"
    assert admitted.event is KVTransferEvent.RECOVERY_ADMITTED
    assert admitted.recovery_epoch == 2
    assert admitted.job_ids == (4, 5)


def test_load_receipt_map_is_bounded():
    meta = OffloadingWorkerMetadata()

    for job_id in range(MAX_LOAD_RECEIPT_JOBS + 1):
        meta.mark_load_completed(job_id, 0)

    assert len(meta.load_receipts) == MAX_LOAD_RECEIPT_JOBS
    assert meta.dropped_load_receipts == 1


def test_worker_metadata_aggregates_load_receipts_by_rank():
    left = OffloadingWorkerMetadata()
    left.mark_completed(7)
    left.mark_load_completed(7, 1)
    right = OffloadingWorkerMetadata()
    right.mark_completed(7)
    right.mark_load_completed(7, 0)

    merged = left.aggregate(right)

    assert isinstance(merged, OffloadingWorkerMetadata)
    assert merged.completed_jobs == {7: 2}
    assert merged.load_receipts == {7: (0, 1)}
    assert merged.dropped_load_receipts == 0


# ---------------------------------------------------------------------------
# Worker-side observations
# ---------------------------------------------------------------------------


def test_load_submission_and_completion_are_observed():
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)
    worker = _make_worker(rank=2)

    worker.start_kv_transfers(_load_metadata(11, "req-11"))
    worker.worker.get_finished.return_value = [
        TransferResult(job_id=11, success=True, transfer_size=4096, transfer_time=0.002)
    ]
    assert worker.get_finished(set()) == (set(), {"req-11"})

    submitted, completed = seen
    assert submitted.event is KVTransferEvent.TRANSFER_SUBMITTED
    assert submitted.operation is TransferOperation.H2D_RESTORE
    assert submitted.job_id == 11
    assert submitted.rank == 2
    assert submitted.request_id == "req-11"
    assert submitted.block_count == 1

    assert completed.event is KVTransferEvent.TRANSFER_COMPLETED
    assert completed.operation is TransferOperation.H2D_RESTORE
    assert completed.request_id == "req-11"
    assert completed.bytes_moved == 4096
    assert completed.duration_ns == 2_000_000

    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.load_receipts == {11: (2,)}


def test_deferred_store_submission_and_completion_are_observed():
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)
    worker = _make_worker()

    worker.prepare_store_kv(_store_metadata(21, "req-21"))
    # prepare_store_kv only queues; it is not an observation point.
    assert seen == []

    worker.start_kv_transfers(_empty_metadata())
    worker.worker.get_finished.return_value = [
        TransferResult(job_id=21, success=True, transfer_size=8192, transfer_time=0.001)
    ]
    worker.get_finished(set())

    submitted, completed = seen
    assert submitted.event is KVTransferEvent.TRANSFER_SUBMITTED
    assert submitted.operation is TransferOperation.D2H_PRESERVE
    assert submitted.request_id == "req-21"
    assert submitted.block_count == 2
    assert completed.event is KVTransferEvent.TRANSFER_COMPLETED
    assert completed.operation is TransferOperation.D2H_PRESERVE
    assert completed.request_id == "req-21"


def test_preemption_flush_submits_and_observes_store():
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)
    worker = _make_worker()

    metadata = _store_metadata(31, "req-31")
    metadata.jobs_to_flush = {31}
    worker.handle_preemptions(metadata)

    (submitted,) = seen
    assert submitted.event is KVTransferEvent.TRANSFER_SUBMITTED
    assert submitted.job_id == 31
    assert submitted.request_id == "req-31"


def test_non_writer_store_confirmation_is_not_an_observation():
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)
    worker = _make_worker(rank=1, replicated_layout=True)

    worker.prepare_store_kv(_store_metadata(41, "req-41"))
    worker.start_kv_transfers(_empty_metadata())

    worker.worker.submit_store.assert_not_called()
    assert seen == []


def test_shutdown_cancels_still_open_transfers():
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)
    worker = _make_worker(rank=3)

    worker.start_kv_transfers(_load_metadata(51, "req-51"))
    worker.shutdown()

    submitted, cancelled = seen
    assert submitted.event is KVTransferEvent.TRANSFER_SUBMITTED
    assert cancelled.event is KVTransferEvent.TRANSFER_CANCELLED
    assert cancelled.operation is TransferOperation.H2D_RESTORE
    assert cancelled.job_id == 51
    assert cancelled.request_id == "req-51"
    assert cancelled.reason is TransferCancellationReason.HOST_SHUTDOWN
    assert worker._load_jobs == {}


def test_worker_is_inert_without_observers(monkeypatch):
    monkeypatch.setattr(observability, "time", _ExplodingClock())
    worker = _make_worker()

    worker.prepare_store_kv(_store_metadata(61, "req-61"))
    worker.start_kv_transfers(_load_metadata(62, "req-62"))
    worker.worker.get_finished.return_value = [TransferResult(job_id=62, success=True)]
    worker.get_finished(set())
    worker.note_recovery_admissions(_admission_metadata("req-63", 1, (63,), "decode"))
    worker.observe_forward_batch(["req-63"])
    worker.shutdown()


def test_first_compute_is_reported_once_with_epoch_and_roster():
    seen = _observe()
    worker = _make_worker(rank=3)
    worker.note_recovery_admissions(
        _admission_metadata("req-1", 2, (11, 12), "prefill")
    )

    worker.observe_forward_batch(["req-other", "req-1"])

    assert len(seen) == 1
    record = seen[0]
    assert record.event is KVTransferEvent.FIRST_COMPUTE
    assert record.request_id == "req-1"
    assert record.recovery_epoch == 2
    assert record.job_ids == (11, 12)
    assert record.compute_kind is ComputeKind.PREFILL
    assert record.rank == 3

    # Consumed exactly once: the same request never reports a second time.
    worker.observe_forward_batch(["req-1"])
    assert len(seen) == 1


def test_first_compute_waits_for_a_batch_that_contains_the_request():
    seen = _observe()
    worker = _make_worker()
    worker.note_recovery_admissions(_admission_metadata("req-1", 1, (7,), "decode"))

    worker.observe_forward_batch(["req-other"])
    assert seen == []

    worker.observe_forward_batch(["req-1"])
    assert len(seen) == 1
    assert seen[0].compute_kind is ComputeKind.DECODE
    assert seen[0].job_ids == (7,)


def test_a_newer_admission_replaces_the_pending_one():
    seen = _observe()
    worker = _make_worker()
    worker.note_recovery_admissions(_admission_metadata("req-1", 1, (7,), "decode"))
    worker.note_recovery_admissions(_admission_metadata("req-1", 2, (9,), "prefill"))

    worker.observe_forward_batch(["req-1"])

    assert len(seen) == 1
    assert seen[0].recovery_epoch == 2
    assert seen[0].job_ids == (9,)
    assert seen[0].compute_kind is ComputeKind.PREFILL


def test_finished_request_drops_its_pending_admission():
    seen = _observe()
    worker = _make_worker()
    worker.note_recovery_admissions(_admission_metadata("req-1", 1, (7,), "decode"))
    worker.worker.get_finished.return_value = []

    worker.get_finished({"req-1"})
    worker.observe_forward_batch(["req-1"])

    assert seen == []


def test_unknown_compute_kind_still_reports_the_forward():
    seen = _observe()
    worker = _make_worker()
    worker.note_recovery_admissions(_admission_metadata("req-1", 1, (7,), "sideways"))

    worker.observe_forward_batch(["req-1"])

    assert len(seen) == 1
    assert seen[0].event is KVTransferEvent.FIRST_COMPUTE
    assert seen[0].compute_kind is None


def test_pending_first_compute_map_is_bounded():
    worker = _make_worker()
    for index in range(MAX_PENDING_FIRST_COMPUTE + 3):
        worker.note_recovery_admissions(
            _admission_metadata(f"req-{index}", 1, (index,), "decode")
        )

    assert len(worker._recovery_admissions) == MAX_PENDING_FIRST_COMPUTE
    # The oldest entries are dropped first, the newest survive.
    assert "req-0" not in worker._recovery_admissions
    assert f"req-{MAX_PENDING_FIRST_COMPUTE + 2}" in worker._recovery_admissions


def test_connector_stashes_admissions_and_forwards_the_batch_to_its_worker():
    connector = object.__new__(OffloadingConnector)
    connector.connector_worker = MagicMock()
    metadata = _admission_metadata("req-1", 1, (7,), "decode")

    connector.bind_connector_metadata(metadata)
    connector.connector_worker.note_recovery_admissions.assert_called_once_with(
        metadata
    )

    connector.observe_forward_batch(["req-1", "req-2"])
    connector.connector_worker.observe_forward_batch.assert_called_once_with(
        ["req-1", "req-2"]
    )
