# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the default-off offloading transfer observation seam."""

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading import observability
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.observability import (
    KV_TRANSFER_OBSERVER_CONTRACT,
    KVTransferEvent,
    TransferCancellationReason,
    TransferOperation,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
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
    worker.shutdown()
