# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for scheduler-side H2D receipts and the recovery observation chain."""

from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_config import (
    _make_mamba_hybrid_kv_cache_config,
    _make_vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.test_observability import (
    _load_metadata,
    _make_worker,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.config import KVEventsConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    correlated_observability as correlated,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import observability
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.observability import (
    KVTransferEvent,
    RecoveryRequeueReason,
    TransferOperation,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
    TransferJobStatus,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import TransferResult
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

REQ_ID = "req-rec"


@pytest.fixture(autouse=True)
def _clean_registry():
    observability.reset_kv_transfer_observers()
    correlated.reset_correlated_observers()
    yield
    observability.reset_kv_transfer_observers()
    correlated.reset_correlated_observers()


def _make_scheduler() -> OffloadingConnectorScheduler:
    vllm_config = _make_vllm_config(extra_config=None)
    vllm_config.speculative_config = None
    vllm_config.kv_events_config = KVEventsConfig(
        enable_kv_cache_events=False, publisher="null"
    )
    kv_cache_config = _make_mamba_hybrid_kv_cache_config()
    spec = MockOffloadingSpec(build_offloading_config(vllm_config, kv_cache_config))
    return OffloadingConnectorScheduler(spec, vllm_config, kv_cache_config)


def _track_request(
    scheduler: OffloadingConnectorScheduler,
    req_id: str = REQ_ID,
    num_preemptions: int = 1,
) -> RequestOffloadState:
    request = MagicMock()
    request.request_id = req_id
    request.kv_transfer_params = None
    request.num_prompt_tokens = 1200
    request.num_tokens = 1200
    request.num_computed_tokens = 0
    request.block_hashes = [BlockHash(f"{req_id}-{i}".encode()) for i in range(150)]
    request.all_token_ids = list(range(1200))
    request.lora_request = None
    request.shared_prefix_boundary = 0
    request.status = RequestStatus.RUNNING
    request.num_preemptions = num_preemptions
    request.is_finished.return_value = False
    scheduler.on_new_request(request)
    return scheduler._req_status[req_id]


def _observe() -> list:
    seen: list = []
    observability.register_kv_transfer_observer("test", seen.append)
    return seen


def _preempt(scheduler: OffloadingConnectorScheduler, req_id: str = REQ_ID) -> None:
    output = SchedulerOutput.make_empty()
    output.preempted_req_ids = {req_id}
    scheduler.build_connector_meta(output)


def _resume(scheduler: OffloadingConnectorScheduler, req_id: str = REQ_ID) -> None:
    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs.resumed_req_ids = {req_id}
    scheduler.build_connector_meta(output)


def _resume_with_meta(scheduler: OffloadingConnectorScheduler, req_id: str = REQ_ID):
    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs.resumed_req_ids = {req_id}
    return scheduler.build_connector_meta(output)


def _complete_load(
    scheduler: OffloadingConnectorScheduler,
    req_state: RequestOffloadState,
    job_id: int,
    ranks: tuple[int, ...] = (0, 1),
) -> None:
    """Deliver one aggregated load completion, as the workers report it."""
    req_id = req_state.req.request_id
    scheduler._jobs[job_id] = TransferJobStatus(
        req_id=req_id, pending_count=len(ranks), keys=set(), is_store=False
    )
    req_state.transfer_jobs.add(job_id)
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job_id: len(ranks)},
                load_receipts={job_id: ranks},
            )
        )
    )


def test_correlated_recovery_attests_exact_worker_generations():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = []
    correlated.register_correlated_observer(seen.append)
    _preempt(scheduler)

    workers = [_make_worker(rank=rank) for rank in (0, 1)]
    completed = []
    for worker in workers:
        job_meta = _load_metadata(7, REQ_ID)
        job_meta.load_jobs[7].scheduler_generation = scheduler._scheduler_generation
        job_meta.load_jobs[7].recovery_epoch = 1
        worker.start_kv_transfers(job_meta)
        worker.worker.get_finished.return_value = [
            TransferResult(job_id=7, success=True, transfer_size=512)
        ]
        worker.get_finished(set())
        completed.append(worker.build_connector_worker_meta())

    scheduler._jobs[7] = TransferJobStatus(
        req_id=REQ_ID,
        pending_count=2,
        keys=set(),
        is_store=False,
        recovery_epoch=1,
    )
    req_state.transfer_jobs.add(7)
    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=completed[0].aggregate(completed[1]))
    )
    admission = _resume_with_meta(scheduler)
    for worker in workers:
        worker.note_recovery_admissions(admission)
        worker.observe_forward_batch([REQ_ID])

    assert [item.event for item in seen] == [
        correlated.CorrelatedEvent.RECOVERY_REQUEUED,
        correlated.CorrelatedEvent.RESTORE_SUBMITTED,
        correlated.CorrelatedEvent.RESTORE_COMPLETED,
        correlated.CorrelatedEvent.RESTORE_SUBMITTED,
        correlated.CorrelatedEvent.RESTORE_COMPLETED,
        correlated.CorrelatedEvent.TRANSFER_RECEIPT,
        correlated.CorrelatedEvent.RECOVERY_ADMITTED,
        correlated.CorrelatedEvent.FIRST_COMPUTE,
        correlated.CorrelatedEvent.FIRST_COMPUTE,
    ]
    expected_workers = tuple(
        sorted(
            correlated.WorkerReceipt(worker._rank, worker._worker_generation)
            for worker in workers
        )
    )
    assert seen[5].workers == expected_workers
    assert seen[6].roster == (correlated.JobReceipt(7, expected_workers),)
    assert all(item.roster == seen[6].roster for item in seen[7:])
    for worker in workers:
        worker.observe_forward_batch([REQ_ID])
    assert len(seen) == 9


def test_correlated_admission_requires_worker_generation_receipts():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = []
    correlated.register_correlated_observer(seen.append)
    _preempt(scheduler)
    _complete_load(scheduler, req_state, 7, ranks=(0, 1))
    _resume(scheduler)
    assert [item.event for item in seen] == [
        correlated.CorrelatedEvent.RECOVERY_REQUEUED
    ]


def test_correlated_first_compute_does_not_require_v1_admission_metadata():
    worker = _make_worker(rank=0)
    seen = []
    correlated.register_correlated_observer(seen.append)
    roster = (
        correlated.JobReceipt(
            7,
            (correlated.WorkerReceipt(0, worker._worker_generation),),
        ),
    )
    worker.note_recovery_admissions(
        OffloadingConnectorMetadata(
            load_jobs={},
            store_jobs={},
            recovery_admissions_v2={REQ_ID: ("a" * 32, 1, roster, "decode")},
        )
    )
    worker.observe_forward_batch([REQ_ID])
    assert [record.event for record in seen] == [
        correlated.CorrelatedEvent.FIRST_COMPUTE
    ]


# ---------------------------------------------------------------------------
# H2D receipts
# ---------------------------------------------------------------------------


def test_restore_completion_emits_receipt_with_exact_ranks():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()

    _complete_load(scheduler, req_state, job_id=7, ranks=(0, 1))

    (receipt,) = seen
    assert receipt.event is KVTransferEvent.TRANSFER_RECEIPT
    assert receipt.operation is TransferOperation.H2D_RESTORE
    assert receipt.job_id == 7
    assert receipt.request_id == REQ_ID
    assert receipt.ranks == (0, 1)
    assert receipt.rank is None


def test_receipt_waits_for_every_worker_before_publishing():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()

    req_id = req_state.req.request_id
    scheduler._jobs[9] = TransferJobStatus(
        req_id=req_id, pending_count=2, keys=set(), is_store=False
    )
    req_state.transfer_jobs.add(9)
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={9: 1}, load_receipts={9: (0,)}
            )
        )
    )
    assert seen == []

    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={9: 1}, load_receipts={9: (1,)}
            )
        )
    )
    (receipt,) = seen
    assert receipt.ranks == (0, 1)


def test_incomplete_receipt_is_not_published_or_admitted():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()
    _preempt(scheduler)

    job_id = 10
    scheduler._jobs[job_id] = TransferJobStatus(
        req_id=REQ_ID, pending_count=2, keys=set(), is_store=False
    )
    req_state.transfer_jobs.add(job_id)
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job_id: 2},
                load_receipts={job_id: (0,)},
                dropped_load_receipts=1,
            )
        )
    )
    _resume(scheduler)

    assert [record.event for record in seen] == [KVTransferEvent.RECOVERY_REQUEUED]


# ---------------------------------------------------------------------------
# Recovery requeue / admission
# ---------------------------------------------------------------------------


def test_preemption_opens_a_recovery_episode():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()

    _preempt(scheduler)

    (requeue,) = seen
    assert requeue.event is KVTransferEvent.RECOVERY_REQUEUED
    assert requeue.request_id == REQ_ID
    assert requeue.recovery_epoch == 1
    assert requeue.requeue_reason is RecoveryRequeueReason.UNCLASSIFIED
    assert req_state.pending_recovery is True
    assert req_state.restored_job_ids == []


def test_admission_reports_the_exact_restored_roster():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()

    _preempt(scheduler)
    _complete_load(scheduler, req_state, job_id=11, ranks=(0, 1))
    _complete_load(scheduler, req_state, job_id=12, ranks=(0, 1))
    _resume(scheduler)

    requeue, receipt_a, receipt_b, admitted = seen
    assert requeue.event is KVTransferEvent.RECOVERY_REQUEUED
    assert receipt_a.event is KVTransferEvent.TRANSFER_RECEIPT
    assert receipt_b.event is KVTransferEvent.TRANSFER_RECEIPT
    assert admitted.event is KVTransferEvent.RECOVERY_ADMITTED
    assert admitted.request_id == REQ_ID
    assert admitted.recovery_epoch == 1
    assert admitted.job_ids == (11, 12)
    assert req_state.pending_recovery is False


def test_requeue_without_restore_emits_no_admission():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()

    _preempt(scheduler)
    _resume(scheduler)

    assert [record.event for record in seen] == [KVTransferEvent.RECOVERY_REQUEUED]
    assert req_state.pending_recovery is False


def test_plain_prefix_load_is_not_a_recovery_admission():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()

    # No preemption: the request only fills from the offload cache.
    _complete_load(scheduler, req_state, job_id=21, ranks=(0,))
    _resume(scheduler)

    assert [record.event for record in seen] == [KVTransferEvent.TRANSFER_RECEIPT]
    assert req_state.restored_job_ids == []


def test_admission_is_emitted_once_per_episode():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()

    _preempt(scheduler)
    _complete_load(scheduler, req_state, job_id=31, ranks=(0,))
    _resume(scheduler)
    _resume(scheduler)

    assert [record.event for record in seen].count(
        KVTransferEvent.RECOVERY_ADMITTED
    ) == 1


def test_second_preemption_starts_a_new_epoch():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()

    _preempt(scheduler)
    _complete_load(scheduler, req_state, job_id=41, ranks=(0,))
    _resume(scheduler)

    req_state.req.num_preemptions = 2
    _preempt(scheduler)

    requeues = [
        record for record in seen if record.event is KVTransferEvent.RECOVERY_REQUEUED
    ]
    assert [record.recovery_epoch for record in requeues] == [1, 2]
    assert req_state.restored_job_ids == []


def test_overflowed_roster_emits_no_admission():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    seen = _observe()

    _preempt(scheduler)
    req_state.restored_job_ids = [51]
    req_state.restored_overflow = True
    _resume(scheduler)

    assert [record.event for record in seen] == [KVTransferEvent.RECOVERY_REQUEUED]
    assert req_state.pending_recovery is False


def test_admission_is_shipped_to_workers_with_its_compute_kind():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    _observe()

    _preempt(scheduler)
    _complete_load(scheduler, req_state, job_id=11, ranks=(0, 1))
    _complete_load(scheduler, req_state, job_id=12, ranks=(0, 1))
    meta = _resume_with_meta(scheduler)

    assert meta.recovery_admissions == {REQ_ID: (1, (11, 12), "prefill")}


def test_admission_ships_as_decode_once_the_prompt_is_computed():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    _observe()

    _preempt(scheduler)
    _complete_load(scheduler, req_state, job_id=11, ranks=(0, 1))
    req_state.req.num_computed_tokens = req_state.req.num_prompt_tokens
    meta = _resume_with_meta(scheduler)

    assert meta.recovery_admissions == {REQ_ID: (1, (11,), "decode")}


def test_admission_metadata_is_shipped_once():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    _observe()

    _preempt(scheduler)
    _complete_load(scheduler, req_state, job_id=11, ranks=(0, 1))
    assert _resume_with_meta(scheduler).recovery_admissions
    assert _resume_with_meta(scheduler).recovery_admissions == {}


def test_overflowed_roster_ships_no_admission_metadata():
    scheduler = _make_scheduler()
    req_state = _track_request(scheduler)
    _observe()

    _preempt(scheduler)
    req_state.restored_job_ids = [51]
    req_state.restored_overflow = True
    meta = _resume_with_meta(scheduler)

    assert meta is None or meta.recovery_admissions == {}
