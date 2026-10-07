# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Final

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.base import LoadStoreSpec

ReqId = str

# Bounds the per-aggregation load-receipt map. The number of in-flight load
# jobs per engine step stays far below this; the cap only guards pathological
# states and overflow is counted, never silently absorbed.
MAX_LOAD_RECEIPT_JOBS: Final = 4096

# Bounds the recovery admissions one metadata can carry to the workers.
MAX_RECOVERY_ADMISSIONS: Final = 4096

# Bounds the worker-side map of admissions that await a first real forward.
MAX_PENDING_FIRST_COMPUTE: Final = 4096


@dataclass(slots=True)
class DirectionalTransferStats:
    bytes: int = 0
    time: float = 0.0
    sizes: list[int | float] = field(default_factory=list)

    def aggregate(
        self, other: "DirectionalTransferStats"
    ) -> "DirectionalTransferStats":
        return DirectionalTransferStats(
            bytes=self.bytes + other.bytes,
            time=self.time + other.time,
            sizes=[*self.sizes, *other.sizes],
        )

    def record(self, num_bytes: int, time: float) -> None:
        self.bytes += num_bytes
        self.time += time
        self.sizes.append(num_bytes)

    def is_empty(self) -> bool:
        return self.bytes == 0 and self.time == 0.0 and not self.sizes


@dataclass(slots=True)
class TransferStats:
    load: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)
    store: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)

    def aggregate(self, other: "TransferStats") -> "TransferStats":
        return TransferStats(
            load=self.load.aggregate(other.load),
            store=self.store.aggregate(other.store),
        )

    def is_empty(self) -> bool:
        return self.load.is_empty() and self.store.is_empty()


@dataclass
class TransferJob:
    """A transfer job bundling request context with transfer spec.

    Used for both loads and stores, keyed by scheduler-assigned job ID.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler processes the completion.
    """

    req_id: ReqId
    src_spec: LoadStoreSpec
    dst_spec: LoadStoreSpec


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None
    # req_id -> (recovery epoch, sorted restored job ids, compute kind value)
    # for episodes the scheduler admitted since the last metadata. Workers
    # consume an entry once, on that request's first real forward.
    recovery_admissions: dict[str, tuple[int, tuple[int, ...], str]] = field(
        default_factory=dict
    )


@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed transfer jobs.

    Each worker reports {job_id: 1} for newly completed transfer jobs
    (load or store). aggregate() sums counts across workers within a step.
    The scheduler accumulates across steps and processes
    a transfer completion only when count reaches num_workers.

    ``load_receipts`` additionally reports, for each completed load job, the
    ranks that finished it on their own device. It is observation-only: the
    scheduling counters above keep their existing semantics.
    """

    completed_jobs: dict[int, int] = field(default_factory=dict)
    transfer_stats: TransferStats = field(default_factory=TransferStats)
    # job_id -> sorted ranks that reported completing this load.
    load_receipts: dict[int, tuple[int, ...]] = field(default_factory=dict)
    # Load receipts dropped because the bounded map was already full.
    dropped_load_receipts: int = 0

    def mark_completed(self, job_id: int) -> None:
        """Record a transfer job completion from this worker."""
        self.completed_jobs[job_id] = 1

    def mark_load_completed(self, job_id: int, rank: int) -> None:
        """Record one rank's completion of a load (H2D) transfer."""
        ranks = self.load_receipts.get(job_id)
        if ranks is None:
            if len(self.load_receipts) >= MAX_LOAD_RECEIPT_JOBS:
                self.dropped_load_receipts += 1
                return
            self.load_receipts[job_id] = (rank,)
        elif rank not in ranks:
            self.load_receipts[job_id] = tuple(sorted((*ranks, rank)))

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        merged_receipts = {
            job_id: set(ranks) for job_id, ranks in self.load_receipts.items()
        }
        for job_id, ranks in other.load_receipts.items():
            merged_receipts.setdefault(job_id, set()).update(ranks)

        sorted_receipts = sorted(merged_receipts.items())
        retained_receipts = sorted_receipts[:MAX_LOAD_RECEIPT_JOBS]
        aggregation_drops = len(sorted_receipts) - len(retained_receipts)

        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
            load_receipts={
                job_id: tuple(sorted(ranks)) for job_id, ranks in retained_receipts
            },
            dropped_load_receipts=(
                self.dropped_load_receipts
                + other.dropped_load_receipts
                + aggregation_drops
            ),
        )
