"""本模块的作用：在调度器与 worker 之间传递物化分支和计时样本。
输入：原生 SimpleCPUOffload 元数据、请求进度及分支测量。
输出：保持原生执行语义的扩展元数据，不改变 load/recompute 动作。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)


@dataclass
class DynamicCPUOffloadMetadata(SimpleCPUOffloadMetadata):
    """扩展原生 offload 元数据以携带重算工作量。"""

    recompute_requests: dict[str, int] = field(default_factory=dict)
    recompute_flops: dict[str, float] = field(default_factory=dict)
    reset_recompute_requests: set[str] = field(default_factory=set)
    completed_recompute_requests: set[str] = field(default_factory=set)
    load_block_counts: dict[str, int] = field(default_factory=dict)
    decision_times: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        base: SimpleCPUOffloadMetadata,
        recompute_requests: dict[str, int],
        reset_recompute_requests: set[str],
        completed_recompute_requests: set[str],
        load_block_counts: dict[str, int],
        decision_times: dict[str, float] | None = None,
        recompute_flops: dict[str, float] | None = None,
    ) -> DynamicCPUOffloadMetadata:
        """复制原生元数据，并保持原生字段不变。"""
        return cls(
            load_event=base.load_event,
            load_gpu_blocks=base.load_gpu_blocks,
            load_cpu_blocks=base.load_cpu_blocks,
            load_event_to_reqs=base.load_event_to_reqs,
            store_event=base.store_event,
            store_gpu_blocks=base.store_gpu_blocks,
            store_cpu_blocks=base.store_cpu_blocks,
            need_flush=base.need_flush,
            recompute_requests=recompute_requests,
            recompute_flops=recompute_flops or {},
            reset_recompute_requests=reset_recompute_requests,
            completed_recompute_requests=completed_recompute_requests,
            load_block_counts=load_block_counts,
            decision_times=decision_times or {},
        )


@dataclass(frozen=True)
class TimingSampleMetadata:
    """一次 worker 侧完成的服务测量。"""

    request_id: str
    size: int
    service_ms: float
    kv_bytes: int = 0
    queue_wait_ms: float = 0.0
    work_flops: float = 0.0


@dataclass
class DynamicCPUOffloadWorkerMetadata(SimpleCPUOffloadWorkerMetadata):
    """携带原生 store 完成信息和插件计时样本。"""

    copy_samples: list[TimingSampleMetadata] = field(default_factory=list)
    recompute_samples: list[TimingSampleMetadata] = field(default_factory=list)

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> DynamicCPUOffloadWorkerMetadata:
        """合并 worker 样本并保留原生 store 计数。"""
        if not isinstance(other, DynamicCPUOffloadWorkerMetadata):
            raise TypeError("Cannot aggregate different worker metadata types")
        merged_store_events = dict(self.completed_store_events)
        for event_idx, count in other.completed_store_events.items():
            merged_store_events[event_idx] = (
                merged_store_events.get(event_idx, 0) + count
            )
        return DynamicCPUOffloadWorkerMetadata(
            completed_store_events=merged_store_events,
            copy_samples=[*self.copy_samples, *other.copy_samples],
            recompute_samples=[*self.recompute_samples, *other.recompute_samples],
        )


def as_worker_metadata(
    base: SimpleCPUOffloadWorkerMetadata | None,
    copy_samples: list[TimingSampleMetadata],
    recompute_samples: list[TimingSampleMetadata],
) -> DynamicCPUOffloadWorkerMetadata | None:
    """仅在存在原生或插件样本时构造扩展元数据。"""
    if base is None and not copy_samples and not recompute_samples:
        return None
    return DynamicCPUOffloadWorkerMetadata(
        completed_store_events=base.completed_store_events if base else {},
        copy_samples=copy_samples,
        recompute_samples=recompute_samples,
    )


__all__ = [
    "DynamicCPUOffloadMetadata",
    "DynamicCPUOffloadWorkerMetadata",
    "TimingSampleMetadata",
]
