"""本模块的作用：记录每次物化决策的预测分项、实际分支和误差。
输入：决策输出、运行时观测和 worker 完成计时。
输出：内存记录及可追加的 NDJSON 审计记录。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from kv_materialization_plugin.cost_model import MaterializationCostEstimate
from kv_materialization_plugin.decision import (
    MaterializationDecision,
    MaterializationObservation,
)


@dataclass(slots=True)
class AuditRecord:
    """一条物化尝试的决策与完成数据。"""

    request_id: str
    hit_tokens: int
    hit_blocks: int
    decision: str
    reason: str
    predicted_load_ms: float | None
    predicted_recompute_ms: float | None
    fallback: bool
    load_estimate: MaterializationCostEstimate | None = None
    recompute_estimate: MaterializationCostEstimate | None = None
    estimate_source: str = "unavailable"
    confidence_guard: str | None = None
    kv_bytes: int = 0
    kv_bytes_source: str = "unavailable"
    prefill_flops: float | None = None
    device_prefix_tokens: int = 0
    batch_size: int = 1
    active_materialization_count: int = 0
    timing_scope: str = (
        "decision_to_worker_sample_return;"
        " phase_queue=admission_to_service_start"
    )
    queue_wait_isolated: bool = False
    load_queue_wait_ms: float | None = None
    load_queue_wait_source: str = "unavailable"
    load_service_ms: float | None = None
    load_extra_wait_ms: float | None = None
    load_observation_age_ms: float | None = None
    load_sample_count: int = 0
    recompute_service_ms: float | None = None
    recompute_queue_wait_ms: float | None = None
    recompute_queue_wait_source: str = "unavailable"
    recompute_extra_wait_ms: float | None = None
    recompute_observation_age_ms: float | None = None
    recompute_sample_count: int = 0
    run_id: str | None = None
    mode: str | None = None
    gpu_local_hit_tokens: int | None = None
    invalid_fields: tuple[str, ...] = ()
    actual_branch: str | None = None
    actual_cost_ms: float | None = None
    service_ms: float | None = None
    extra_wait_ms: float | None = None
    queue_wait_ms: float | None = None
    prediction_error_ms: float | None = None
    prediction_error_ratio: float | None = None
    status: str = "decided"


class AuditLog:
    """收集审计数据，不要求 vLLM 引入额外日志框架。"""

    def __init__(
        self,
        output_path: str | Path | None = None,
        run_id: str | None = None,
        mode: str | None = None,
    ) -> None:
        self._records: dict[str, AuditRecord] = {}
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._output = destination.open("a", encoding="utf-8", buffering=1)
        else:
            self._output = None
        self._run_id = run_id
        self._mode = mode

    def start(
        self,
        request_id: str,
        hit_tokens: int,
        hit_blocks: int,
        decision: MaterializationDecision,
        gpu_local_hit_tokens: int | None = None,
        observation: MaterializationObservation | None = None,
    ) -> None:
        """开始或替换一条请求记录。"""
        self._records[request_id] = AuditRecord(
            request_id=request_id,
            hit_tokens=hit_tokens,
            hit_blocks=hit_blocks,
            decision=decision.mode,
            reason=decision.reason,
            predicted_load_ms=decision.predicted_load_ms,
            predicted_recompute_ms=decision.predicted_recompute_ms,
            fallback=decision.fallback,
            load_estimate=decision.load_estimate,
            recompute_estimate=decision.recompute_estimate,
            estimate_source=decision.estimate_source,
            confidence_guard=decision.confidence_guard,
            kv_bytes=observation.kv_bytes if observation else 0,
            kv_bytes_source=(
                observation.kv_bytes_source if observation else "unavailable"
            ),
            prefill_flops=observation.prefill_flops if observation else None,
            device_prefix_tokens=(
                observation.device_prefix_tokens if observation else 0
            ),
            batch_size=observation.batch_size if observation else 1,
            active_materialization_count=(
                observation.active_materialization_count if observation else 0
            ),
            queue_wait_isolated=bool(
                observation
                and observation.load_queue_wait_ms is not None
                and observation.recompute_queue_wait_ms is not None
            ),
            load_service_ms=observation.load_service_ms if observation else None,
            load_queue_wait_ms=(
                observation.load_queue_wait_ms if observation else None
            ),
            load_queue_wait_source=(
                observation.load_queue_wait_source if observation else "unavailable"
            ),
            load_extra_wait_ms=(
                observation.load_extra_wait_ms if observation else None
            ),
            load_observation_age_ms=(
                observation.load_observation_age_ms if observation else None
            ),
            load_sample_count=observation.load_sample_count if observation else 0,
            recompute_service_ms=(
                observation.recompute_service_ms if observation else None
            ),
            recompute_queue_wait_ms=(
                observation.recompute_queue_wait_ms if observation else None
            ),
            recompute_queue_wait_source=(
                observation.recompute_queue_wait_source
                if observation
                else "unavailable"
            ),
            recompute_extra_wait_ms=(
                observation.recompute_extra_wait_ms if observation else None
            ),
            recompute_observation_age_ms=(
                observation.recompute_observation_age_ms if observation else None
            ),
            recompute_sample_count=(
                observation.recompute_sample_count if observation else 0
            ),
            run_id=self._run_id,
            mode=self._mode,
            gpu_local_hit_tokens=gpu_local_hit_tokens,
            invalid_fields=decision.invalid_fields,
        )

    def complete(
        self,
        request_id: str,
        actual_branch: str,
        actual_cost_ms: float,
        service_ms: float | None = None,
        extra_wait_ms: float | None = None,
        queue_wait_ms: float | None = None,
        status: str = "completed",
    ) -> None:
        """完成一条记录并计算已执行分支的预测误差。"""
        record = self._records.get(request_id)
        if record is None:
            return
        record.actual_branch = actual_branch
        record.actual_cost_ms = actual_cost_ms
        record.service_ms = service_ms
        record.extra_wait_ms = extra_wait_ms
        record.queue_wait_ms = queue_wait_ms
        prediction = (
            record.predicted_load_ms
            if actual_branch == "cpu_kv_load"
            else record.predicted_recompute_ms
        )
        if (
            prediction is not None
            and math.isfinite(float(prediction))
            and prediction > 0.0
            and math.isfinite(float(actual_cost_ms))
        ):
            record.prediction_error_ms = float(actual_cost_ms) - float(prediction)
            record.prediction_error_ratio = (
                record.prediction_error_ms / float(prediction)
            )
        record.status = status
        self._write(record)

    def close(self) -> None:
        """刷新并关闭可选的 NDJSON 输出。"""
        if self._output is not None:
            self._output.close()
            self._output = None

    def records(self) -> list[AuditRecord]:
        """按插入顺序返回记录。"""
        return list(self._records.values())

    def json_lines(self) -> str:
        """序列化为换行分隔 JSON。"""
        return "\n".join(
            json.dumps(asdict(record), sort_keys=True) for record in self.records()
        )

    def _write(self, record: AuditRecord) -> None:
        """追加写出一条完成记录。"""
        if self._output is None:
            return
        self._output.write(json.dumps(asdict(record), sort_keys=True) + "\n")


__all__ = ["AuditLog", "AuditRecord"]
