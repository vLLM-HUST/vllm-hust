"""本模块的作用：保存近期加载/重算测量，并生成决策时的等待观测。
输入：worker 返回的服务时间、队列等待、实际 KV 字节数和完整前向 FLOPs。
输出：带新鲜度、样本数和来源标记的决策观测，以及可复用的 JSON 状态。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from kv_materialization_plugin.decision import MaterializationObservation


@dataclass(frozen=True, slots=True)
class TimingSample:
    """一次完成的物化测量；服务时间不含单独记录的队列等待。"""

    size: int
    total_ms: float
    service_ms: float
    extra_wait_ms: float
    timestamp: float
    kv_bytes: int = 0
    queue_wait_ms: float | None = None
    work_flops: float = 0.0


class TelemetryWindow:
    """按规模保存近期测量，且不把等待观测硬编码为零。"""

    def __init__(self, max_samples: int = 32) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._max_samples = max_samples
        self._load: dict[int, deque[TimingSample]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._recompute: dict[int, deque[TimingSample]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

    def observe_load(
        self,
        blocks: int,
        total_ms: float,
        service_ms: float,
        kv_bytes: int = 0,
        queue_wait_ms: float | None = None,
        timestamp: float | None = None,
    ) -> None:
        """记录一次 CPU 到设备的真实加载。"""
        self._observe(
            self._load,
            blocks,
            total_ms,
            service_ms,
            kv_bytes,
            queue_wait_ms,
            0.0,
            timestamp,
        )

    def observe_recompute(
        self,
        tokens: int,
        total_ms: float,
        service_ms: float,
        queue_wait_ms: float | None = None,
        timestamp: float | None = None,
        work_flops: float = 0.0,
    ) -> None:
        """记录一次完整前向重算。"""
        self._observe(
            self._recompute,
            tokens,
            total_ms,
            service_ms,
            0,
            queue_wait_ms,
            work_flops,
            timestamp,
        )

    def snapshot(
        self,
        hit_tokens: int,
        hit_blocks: int,
        kv_bytes: int = 0,
        max_age_ms: float | None = None,
        prefill_flops: float | None = None,
        device_prefix_tokens: int = 0,
        batch_size: int = 1,
        formula_mode: bool = False,
    ) -> MaterializationObservation:
        """生成一次决策观测。

        历史中位数路径只使用同一 token/block 桶的总时延。公式路径只
        借用近期队列等待中位数；当当前尺寸没有样本时，明确标记为
        recent_completed_median，而不是假装设备队列等待为零。
        """
        now = time.time()
        if max_age_ms is not None and (
            not math.isfinite(max_age_ms) or max_age_ms < 0.0
        ):
            raise ValueError("max_age_ms must be finite and non-negative")
        load = self._load.get(hit_blocks)
        recompute = self._recompute.get(hit_tokens)
        load_stats = self._stats(load, now, max_age_ms)
        recompute_stats = self._stats(recompute, now, max_age_ms)

        load_queue_stats = load_stats
        recompute_queue_stats = recompute_stats
        load_queue_source = (
            "recent_completed_same_size" if load_stats[2] is not None else "unavailable"
        )
        recompute_queue_source = (
            "recent_completed_same_size"
            if recompute_stats[2] is not None
            else "unavailable"
        )
        if formula_mode:
            if load_stats[2] is None:
                load_queue_stats = self._stats(
                    self._all_samples(self._load), now, max_age_ms
                )
                load_queue_source = (
                    "recent_completed_median"
                    if load_queue_stats[2] is not None
                    else "unavailable"
                )
            if recompute_stats[2] is None:
                recompute_queue_stats = self._stats(
                    self._all_samples(self._recompute), now, max_age_ms
                )
                recompute_queue_source = (
                    "recent_completed_median"
                    if recompute_queue_stats[2] is not None
                    else "unavailable"
                )

        effective_kv_bytes = int(kv_bytes)
        kv_bytes_source = "runtime_block_bytes" if effective_kv_bytes > 0 else "unavailable"
        if formula_mode and effective_kv_bytes <= 0:
            inferred = self._median_kv_bytes(
                self._all_samples(self._load), now, max_age_ms
            )
            if inferred is not None and inferred > 0:
                effective_kv_bytes = int(round(inferred))
                kv_bytes_source = "recent_completed_load_sample_median"

        return MaterializationObservation(
            hit_tokens=hit_tokens,
            hit_blocks=hit_blocks,
            kv_bytes=effective_kv_bytes,
            kv_bytes_source=kv_bytes_source,
            device_prefix_tokens=device_prefix_tokens,
            batch_size=batch_size,
            prefill_flops=prefill_flops,
            load_total_ms=load_stats[0],
            load_service_ms=load_stats[1],
            load_queue_wait_ms=load_queue_stats[2],
            load_queue_wait_source=load_queue_source,
            load_extra_wait_ms=load_stats[3],
            load_observation_age_ms=load_queue_stats[4],
            load_sample_count=load_queue_stats[5],
            recompute_total_ms=recompute_stats[0],
            recompute_service_ms=recompute_stats[1],
            recompute_queue_wait_ms=recompute_queue_stats[2],
            recompute_queue_wait_source=recompute_queue_source,
            recompute_extra_wait_ms=recompute_stats[3],
            recompute_observation_age_ms=recompute_queue_stats[4],
            recompute_sample_count=recompute_queue_stats[5],
        )

    def state(self) -> dict[str, list[dict[str, Any]]]:
        """返回 JSON 兼容的校准状态。"""
        return {
            "load": [
                asdict(sample)
                for samples in self._load.values()
                for sample in samples
            ],
            "recompute": [
                asdict(sample)
                for samples in self._recompute.values()
                for sample in samples
            ],
        }

    def save_json(self, path: str | Path) -> None:
        """原子保存校准状态。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            json.dump(self.state(), temporary, sort_keys=True)
            temporary.write("\n")
            temporary_path = temporary.name
        os.replace(temporary_path, destination)

    def load_json(self, path: str | Path) -> None:
        """合并读取校准状态；兼容缺少新增字段的旧样本。"""
        source = Path(path)
        if not source.is_file():
            return
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Telemetry state must be an object: {source}")
        for branch, buckets in (("load", self._load), ("recompute", self._recompute)):
            samples = value.get(branch, [])
            if not isinstance(samples, list):
                raise ValueError(f"Telemetry state field must be a list: {branch}")
            for raw in samples:
                if not isinstance(raw, dict):
                    raise ValueError(f"Telemetry sample must be an object: {branch}")
                sample = TimingSample(**raw)
                self._validate_sample(sample)
                buckets[sample.size].append(sample)

    def _observe(
        self,
        buckets: dict[int, deque[TimingSample]],
        size: int,
        total_ms: float,
        service_ms: float,
        kv_bytes: int,
        queue_wait_ms: float | None,
        work_flops: float,
        timestamp: float | None,
    ) -> None:
        if timestamp is None:
            timestamp = time.time()
        extra_wait_ms = total_ms - service_ms
        sample = TimingSample(
            size=size,
            total_ms=total_ms,
            service_ms=service_ms,
            extra_wait_ms=extra_wait_ms,
            timestamp=timestamp,
            kv_bytes=kv_bytes,
            queue_wait_ms=queue_wait_ms,
            work_flops=work_flops,
        )
        self._validate_sample(sample)
        buckets[size].append(sample)

    @staticmethod
    def _all_samples(
        buckets: dict[int, deque[TimingSample]],
    ) -> deque[TimingSample]:
        """合并多个尺寸的样本，仅用于公式路径的近期等待近似。"""
        samples: deque[TimingSample] = deque()
        for bucket in buckets.values():
            samples.extend(bucket)
        return samples

    @staticmethod
    def _eligible(
        samples: deque[TimingSample] | None,
        now: float,
        max_age_ms: float | None,
    ) -> list[TimingSample]:
        """筛选尚未过期的样本。"""
        if not samples:
            return []
        return [
            sample
            for sample in samples
            if max_age_ms is None
            or max(0.0, (now - sample.timestamp) * 1000.0) <= max_age_ms
        ]

    @classmethod
    def _median_kv_bytes(
        cls,
        samples: deque[TimingSample],
        now: float,
        max_age_ms: float | None,
    ) -> float | None:
        """返回近期样本中的 KV 字节中位数。"""
        eligible = [
            sample.kv_bytes
            for sample in cls._eligible(samples, now, max_age_ms)
            if sample.kv_bytes > 0
        ]
        return float(median(eligible)) if eligible else None

    @classmethod
    def _stats(
        cls,
        samples: deque[TimingSample] | None,
        now: float,
        max_age_ms: float | None,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        int,
    ]:
        eligible = cls._eligible(samples, now, max_age_ms)
        if not eligible:
            return None, None, None, None, None, 0
        newest = max(sample.timestamp for sample in eligible)
        return (
            float(median(sample.total_ms for sample in eligible)),
            float(median(sample.service_ms for sample in eligible)),
            (
                float(median(sample.queue_wait_ms for sample in eligible))
                if all(sample.queue_wait_ms is not None for sample in eligible)
                else None
            ),
            float(median(sample.extra_wait_ms for sample in eligible)),
            max(0.0, (now - newest) * 1000.0),
            len(eligible),
        )

    @staticmethod
    def _validate_sample(sample: TimingSample) -> None:
        numeric_values = (
            sample.total_ms,
            sample.service_ms,
            sample.extra_wait_ms,
            sample.timestamp,
            sample.work_flops,
        )
        if (
            not isinstance(sample.size, int)
            or isinstance(sample.size, bool)
            or sample.size <= 0
            or not isinstance(sample.kv_bytes, int)
            or isinstance(sample.kv_bytes, bool)
            or sample.kv_bytes < 0
            or any(
                not isinstance(value, int | float) or isinstance(value, bool)
                for value in numeric_values
            )
            or not all(
                math.isfinite(float(value)) and float(value) >= 0.0
                for value in numeric_values
            )
            or (
                sample.queue_wait_ms is not None
                and (
                    not isinstance(sample.queue_wait_ms, int | float)
                    or isinstance(sample.queue_wait_ms, bool)
                    or not math.isfinite(float(sample.queue_wait_ms))
                    or sample.queue_wait_ms < 0.0
                )
            )
        ):
            raise ValueError(
                "Telemetry measurements must be finite, non-negative numbers"
            )


__all__ = ["TelemetryWindow", "TimingSample"]
