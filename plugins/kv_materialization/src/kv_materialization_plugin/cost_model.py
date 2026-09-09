"""本模块的作用：定义 KV 加载与完整前向重算的成本模型及其输入。
输入：实际 KV 字节数、完整前向 FLOPs、运行时可见的等待时间和冻结校准参数。
输出：两条路径的可审计成本分项，或适用性校验错误。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _finite_nonnegative(value: object) -> bool:
    """判断值是否为有限的非负数。"""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _positive(value: object) -> bool:
    """判断值是否为有限的正数。"""
    return _finite_nonnegative(value) and float(value) > 0.0


@dataclass(frozen=True, slots=True)
class MaterializationCostEstimate:
    """一条路径的成本分项及其工作量来源。"""

    total_ms: float
    queue_wait_ms: float
    fixed_ms: float
    service_ms: float
    work_amount: float
    work_unit: str
    effective_rate_per_s: float
    queue_source: str


@dataclass(frozen=True, slots=True)
class MaterializationModelShape:
    """用于展开完整 Transformer 前向 FLOPs 的模型结构。"""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    gated_mlp: bool = True
    vocab_size: int | None = None

    def __post_init__(self) -> None:
        """校验模型结构，避免用不完整形状生成代价。"""
        positive_fields = (
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_kv_heads",
            "head_dim",
            "intermediate_size",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.num_kv_heads > self.num_attention_heads:
            raise ValueError("num_kv_heads cannot exceed num_attention_heads")
        if self.vocab_size is not None and (
            not isinstance(self.vocab_size, int)
            or isinstance(self.vocab_size, bool)
            or self.vocab_size <= 0
        ):
            raise ValueError("vocab_size must be a positive integer when provided")


def estimate_prefill_flops(
    shape: MaterializationModelShape,
    recompute_tokens: int,
    device_prefix_tokens: int = 0,
    batch_size: int = 1,
) -> float:
    """估算重算 token 段的完整 Transformer 前向 FLOPs。

    计入 Q/K/V 与输出投影、Attention 的 QK/AV、FFN 两或三次线性投影，
    以及每个序列最后一个位置的 logits 投影。Attention 的 key 长度包含
    决策时已经在设备上的前缀和当前重算段内的因果位置。
    """
    for name, value in (
        ("recompute_tokens", recompute_tokens),
        ("device_prefix_tokens", device_prefix_tokens),
        ("batch_size", batch_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if recompute_tokens <= 0 or batch_size <= 0:
        raise ValueError("recompute_tokens and batch_size must be positive")

    query_tokens = float(batch_size * recompute_tokens)
    kv_width = float(shape.num_kv_heads * shape.head_dim)
    hidden = float(shape.hidden_size)
    intermediate = float(shape.intermediate_size)
    query_heads = float(shape.num_attention_heads)
    attention_positions = float(
        recompute_tokens * device_prefix_tokens
        + recompute_tokens * (recompute_tokens + 1) / 2
    )

    qkv_flops = 2.0 * query_tokens * hidden * (hidden + 2.0 * kv_width)
    output_projection_flops = 2.0 * query_tokens * hidden * hidden
    ffn_projection_count = 3.0 if shape.gated_mlp else 2.0
    ffn_flops = (
        2.0
        * query_tokens
        * hidden
        * intermediate
        * ffn_projection_count
    )
    attention_flops = (
        4.0
        * float(batch_size)
        * query_heads
        * float(shape.head_dim)
        * attention_positions
    )
    block_flops = (
        qkv_flops + output_projection_flops + ffn_flops + attention_flops
    ) * float(shape.num_layers)
    logits_flops = 0.0
    if shape.vocab_size is not None:
        logits_flops = 2.0 * float(batch_size) * hidden * shape.vocab_size
    return block_flops + logits_flops


@dataclass(frozen=True, slots=True)
class MaterializationCostModel:
    """冻结的服务时间模型；队列等待始终由运行时观测单独提供。"""

    load_fixed_ms: float
    load_bandwidth_bytes_per_s: float
    recompute_fixed_ms: float
    recompute_flops_per_s: float
    calibration_id: str = ""
    load_min_bytes: int | None = None
    load_max_bytes: int | None = None
    recompute_min_flops: float | None = None
    recompute_max_flops: float | None = None
    load_fit_r2: float | None = None
    recompute_fit_r2: float | None = None
    load_sample_count: int = 0
    recompute_sample_count: int = 0

    def __post_init__(self) -> None:
        """校验冻结参数，禁止把无效校准静默带入在线决策。"""
        for name in ("load_fixed_ms", "recompute_fixed_ms"):
            if not _finite_nonnegative(getattr(self, name)):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("load_bandwidth_bytes_per_s", "recompute_flops_per_s"):
            if not _positive(getattr(self, name)):
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "load_min_bytes",
            "load_max_bytes",
            "load_sample_count",
            "recompute_sample_count",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("recompute_min_flops", "recompute_max_flops"):
            value = getattr(self, name)
            if value is not None and not _positive(value):
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.load_min_bytes is not None
            and self.load_max_bytes is not None
            and self.load_min_bytes > self.load_max_bytes
        ):
            raise ValueError("load byte range is inverted")
        if (
            self.recompute_min_flops is not None
            and self.recompute_max_flops is not None
            and self.recompute_min_flops > self.recompute_max_flops
        ):
            raise ValueError("recompute FLOPs range is inverted")
        for name in ("load_fit_r2", "recompute_fit_r2"):
            value = getattr(self, name)
            if value is not None and (
                not _finite_nonnegative(value) or float(value) > 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1]")

    @staticmethod
    def _check_work_range(
        value: float,
        minimum: float | int | None,
        maximum: float | int | None,
        field_name: str,
    ) -> None:
        """检查在线工作量是否落在冻结校准范围内。"""
        if not _positive(value):
            raise ValueError(f"{field_name} must be positive")
        if minimum is not None and value < float(minimum):
            raise ValueError(f"{field_name} is below calibration range")
        if maximum is not None and value > float(maximum):
            raise ValueError(f"{field_name} is above calibration range")

    def predict_load(
        self,
        kv_bytes: int,
        queue_wait_ms: float,
        queue_source: str,
    ) -> MaterializationCostEstimate:
        """按实际 KV 字节数预测加载总成本。"""
        if not isinstance(kv_bytes, int) or isinstance(kv_bytes, bool):
            raise ValueError("kv_bytes must be an integer")
        self._check_work_range(
            float(kv_bytes),
            self.load_min_bytes,
            self.load_max_bytes,
            "kv_bytes",
        )
        if (
            not isinstance(queue_source, str)
            or not queue_source
            or queue_source == "unavailable"
        ):
            raise ValueError("load queue wait source is unavailable")
        if not _finite_nonnegative(queue_wait_ms):
            raise ValueError("load queue wait is unavailable")
        service_ms = (
            self.load_fixed_ms
            + float(kv_bytes) / self.load_bandwidth_bytes_per_s * 1000.0
        )
        return MaterializationCostEstimate(
            total_ms=float(queue_wait_ms) + service_ms,
            queue_wait_ms=float(queue_wait_ms),
            fixed_ms=self.load_fixed_ms,
            service_ms=service_ms,
            work_amount=float(kv_bytes),
            work_unit="bytes",
            effective_rate_per_s=self.load_bandwidth_bytes_per_s,
            queue_source=queue_source,
        )

    def predict_recompute(
        self,
        prefill_flops: float,
        queue_wait_ms: float,
        queue_source: str,
    ) -> MaterializationCostEstimate:
        """按完整前向 FLOPs 预测重算总成本。"""
        self._check_work_range(
            float(prefill_flops),
            self.recompute_min_flops,
            self.recompute_max_flops,
            "prefill_flops",
        )
        if (
            not isinstance(queue_source, str)
            or not queue_source
            or queue_source == "unavailable"
        ):
            raise ValueError("recompute queue wait source is unavailable")
        if not _finite_nonnegative(queue_wait_ms):
            raise ValueError("recompute queue wait is unavailable")
        service_ms = (
            self.recompute_fixed_ms
            + float(prefill_flops) / self.recompute_flops_per_s * 1000.0
        )
        return MaterializationCostEstimate(
            total_ms=float(queue_wait_ms) + service_ms,
            queue_wait_ms=float(queue_wait_ms),
            fixed_ms=self.recompute_fixed_ms,
            service_ms=service_ms,
            work_amount=float(prefill_flops),
            work_unit="flops",
            effective_rate_per_s=self.recompute_flops_per_s,
            queue_source=queue_source,
        )

    def to_dict(self) -> dict[str, Any]:
        """导出稳定的 JSON 参数结构。"""
        return {
            "schema_version": 1,
            "calibration_id": self.calibration_id,
            "load": {
                "fixed_ms": self.load_fixed_ms,
                "bandwidth_bytes_per_s": self.load_bandwidth_bytes_per_s,
                "min_bytes": self.load_min_bytes,
                "max_bytes": self.load_max_bytes,
                "fit_r2": self.load_fit_r2,
                "sample_count": self.load_sample_count,
            },
            "recompute": {
                "fixed_ms": self.recompute_fixed_ms,
                "prefill_flops_per_s": self.recompute_flops_per_s,
                "min_flops": self.recompute_min_flops,
                "max_flops": self.recompute_max_flops,
                "fit_r2": self.recompute_fit_r2,
                "sample_count": self.recompute_sample_count,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MaterializationCostModel:
        """从 JSON 对象读取冻结参数。"""
        if value.get("schema_version", 1) != 1:
            raise ValueError("Unsupported cost model schema_version")
        load = value.get("load")
        recompute = value.get("recompute")
        if not isinstance(load, dict) or not isinstance(recompute, dict):
            raise ValueError("Cost model must contain load and recompute objects")
        return cls(
            load_fixed_ms=float(load["fixed_ms"]),
            load_bandwidth_bytes_per_s=float(load["bandwidth_bytes_per_s"]),
            recompute_fixed_ms=float(recompute["fixed_ms"]),
            recompute_flops_per_s=float(recompute["prefill_flops_per_s"]),
            calibration_id=str(value.get("calibration_id", "")),
            load_min_bytes=(
                int(load["min_bytes"]) if load.get("min_bytes") is not None else None
            ),
            load_max_bytes=(
                int(load["max_bytes"]) if load.get("max_bytes") is not None else None
            ),
            recompute_min_flops=(
                float(recompute["min_flops"])
                if recompute.get("min_flops") is not None
                else None
            ),
            recompute_max_flops=(
                float(recompute["max_flops"])
                if recompute.get("max_flops") is not None
                else None
            ),
            load_fit_r2=(
                float(load["fit_r2"]) if load.get("fit_r2") is not None else None
            ),
            recompute_fit_r2=(
                float(recompute["fit_r2"])
                if recompute.get("fit_r2") is not None
                else None
            ),
            load_sample_count=int(load.get("sample_count", 0)),
            recompute_sample_count=int(recompute.get("sample_count", 0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> MaterializationCostModel:
        """从 JSON 文件读取冻结参数。"""
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Cost model must be a JSON object: {source}")
        return cls.from_dict(value)

    def save_json(self, path: str | Path) -> None:
        """保存冻结参数 JSON 文件。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "MaterializationCostEstimate",
    "MaterializationCostModel",
    "MaterializationModelShape",
    "estimate_prefill_flops",
]
