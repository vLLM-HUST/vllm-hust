"""本模块的作用：用多尺寸、多次真实服务样本拟合两条路径的线性服务成本。
输入：TelemetryWindow 导出的 JSON 样本，或样本字典序列。
输出：带拟合范围和 R² 诊断的冻结 MaterializationCostModel。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from kv_materialization_plugin.cost_model import MaterializationCostModel


@dataclass(frozen=True, slots=True)
class LinearFit:
    """一个服务时间对工作量的线性拟合结果。"""

    fixed_ms: float
    rate_per_s: float
    min_work: float
    max_work: float
    r2: float
    sample_count: int
    distinct_work_count: int
    max_abs_residual_ms: float


def _field(sample: object, name: str) -> object:
    """从 dataclass 或 JSON 字典读取字段。"""
    if isinstance(sample, Mapping):
        return sample.get(name)
    return getattr(sample, name, None)


def fit_linear_service(
    samples: Iterable[object],
    work_field: str,
    min_samples: int = 3,
) -> LinearFit:
    """联合拟合 service_ms = fixed_ms + work / effective_rate。"""
    points: list[tuple[float, float]] = []
    for sample in samples:
        work = _field(sample, work_field)
        service_ms = _field(sample, "service_ms")
        if (
            isinstance(work, int | float)
            and not isinstance(work, bool)
            and isinstance(service_ms, int | float)
            and not isinstance(service_ms, bool)
            and math.isfinite(float(work))
            and float(work) > 0.0
            and math.isfinite(float(service_ms))
            and float(service_ms) >= 0.0
        ):
            points.append((float(work), float(service_ms)))
    distinct = {work for work, _ in points}
    if len(points) < min_samples or len(distinct) < 2:
        raise ValueError(
            f"{work_field} requires {min_samples} samples and two work sizes"
        )

    mean_x = sum(work for work, _ in points) / len(points)
    mean_y = sum(service for _, service in points) / len(points)
    denominator = sum((work - mean_x) ** 2 for work, _ in points)
    if denominator <= 0.0:
        raise ValueError(f"{work_field} has no work-size variation")
    slope = sum(
        (work - mean_x) * (service - mean_y) for work, service in points
    ) / denominator
    fixed_ms = mean_y - slope * mean_x
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError(f"{work_field} has non-positive service slope")
    if not math.isfinite(fixed_ms) or fixed_ms < 0.0:
        raise ValueError(
            f"{work_field} has a negative fixed cost; linear model is not stable"
        )

    residuals = [
        service - (fixed_ms + slope * work) for work, service in points
    ]
    squared_error = sum(residual * residual for residual in residuals)
    total_variation = sum((service - mean_y) ** 2 for _, service in points)
    r2 = 1.0 if total_variation == 0.0 else 1.0 - squared_error / total_variation
    r2 = min(1.0, max(0.0, r2))
    return LinearFit(
        fixed_ms=fixed_ms,
        rate_per_s=1000.0 / slope,
        min_work=min(work for work, _ in points),
        max_work=max(work for work, _ in points),
        r2=r2,
        sample_count=len(points),
        distinct_work_count=len(distinct),
        max_abs_residual_ms=max(abs(residual) for residual in residuals),
    )


def load_telemetry_files(paths: Iterable[str | Path]) -> dict[str, list[dict[str, Any]]]:
    """读取并合并多个强制模式导出的 telemetry.json。"""
    merged: dict[str, list[dict[str, Any]]] = {"load": [], "recompute": []}
    for path in paths:
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Telemetry file must be an object: {source}")
        for branch in merged:
            samples = value.get(branch, [])
            if not isinstance(samples, list):
                raise ValueError(f"Telemetry branch must be a list: {branch}")
            if not all(isinstance(sample, dict) for sample in samples):
                raise ValueError(f"Telemetry samples must be objects: {branch}")
            merged[branch].extend(samples)
    return merged


def fit_cost_model(
    load_samples: Iterable[object],
    recompute_samples: Iterable[object],
    calibration_id: str,
    min_samples: int = 3,
) -> tuple[MaterializationCostModel, dict[str, Any]]:
    """从两条路径的服务样本拟合公式并返回诊断信息。"""
    load_fit = fit_linear_service(load_samples, "kv_bytes", min_samples)
    recompute_fit = fit_linear_service(
        recompute_samples, "work_flops", min_samples
    )
    model = MaterializationCostModel(
        load_fixed_ms=load_fit.fixed_ms,
        load_bandwidth_bytes_per_s=load_fit.rate_per_s,
        recompute_fixed_ms=recompute_fit.fixed_ms,
        recompute_flops_per_s=recompute_fit.rate_per_s,
        calibration_id=calibration_id,
        load_min_bytes=int(load_fit.min_work),
        load_max_bytes=int(load_fit.max_work),
        recompute_min_flops=recompute_fit.min_work,
        recompute_max_flops=recompute_fit.max_work,
        load_fit_r2=load_fit.r2,
        recompute_fit_r2=recompute_fit.r2,
        load_sample_count=load_fit.sample_count,
        recompute_sample_count=recompute_fit.sample_count,
    )
    diagnostics = {
        "load": {
            "fixed_ms": load_fit.fixed_ms,
            "bandwidth_bytes_per_s": load_fit.rate_per_s,
            "min_bytes": load_fit.min_work,
            "max_bytes": load_fit.max_work,
            "fit_r2": load_fit.r2,
            "sample_count": load_fit.sample_count,
            "distinct_work_count": load_fit.distinct_work_count,
            "max_abs_residual_ms": load_fit.max_abs_residual_ms,
        },
        "recompute": {
            "fixed_ms": recompute_fit.fixed_ms,
            "prefill_flops_per_s": recompute_fit.rate_per_s,
            "min_flops": recompute_fit.min_work,
            "max_flops": recompute_fit.max_work,
            "fit_r2": recompute_fit.r2,
            "sample_count": recompute_fit.sample_count,
            "distinct_work_count": recompute_fit.distinct_work_count,
            "max_abs_residual_ms": recompute_fit.max_abs_residual_ms,
        },
    }
    return model, diagnostics


__all__ = [
    "LinearFit",
    "fit_cost_model",
    "fit_linear_service",
    "load_telemetry_files",
]
