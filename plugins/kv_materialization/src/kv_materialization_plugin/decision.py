"""本模块的作用：在 CPU KV 加载和完整前向重算之间做纯数据决策。
输入：缓存命中、实际 KV 字节数、完整前向 FLOPs、近期等待观测和冻结模型。
输出：可审计的选择、预测总时延、成本分项或明确回退原因。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from kv_materialization_plugin.cost_model import (
    MaterializationCostEstimate,
    MaterializationCostModel,
)

MaterializationMode = Literal["load", "recompute"]
MaterializationPredictor = Literal["historical", "formula"]


def _finite_nonnegative(value: object) -> bool:
    """判断值是否为有限的非负数。"""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _finite_positive(value: object) -> bool:
    """判断值是否为有限的正数。"""
    return _finite_nonnegative(value) and float(value) > 0.0


@dataclass(frozen=True, slots=True)
class MaterializationDecisionConfig:
    """二元物化决策的运行配置。"""

    enabled: bool = False
    forced_mode: MaterializationMode | None = None
    fallback_mode: MaterializationMode = "load"
    predictor: MaterializationPredictor = "historical"
    cost_model: MaterializationCostModel | None = None
    min_copy_samples: int = 1
    min_recompute_samples: int = 1
    max_observation_age_ms: float = 5000.0

    def __post_init__(self) -> None:
        """在运行前校验配置。"""
        if self.forced_mode not in (None, "load", "recompute"):
            raise ValueError(f"Invalid forced mode: {self.forced_mode!r}")
        if self.fallback_mode not in ("load", "recompute"):
            raise ValueError(f"Invalid fallback mode: {self.fallback_mode!r}")
        if self.predictor not in ("historical", "formula"):
            raise ValueError(f"Invalid predictor: {self.predictor!r}")
        if self.min_copy_samples < 0 or self.min_recompute_samples < 0:
            raise ValueError("Minimum sample counts must be non-negative")
        if not _finite_nonnegative(self.max_observation_age_ms):
            raise ValueError("max_observation_age_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MaterializationObservation:
    """运行时提供给决策器的测量、工作量和等待估计。"""

    hit_tokens: int
    hit_blocks: int
    kv_bytes: int = 0
    kv_bytes_source: str = "unavailable"
    device_prefix_tokens: int = 0
    batch_size: int = 1
    prefill_flops: float | None = None
    active_materialization_count: int = 0
    load_total_ms: float | None = None
    load_service_ms: float | None = None
    load_queue_wait_ms: float | None = None
    load_queue_wait_source: str = "unavailable"
    load_extra_wait_ms: float | None = None
    load_observation_age_ms: float | None = None
    load_sample_count: int = 0
    recompute_total_ms: float | None = None
    recompute_service_ms: float | None = None
    recompute_queue_wait_ms: float | None = None
    recompute_queue_wait_source: str = "unavailable"
    recompute_extra_wait_ms: float | None = None
    recompute_observation_age_ms: float | None = None
    recompute_sample_count: int = 0


@dataclass(frozen=True, slots=True)
class MaterializationDecision:
    """可审计的物化选择及两条路径的预测。"""

    mode: MaterializationMode
    reason: str
    predicted_load_ms: float | None = None
    predicted_recompute_ms: float | None = None
    load_estimate: MaterializationCostEstimate | None = None
    recompute_estimate: MaterializationCostEstimate | None = None
    fallback: bool = False
    invalid_fields: tuple[str, ...] = ()
    estimate_source: str = "unavailable"
    confidence_guard: str | None = None


def _fallback(
    config: MaterializationDecisionConfig,
    reason: str,
    invalid_fields: tuple[str, ...] = (),
) -> MaterializationDecision:
    """构造明确的安全回退。"""
    return MaterializationDecision(
        mode=config.fallback_mode,
        reason=reason,
        fallback=True,
        invalid_fields=invalid_fields,
    )


def _validate_observation(
    observation: MaterializationObservation,
    config: MaterializationDecisionConfig,
) -> tuple[str, ...]:
    """返回不能支持当前预测的观测字段。"""
    invalid: list[str] = []
    if not isinstance(observation.hit_tokens, int) or observation.hit_tokens <= 0:
        invalid.append("hit_tokens")
    if not isinstance(observation.hit_blocks, int) or observation.hit_blocks <= 0:
        invalid.append("hit_blocks")
    if not isinstance(observation.kv_bytes, int) or observation.kv_bytes < 0:
        invalid.append("kv_bytes")
    if (
        not isinstance(observation.active_materialization_count, int)
        or isinstance(observation.active_materialization_count, bool)
        or observation.active_materialization_count < 0
    ):
        invalid.append("active_materialization_count")
    for field_name in ("device_prefix_tokens", "batch_size"):
        value = getattr(observation, field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            invalid.append(field_name)
    if observation.batch_size <= 0:
        invalid.append("batch_size")
    if config.predictor == "historical":
        if not _finite_nonnegative(observation.load_total_ms):
            invalid.append("load_total_ms")
        if not _finite_nonnegative(observation.recompute_total_ms):
            invalid.append("recompute_total_ms")
    else:
        if observation.kv_bytes <= 0:
            invalid.append("kv_bytes")
        if observation.kv_bytes_source == "unavailable":
            invalid.append("kv_bytes_source")
        if not _finite_positive(observation.prefill_flops):
            invalid.append("prefill_flops")
        if config.cost_model is None:
            invalid.append("cost_model")
    if observation.load_sample_count < config.min_copy_samples:
        invalid.append("load_sample_count")
    if observation.recompute_sample_count < config.min_recompute_samples:
        invalid.append("recompute_sample_count")

    for field_name in (
        "load_queue_wait_ms",
        "recompute_queue_wait_ms",
        "load_observation_age_ms",
        "recompute_observation_age_ms",
    ):
        if not _finite_nonnegative(getattr(observation, field_name)):
            invalid.append(field_name)
    if config.predictor == "formula":
        for field_name in (
            "load_queue_wait_source",
            "recompute_queue_wait_source",
        ):
            source = getattr(observation, field_name)
            if not isinstance(source, str) or not source or source == "unavailable":
                invalid.append(field_name)
    for field_name in (
        "load_observation_age_ms",
        "recompute_observation_age_ms",
    ):
        age = getattr(observation, field_name)
        if _finite_nonnegative(age) and float(age) > config.max_observation_age_ms:
            invalid.append(field_name)
    return tuple(invalid)


def _confidence_fallback(
    config: MaterializationDecisionConfig,
    invalid_fields: tuple[str, ...],
) -> MaterializationDecision:
    """处理仅因近期等待观测不足而不能决策的情况。"""
    confidence_fields = {
        "load_sample_count",
        "recompute_sample_count",
        "load_observation_age_ms",
        "recompute_observation_age_ms",
        "load_queue_wait_ms",
        "recompute_queue_wait_ms",
        "load_queue_wait_source",
        "recompute_queue_wait_source",
        "kv_bytes_source",
    }
    if set(invalid_fields).issubset(confidence_fields):
        return MaterializationDecision(
            mode=config.fallback_mode,
            reason="insufficient_observation_confidence",
            fallback=True,
            invalid_fields=invalid_fields,
            confidence_guard="recent_samples_and_phase_timestamps",
        )
    return _fallback(config, "invalid_or_missing_observation", invalid_fields)


def _select(
    config: MaterializationDecisionConfig,
    load_estimate: MaterializationCostEstimate,
    recompute_estimate: MaterializationCostEstimate,
    estimate_source: str,
) -> MaterializationDecision:
    """按固定边界规则选择较早完成的路径。"""
    if load_estimate.total_ms < recompute_estimate.total_ms:
        return MaterializationDecision(
            mode="load",
            reason="predicted_load_is_lower",
            predicted_load_ms=load_estimate.total_ms,
            predicted_recompute_ms=recompute_estimate.total_ms,
            load_estimate=load_estimate,
            recompute_estimate=recompute_estimate,
            estimate_source=estimate_source,
        )
    return MaterializationDecision(
        mode="recompute",
        reason="predicted_recompute_is_not_slower",
        predicted_load_ms=load_estimate.total_ms,
        predicted_recompute_ms=recompute_estimate.total_ms,
        load_estimate=load_estimate,
        recompute_estimate=recompute_estimate,
        estimate_source=estimate_source,
    )


def choose_materialization(
    observation: MaterializationObservation,
    config: MaterializationDecisionConfig,
) -> MaterializationDecision:
    """选择 CPU KV 加载或完整前向重算，不执行设备操作。"""
    if observation.hit_tokens <= 0 or observation.hit_blocks <= 0:
        return _fallback(config, "no_complete_cpu_hit")

    if config.forced_mode is not None:
        return MaterializationDecision(
            mode=config.forced_mode,
            reason=f"forced_{config.forced_mode}",
        )

    if not config.enabled:
        return MaterializationDecision(mode=config.fallback_mode, reason="disabled")

    if observation.active_materialization_count > 0:
        return _fallback(
            config,
            "unsupported_concurrent_context",
            ("active_materialization_count",),
        )

    invalid_fields = _validate_observation(observation, config)
    if invalid_fields:
        return _confidence_fallback(config, invalid_fields)

    if config.predictor == "historical":
        assert observation.load_total_ms is not None
        assert observation.recompute_total_ms is not None
        load_estimate = MaterializationCostEstimate(
            total_ms=float(observation.load_total_ms),
            queue_wait_ms=float(observation.load_queue_wait_ms or 0.0),
            fixed_ms=0.0,
            service_ms=float(observation.load_service_ms or 0.0),
            work_amount=float(observation.kv_bytes),
            work_unit="historical_total_ms",
            effective_rate_per_s=0.0,
            queue_source=observation.load_queue_wait_source,
        )
        recompute_estimate = MaterializationCostEstimate(
            total_ms=float(observation.recompute_total_ms),
            queue_wait_ms=float(observation.recompute_queue_wait_ms or 0.0),
            fixed_ms=0.0,
            service_ms=float(observation.recompute_service_ms or 0.0),
            work_amount=float(observation.prefill_flops or 0.0),
            work_unit="historical_total_ms",
            effective_rate_per_s=0.0,
            queue_source=observation.recompute_queue_wait_source,
        )
        return _select(
            config,
            load_estimate,
            recompute_estimate,
            "end_to_end_median",
        )

    assert config.cost_model is not None
    assert observation.load_queue_wait_ms is not None
    assert observation.recompute_queue_wait_ms is not None
    assert observation.prefill_flops is not None
    try:
        load_estimate = config.cost_model.predict_load(
            observation.kv_bytes,
            observation.load_queue_wait_ms,
            observation.load_queue_wait_source,
        )
        recompute_estimate = config.cost_model.predict_recompute(
            observation.prefill_flops,
            observation.recompute_queue_wait_ms,
            observation.recompute_queue_wait_source,
        )
    except ValueError as error:
        return _fallback(
            config,
            "cost_model_not_applicable",
            (str(error).split()[0],),
        )
    estimate_source = (
        f"formula_calibration:{config.cost_model.calibration_id or 'unnamed'}"
    )
    return _select(
        config,
        load_estimate,
        recompute_estimate,
        estimate_source,
    )


__all__ = [
    "MaterializationDecision",
    "MaterializationDecisionConfig",
    "MaterializationMode",
    "MaterializationObservation",
    "MaterializationPredictor",
    "choose_materialization",
]
