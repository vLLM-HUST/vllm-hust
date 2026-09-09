"""纯 CPU 测试公式成本模型、完整前向工作量和多尺寸拟合。"""

from __future__ import annotations

import pytest

from kv_materialization_plugin.calibration import fit_cost_model
from kv_materialization_plugin.cost_model import (
    MaterializationCostModel,
    MaterializationModelShape,
    estimate_prefill_flops,
)
from kv_materialization_plugin.decision import (
    MaterializationDecisionConfig,
    MaterializationObservation,
    choose_materialization,
)


def make_formula_observation(**overrides: object) -> MaterializationObservation:
    """构造一条具有完整公式输入的观测。"""
    values: dict[str, object] = {
        "hit_tokens": 256,
        "hit_blocks": 4,
        "kv_bytes": 4096,
        "kv_bytes_source": "runtime_block_bytes",
        "device_prefix_tokens": 128,
        "batch_size": 1,
        "prefill_flops": 20_000.0,
        "load_queue_wait_ms": 3.0,
        "load_queue_wait_source": "recent_completed_median",
        "load_sample_count": 3,
        "load_observation_age_ms": 10.0,
        "recompute_queue_wait_ms": 2.0,
        "recompute_queue_wait_source": "recent_completed_median",
        "recompute_sample_count": 3,
        "recompute_observation_age_ms": 10.0,
    }
    values.update(overrides)
    return MaterializationObservation(**values)  # type: ignore[arg-type]


def test_prefill_flops_includes_device_prefix_attention_work() -> None:
    """已有设备前缀会增加重算段的 Attention 工作量。"""
    shape = MaterializationModelShape(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=4,
        intermediate_size=32,
        gated_mlp=True,
    )
    without_prefix = estimate_prefill_flops(shape, 8, device_prefix_tokens=0)
    with_prefix = estimate_prefill_flops(shape, 8, device_prefix_tokens=16)

    assert with_prefix > without_prefix


def test_formula_decision_records_all_cost_components() -> None:
    """公式决策保留两条路径的等待、固定项和服务项。"""
    model = MaterializationCostModel(
        load_fixed_ms=1.0,
        load_bandwidth_bytes_per_s=1_000_000.0,
        recompute_fixed_ms=2.0,
        recompute_flops_per_s=1_000_000.0,
        calibration_id="unit",
        load_min_bytes=1024,
        load_max_bytes=8192,
        recompute_min_flops=10_000.0,
        recompute_max_flops=30_000.0,
    )
    decision = choose_materialization(
        make_formula_observation(),
        MaterializationDecisionConfig(
            enabled=True,
            predictor="formula",
            cost_model=model,
            min_copy_samples=2,
            min_recompute_samples=2,
        ),
    )

    assert decision.mode == "load"
    assert decision.predicted_load_ms == pytest.approx(8.096)
    assert decision.predicted_recompute_ms == pytest.approx(24.0)
    assert decision.load_estimate is not None
    assert decision.load_estimate.queue_wait_ms == pytest.approx(3.0)
    assert decision.load_estimate.fixed_ms == pytest.approx(1.0)
    assert decision.load_estimate.service_ms == pytest.approx(5.096)
    assert decision.recompute_estimate is not None
    assert decision.estimate_source == "formula_calibration:unit"


def test_formula_decision_does_not_assume_missing_queue_is_zero() -> None:
    """缺少任一等待估计时必须回退。"""
    model = MaterializationCostModel(
        load_fixed_ms=1.0,
        load_bandwidth_bytes_per_s=1_000_000.0,
        recompute_fixed_ms=2.0,
        recompute_flops_per_s=1_000_000.0,
    )
    decision = choose_materialization(
        make_formula_observation(load_queue_wait_ms=None),
        MaterializationDecisionConfig(
            enabled=True,
            predictor="formula",
            cost_model=model,
        ),
    )

    assert decision.fallback is True
    assert decision.reason == "insufficient_observation_confidence"
    assert "load_queue_wait_ms" in decision.invalid_fields


def test_formula_decision_does_not_accept_unlabeled_zero_queue() -> None:
    """数值为零但来源未知时也必须回退。"""
    model = MaterializationCostModel(
        load_fixed_ms=1.0,
        load_bandwidth_bytes_per_s=1_000_000.0,
        recompute_fixed_ms=2.0,
        recompute_flops_per_s=1_000_000.0,
    )
    decision = choose_materialization(
        make_formula_observation(load_queue_wait_source="unavailable"),
        MaterializationDecisionConfig(
            enabled=True,
            predictor="formula",
            cost_model=model,
        ),
    )

    assert decision.fallback is True
    assert decision.reason == "insufficient_observation_confidence"
    assert "load_queue_wait_source" in decision.invalid_fields


def test_formula_decision_rejects_out_of_range_work() -> None:
    """超出冻结校准范围的工作量不能外推。"""
    model = MaterializationCostModel(
        load_fixed_ms=1.0,
        load_bandwidth_bytes_per_s=1_000_000.0,
        recompute_fixed_ms=2.0,
        recompute_flops_per_s=1_000_000.0,
        load_min_bytes=8192,
    )
    decision = choose_materialization(
        make_formula_observation(),
        MaterializationDecisionConfig(
            enabled=True,
            predictor="formula",
            cost_model=model,
        ),
    )

    assert decision.fallback is True
    assert decision.reason == "cost_model_not_applicable"


def test_calibration_uses_multiple_sizes_and_service_only() -> None:
    """拟合使用不含队列的 service_ms，避免把等待重复加入速度。"""
    load = [
        {"kv_bytes": 1000, "service_ms": 3.0, "total_ms": 30.0},
        {"kv_bytes": 2000, "service_ms": 5.0, "total_ms": 50.0},
        {"kv_bytes": 4000, "service_ms": 9.0, "total_ms": 90.0},
    ]
    recompute = [
        {"work_flops": 1000.0, "service_ms": 4.0, "total_ms": 40.0},
        {"work_flops": 2000.0, "service_ms": 6.0, "total_ms": 60.0},
        {"work_flops": 4000.0, "service_ms": 10.0, "total_ms": 100.0},
    ]

    model, diagnostics = fit_cost_model(load, recompute, "fit", min_samples=3)

    assert model.load_fixed_ms == pytest.approx(1.0)
    assert model.load_bandwidth_bytes_per_s == pytest.approx(500_000.0)
    assert model.recompute_fixed_ms == pytest.approx(2.0)
    assert model.recompute_flops_per_s == pytest.approx(500_000.0)
    assert diagnostics["load"]["distinct_work_count"] == 3
    assert diagnostics["recompute"]["fit_r2"] == pytest.approx(1.0)
