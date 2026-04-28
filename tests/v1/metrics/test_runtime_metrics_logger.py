# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.reader import Gauge, get_metrics_snapshot

pytestmark = pytest.mark.skip_global_cleanup


class _DummyMetricsAdapter:
    def __init__(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass


def _build_fake_vllm_config():
    return SimpleNamespace(
        observability_config=SimpleNamespace(
            show_hidden_metrics=False,
            kv_cache_metrics=False,
            enable_mfu_metrics=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        model_config=SimpleNamespace(
            served_model_name="test-model",
            max_model_len=1024,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=0,
            metrics_info=lambda: {},
        ),
        lora_config=None,
    )


def _find_gauge(metrics: list[object], name: str, **labels: str) -> Gauge:
    for metric in metrics:
        if not isinstance(metric, Gauge):
            continue
        if metric.name != name:
            continue
        if all(metric.labels.get(key) == value for key, value in labels.items()):
            return metric
    raise AssertionError(f"Gauge {name} with labels {labels} not found")


def test_prometheus_runtime_metrics_are_exported(monkeypatch):
    unregister_vllm_metrics()
    monkeypatch.setattr(PrometheusStatLogger, "_spec_decoding_cls", _DummyMetricsAdapter)
    monkeypatch.setattr(PrometheusStatLogger, "_kv_connector_cls", _DummyMetricsAdapter)
    monkeypatch.setattr(PrometheusStatLogger, "_perf_metrics_cls", _DummyMetricsAdapter)

    logger = PrometheusStatLogger(_build_fake_vllm_config(), engine_indexes=[0, 1])
    logger.record_runtime_state(
        {
            0: {
                "free_vram_bytes": 123,
                "reserved_vram_bytes": 45,
                "model_weight_bytes": 67,
                "available_kv_cache_memory_bytes": 89,
                "model_residency_state": "resident",
                "model_load_state": "ready",
            },
            1: {
                "free_vram_bytes": 321,
                "reserved_vram_bytes": 54,
                "model_weight_bytes": 76,
                "available_kv_cache_memory_bytes": 98,
                "model_residency_state": "weights_offloaded",
                "model_load_state": "initializing",
            },
        }
    )

    metrics = get_metrics_snapshot()

    assert _find_gauge(metrics, "vllm:free_vram_bytes", engine="0").value == 123
    assert _find_gauge(metrics, "vllm:reserved_vram_bytes", engine="1").value == 54
    assert _find_gauge(metrics, "vllm:model_weight_bytes", engine="0").value == 67
    assert (
        _find_gauge(metrics, "vllm:available_kv_cache_memory_bytes", engine="1").value
        == 98
    )

    assert (
        _find_gauge(
            metrics,
            "vllm:model_residency_state",
            engine="0",
            state="resident",
        ).value
        == 1.0
    )
    assert (
        _find_gauge(
            metrics,
            "vllm:model_residency_state",
            engine="1",
            state="weights_offloaded",
        ).value
        == 1.0
    )
    assert (
        _find_gauge(metrics, "vllm:model_load_state", engine="0", state="ready").value
        == 1.0
    )
    assert (
        _find_gauge(
            metrics,
            "vllm:model_load_state",
            engine="1",
            state="initializing",
        ).value
        == 1.0
    )

    unregister_vllm_metrics()