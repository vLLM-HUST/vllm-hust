# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu_worker import Worker

pytestmark = pytest.mark.skip_global_cleanup


def test_gpu_worker_runtime_metrics_report_residency_and_memory(monkeypatch):
    worker = Worker.__new__(Worker)
    worker.device = torch.device("cuda:0")
    worker.model_runner = SimpleNamespace(model_memory_usage=321)
    worker.available_kv_cache_memory_bytes = 654
    worker._sleep_level = 1

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device=None: (1234, 4321))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device=None: 987)

    runtime_metrics = worker.get_runtime_metrics()

    assert runtime_metrics == {
        "free_vram_bytes": 1234,
        "reserved_vram_bytes": 987,
        "model_weight_bytes": 321,
        "available_kv_cache_memory_bytes": 654,
        "model_residency_state": "weights_offloaded",
        "model_load_state": "ready",
    }


def test_gpu_worker_runtime_metrics_report_discard_all_when_sleep_level_two(monkeypatch):
    worker = Worker.__new__(Worker)
    worker.device = torch.device("cuda:0")
    worker.model_runner = None
    worker.available_kv_cache_memory_bytes = 0
    worker._sleep_level = 2

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device=None: (11, 22))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device=None: 33)

    runtime_metrics = worker.get_runtime_metrics()

    assert runtime_metrics["model_residency_state"] == "discard_all"
    assert runtime_metrics["model_load_state"] == "initializing"
    assert runtime_metrics["free_vram_bytes"] == 11
    assert runtime_metrics["reserved_vram_bytes"] == 33