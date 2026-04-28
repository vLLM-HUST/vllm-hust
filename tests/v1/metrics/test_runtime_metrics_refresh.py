# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine

pytestmark = pytest.mark.skip_global_cleanup


class _RecordingLoggerManager:
    def __init__(self) -> None:
        self.engine_indexes = [0]
        self.recorded: list[dict[int, dict[str, object]]] = []

    def record_runtime_state(self, runtime_metrics_by_engine):
        self.recorded.append(runtime_metrics_by_engine)


def test_llm_engine_runtime_metrics_refresh_is_fail_open():
    engine = LLMEngine.__new__(LLMEngine)
    engine.logger_manager = _RecordingLoggerManager()

    def _raise(_method):
        raise RuntimeError("boom")

    engine.collective_rpc = _raise

    engine._refresh_runtime_metrics()

    assert engine.logger_manager.recorded == []


@pytest.mark.asyncio
async def test_async_llm_runtime_metrics_refresh_is_fail_open():
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.logger_manager = _RecordingLoggerManager()

    async def _raise(_method, timeout=None, args=(), kwargs=None):
        raise RuntimeError("boom")

    engine.collective_rpc = _raise

    await engine._refresh_runtime_metrics()

    assert engine.logger_manager.recorded == []


def test_llm_engine_runtime_metrics_refresh_records_valid_payload():
    engine = LLMEngine.__new__(LLMEngine)
    engine.logger_manager = _RecordingLoggerManager()
    engine.collective_rpc = lambda _method: [
        {"free_vram_bytes": 123, "model_load_state": "ready"}
    ]

    engine._refresh_runtime_metrics()

    assert engine.logger_manager.recorded == [
        {0: {"free_vram_bytes": 123, "model_load_state": "ready"}}
    ]


@pytest.mark.asyncio
async def test_async_llm_runtime_metrics_refresh_records_valid_payload():
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.logger_manager = _RecordingLoggerManager()

    async def _collect(_method, timeout=None, args=(), kwargs=None):
        return [{"free_vram_bytes": 456, "model_load_state": "initializing"}]

    engine.collective_rpc = _collect

    await engine._refresh_runtime_metrics()

    assert engine.logger_manager.recorded == [
        {0: {"free_vram_bytes": 456, "model_load_state": "initializing"}}
    ]