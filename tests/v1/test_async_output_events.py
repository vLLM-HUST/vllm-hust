# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
EVENTS_PATH = ROOT / "vllm" / "v1" / "events.py"
SPEC = importlib.util.spec_from_file_location("_async_output_events", EVENTS_PATH)
assert SPEC is not None and SPEC.loader is not None
events = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = events
SPEC.loader.exec_module(events)


@pytest.fixture(autouse=True)
def reset_event_bus():
    events.EventBus._sinks = []
    events.EventBus.enabled = False
    yield
    events.EventBus._sinks = []
    events.EventBus.enabled = False


def test_event_bus_is_default_off_and_idempotent() -> None:
    class Sink:
        def emit(self, event: Any) -> None:
            raise AssertionError("disabled bus dispatched an event")

    sink = Sink()
    assert events.EventBus.enabled is False
    assert events.EventBus._sinks == []

    events.EventBus.register_sink(sink)
    events.EventBus.register_sink(sink)
    assert events.EventBus.enabled is True
    assert events.EventBus._sinks == [sink]

    events.EventBus.unregister_sink(sink)
    assert events.EventBus.enabled is False
    assert events.EventBus._sinks == []


def test_sink_failure_isolated_and_disabled() -> None:
    received: list[Any] = []

    class FailingSink:
        def emit(self, event: Any) -> None:
            raise RuntimeError("observer failure")

    class HealthySink:
        def emit(self, event: Any) -> None:
            received.append(event)

    failing_sink = FailingSink()
    healthy_sink = HealthySink()
    events.EventBus.register_sink(failing_sink)
    events.EventBus.register_sink(healthy_sink)

    event = events.AsyncOutputCopyFailed(1, "sampled_token_d2h")
    events.EventBus.emit(event)

    assert received == [event]
    assert events.EventBus._sinks == [healthy_sink]
    assert events.EventBus.enabled is True


def test_async_output_event_payloads_are_typed() -> None:
    lifecycle = [
        events.AsyncOutputCreated(1, ("req-1",), (1, 1), "int64"),
        events.AsyncOutputCopyIssued(1, 2, 3, 8, 4),
        events.AsyncOutputWaitComplete(1, 5),
        events.AsyncOutputMaterialized(1, 6),
        events.AsyncOutputRetained(2),
        events.AsyncOutputConsumed(2, 7, 8),
    ]

    assert all(isinstance(event, events.WorkerEvent) for event in lifecycle)
    assert all(event.ts_monotonic_ns > 0 for event in lifecycle)


def test_hot_path_constructs_events_only_when_enabled() -> None:
    event_names = {
        "AsyncOutputCreated",
        "AsyncOutputCopyIssued",
        "AsyncOutputCopyFailed",
        "AsyncOutputWaitComplete",
        "AsyncOutputMaterialized",
        "AsyncOutputRetained",
        "AsyncOutputConsumed",
    }
    guarded_events: set[str] = set()
    for relative_path in (
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/worker/gpu_input_batch.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "events_enabled = EventBus.enabled" in source
        parents: dict[ast.AST, ast.AST] = {}
        for candidate_parent in ast.walk(tree):
            for child in ast.iter_child_nodes(candidate_parent):
                parents[child] = candidate_parent
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in event_names:
                continue
            guard: ast.AST | None = parents.get(node)
            while guard is not None and not isinstance(guard, ast.If):
                guard = parents.get(guard)
            assert isinstance(guard, ast.If)
            assert ast.unparse(guard.test) in {"EventBus.enabled", "events_enabled"}
            guarded_events.add(node.func.id)

    assert guarded_events == event_names
