# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure source shutdown contracts: no model, device or external MOD dependency."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def events_module():
    # The production file is standalone; avoid importing the heavy engine graph.
    name = "event_bus_shutdown_contract"
    spec = importlib.util.spec_from_file_location(name, ROOT / "vllm/v1/events.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def shutdown_function(path, class_name, bus):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown"
    )
    namespace = {
        "EventBus": bus,
        "logger": SimpleNamespace(
            debug_once=lambda *a: None,
            debug=lambda *a: None,
            warning=lambda *a: None,
            exception=lambda *a: None,
        ),
        "gc": SimpleNamespace(unfreeze=lambda: None),
        "cleanup_dist_env_and_memory": lambda: None,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / path), "exec"),
        namespace,
    )
    return namespace["shutdown"]


def test_close_detaches_before_callback_and_preserves_actual_result():
    bus = events_module().EventBus
    expected = object()
    calls = []

    class Sink:
        def emit(self, event):
            calls.append(event)

        def close(self):
            assert bus.enabled is False
            calls.append("close")
            return expected

    bus.register_sink(Sink())
    bus.emit("event")
    assert bus.close_sinks() == (expected,)
    assert bus.close_sinks() == ()
    assert calls == ["event", "close"]


def test_emit_only_sink_remains_valid_without_invented_close_success():
    bus = events_module().EventBus
    calls = []

    class Sink:
        def emit(self, event):
            calls.append(event)

    bus.register_sink(Sink())
    bus.emit("event")
    assert bus.close_sinks() == ()
    bus.emit("after close")
    assert calls == ["event"]


@pytest.mark.parametrize("property_failure", [False, True])
def test_failed_close_is_retained_and_other_sink_still_closes(property_failure):
    bus = events_module().EventBus
    error = OSError("real callback failure")
    expected = object()

    class Broken:
        def emit(self, event):
            pass

        @property
        def close(self):
            if property_failure:
                raise error

            def fail():
                raise error

            return fail

    class Other:
        def emit(self, event):
            pass

        def close(self):
            return expected

    bus.register_sink(Broken())
    bus.register_sink(Other())
    assert bus.close_sinks() == (error, expected)
    assert not bus.enabled


@pytest.mark.parametrize("worker_fails", [False, True])
def test_engine_drains_before_worker_teardown(worker_fails):
    seen = []

    class Bus:
        @staticmethod
        def close_sinks():
            seen.append("drain")
            return (None,)

    class Executor:
        def shutdown(self):
            seen.append("workers")
            if worker_fails:
                raise OSError("teardown error")

    close = shutdown_function("vllm/v1/engine/core.py", "EngineCore", Bus)
    engine = SimpleNamespace(
        structured_output_manager=SimpleNamespace(clear_backend=lambda: None),
        model_executor=Executor(),
        scheduler=None,
    )
    if worker_fails:
        with pytest.raises(OSError, match="teardown error"):
            close(engine)
    else:
        close(engine)
    assert seen == ["drain", "workers"]


@pytest.mark.parametrize("failure", [None, "extension", "worker"])
def test_worker_wrapper_drains_before_extension_and_device_teardown(failure):
    seen = []

    class Bus:
        @staticmethod
        def close_sinks():
            seen.append("drain")
            return (None,)

    class Extension:
        @staticmethod
        def destroy_worker_extension(worker):
            seen.append("extension")
            if failure == "extension":
                raise OSError("extension error")

    class Worker:
        def shutdown(self):
            seen.append("worker")
            if failure == "worker":
                raise OSError("device error")

    close = shutdown_function("vllm/v1/worker/worker_base.py", "WorkerWrapperBase", Bus)
    wrapper = SimpleNamespace(worker=Worker(), _worker_extension_cls=Extension)
    if failure == "worker":
        with pytest.raises(OSError, match="device error"):
            close(wrapper)
    else:
        close(wrapper)
    assert seen == ["drain", "extension", "worker"]
