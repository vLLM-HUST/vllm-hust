# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only source selector checks; no native proof or device qualification."""

import builtins
import importlib
import json
import os
import socket
import sys
import threading
import types
from pathlib import Path

import pytest


@pytest.fixture
def host(monkeypatch):
    path = Path(
        os.environ.get(
            "H4_SELECTOR_HOST_SOURCE",
            str(Path(__file__).resolve().parents[2] / "vllm/plugins"),
        )
    )
    package = types.ModuleType("h4_selector_test_host")
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in list(sys.modules):
        if name.startswith(package.__name__ + "."):
            monkeypatch.delitem(sys.modules, name)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "vllm"}:
            raise AssertionError(
                "Host/Torch/model import prohibited in CPU selector tests"
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    selection = importlib.import_module(package.__name__ + ".native_sampled_selection")
    monkeypatch.delenv(selection.ENV, raising=False)
    return package, selection


def spec_for(sel, role="worker"):
    kernel = sel.native.identity(os.getpid())
    start_identity = f"pid:{kernel['pid']}:start_ticks:{kernel['start_ticks']}"
    process = {
        **kernel,
        "host": "test-source-only",
        "role": role,
        "ordinal": 0,
        "start_identity": start_identity,
        "process_epoch": 0,
        "assignment_source": "host",
    }
    phases = [
        {"phase_id": f"p{i}", "external_request_id": f"cmpl-new-{i}-0"}
        for i in range(32)
    ]
    return {"phases": phases, "senders": [{"process": process}]}, process


def test_default_keeps_original_instrumentation_without_registration(host):
    _, sel = host
    assert sel.register({}, {}, lambda: None) is None
    assert sel.runtime_request_selected("opaque-old-domain") is True


def test_exact32_api_selection_uses_only_actual_external_mapping(host):
    _, sel = host
    spec, process = spec_for(sel, "api-server")
    calls = []
    selector = sel.SampledSelection({}, spec, lambda: calls.append("current"))
    actual = {
        "process": process,
        "schema": "ecpa-native-request-id-mapping/v1",
        "external_request_id": "cmpl-new-31-0",
    }
    assert selector.fact(actual) is True
    assert (
        selector.fact({**actual, "external_request_id": "cmpl-not-selected-0"}) is False
    )
    assert calls == ["current", "current"]
    with pytest.raises(ValueError):
        selector.fact({**actual, "schema": "fake-mapping"})
    with pytest.raises(ValueError):
        selector.fact({**actual, "process": {**process, "pid": process["pid"] + 1}})


@pytest.mark.parametrize("count", [0, 31, 33])
def test_other_sample_counts_not_accepted(host, count):
    _, sel = host
    spec, _ = spec_for(sel)
    spec["phases"] = [{"external_request_id": str(i)} for i in range(count)]
    with pytest.raises(ValueError):
        sel.SampledSelection({}, spec, lambda: None)


def test_worker_opaque_mapping_does_not_guess_http_ids_and_rechecks_source(
    host, monkeypatch
):
    _, sel = host
    spec, _ = spec_for(sel)
    checks = []
    selector = sel.SampledSelection({}, spec, lambda: checks.append("current"))
    reads = []

    def original_mappings():
        reads.append(1)
        return {"actual-random-opaque": {"phase": spec["phases"][0]}}

    monkeypatch.setattr(selector, "_actual_mappings", original_mappings)
    assert selector.runtime("actual-random-opaque") is True
    assert selector.runtime("actual-random-opaque") is True
    assert selector.runtime("cmpl-new-0-0") is False
    assert len(reads) == 2 and len(checks) == 3

    def drift():
        raise ValueError("original private context changed")

    selector.current = drift
    with pytest.raises(ValueError):
        selector.runtime("actual-random-opaque")


@pytest.fixture
def describe_server(host, tmp_path):
    _, sel = host
    # The actual Unix SO_PEERCRED identity is this fixture process, never a
    # forged future PID. The status remains only an instrumentation selector.
    root = tmp_path / "s"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    path = root / "c"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    path.chmod(0o600)
    listener.listen(1)
    listener.settimeout(0.3)
    descriptor = {
        "socket_path": str(path),
        "socket_identity": [path.stat().st_dev, path.stat().st_ino],
        "collector": sel.native.identity(os.getpid()),
        "spec_sha256": "a" * 64,
        "collector_source_sha256": "b" * 64,
        "archive_identity": [1, 2],
    }
    spec, _ = spec_for(sel)
    status = {
        "schema": "ecpa-h4-sampled-concurrent-native-collector-status/v1",
        **{k: v for k, v in descriptor.items() if k != "socket_path"},
        "failed": False,
        "source_drain": {
            "runtimes": {"opaque-real-mapped": {"phase": spec["phases"][0]}}
        },
    }
    errors = []
    threads = []

    def serve():
        try:
            connection, _ = listener.accept()
            with connection:
                assert connection.recv(1024) == b'{"op":"describe"}\n'
                connection.sendall(json.dumps(status).encode() + b"\n")
        except (OSError, AssertionError) as exc:
            errors.append(repr(exc))

    def start():
        t = threading.Thread(target=serve, daemon=True)
        threads.append(t)
        t.start()
        return sel.SampledSelection(descriptor, spec, lambda: None), status, descriptor

    yield start
    listener.close()
    for t in threads:
        t.join(timeout=0.5)


def test_real_unix_peer_and_original_accepted_mapping(host, describe_server):
    selector, _, _ = describe_server()
    assert selector.runtime("opaque-real-mapped") is True


@pytest.mark.parametrize(
    "field", ["schema", "spec_sha256", "archive_identity", "failed", "phase"]
)
def test_original_describe_join_rejects_wrong_source_or_unregistered_phase(
    describe_server, field
):
    selector, status, _ = describe_server()
    if field == "phase":
        status["source_drain"]["runtimes"]["opaque-real-mapped"]["phase"] = {
            "phase_id": "unregistered"
        }
    elif field == "failed":
        status[field] = True
    elif field == "archive_identity":
        status[field] = [99, 99]
    else:
        status[field] = "wrong"
    with pytest.raises(ValueError):
        selector.runtime("opaque-real-mapped")


def test_unsampled_upload_never_imports_torch_or_reads_device_tables(host, monkeypatch):
    package, sel = host
    module = importlib.import_module(package.__name__ + ".native_block_table_observer")
    monkeypatch.setattr(module, "_observer", lambda fact: None)
    monkeypatch.setattr(sel, "runtime_request_selected", lambda rid: False)
    runner = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(req_ids=["opaque-unsampled"], num_reqs=1)
    )
    module.observe_block_table_upload(
        runner, types.SimpleNamespace(num_scheduled_tokens={"opaque-unsampled": 1})
    )


def test_unsampled_operator_returns_before_actual_metadata_read(host, monkeypatch):
    package, sel = host
    module = importlib.import_module(
        package.__name__ + ".native_attention_operator_observer"
    )
    monkeypatch.setattr(module, "_publisher", lambda fact: None)
    monkeypatch.setattr(sel, "runtime_request_selected", lambda rid: False)
    runner = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(req_ids=["opaque-unsampled"], num_reqs=1)
    )
    module.bind_actual_input_batch(runner, {}, eager=True)
