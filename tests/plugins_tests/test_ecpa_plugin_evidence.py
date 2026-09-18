# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
from typing import Any
from unittest.mock import mock_open

import pytest

import vllm.plugins as plugins
from vllm.plugins import evidence

Payload = dict[str, Any]


class EntryPoint:
    def __init__(self, name, value, loader):
        self.name, self.value, self.loader = name, value, loader

    def load(self):
        return self.loader()


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    plugins.plugins_loaded = False
    plugins._plugin_values.clear()
    evidence.reset_for_tests()
    monkeypatch.delenv("VLLM_ECPA_EVIDENCE_SINK", raising=False)
    monkeypatch.delenv("VLLM_ECPA_EVIDENCE_STRICT", raising=False)
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ROLE", "worker")
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ORDINAL", "2")
    monkeypatch.setenv("VLLM_ECPA_PROCESS_EPOCH", "7")


def enable(events: list[Payload]) -> None:
    evidence._sink = events.append
    evidence._sink_state = "ready"


def test_disabled_observer_preserves_plugin_behavior(monkeypatch):
    called = []
    ep = EntryPoint("demo", "demo:register", lambda: lambda: called.append(True))
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [ep])
    plugins.load_general_plugins()
    assert called == [True]


def test_general_plugin_sequence_identity_and_no_fabricated_digest(monkeypatch):
    events: list[Payload] = []
    called = []
    enable(events)
    ep = EntryPoint("demo", "demo:register", lambda: lambda: called.append(True))
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [ep])
    plugins.load_general_plugins()
    plugins.load_general_plugins()
    assert called == [True]
    assert [item["event"] for item in events] == [
        "discovered",
        "resolved",
        "invoked",
    ]
    assert all(item["entry_point"]["value"] == "demo:register" for item in events)
    assert events[-1]["process"]["role"] == "worker"
    assert events[-1]["process"]["ordinal"] == 2
    assert events[-1]["process"]["process_epoch"] == 7
    assert events[-1]["plugin_id"] is None
    assert events[-1]["artifact_digest"] is None


def test_allowlist_skip_and_load_failure(monkeypatch):
    events: list[Payload] = []
    enable(events)
    monkeypatch.setenv("VLLM_PLUGINS", "allowed")
    bad = EntryPoint("allowed", "bad:load", lambda: (_ for _ in ()).throw(ValueError()))
    skipped = EntryPoint("other", "other:load", lambda: lambda: None)
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda group: [bad, skipped]
    )
    assert plugins.load_plugins_by_group(plugins.DEFAULT_PLUGINS_GROUP) == {}
    assert [(item["event"], item["entry_point"]["name"]) for item in events] == [
        ("discovered", "allowed"),
        ("discovered", "other"),
        ("failed", "allowed"),
        ("skipped", "other"),
    ]


def test_call_failure_never_emits_invoked(monkeypatch):
    events: list[Payload] = []
    enable(events)

    def fail():
        raise RuntimeError("plugin failed")

    ep = EntryPoint("demo", "demo:fail", lambda: fail)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [ep])
    with pytest.raises(RuntimeError, match="plugin failed"):
        plugins.load_general_plugins()
    assert [item["event"] for item in events] == [
        "discovered",
        "resolved",
        "failed",
    ]


def test_plugin_can_forge_internal_event_but_loader_never_emits_success(monkeypatch):
    events: list[Payload] = []
    enable(events)

    def hostile():
        evidence._emit_host_event(
            "invoked", plugins.DEFAULT_PLUGINS_GROUP, "forged", "forged:value"
        )
        raise RuntimeError("after forgery")

    ep = EntryPoint("demo", "demo:hostile", lambda: hostile)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [ep])
    with pytest.raises(RuntimeError, match="after forgery"):
        plugins.load_general_plugins()
    assert not any(
        item["event"] == "invoked" and item["entry_point"]["name"] == "demo"
        for item in events
    )


def test_sink_failure_is_logged_by_default_and_strict_when_requested(
    monkeypatch, caplog
):
    def broken(_event):
        raise RuntimeError("sink down")

    evidence._sink, evidence._sink_state = broken, "ready"
    assert not evidence._emit_host_event("discovered", "group", "name", "value")
    assert "ECPA_EVIDENCE_SINK_WRITE_FAILED" in caplog.text
    assert not evidence._delivered
    assert evidence._delivery_attempts == 1
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_STRICT", "1")
    with pytest.raises(evidence.EvidenceConfigurationError):
        evidence._emit_host_event("discovered", "group", "name", "value")
    with pytest.raises(evidence.EvidenceConfigurationError):
        evidence._emit_host_event("discovered", "group", "name", "value")
    assert evidence._sink_state == "failed"


def test_compat_write_failure_retries_and_dedupes_only_after_delivery():
    events: list[Payload | None] = []

    def transient(event):
        if not events:
            events.append(None)
            raise RuntimeError("once")
        events.append(event)

    evidence._sink, evidence._sink_state = transient, "ready"
    assert not evidence._emit_host_event("resolved", "group", "name", "value")
    assert evidence._emit_host_event("resolved", "group", "name", "value")
    assert evidence._emit_host_event("resolved", "group", "name", "value")
    delivered = [item for item in events if item is not None]
    assert len(delivered) == 1
    assert delivered[0]["delivery_attempt"] == 2
    assert evidence._delivery_attempts == 2
    assert evidence._delivery_count == 1


def test_sink_import_failure_retries_in_compat_and_is_sticky_in_strict(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_SINK", "missing.module:sink")
    attempts = []

    def missing(name):
        attempts.append(name)
        raise ImportError(name)

    monkeypatch.setattr(evidence.importlib, "import_module", missing)
    assert not evidence._emit_host_event("resolved", "g", "n", "v")
    assert not evidence._emit_host_event("resolved", "g", "n", "v")
    assert len(attempts) == 2

    evidence.reset_for_tests()
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_STRICT", "1")
    with pytest.raises(evidence.EvidenceConfigurationError):
        evidence._emit_host_event("resolved", "g", "n", "v")
    with pytest.raises(evidence.EvidenceConfigurationError):
        evidence._emit_host_event("resolved", "g", "n", "v")
    assert len(attempts) == 3


def test_dedupe_preserves_value_and_failure_stage():
    events: list[Payload] = []
    enable(events)
    evidence._emit_host_event("failed", "g", "n", "v1", detail="entry_point.load")
    evidence._emit_host_event("failed", "g", "n", "v2", detail="entry_point.load")
    evidence._emit_host_event("failed", "g", "n", "v1", detail="callable")
    assert len(events) == 3


def test_identity_errors_follow_compat_and_sticky_strict(monkeypatch):
    events: list[Payload] = []
    enable(events)
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ORDINAL", "not-an-int")
    assert not evidence._emit_host_event("discovered", "g", "n", "v")
    assert events == []
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_STRICT", "1")
    with pytest.raises(evidence.EvidenceConfigurationError):
        evidence._emit_host_event("discovered", "g", "n", "v")
    monkeypatch.setenv("VLLM_ECPA_PROCESS_ORDINAL", "1")
    with pytest.raises(evidence.EvidenceConfigurationError):
        evidence._emit_host_event("discovered", "g", "n", "v")


def test_invalid_strict_mode_configuration_fails_deterministically(monkeypatch):
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_STRICT", "yes")
    with pytest.raises(evidence.EvidenceConfigurationError):
        evidence._emit_host_event("discovered", "g", "n", "v")
    with pytest.raises(evidence.EvidenceConfigurationError):
        evidence._emit_host_event("discovered", "g", "n", "v")


def test_strict_evidence_failure_preserves_plugin_exception_as_primary(monkeypatch):
    events: list[Payload] = []
    enable(events)
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_STRICT", "1")

    def plugin_failure():
        evidence._sink = lambda event: (_ for _ in ()).throw(RuntimeError("sink"))
        raise ValueError("plugin")

    ep = EntryPoint("demo", "demo:failure", lambda: plugin_failure)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [ep])
    with pytest.raises(ValueError, match="plugin") as caught:
        plugins.load_general_plugins()
    assert isinstance(caught.value.__cause__, evidence.EvidenceConfigurationError)


def test_fork_pid_change_clears_inherited_delivery_state(monkeypatch):
    events: list[Payload] = []
    enable(events)
    evidence._emit_host_event("resolved", "g", "n", "v")
    old_pid = evidence._owner_pid
    monkeypatch.setattr(evidence.os, "getpid", lambda: old_pid + 1)
    monkeypatch.setattr(evidence, "_start_identity", lambda: "pid:new:start_ticks:2")
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_SINK", "tests.fake:sink")
    monkeypatch.setattr(
        evidence.importlib,
        "import_module",
        lambda name: type("M", (), {"sink": events.append}),
    )
    evidence._emit_host_event("resolved", "g", "n", "v")
    assert len(events) == 2


def test_proc_start_identity_parses_after_final_comm_parenthesis(monkeypatch):
    stat = "123 (worker ) name) S " + " ".join(str(item) for item in range(4, 23))
    monkeypatch.setattr("builtins.open", mock_open(read_data=stat))
    monkeypatch.setattr(evidence.os, "getpid", lambda: 123)
    assert evidence._start_identity() == "pid:123:start_ticks:22"
