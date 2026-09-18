import importlib.metadata

import pytest

import vllm.plugins as plugins
from vllm.plugins import evidence


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


def enable(events):
    evidence._sink = events.append
    evidence._sink_loaded = True


def test_disabled_observer_preserves_plugin_behavior(monkeypatch):
    called = []
    ep = EntryPoint("demo", "demo:register", lambda: lambda: called.append(True))
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [ep])
    plugins.load_general_plugins()
    assert called == [True]


def test_general_plugin_sequence_identity_and_no_fabricated_digest(monkeypatch):
    events, called = [], []
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
    events = []
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
    events = []
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


def test_sink_failure_is_logged_by_default_and_strict_when_requested(
    monkeypatch, caplog
):
    def broken(_event):
        raise RuntimeError("sink down")

    evidence._sink, evidence._sink_loaded = broken, True
    evidence.emit("discovered", "group", "name", "value")
    assert "ECPA_EVIDENCE_SINK_WRITE_FAILED" in caplog.text
    monkeypatch.setenv("VLLM_ECPA_EVIDENCE_STRICT", "1")
    evidence._emitted.clear()
    with pytest.raises(RuntimeError, match="sink down"):
        evidence.emit("discovered", "group", "name", "value")
