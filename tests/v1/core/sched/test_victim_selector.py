"""Unit tests for vllm.v1.core.sched.victim_selector (runs WITHOUT vllm installed).

sys.modules stubbing technique (see vllm-plugin-authoring skill): the module
under test only needs `vllm.v1.core.sched.request_queue.SchedulingPolicy` and
`vllm.v1.request.Request`, which we stub here.
"""
import enum
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub the vllm module tree BEFORE importing the module under test
# ---------------------------------------------------------------------------

def _make_module(name, path=None, **attrs):
    m = types.ModuleType(name)
    m.__path__ = path if path is not None else []  # real dirs for real submodule loads
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


@dataclass
class _StubRequest:
    request_id: str
    priority: int = 0
    arrival_time: float = 0.0
    predicted_length: int | None = None


def _install_stubs():
    # repo root = tests/v1/core/sched -> up 3 = tests -> up 4 = repo
    repo = Path(__file__).resolve().parent.parents[3]
    vllm_root = repo / "vllm"
    for name, sub in [("vllm", vllm_root), ("vllm.v1", vllm_root / "v1"),
                      ("vllm.v1.core", vllm_root / "v1" / "core"),
                      ("vllm.v1.core.sched", vllm_root / "v1" / "core" / "sched")]:
        if name not in sys.modules:
            _make_module(name, path=[str(sub)])

    class SchedulingPolicy(enum.Enum):
        FCFS = "fcfs"
        PRIORITY = "priority"

    req_mod = _make_module("vllm.v1.request", Request=_StubRequest)
    rq_mod = _make_module(
        "vllm.v1.core.sched.request_queue", SchedulingPolicy=SchedulingPolicy
    )
    return req_mod, rq_mod


_install_stubs()

# Import module under test from the local checkout (not installed).
from vllm.v1.core.sched.victim_selector import (  # noqa: E402
    NoOpVictimSelector,
    get_victim_selector,
)


class _FakeVllmConfig:
    additional_config: dict = {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoOpVictimSelector:
    def test_fcfs_picks_last_running(self):
        sel = NoOpVictimSelector()
        running = [
            _StubRequest(request_id="a", arrival_time=1.0),
            _StubRequest(request_id="b", arrival_time=2.0),
            _StubRequest(request_id="c", arrival_time=3.0),
        ]
        from vllm.v1.core.sched.request_queue import SchedulingPolicy
        victim = sel.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "c"  # last in running == upstream pop()

    def test_priority_picks_highest_priority(self):
        sel = NoOpVictimSelector()
        running = [
            _StubRequest(request_id="low", priority=0, arrival_time=5.0),
            _StubRequest(request_id="high", priority=3, arrival_time=1.0),
        ]
        from vllm.v1.core.sched.request_queue import SchedulingPolicy
        victim = sel.pick_victim(running, SchedulingPolicy.PRIORITY)
        assert victim.request_id == "high"

    def test_priority_tie_breaks_by_latest_arrival(self):
        sel = NoOpVictimSelector()
        running = [
            _StubRequest(request_id="old", priority=1, arrival_time=1.0),
            _StubRequest(request_id="new", priority=1, arrival_time=9.0),
        ]
        from vllm.v1.core.sched.request_queue import SchedulingPolicy
        victim = sel.pick_victim(running, SchedulingPolicy.PRIORITY)
        assert victim.request_id == "new"

    def test_empty_running_raises(self):
        sel = NoOpVictimSelector()
        from vllm.v1.core.sched.request_queue import SchedulingPolicy
        with pytest.raises(ValueError):
            sel.pick_victim([], SchedulingPolicy.FCFS)

    def test_export_metrics_empty(self):
        assert NoOpVictimSelector().export_metrics() == {}


class TestGetVictimSelector:
    def test_falls_back_to_noop_when_no_plugin(self, monkeypatch):
        # No entry points installed in this environment -> NoOp.
        # (Explicitly empty so a locally-installed legacy plugin such as
        # vllm-hust-dla cannot leak into this test.)
        from importlib import metadata

        monkeypatch.setattr(metadata, "entry_points", lambda *, group: [])
        sel = get_victim_selector(_FakeVllmConfig())
        assert isinstance(sel, NoOpVictimSelector)

    def test_respects_disabled_flag(self, monkeypatch):
        class FakeCfg:
            additional_config = {"victim_selector_plugin_disabled": True}
        sel = get_victim_selector(FakeCfg())
        assert isinstance(sel, NoOpVictimSelector)

    def test_loads_registered_plugin(self, monkeypatch):
        from importlib import metadata

        class FakeSelector:
            @classmethod
            def from_vllm_config(cls, vllm_config):
                return cls()

        class FakeEP:
            def load(self):
                return FakeSelector

        fake_eps = [FakeEP()]

        def fake_entry_points(*, group=None):
            assert group == "vllm.victim_selector"
            return fake_eps

        monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
        sel = get_victim_selector(_FakeVllmConfig())
        assert isinstance(sel, FakeSelector)

    def test_plugin_load_failure_falls_back(self, monkeypatch):
        from importlib import metadata

        class BadEP:
            def load(self):
                raise ImportError("plugin broken")

        monkeypatch.setattr(metadata, "entry_points", lambda *, group: [BadEP()])
        sel = get_victim_selector(_FakeVllmConfig())
        assert isinstance(sel, NoOpVictimSelector)
