# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Actual Host observer Router contracts; CPU fixtures never qualify recovery."""

import os
from pathlib import Path

import pytest

from vllm.plugins import native_observer_rebind as host


def test_successor_rejects_live_old_collector_before_probe(monkeypatch):
    old = {
        "source_files": ["fixture-source"],
        "spec_path": "fixture-old",
        "spec_sha256": "old",
        "collector": {"pid": 1},
        "archive_identity": {"epoch": "old"},
        "socket_identity": [1, 1],
        "socket_path": "old",
    }
    new = {
        **old,
        "spec_path": "fixture-new",
        "spec_sha256": "new",
        "collector": {"pid": 2},
        "archive_identity": {"epoch": "new"},
        "socket_identity": [1, 2],
        "socket_path": "new",
    }
    scope = {
        "native_plan_id": "fixture-plan",
        "native_launch_id": "fixture-launch",
        "phases": list(range(60)),
    }
    monkeypatch.setattr(host, "stable_spec", lambda context: scope)
    monkeypatch.setattr(host, "same_kernel_alive", lambda kernel: True)
    calls = []
    monkeypatch.setattr(host, "probe", lambda *a: calls.append(True))
    with pytest.raises(ValueError, match="old collector kernel remains alive"):
        host.validate_successor(old, new, "semantic", scope)
    assert calls == []


@pytest.fixture
def pair_router(tmp_path, monkeypatch):
    """Exercise actual Router atomic swap logic, isolating runtime source probes."""
    producer = {
        "pid": 123,
        "start_ticks": 456,
        "session": 123,
        "start_identity": "fixture",
        "boot_id": "fixture",
    }
    policy = {
        "schema": host.POLICY,
        "domain": host.DOMAIN,
        "start_id": "observer-loss-r1-ecpa",
        "receipt_sha256": "fixture-only-receipt",
        "producer": producer,
        "controller": producer,
        "controller_cmdline_sha256": "fixture",
        "controller_source_files": ["fixture"],
        "native_plan_id": "fixture-plan",
        "native_launch_id": "fixture-launch",
        "socket_path": str(tmp_path / "control.sock"),
        "journal_path": str(tmp_path / "control.jsonl"),
        "control_nonce": "a" * 64,
        "initial_contexts": {},
        "max_recovery_ns": 30_000_000_000,
        "ready_path": str(tmp_path / "ready.json"),
        "producer_source_files": ["fixture"],
        "producer_group": [
            {"producer": {"pid": 55}, "role": "api-server", "ordinal": 0},
            {"producer": producer, "role": "worker", "ordinal": 0},
            {"producer": {"pid": 124}, "role": "worker", "ordinal": 1},
        ],
    }
    contexts = {}
    callbacks = []
    builders = []
    for channel in ("semantic", "tensor"):
        path = tmp_path / (channel + "-old.json")
        path.write_text("{}")
        path.chmod(0o600)
        policy["initial_contexts"][channel] = {
            "path": str(path),
            "sha256": "fixture-old-" + channel,
        }
        newpath = tmp_path / (channel + "-new.json")
        newpath.write_text("{}")
        newpath.chmod(0o600)
        contexts[channel] = {"path": str(newpath), "sha256": "fixture-new-" + channel}
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(host.native.canonical(policy))
    policy_path.chmod(0o600)
    tmp_path.chmod(0o700)
    monkeypatch.setattr(host.native, "identity", lambda pid: producer)
    monkeypatch.setattr(host.native, "source_digests", lambda *a: None)
    monkeypatch.setattr(host.Router, "_serve", lambda self: None)
    monkeypatch.setattr(host.Router, "_capture_collector", lambda *a: None)
    monkeypatch.setattr(
        host,
        "checked",
        lambda ref: (
            {
                "spec_path": "fixture",
                "spec_sha256": "fixture",
                "collector": {"pid": 2},
                "ready": {"collector": {"pid": 2}},
            },
            b"fixture",
            None,
        ),
    )
    monkeypatch.setattr(
        host,
        "stable_spec",
        lambda context: {
            "native_plan_id": "fixture-plan",
            "phases": [
                {"client_request_id": f"fixture/request-{i:05d}"} for i in range(60)
            ],
        },
    )
    monkeypatch.setattr(host, "validate_successor", lambda *a: None)
    monkeypatch.setattr(host.time, "monotonic_ns", lambda: 200)
    router = host.Router(policy_path)
    router._resolve_api_mapping = lambda rid: {
        "runtime_request_id": rid,
        "generation": router.requests.get(rid, router.generation),
    }
    publications = {}
    for channel in ("semantic", "tensor"):

        def builder(path, channel=channel):
            builders.append((channel, str(path)))

            def successor(fact):
                callbacks.append((channel, "new", fact["runtime_request_id"]))
                return {"actual_callback_fixture": "new"}

            return successor

        def original(fact, channel=channel):
            callbacks.append((channel, "old", fact["runtime_request_id"]))
            return {"actual_callback_fixture": "old"}

        publications[channel] = router.add(
            channel, policy["initial_contexts"][channel]["path"], original, builder
        )
    command = dict(
        schema=host.COMMAND,
        domain=host.DOMAIN,
        operation="prepare",
        start_id=policy["start_id"],
        receipt_sha256=policy["receipt_sha256"],
        control_nonce=policy["control_nonce"],
        producer=producer,
        expected_generation=0,
        transition_nonce="b" * 64,
        fault_ns=100,
        deadline_ns=30_000_000_100,
        contexts=contexts,
        old_client_request_ids=["fixture/request-00000"],
    )
    yield router, publications, command, callbacks, builders
    os.close(router.journal_fd)


def commit(router, command):
    prepared = router.apply(command)
    final = {k: v for k, v in command.items() if k != "contexts"}
    final.update(
        operation="commit",
        prepared_sha256=host.native.sha(host.native.canonical(prepared)),
    )
    return router.apply(final)


def test_pair_commit_changes_both_transports_without_reset_or_fact_replay(pair_router):
    router, publish, command, calls, builders = pair_router
    publish["semantic"]({"runtime_request_id": "fixture-old-request"})
    response = commit(router, command)
    assert response["state"] == "committed" and response["generation"] == 1
    assert response["native_completion_proved"] is False
    assert len(builders) == 2 and len(calls) == 1  # Prepare/commit publishes no fact.
    assert router.requests == {"fixture-old-request": 0}
    publish["semantic"]({"runtime_request_id": "fixture-new-request"})
    publish["tensor"]({"runtime_request_id": "fixture-new-request"})
    assert calls[-2:] == [
        ("semantic", "new", "fixture-new-request"),
        ("tensor", "new", "fixture-new-request"),
    ]


def test_inflight_original_request_cannot_cross_epoch_or_be_retried(pair_router):
    router, publish, command, calls, _ = pair_router
    publish["semantic"]({"runtime_request_id": "fixture-inflight"})
    commit(router, command)
    result = publish["tensor"]({"runtime_request_id": "fixture-inflight"})
    assert host.is_unavailable(result) and not isinstance(result, dict)
    assert "fixture-inflight" in router.unavailable_requests
    assert calls == [("semantic", "old", "fixture-inflight")]


def test_one_channel_prepare_cannot_split_worker_transport(pair_router):
    router, _, command, _, builders = pair_router
    command["contexts"].pop("tensor")
    with pytest.raises(ValueError, match="prepare jointly"):
        router.apply(command)
    assert router.generation == 0 and builders == []


def test_nonce_and_generation_replay_rejected(pair_router):
    router, _, command, _, _ = pair_router
    commit(router, command)
    with pytest.raises(ValueError, match="epoch/receipt/nonce"):
        router.apply(command)


def test_extended_recovery_deadline_is_rejected(pair_router):
    router, _, command, _, builders = pair_router
    command["deadline_ns"] += 1
    with pytest.raises(ValueError, match="original30s"):
        router.apply(command)
    assert builders == []


def test_wrong_controller_kernel_rejected_before_command(pair_router, monkeypatch):
    router, _, _, _, _ = pair_router

    class Connection:
        def getsockopt(self, *args):
            return host.struct.pack("3i", 999, os.getuid(), os.getgid())

    monkeypatch.setattr(host.native, "identity", lambda pid: {"pid": pid})
    with pytest.raises(ValueError, match="original authority/Host kernels"):
        router._peer(Connection())


def test_wrong_prepared_digest_never_swaps_active_callbacks(pair_router):
    router, _, command, _, _ = pair_router
    router.apply(command)
    final = {k: v for k, v in command.items() if k != "contexts"}
    final.update(operation="commit", prepared_sha256="forged-fixture")
    with pytest.raises(ValueError, match="exact prepared"):
        router.apply(final)
    assert router.generation == 0 and router.pending is not None


def test_socket_timeout_of_live_collector_is_not_kernel_death(monkeypatch):
    monkeypatch.setenv(host.ENV, "fixture-only-opt-in")
    monkeypatch.setattr(host, "same_kernel_alive", lambda kernel: True)

    def timeout(*args, **kwargs):
        raise TimeoutError("fixture actual live collector timeout")

    monkeypatch.setattr(host.native, "_submit_fact", timeout)
    with pytest.raises(TimeoutError):
        host.native.submit_fact(
            Path("/fixture/socket"),
            {},
            expected_collector={"pid": 1},
            expected_socket_identity=(1, 1),
            expected_collector_source_sha256="fixture",
            expected_archive_identity={},
        )


def test_unknown_peer_or_invalid_ack_never_becomes_observer_unavailable(monkeypatch):
    monkeypatch.setenv(host.ENV, "fixture-only-opt-in")
    monkeypatch.setattr(host, "collector_kernel_gone", lambda kernel: True)

    def invalid(*args, **kwargs):
        raise ValueError("fixture unknown peer or invalid ACK")

    monkeypatch.setattr(host.native, "_submit_fact", invalid)
    with pytest.raises(ValueError, match="unknown peer"):
        host.native.submit_fact(
            Path("/fixture/socket"),
            {},
            expected_collector={"pid": 1},
            expected_socket_identity=(1, 1),
            expected_collector_source_sha256="fixture",
            expected_archive_identity={},
        )


def test_missing_boot_identity_is_unknown_and_cannot_prove_collector_gone(monkeypatch):
    collector = {
        "pid": 123,
        "start_ticks": 456,
        "session": 123,
        "start_identity": "pid:123:start_ticks:456",
        "boot_id": "fixture",
    }

    def unavailable(pid):
        raise FileNotFoundError(
            2, "fixture missing boot evidence", "/proc/sys/kernel/random/boot_id"
        )

    monkeypatch.setattr(host.native, "identity", unavailable)
    with pytest.raises(ValueError, match="outside exact proc stat"):
        host.same_kernel_alive(collector)


def test_session_change_of_same_live_kernel_is_not_death(monkeypatch):
    collector = {
        "pid": 123,
        "start_ticks": 456,
        "session": 123,
        "start_identity": "pid:123:start_ticks:456",
        "boot_id": "fixture",
    }
    monkeypatch.setattr(
        host.native, "identity", lambda pid: {**collector, "session": 999}
    )
    with pytest.raises(ValueError, match="live collector session"):
        host.same_kernel_alive(collector)
