# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in live transport epoch replacement, never a native completion proof.

The source-owned launch owner must preregister the immutable policy before first
publication. Control peers are pinned actual kernels, not claimed PID fields.
Callbacks remain registered once; their semantic/tensor transports swap jointly.
No fact is replayed and no attention inventory or drain history is reset.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import select
import socket
import stat
import struct
import threading
import time
from pathlib import Path

import regex as re

from . import native_semantic_collector as native

ENV = "ECPA_NATIVE_OBSERVER_REBIND_POLICY"
DIRECTORY_ENV = "ECPA_NATIVE_OBSERVER_REBIND_DIRECTORY"
bootstrap_lock = threading.RLock()
_bootstrap_pid = None
POLICY = "ecpa-native-observer-rebind-policy/v2"
COMMAND = "ecpa-native-observer-rebind-command/v1"
REPLY = "ecpa-native-observer-rebind-reply/v1"
DOMAIN = "original-H3-observer-loss-live-transport-only"
_router = None
_UNAVAILABLE = object()
_install_lock = threading.Lock()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def checked(reference):
    path = Path(reference["path"])
    require(
        path.is_absolute() and path.resolve() == path,
        "noncanonical private rebind input",
    )
    raw, info = native.read_private(path, 0o600)
    require(
        native.sha(raw) == reference["sha256"], "private rebind input bytes changed"
    )
    return json.loads(raw), raw, info


def same_kernel_alive(kernel):
    require(
        set(kernel) == {"pid", "start_ticks", "start_identity", "session", "boot_id"}
        and kernel["start_identity"]
        == f"pid:{kernel['pid']}:start_ticks:{kernel['start_ticks']}",
        "exact original native collector identity required",
    )
    try:
        actual = native.identity(kernel["pid"])
    except FileNotFoundError as error:
        # Missing boot/source/context files are integrity errors, not process death.
        require(
            error.filename == f"/proc/{kernel['pid']}/stat",
            "collector liveness proof unavailable outside exact proc stat",
        )
        return False
    except ProcessLookupError:
        require(
            not Path(f"/proc/{kernel['pid']}").exists(), "collector liveness is unknown"
        )
        return False
    same = all(actual[key] == kernel[key] for key in ("pid", "start_ticks", "boot_id"))
    if same:
        require(actual == kernel, "live collector session/identity metadata changed")
    return same


def ready_of(context, channel):
    return context if channel == "semantic" else context["ready"]


def probe(context, channel):
    """Authenticate the actual H3 successor, never the generic semantic ABI.

    Contexts pin the original native spec. The original H3 bootstrap puts its
    complete tensor/drain policies beside that spec; bind those actual bytes
    before accepting the collector's source-drain extension. This read-only
    status establishes transport ownership only, never native completion.
    """
    ready = ready_of(context, channel)
    spec_path = Path(context["spec_path"])
    spec, _, _ = checked({"path": str(spec_path), "sha256": context["spec_sha256"]})
    tensor_raw, _ = native.read_private(spec_path.parent / "tensor-policy.json", 0o600)
    drain_raw, _ = native.read_private(spec_path.parent / "drain-policy.json", 0o600)
    tensor, drain = json.loads(tensor_raw), json.loads(drain_raw)
    phase_ids = [phase["phase_id"] for phase in spec["phases"]]
    require(
        len(phase_ids) == 60
        and len(set(phase_ids)) == 60
        and tensor["native_spec_sha256"] == context["spec_sha256"]
        and drain
        == {
            "schema": "ecpa-h3-concurrent-native-drain-policy/v1",
            "native_spec_sha256": context["spec_sha256"],
            "tensor_policy_sha256": native.sha(tensor_raw),
            "phase_ids": phase_ids,
            "required_ranks": [0, 1],
            "formal_registration": False,
        },
        "original H3 successor spec/tensor/drain policy binding differs",
    )
    path = Path(ready["socket_path"])
    require(path.is_absolute() and path.resolve() == path, "noncanonical H3 socket")
    info, parent = path.lstat(), path.parent.lstat()
    archive = ready["archive_identity"]
    require(
        stat.S_ISSOCK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == 0o600
        and [info.st_dev, info.st_ino] == ready["socket_identity"]
        and stat.S_ISDIR(parent.st_mode)
        and parent.st_uid == os.getuid()
        and stat.S_IMODE(parent.st_mode) == 0o700
        and [parent.st_dev, parent.st_ino]
        == [archive["directory_device"], archive["directory_inode"]],
        "H3 successor socket/archive owner, mode or inode changed",
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(str(path))
        pid, uid, gid = struct.unpack(
            "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        require(
            uid == os.getuid()
            and gid == os.getgid()
            and native.identity(pid) == ready["collector"],
            "H3 successor status peer is not the exact frozen collector",
        )
        connection.sendall(native.canonical({"op": "describe"}) + b"\n")
        raw = b""
        while not raw.endswith(b"\n"):
            piece = connection.recv(native.MAX_FRAME + 1)
            require(
                piece and len(raw) + len(piece) <= native.MAX_FRAME,
                "H3 successor status incomplete or oversized",
            )
            raw += piece
    require(
        native.identity(pid) == ready["collector"],
        "H3 successor collector epoch changed during status authentication",
    )
    status = json.loads(raw)
    require(raw == native.canonical(status) + b"\n", "noncanonical H3 status")
    prefix, source_drain = status.get("acknowledged_prefix"), status.get("source_drain")
    require(
        isinstance(prefix, dict)
        and set(prefix) == {"next_sequence", "last_event_sha256"}
        and type(prefix["next_sequence"]) is int
        and prefix["next_sequence"] >= 1
        and type(prefix["last_event_sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", prefix["last_event_sha256"])
        and isinstance(source_drain, dict)
        and set(source_drain) == {"schema", "policy_sha256", "runtimes"}
        and source_drain["schema"] == drain["schema"]
        and source_drain["policy_sha256"] == native.sha(drain_raw)
        and isinstance(source_drain["runtimes"], dict),
        "H3 successor durable prefix or source-drain binding differs",
    )
    require(
        status.get("failed") is False
        and status.get("formal_producer_registration") is False
        and {
            k: v
            for k, v in status.items()
            if k not in {"acknowledged_prefix", "issued_monotonic_ns", "source_drain"}
        }
        == {
            "schema": "ecpa-h3-concurrent-native-collector-status/v1",
            "collector": ready["collector"],
            "socket_identity": ready["socket_identity"],
            "socket_path": str(path),
            "collector_source_sha256": ready["collector_source_sha256"],
            "spec_sha256": context["spec_sha256"],
            "archive_identity": archive,
            "classification": "REAL_RUNTIME_UNQUALIFIED",
            "formal_producer_registration": False,
            "failed": False,
        },
        "H3 successor status differs from frozen owner/source/spec/archive",
    )
    return status


def stable_spec(context):
    value, _, _ = checked(
        {"path": context["spec_path"], "sha256": context["spec_sha256"]}
    )
    # Actual freeze clock is observer-local. Every Plan/launch, sender, phase,
    # challenge nonce, layer/rank obligation and event-age/upload bound stays exact.
    require(type(value["frozen_monotonic_ns"]) is int, "collector freeze clock missing")
    return {key: item for key, item in value.items() if key != "frozen_monotonic_ns"}


def validate_successor(previous, successor, channel, policy):
    require(
        previous["source_files"] == successor["source_files"],
        "producer source closure changed",
    )
    old_spec, new_spec = stable_spec(previous), stable_spec(successor)
    require(
        old_spec == new_spec
        and old_spec["native_plan_id"] == policy["native_plan_id"]
        and old_spec["native_launch_id"] == policy["native_launch_id"]
        and len(old_spec["phases"]) == 60,
        "successor changed original full native scope",
    )
    old, new = ready_of(previous, channel), ready_of(successor, channel)
    require(
        old["collector"] != new["collector"]
        and old["archive_identity"] != new["archive_identity"]
        and old["socket_identity"] != new["socket_identity"]
        and old["socket_path"] != new["socket_path"],
        "successor reused observer/archive/socket epoch",
    )
    require(
        not same_kernel_alive(old["collector"]), "old collector kernel remains alive"
    )
    native.source_digests(successor["source_files"])
    if channel == "tensor":
        old_policy, _, _ = checked(
            {"path": previous["policy_path"], "sha256": previous["policy_sha256"]}
        )
        new_policy, _, _ = checked(
            {"path": successor["policy_path"], "sha256": successor["policy_sha256"]}
        )
        require(
            new_policy["native_spec_sha256"] == successor["spec_sha256"],
            "new tensor policy/spec differs",
        )
        require(
            {
                k: v
                for k, v in old_policy.items()
                if k not in {"native_spec_sha256", "worker_directories"}
            }
            == {
                k: v
                for k, v in new_policy.items()
                if k not in {"native_spec_sha256", "worker_directories"}
            },
            "tensor obligations changed during observer rebind",
        )
        require(
            set(new_policy["worker_directories"])
            == set(old_policy["worker_directories"]),
            "tensor sender membership changed",
        )
    probe(successor, channel)


class Router:
    def __init__(self, policy_path):
        self.policy_path = Path(policy_path)
        raw, info = native.read_private(self.policy_path, 0o600)
        self.policy_raw, self.policy_inode = raw, (info.st_dev, info.st_ino)
        self.policy = json.loads(raw)
        p = self.policy
        require(
            set(p)
            == {
                "schema",
                "domain",
                "start_id",
                "receipt_sha256",
                "producer",
                "controller",
                "controller_cmdline_sha256",
                "controller_source_files",
                "native_plan_id",
                "native_launch_id",
                "socket_path",
                "journal_path",
                "control_nonce",
                "initial_contexts",
                "max_recovery_ns",
                "ready_path",
                "producer_group",
                "producer_source_files",
            },
            "exact preregistered rebind policy required",
        )
        require(
            p["schema"] == POLICY
            and p["domain"] == DOMAIN
            and p["start_id"].startswith("observer-loss-r")
            and p["producer"] == native.identity(os.getpid())
            and p["max_recovery_ns"] == 30_000_000_000
            and re.fullmatch("[0-9a-f]{64}", p["control_nonce"]),
            "wrong actual original observer rebind scope",
        )
        require(
            set(p["initial_contexts"]) in ({"semantic"}, {"semantic", "tensor"}),
            "unsupported producer channels",
        )
        self.lock = threading.RLock()
        self.mapping_lock = threading.RLock()
        self.channels, self.pending, self.used = {}, None, set()
        self.generation, self.requests = 0, {}
        self.unavailable_requests = set()
        self.api_mappings, self.old_client_ids, self.pidfds = {}, set(), {}
        self.api_mapping_cache = {}
        self.server = None
        journal = Path(p["journal_path"])
        self._private_parent(journal)
        self.journal_fd = os.open(
            journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        self.current = {key: value for key, value in p["initial_contexts"].items()}

    @staticmethod
    def _private_parent(path):
        require(
            path.is_absolute() and path.resolve() == path,
            "private endpoint path must be canonical",
        )
        parent = path.parent.lstat()
        require(
            stat.S_ISDIR(parent.st_mode)
            and parent.st_uid == os.getuid()
            and stat.S_IMODE(parent.st_mode) == 0o700,
            "control endpoint parent must already be private700",
        )

    def _current_policy(self):
        raw, info = native.read_private(self.policy_path, 0o600)
        require(
            raw == self.policy_raw and (info.st_dev, info.st_ino) == self.policy_inode,
            "immutable rebind grant changed",
        )
        require(
            self.policy["producer"] == native.identity(os.getpid()),
            "producer epoch changed",
        )
        native.source_digests(self.policy["controller_source_files"])
        native.source_digests(self.policy["producer_source_files"])

    def _record(self, event):
        raw = native.canonical(event) + b"\n"
        view = memoryview(raw)
        while view:
            count = os.write(self.journal_fd, view)
            require(count > 0, "rebind journal short write")
            view = view[count:]
        os.fsync(self.journal_fd)

    def add(self, channel, context_path, callback, builder):
        with self.lock:
            self._current_policy()
            require(
                channel not in self.channels and channel in self.current,
                "channel installed twice or unregistered",
            )
            require(
                str(context_path) == self.current[channel]["path"],
                "initial context path differs from grant",
            )
            context, _, _ = checked(self.current[channel])
            self._capture_collector(ready_of(context, channel)["collector"])
            require(
                stable_spec(context)["native_plan_id"] == self.policy["native_plan_id"],
                "initial fullPlan differs",
            )
            self.channels[channel] = (callback, builder, context)
            if set(self.channels) == set(self.current):
                self._serve()

        def publication(fact):
            # Same order as the original semantic drain wrapper. Tensor's
            # acknowledged_publication reenters this RLock; never invert it.
            from .native_phase_drain import _lock as drain_lock

            with drain_lock, self.lock:
                self._current_policy()
                rid = fact.get("runtime_request_id") or fact.get("binding", {}).get(
                    "runtime_request_id"
                )
                if rid is not None:
                    require(
                        type(rid) is str and rid, "actual runtime request ID required"
                    )
                    require(
                        rid in self.requests or len(self.requests) < 4096,
                        "bounded request epoch map exhausted",
                    )
                    own = next(
                        member
                        for member in self.policy["producer_group"]
                        if member["producer"] == self.policy["producer"]
                    )
                    if own["role"] == "api-server":
                        require(
                            fact.get("schema") == "ecpa-native-request-id-mapping/v1",
                            "API epoch binding requires actual source mapping fact",
                        )
                        spec = stable_spec(self.channels["semantic"][2])
                        phases = [
                            phase
                            for phase in spec["phases"]
                            if phase["external_request_id"]
                            == fact.get("external_request_id")
                        ]
                        require(
                            len(phases) == 1,
                            "API mapping is not one frozen original request",
                        )
                        sender = next(
                            row
                            for row in spec["senders"]
                            if row["process"]["pid"] == os.getpid()
                        )
                        require(
                            fact["process"] == sender["process"],
                            "API epoch mapping source process changed",
                        )
                        client_id = phases[0]["client_request_id"]
                        generation = (
                            0 if client_id in self.old_client_ids else self.generation
                        )
                        mapping = dict(
                            runtime_request_id=rid,
                            client_request_id=client_id,
                            generation=generation,
                            source_fact_sha256=native.sha(native.canonical(fact)),
                        )
                        with self.mapping_lock:
                            require(
                                rid not in self.api_mappings,
                                "actual API runtime ID mapping replayed",
                            )
                            self.api_mappings[rid] = mapping
                    else:
                        if rid not in self.api_mapping_cache:
                            resolved = self._resolve_api_mapping(rid)
                            self.api_mapping_cache[rid] = json.loads(
                                native.canonical(resolved)
                            )
                        mapping = self.api_mapping_cache[rid]
                        generation = mapping["generation"]
                    assigned = self.requests.setdefault(rid, generation)
                    require(
                        assigned == generation,
                        "actual API transport epoch changed for existing "
                        "runtime request",
                    )
                    if assigned != self.generation or rid in self.unavailable_requests:
                        self._unavailable(
                            rid, channel, "in-flight old epoch; never replay"
                        )
                        return _UNAVAILABLE
                try:
                    return self.channels[channel][0](fact)
                except native.CollectorKernelGone as error:
                    current = ready_of(self.channels[channel][2], channel)
                    require(
                        error.expected_collector == current["collector"],
                        "typed collector-loss diagnostic refers to another epoch",
                    )
                    # A live collector error, malformed source, rejected fact,
                    # wrong peer or bad ACK must retain its original exception.
                    if not self._collector_gone(current["collector"]):
                        raise
                    require(
                        rid is not None,
                        "collector loss without an actual runtime request",
                    )
                    self._unavailable(
                        rid, channel, "pinned collector kernel actually gone"
                    )
                    return _UNAVAILABLE

        return publication

    def _capture_collector(self, collector):
        key = native.canonical(collector)
        if key in self.pidfds:
            return
        require(
            same_kernel_alive(collector),
            "collector must be exact live epoch before pidfd capture",
        )
        fd = os.pidfd_open(collector["pid"], 0)
        try:
            require(
                same_kernel_alive(collector),
                "collector epoch changed during pidfd capture",
            )
        except BaseException:
            os.close(fd)
            raise
        self.pidfds[key] = fd

    def _collector_gone(self, collector):
        fd = self.pidfds.get(native.canonical(collector))
        if fd is None:
            raise ValueError("exact preregistered collector pidfd unavailable")
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        events = poll.poll(0)
        require(
            not any(mask & (select.POLLERR | select.POLLNVAL) for _, mask in events),
            "collector pidfd death witness is invalid/unknown",
        )
        return any(mask & select.POLLIN for _, mask in events)

    def _resolve_api_mapping(self, rid):
        api = next(
            member
            for member in self.policy["producer_group"]
            if member["role"] == "api-server"
        )
        ready, _, _ = checked(
            {
                "path": api["ready_path"],
                "sha256": native.sha(
                    native.read_private(Path(api["ready_path"]), 0o600)[0]
                ),
            }
        )
        # The file cannot authorize readiness; the connected source-owned API
        # kernel and challenge echo independently authenticate every lookup.
        info = Path(api["socket_path"]).lstat()
        require(
            ready["producer"] == api["producer"]
            and ready["socket_identity"] == [info.st_dev, info.st_ino],
            "API mapping endpoint changed from actual ready inode",
        )
        nonce = secrets.token_hex(32)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(1)
            channel.connect(api["socket_path"])
            pid, uid, gid = struct.unpack(
                "3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            require(
                uid == os.getuid()
                and gid == os.getgid()
                and native.identity(pid) == api["producer"],
                "mapping resolver peer is not original actual API kernel",
            )
            command = dict(
                schema=COMMAND,
                domain=DOMAIN,
                operation="resolve",
                start_id=self.policy["start_id"],
                receipt_sha256=self.policy["receipt_sha256"],
                producer=self.policy["producer"],
                runtime_request_id=rid,
                challenge_nonce=nonce,
            )
            channel.sendall(native.canonical(command) + b"\n")
            raw = b""
            while not raw.endswith(b"\n"):
                piece = channel.recv(native.MAX_FRAME + 1)
                require(
                    piece and len(raw) + len(piece) <= native.MAX_FRAME,
                    "API mapping response unavailable/oversized",
                )
                raw += piece
        response = json.loads(raw)
        require(
            raw == native.canonical(response) + b"\n"
            and response["state"] == "resolved"
            and response["producer"] == api["producer"]
            and response["challenge_nonce"] == nonce
            and response["mapping"]["runtime_request_id"] == rid,
            "actual API mapping/nonce differs",
        )
        mapping = response["mapping"]
        require(
            set(mapping)
            == {
                "runtime_request_id",
                "client_request_id",
                "generation",
                "source_fact_sha256",
            }
            and type(mapping["generation"]) is int
            and mapping["generation"] in (0, 1)
            and mapping["client_request_id"]
            in {
                p["client_request_id"]
                for p in stable_spec(self.channels["semantic"][2])["phases"]
            }
            and re.fullmatch("[0-9a-f]{64}", mapping["source_fact_sha256"]),
            "API epoch mapping is outside exact original60/source digest domain",
        )
        return mapping

    def _unavailable(self, rid, channel, reason):
        if rid not in self.unavailable_requests:
            self._record(
                dict(
                    kind="native.observation.unavailable",
                    runtime_request_id=rid,
                    producer=self.policy["producer"],
                    generation=self.requests[rid],
                    channel=channel,
                    reason=reason,
                    observed_ns=time.monotonic_ns(),
                    ACK=False,
                    native_completion_proved=False,
                )
            )
            self.unavailable_requests.add(rid)

    def _peer(self, connection):
        pid, uid, gid = struct.unpack(
            "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        require(
            uid == os.getuid() and gid == os.getgid(), "control peer uid/gid differs"
        )
        kernel = native.identity(pid)
        allowed = [self.policy["controller"]] + [
            m["producer"] for m in self.policy["producer_group"]
        ]
        require(
            kernel in allowed,
            "control peer differs from original authority/Host kernels",
        )
        for member in self.policy["producer_group"]:
            if member["producer"] == kernel:
                require(
                    native.sha(Path(f"/proc/{pid}/cmdline").read_bytes())
                    == member["cmdline_sha256"],
                    "actual Host mapping peer complete argv changed",
                )
                native.source_digests(member["source_files"])
        self._current_policy()
        return kernel

    def _read_only(self, command, peer):
        operation = command["operation"]
        common = {
            "schema",
            "domain",
            "operation",
            "start_id",
            "receipt_sha256",
            "producer",
            "challenge_nonce",
        }
        require(
            set(command)
            == common
            | ({"runtime_request_id"} if operation == "resolve" else {"control_nonce"})
            and command["schema"] == COMMAND
            and command["domain"] == DOMAIN
            and command["start_id"] == self.policy["start_id"]
            and command["receipt_sha256"] == self.policy["receipt_sha256"]
            and command["producer"] == peer
            and re.fullmatch("[0-9a-f]{64}", command["challenge_nonce"]),
            "read-only source epoch request differs",
        )
        if operation == "resolve":
            # Mapping records are immutable once actual API source publishes.
            # Resolve never waits for the transport commit lock/probe: a queued
            # old request must not kill a healthy worker during collector repair.
            with self.mapping_lock:
                self._current_policy()
                require(
                    peer
                    in [
                        m["producer"]
                        for m in self.policy["producer_group"]
                        if m["role"] == "worker"
                    ],
                    "only actual original worker may resolve API epoch",
                )
                require(
                    command["runtime_request_id"] in self.api_mappings,
                    "actual source API mapping not available",
                )
                return dict(
                    state="resolved",
                    producer=self.policy["producer"],
                    challenge_nonce=command["challenge_nonce"],
                    mapping=self.api_mappings[command["runtime_request_id"]],
                )
        with self.lock:
            self._current_policy()
            require(
                operation == "describe"
                and peer == self.policy["controller"]
                and command["control_nonce"] == self.policy["control_nonce"],
                "only original authority may attest producer readiness",
            )
            return dict(
                schema=REPLY,
                domain=DOMAIN,
                state="described",
                producer=self.policy["producer"],
                receipt_sha256=self.policy["receipt_sha256"],
                start_id=self.policy["start_id"],
                controller=self.policy["controller"],
                challenge_nonce=command["challenge_nonce"],
                generation=self.generation,
                contexts=self.current,
                policy_sha256=native.sha(self.policy_raw),
                socket_identity=self.socket_identity,
                source_sha256=native.sha(Path(__file__).read_bytes()),
                native_completion_proved=False,
                observed_ns=time.monotonic_ns(),
            )

    def apply(self, command):
        # Called only after actual SO_PEERCRED authentication by _serve.
        with self.lock:
            self._current_policy()
            p = self.policy
            common = {
                "schema",
                "domain",
                "operation",
                "start_id",
                "receipt_sha256",
                "control_nonce",
                "producer",
                "expected_generation",
                "transition_nonce",
                "fault_ns",
                "deadline_ns",
                "old_client_request_ids",
            }
            operation = command.get("operation")
            require(
                operation in {"prepare", "commit"}
                and set(command)
                == common
                | ({"contexts"} if operation == "prepare" else {"prepared_sha256"}),
                "rebind control fields unsupported",
            )
            require(
                command["schema"] == COMMAND
                and command["domain"] == DOMAIN
                and all(
                    command[k] == p[k]
                    for k in ("start_id", "receipt_sha256", "control_nonce", "producer")
                )
                and command["expected_generation"] == self.generation
                and type(command["fault_ns"]) is int
                and type(command["deadline_ns"]) is int
                and 0
                < command["fault_ns"]
                <= time.monotonic_ns()
                < command["deadline_ns"]
                and command["deadline_ns"] == command["fault_ns"] + p["max_recovery_ns"]
                and re.fullmatch("[0-9a-f]{64}", command["transition_nonce"]),
                "rebind epoch/receipt/nonce/original30s fence differs",
            )
            if operation == "prepare":
                require(
                    self.pending is None
                    and command["transition_nonce"] not in self.used,
                    "prepared/replayed observer transition",
                )
                require(
                    set(command["contexts"]) == set(self.channels),
                    "semantic and tensor must prepare jointly",
                )
                require(
                    type(command["old_client_request_ids"]) is list
                    and len(set(command["old_client_request_ids"]))
                    == len(command["old_client_request_ids"]),
                    "frozen original admission IDs repeated",
                )
                expected_ids = {
                    phase["client_request_id"]
                    for phase in stable_spec(self.channels["semantic"][2])["phases"]
                }
                require(
                    set(command["old_client_request_ids"]) <= expected_ids
                    and command["old_client_request_ids"]
                    == sorted(command["old_client_request_ids"]),
                    "old original admissions outside exact frozen60 declaration",
                )
                self.old_client_ids.update(command["old_client_request_ids"])
                next_channels = {}
                collector = None
                for channel, ref in command["contexts"].items():
                    value, _, _ = checked(ref)
                    validate_successor(self.channels[channel][2], value, channel, p)
                    ready = ready_of(value, channel)
                    require(
                        collector is None or collector == ready["collector"],
                        "split semantic/tensor successor collector",
                    )
                    collector = ready["collector"]
                    self._capture_collector(collector)
                    builder = self.channels[channel][1]
                    next_channels[channel] = (
                        builder(Path(ref["path"])),
                        builder,
                        value,
                    )
                acknowledgement = self._ack(command, "prepared", collector)
                self._record(acknowledgement)
                self.used.add(command["transition_nonce"])
                self.pending = (command, next_channels, acknowledgement)
                return acknowledgement
            pending = self.pending
            if pending is None:
                raise ValueError("no prepared successor")
            prepared, next_channels, acknowledgement = pending
            require(
                all(command[k] == prepared[k] for k in common if k != "operation")
                and command["prepared_sha256"]
                == native.sha(native.canonical(acknowledgement)),
                "commit differs from exact prepared original transition",
            )
            for channel, (_, _, value) in next_channels.items():
                validate_successor(self.channels[channel][2], value, channel, p)
            # All in-progress publication calls share this lock and have finished.
            # No retry, phase-drain reset, materialization reset or callback
            # registration.
            self.old_client_ids.update(command["old_client_request_ids"])
            self.channels = next_channels
            self.current = prepared["contexts"]
            self.generation += 1
            self.pending = None
            committed = self._ack(command, "committed", acknowledgement["collector"])
            self._record(committed)
            return committed

    def _ack(self, command, state, collector):
        return dict(
            schema=REPLY,
            domain=DOMAIN,
            state=state,
            producer=self.policy["producer"],
            start_id=self.policy["start_id"],
            receipt_sha256=self.policy["receipt_sha256"],
            transition_nonce=command["transition_nonce"],
            generation=self.generation,
            collector=collector,
            observed_ns=time.monotonic_ns(),
            contexts=command.get("contexts", self.current),
            old_client_request_ids=command["old_client_request_ids"],
            native_completion_proved=False,
        )

    def _serve(self):
        path = Path(self.policy["socket_path"])
        self._private_parent(path)
        raw_path = os.fsencode(str(path))
        require(
            path.is_absolute() and b"\0" not in raw_path and len(raw_path) <= 107,
            "actual producer control pathname exceeds Linux sun_path byte bound",
        )
        require(
            self.policy["producer"] == native.identity(os.getpid()),
            "actual control bind producer epoch differs from immutable grant",
        )
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))  # Existing path always rejects; never unlink/reuse.
        os.chmod(path, 0o600)
        server.listen(4)
        meta = path.lstat()
        self.socket_identity = [meta.st_dev, meta.st_ino]
        ready = dict(
            schema="ecpa-native-observer-control-ready/v2",
            producer=self.policy["producer"],
            socket_identity=self.socket_identity,
            socket_path=str(path),
            policy_sha256=native.sha(self.policy_raw),
            receipt_sha256=self.policy["receipt_sha256"],
        )
        ready_path = Path(self.policy["ready_path"])
        self._private_parent(ready_path)
        fd = os.open(
            ready_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        try:
            raw = native.canonical(ready)
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                require(count > 0, "ready publication short write")
                view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        self.server = server
        self._record(
            dict(
                kind="producer.rebind-control.ready",
                producer=self.policy["producer"],
                socket_path=str(path),
                observed_ns=time.monotonic_ns(),
            )
        )
        slots = threading.BoundedSemaphore(4)

        def handle(connection):
            try:
                with connection:
                    connection.settimeout(1)
                    try:
                        peer = self._peer(connection)
                        raw = b""
                        while not raw.endswith(b"\n"):
                            piece = connection.recv(native.MAX_FRAME + 1)
                            require(
                                piece and len(raw) + len(piece) <= native.MAX_FRAME,
                                "bounded canonical command required",
                            )
                            raw += piece
                        value = json.loads(raw)
                        require(
                            raw == native.canonical(value) + b"\n",
                            "noncanonical rebind command",
                        )
                        if value.get("operation") in {"describe", "resolve"}:
                            response = self._read_only(value, peer)
                        else:
                            require(
                                peer == self.policy["controller"],
                                "only authority may mutate observer epoch",
                            )
                            require(
                                native.sha(
                                    Path(f"/proc/{peer['pid']}/cmdline").read_bytes()
                                )
                                == self.policy["controller_cmdline_sha256"],
                                "authority complete argv changed",
                            )
                            response = self.apply(value)
                    except Exception as error:
                        response = dict(
                            schema=REPLY,
                            state="rejected",
                            reason=str(error),
                            native_completion_proved=False,
                        )
                    # A missing ACK leaves authority route closed.
                    with contextlib.suppress(OSError):
                        connection.sendall(native.canonical(response) + b"\n")
            finally:
                slots.release()

        def loop():
            while True:
                connection, _ = server.accept()
                if not slots.acquire(blocking=False):
                    connection.close()
                    continue
                threading.Thread(target=handle, args=(connection,), daemon=True).start()

        threading.Thread(
            target=loop, name="native-observer-rebind-control", daemon=True
        ).start()


def install(channel, context_path, callback, builder):
    path = os.environ.get(ENV)
    directory = os.environ.get(DIRECTORY_ENV)
    require(not (path and directory), "ambiguous old/new observer rebind bootstrap")
    if directory:
        root = Path(directory)
        Router._private_parent(root / "grant")
        path = str(root / f"{os.getpid()}.json")
    if not path:
        return callback
    global _router
    with _install_lock:
        if _router is None:
            _router = Router(Path(path))
        require(_router.policy_path == Path(path), "producer rebind policy changed")
        return _router.add(channel, context_path, callback, builder)


def is_unavailable(value):
    """A source-local loss token, deliberately never an ACK dictionary."""
    return value is _UNAVAILABLE


def request_unavailable(runtime_request_id):
    if _router is None:
        return False
    with _router.lock:
        return runtime_request_id in _router.unavailable_requests


def begin_host_bootstrap(role, ordinal):
    """Called by actual Host identity binding, before public admission exists."""
    directory = os.environ.get(DIRECTORY_ENV)
    if not directory or (role, ordinal) not in {
        ("api-server", 0),
        ("worker", 0),
        ("worker", 1),
    }:
        return
    global _bootstrap_pid
    with bootstrap_lock:
        if _bootstrap_pid == os.getpid():
            return
        _bootstrap_pid = os.getpid()

    def initialize():
        grant = Path(directory) / f"{os.getpid()}.json"
        # Existence is only a wakeup. Private bytes, actual kernels/source,
        # callback installation and independently authenticated describe are gates.
        while not grant.exists():
            time.sleep(0.02)
        try:
            policy_raw, _ = native.read_private(grant, 0o600)
            policy = json.loads(policy_raw)
            for reference in policy["initial_contexts"].values():
                while not Path(reference["path"]).exists():
                    time.sleep(0.02)
            with bootstrap_lock:
                from . import native_block_table_observer as semantic
                from .native_semantic_bootstrap import ensure_registered

                ensure_registered(semantic.register_observer)
                if role == "worker":
                    from . import native_tensor_witness as witness
                    from .native_tensor_artifact_bootstrap import (
                        ensure_registered as tensor,
                    )

                    tensor(witness.register_candidate_sink)
        except Exception as error:
            # No control-ready/describe can pass; original launcher remains closed.
            # The failure is retained, never converted into source readiness.
            error_path = Path(directory) / f"{os.getpid()}.bootstrap-error.json"
            fd = os.open(
                error_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            try:
                os.write(
                    fd,
                    native.canonical(
                        {
                            "producer": native.identity(os.getpid()),
                            "error": type(error).__name__ + ": " + str(error),
                            "native_completion_proved": False,
                        }
                    ),
                )
                os.fsync(fd)
            finally:
                os.close(fd)

    threading.Thread(
        target=initialize, name="native-observer-source-bootstrap", daemon=True
    ).start()


def collector_kernel_gone(collector):
    """A pinned pidfd proves exit even before the authority reaps a zombie."""
    if _router is None:
        return False
    with _router.lock:
        return _router._collector_gone(collector)
