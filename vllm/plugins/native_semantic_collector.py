# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Private Unix authority-owned raw collector; qualification candidate only.

Kernel peer identity plus frozen source and challenge bindings precede admission
of raw facts. Native fact authenticity still requires the newly built source
candidate and real qualification; this module never opens a formal traffic gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import socketserver
import stat
import struct
import threading
import time
from pathlib import Path
from typing import Any

import pybase64 as base64
import regex as re

MAX_FRAME = 65536
MAPPING = "ecpa-native-request-id-mapping/v1"
UPLOAD = "ecpa-native-block-table-upload/v1"
SPEC = "ecpa-native-semantic-collector-prereg/v1"
ACK = "ecpa-native-semantic-collector-ack/v1"
KV_SPEC = "ecpa-native-kv-semantic-collector-prereg/v1"
PROJECTION = "ecpa-native-kv-blocktable-projection/v1"
PROJECTION_V2 = "ecpa-native-kv-blocktable-projection/v2"
KV_SPEC_V2 = "ecpa-native-kv-semantic-collector-prereg/v2"
PROJECTION_OPERATION = "KVCompress-worker-host-blocktable-projection-applied"
PROJECTION_SCOPE = "compression-host-metadata-and-uploaded-destination-binding-only"


def projection_predicate(fact, requirement):
    transaction_keys = {
        "runtime_request_id",
        "attention_group",
        "semantic_anchor",
        "physical_anchor",
        "destination_block_ids",
        "host_cpu_block_ids",
        "allocation_block_size",
        "compression_threshold_tokens",
    }
    materialized = fact.get("schema") == PROJECTION_V2
    if materialized:
        transaction_keys |= {
            "scheduled_appended_block_ids",
            "metadata_effect_monotonic_ns",
        }
    exact = transaction_keys | {
        "schema",
        "process",
        "launch_id",
        "plan_id",
        "monotonic_ns",
        "metadata_row",
        "transaction_id",
        "operation",
        "scope",
        "host_mapping",
        "kv_tensor_equivalence",
        "attention_consumed_observed_tensor",
        "full_generation_quality",
    }
    if set(fact) != exact:
        raise ValueError("KV projection fields are unsupported")
    blocks = fact["destination_block_ids"]
    appended = fact.get("scheduled_appended_block_ids", [])
    if materialized and (
        not isinstance(appended, list)
        or any(type(x) is not int or x < 0 for x in appended)
        or type(fact["metadata_effect_monotonic_ns"]) is not int
        or not 0 < fact["metadata_effect_monotonic_ns"] <= fact["monotonic_ns"]
    ):
        raise ValueError("materialized projection lacks actual append/effect clock")
    integer_keys = [
        "attention_group",
        "semantic_anchor",
        "physical_anchor",
        "allocation_block_size",
        "compression_threshold_tokens",
        "metadata_row",
    ]
    if (
        any(type(fact[k]) is not int or fact[k] < 0 for k in integer_keys)
        or not isinstance(blocks, list)
        or not blocks
        or any(type(x) is not int or x < 0 for x in blocks)
        or len(set(blocks)) != len(blocks)
        or fact["host_cpu_block_ids"] != blocks + appended
        or len(set(blocks + appended)) != len(blocks + appended)
        or not 0
        < fact["physical_anchor"]
        < fact["compression_threshold_tokens"]
        <= fact["semantic_anchor"]
        or fact["allocation_block_size"] <= 0
        or fact["physical_anchor"] != len(blocks) * fact["allocation_block_size"]
        or any(
            fact[k] != requirement[k]
            for k in [
                "attention_group",
                "semantic_anchor",
                "physical_anchor",
                "allocation_block_size",
                "compression_threshold_tokens",
            ]
        )
        or len(blocks) != requirement["destination_block_count"]
        or fact["operation"] != PROJECTION_OPERATION
        or fact["scope"] != PROJECTION_SCOPE
        or fact["host_mapping"]
        != (
            "identity-host-block-id-projection-with-scheduled-append"
            if materialized
            else "identity-host-block-id-projection"
        )
        or any(
            fact[k] is not None
            for k in [
                "kv_tensor_equivalence",
                "attention_consumed_observed_tensor",
                "full_generation_quality",
            ]
        )
        or fact["transaction_id"]
        != sha(canonical({k: fact[k] for k in transaction_keys}))
    ):
        raise ValueError("KV projection lacks frozen actual destination/anchor binding")


def projection_upload_match(projection, upload):
    """Only later same-worker destination-prefix binding; no tensor equivalence."""
    if (
        projection["process"] != upload["process"]
        or any(
            projection[k] != upload[k]
            for k in ["runtime_request_id", "launch_id", "plan_id"]
        )
        or upload["monotonic_ns"] < projection["monotonic_ns"]
        or upload["observed_after_barrier_monotonic_ns"] < projection["monotonic_ns"]
    ):
        return False
    destination = projection["host_cpu_block_ids"]
    return any(
        row["group"] == projection["attention_group"]
        and row["blocks"][: len(destination)] == destination
        for row in upload["uploaded_rows"]
    )


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def identity(pid: int) -> dict:
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    fields = raw[raw.rfind(b")") + 2 :].split()
    ticks = int(fields[19])
    return {
        "pid": pid,
        "start_ticks": ticks,
        "start_identity": f"pid:{pid}:start_ticks:{ticks}",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "session": os.getsid(pid),
    }


def read_private(path: Path, mode: int) -> tuple[bytes, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != mode
            or metadata.st_size > MAX_FRAME
        ):
            raise ValueError("frozen input is not a bounded private regular file")
        return os.read(fd, MAX_FRAME + 1), metadata
    finally:
        os.close(fd)


def source_digests(files: list[dict]) -> dict:
    if not files:
        raise ValueError("sender source bytes are unavailable")
    result = {}
    for binding in files:
        path = Path(binding["path"])
        if not path.is_absolute() or path.is_symlink() or str(path) in result:
            raise ValueError("sender source path is not immutable and unique")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("sender source is not a regular file")
            chunks = []
            while chunk := os.read(fd, MAX_FRAME):
                chunks.append(chunk)
            actual = sha(b"".join(chunks))
        finally:
            os.close(fd)
        if actual != binding["sha256"]:
            raise ValueError("sender source bytes changed")
        result[str(path)] = actual
    return result


def native_fact_predicate(
    raw: bytes,
    sender: dict,
    spec: dict,
    *,
    accepted_monotonic_ns: int | None,
    mappings: dict,
    mapping_stamps: dict,
    upload_stamps: dict,
    upload_counts: dict,
) -> tuple:
    """Same pure source/field/predicate/chronology gate for live and raw replay.

    None accept time retains legacy static validation, never full chronology.
    This function performs no filesystem, PID, clock or network reads.
    """
    if accepted_monotonic_ns is not None and (
        type(accepted_monotonic_ns) is not int or accepted_monotonic_ns <= 0
    ):
        raise ValueError("native collector accept clock is invalid")
    phases = {x["external_request_id"]: x for x in spec["phases"]}
    fact = json.loads(raw)
    if not isinstance(fact, dict):
        raise ValueError("native raw fact is not an object")
    common_fields = {
        "schema",
        "process",
        "launch_id",
        "plan_id",
        "monotonic_ns",
    }
    mapping_fields = common_fields | {
        "external_request_id",
        "runtime_request_id",
    }
    upload_fields = common_fields | {
        "observed_after_barrier_monotonic_ns",
        "runtime_request_id",
        "device",
        "uploaded_rows",
        "uploaded_rows_sha256",
        "scope",
        "kv_tensor_equivalence",
        "attention_consumed_observed_tensor",
        "full_generation_quality",
    }
    if set(fact) != (
        mapping_fields
        if fact.get("schema") == MAPPING
        else (
            set(fact)
            if fact.get("schema") in {PROJECTION, PROJECTION_V2}
            else upload_fields
        )
    ):
        raise ValueError("native fact fields do not match a supported source schema")
    if set(fact.get("process", {})) != set(sender["process"]):
        raise ValueError("native process fields differ from frozen Host source")
    if canonical(fact) != raw:
        raise ValueError("native raw bytes are noncanonical or replayed")
    if any(fact.get("process", {}).get(k) != v for k, v in sender["process"].items()):
        raise ValueError("native sender role/process differs from kernel registration")
    if (
        fact.get("plan_id") != spec["native_plan_id"]
        or fact.get("launch_id") != spec["native_launch_id"]
    ):
        raise ValueError("native plan/launch differs from frozen scope")
    now = accepted_monotonic_ns
    stamp = fact.get("monotonic_ns")
    if (
        type(stamp) is not int
        or stamp < spec["frozen_monotonic_ns"]
        or (now is not None and (stamp > now or now - stamp > spec["max_event_age_ns"]))
    ):
        raise ValueError("native event is stale or from another monotonic epoch")
    runtime = fact.get("runtime_request_id")
    if not isinstance(runtime, str) or not runtime:
        raise ValueError("native runtime request identity is absent")
    if fact.get("schema") == MAPPING:
        if sender["process"]["role"] != "api-server":
            raise ValueError("request mapping sender is not native API")
        external = fact.get("external_request_id")
        phase = phases.get(external)
        if (
            phase is None
            or runtime in mappings
            or any(x["external_request_id"] == external for x in mappings.values())
        ):
            raise ValueError("actual native API mapping is unknown or repeated")
        update = ("mapping", runtime)
    elif fact.get("schema") in {PROJECTION, PROJECTION_V2}:
        if (
            spec.get("schema")
            != (KV_SPEC_V2 if fact["schema"] == PROJECTION_V2 else KV_SPEC)
            or sender["process"]["role"] != "worker"
            or runtime not in mappings
        ):
            raise ValueError(
                "KV projection lacks opt-in scope/native API mapping/worker source"
            )
        phase = mappings[runtime]
        if stamp < mapping_stamps[runtime]:
            raise ValueError("KV projection predates actual native API mapping")
        projection_predicate(fact, spec["kv_requirements"][phase["phase_id"]])
        if (
            fact["schema"] == PROJECTION_V2
            and not mapping_stamps[runtime]
            <= fact.get("metadata_effect_monotonic_ns", 0)
            <= stamp
        ):
            raise ValueError("materialized projection effect/row chronology differs")
        if upload_counts.get(("projection", runtime, sender["process"]["ordinal"]), 0):
            raise ValueError(
                "KV phase permits one actual projection per registered rank"
            )
        update = ("projection", sender["process"]["ordinal"])
    elif fact.get("schema") == UPLOAD:
        if sender["process"]["role"] != "worker" or runtime not in mappings:
            raise ValueError("GPU upload lacks native API mapping or worker source")
        phase = mappings[runtime]
        if stamp < mapping_stamps[runtime]:
            raise ValueError("upload predates actual native API request mapping")
        rank = sender["process"]["ordinal"]
        if upload_counts.get((runtime, rank), 0) >= spec["max_uploads_per_rank"]:
            raise ValueError("native upload exceeds frozen finite per-rank bound")
        after = fact.get("observed_after_barrier_monotonic_ns")
        if type(after) is int and after <= upload_stamps.get((runtime, rank), 0):
            raise ValueError("native upload barrier timestamp is replayed")
        rows = fact.get("uploaded_rows")
        if (
            type(after) is not int
            or not stamp <= after
            or (now is not None and after > now)
            or not isinstance(rows, list)
            or not rows
            or fact.get("uploaded_rows_sha256") != sha(canonical(rows))
            or not isinstance(fact.get("device"), str)
            or re.fullmatch(r"npu:[0-9]+", fact["device"]) is None
            or fact.get("scope") != "NPU-block-table-prefix-readback-only"
            or any(
                k not in fact or fact[k] is not None
                for k in (
                    "kv_tensor_equivalence",
                    "attention_consumed_observed_tensor",
                    "full_generation_quality",
                )
            )
        ):
            raise ValueError("native uploaded-row evidence is unsupported or promoted")
        slots = set()
        for row in rows:
            if (
                set(row) != {"group", "row", "blocks"}
                or type(row["group"]) is not int
                or row["group"] < 0
                or type(row["row"]) is not int
                or row["row"] < 0
                or not isinstance(row["blocks"], list)
                or not row["blocks"]
                or any(type(x) is not int or x < 0 for x in row["blocks"])
                or (row["group"], row["row"]) in slots
            ):
                raise ValueError(
                    "native uploaded row has no exact physical block binding"
                )
            slots.add((row["group"], row["row"]))
        update = ("upload", rank)
    else:
        raise ValueError("native schema is unsupported")
    return fact, phase, update


class Collector:
    def __init__(self, spec_path: Path, journal_path: Path):
        raw, _ = read_private(spec_path, 0o600)
        self.spec = json.loads(raw)
        fields = {
            "schema",
            "classification",
            "formal_producer_registration",
            "native_plan_id",
            "native_launch_id",
            "frozen_monotonic_ns",
            "max_event_age_ns",
            "max_uploads_per_rank",
            "senders",
            "phases",
        }
        if self.spec.get("schema") in {KV_SPEC, KV_SPEC_V2}:
            fields.add("kv_requirements")
        if set(self.spec) != fields:
            raise ValueError("collector prereg fields are unsupported")
        self.spec_sha256 = sha(raw)
        self.source_sha256 = sha(Path(__file__).read_bytes())
        self.failed = False
        if (
            self.spec.get("schema") not in {SPEC, KV_SPEC, KV_SPEC_V2}
            or self.spec.get("classification") != "CPU_REVIEW_CANDIDATE_ONLY"
            or self.spec.get("formal_producer_registration") is not False
            or not self.spec.get("native_plan_id")
            or not self.spec.get("native_launch_id")
            or type(self.spec.get("frozen_monotonic_ns")) is not int
            or type(self.spec.get("max_event_age_ns")) is not int
            or self.spec["max_event_age_ns"] <= 0
            or type(self.spec.get("max_uploads_per_rank")) is not int
            or not 0 < self.spec["max_uploads_per_rank"] <= 4096
        ):
            raise ValueError("collector scope is incomplete or promoted")
        if self.spec["schema"] in {KV_SPEC, KV_SPEC_V2}:
            requirements = self.spec["kv_requirements"]
            if set(requirements) != {p["phase_id"] for p in self.spec["phases"]}:
                raise ValueError("KV projection phase preregistration is incomplete")
            for requirement in requirements.values():
                if set(requirement) != {
                    "attention_group",
                    "semantic_anchor",
                    "physical_anchor",
                    "allocation_block_size",
                    "compression_threshold_tokens",
                    "destination_block_count",
                } or any(type(v) is not int or v < 0 for v in requirement.values()):
                    raise ValueError("KV preregistration predicate is malformed")
                if (
                    not 0
                    < requirement["physical_anchor"]
                    < requirement["compression_threshold_tokens"]
                    <= requirement["semantic_anchor"]
                    or requirement["allocation_block_size"] <= 0
                    or requirement["destination_block_count"] <= 0
                    or requirement["physical_anchor"]
                    != requirement["destination_block_count"]
                    * requirement["allocation_block_size"]
                ):
                    raise ValueError("KV preregistration anchors are inconsistent")
        self.senders = {x["process"]["pid"]: x for x in self.spec["senders"]}
        if len(self.senders) != len(self.spec["senders"]):
            raise ValueError("sender PID is repeated")
        slots = set()
        for sender in self.senders.values():
            if set(sender) != {
                "process",
                "peer_uid",
                "peer_gid",
                "cmdline_sha256",
                "source_files",
            }:
                raise ValueError("sender prereg fields are unsupported")
            process = sender["process"]
            if set(process) != {
                "host",
                "role",
                "ordinal",
                "pid",
                "start_identity",
                "process_epoch",
                "assignment_source",
                "session",
                "start_ticks",
                "boot_id",
            }:
                raise ValueError("Host sender process fields are unsupported")
            if (
                process.get("role") not in {"api-server", "worker"}
                or type(process.get("ordinal")) is not int
                or type(process.get("process_epoch")) is not int
                or process.get("assignment_source") != "host"
            ):
                raise ValueError("sender role/epoch is unsupported")
            slot = (process["role"], process["ordinal"])
            if slot in slots:
                raise ValueError("sender role slot is repeated")
            slots.add(slot)
            if any(process.get(k) != v for k, v in identity(process["pid"]).items()):
                raise ValueError("sender process identity changed before freeze")
            if (
                type(sender["peer_uid"]) is not int
                or type(sender["peer_gid"]) is not int
                or sender["peer_uid"] != os.getuid()
                or sender["peer_gid"] != os.getgid()
            ):
                raise ValueError("sender UID differs from deployment")
            if sender["cmdline_sha256"] != sha(
                Path(f"/proc/{process['pid']}/cmdline").read_bytes()
            ):
                raise ValueError("sender argv differs from freeze")
            source_digests(sender["source_files"])
        if slots != {("api-server", 0), ("worker", 0), ("worker", 1)}:
            raise ValueError("exact API and two worker sender set is unavailable")
        self.phases: dict[Any, Any] = {}
        client_ids = set()
        nonces = set()
        phase_ids = set()
        for phase in self.spec["phases"]:
            if set(phase) != {
                "phase_id",
                "client_request_id",
                "challenge_nonce",
                "external_request_id",
                "expected_ranks",
            }:
                raise ValueError("challenge prereg fields are unsupported")
            if (
                not all(
                    isinstance(phase.get(k), str) and phase[k]
                    for k in (
                        "phase_id",
                        "client_request_id",
                        "challenge_nonce",
                        "external_request_id",
                    )
                )
                or phase.get("expected_ranks") != [0, 1]
                or phase["external_request_id"] in self.phases
                or phase["client_request_id"] in client_ids
                or phase["challenge_nonce"] in nonces
                or phase["phase_id"] in phase_ids
            ):
                raise ValueError(
                    "challenge phase/nonce/request binding is absent or repeated"
                )
            self.phases[phase["external_request_id"]] = phase
            client_ids.add(phase["client_request_id"])
            nonces.add(phase["challenge_nonce"])
            phase_ids.add(phase["phase_id"])
        if not self.phases:
            raise ValueError("challenge set is empty")
        parent = journal_path.parent.stat()
        if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700:
            raise ValueError("collector directory is not private")
        self.fd = os.open(
            journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        self.archive_identity = {
            "directory_device": parent.st_dev,
            "directory_inode": parent.st_ino,
            "journal_device": os.fstat(self.fd).st_dev,
            "journal_inode": os.fstat(self.fd).st_ino,
        }
        self.lock = threading.Lock()
        self.sequence = 0
        self.previous = "0" * 64
        self.maps: dict[Any, Any] = {}
        self.mapping_stamps: dict[Any, Any] = {}
        self.ranks: dict[Any, Any] = {}
        self.upload_stamps: dict[Any, Any] = {}
        self.upload_counts: dict[Any, Any] = {}
        self.seen_raw: set[Any] = set()
        self.collector_identity = identity(os.getpid())
        self._append(
            {
                "kind": "collector.start",
                "started_monotonic_ns": time.monotonic_ns(),
                "intake_predicate_schema": (
                    "ecpa-native-semantic-intake/v3"
                    if self.spec["schema"] == KV_SPEC_V2
                    else "ecpa-native-semantic-intake/v2"
                ),
                "spec_sha256": sha(raw),
                "collector": self.collector_identity,
                "collector_source_sha256": sha(Path(__file__).read_bytes()),
                "archive_identity": self.archive_identity,
            }
        )
        directory = os.open(journal_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _append(self, value: dict) -> dict:
        event = value | {"sequence": self.sequence, "previous_sha256": self.previous}
        event["event_sha256"] = sha(canonical(event))
        data = memoryview(canonical(event) + b"\n")
        try:
            while data:
                n = os.write(self.fd, data)
                if n <= 0:
                    raise OSError("collector journal write failed")
                data = data[n:]
            os.fsync(self.fd)
        except OSError:
            self.failed = True
            raise
        self.sequence += 1
        self.previous = event["event_sha256"]
        return event

    def accept(self, peer: tuple[int, int, int], raw: bytes) -> dict:
        with self.lock:
            if self.failed:
                raise ValueError(
                    "collector persistence failed; no further ACK is possible"
                )
            pid, uid, gid = peer
            sender = self.senders.get(pid)
            if (
                sender is None
                or sender["peer_uid"] != uid
                or uid != os.getuid()
                or sender["peer_gid"] != gid
            ):
                raise ValueError("kernel peer is not a registered source sender")
            actual = identity(pid)
            if any(sender["process"].get(k) != v for k, v in actual.items()):
                raise ValueError("kernel peer epoch/boot/session changed")
            if (
                sha(Path(f"/proc/{pid}/cmdline").read_bytes())
                != sender["cmdline_sha256"]
            ):
                raise ValueError("kernel peer argv changed")
            sources = source_digests(sender["source_files"])
            if sha(raw) in self.seen_raw:
                raise ValueError("native raw bytes are replayed")
            accepted_ns = time.monotonic_ns()
            fact, phase, update = native_fact_predicate(
                raw,
                sender,
                self.spec,
                accepted_monotonic_ns=accepted_ns,
                mappings=self.maps,
                mapping_stamps=self.mapping_stamps,
                upload_stamps=self.upload_stamps,
                upload_counts=self.upload_counts,
            )
            runtime = fact["runtime_request_id"]
            stamp = fact["monotonic_ns"]
            event = self._append(
                {
                    "kind": "native.accepted",
                    "accepted_monotonic_ns": accepted_ns,
                    "raw_base64": base64.b64encode(raw).decode(),
                    "raw_sha256": sha(raw),
                    "peer": {"pid": pid, "uid": uid, "gid": gid},
                    "source_sha256": sources,
                    "process": actual,
                    "phase": phase,
                    "runtime_request_id": runtime,
                    "archive_identity": self.archive_identity,
                }
            )
            self.seen_raw.add(sha(raw))
            if update[0] == "mapping":
                self.maps[runtime] = phase
                self.mapping_stamps[runtime] = stamp
            elif update[0] == "projection":
                self.upload_counts[("projection", runtime, update[1])] = 1
            elif update[0] == "upload":
                self.ranks.setdefault(runtime, set()).add(update[1])
                self.upload_stamps[(runtime, update[1])] = fact[
                    "observed_after_barrier_monotonic_ns"
                ]
                self.upload_counts[(runtime, update[1])] = (
                    self.upload_counts.get((runtime, update[1]), 0) + 1
                )
            ack = {
                "schema": ACK,
                "raw_sha256": sha(raw),
                "event_sha256": event["event_sha256"],
                "sequence": event["sequence"],
                "phase": phase,
                "collector": self.collector_identity,
                "collector_source_sha256": self.source_sha256,
                "archive_identity": self.archive_identity,
                "formal_producer_registration": False,
                "classification": "CPU_REVIEW_CANDIDATE_ONLY",
            }
            ack["ack_sha256"] = sha(canonical(ack))
            return ack


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, collector: Collector):
        self.collector = collector
        metadata = path.parent.stat()
        if (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (metadata.st_dev, metadata.st_ino)
            != (
                collector.archive_identity["directory_device"],
                collector.archive_identity["directory_inode"],
            )
        ):
            raise ValueError("collector socket directory differs from private archive")
        super().__init__(str(path), Handler)
        os.chmod(path, 0o600)


class Handler(socketserver.StreamRequestHandler):
    server: Any

    def handle(self):
        peer = struct.unpack(
            "3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        try:
            frame = self.rfile.readline(MAX_FRAME + 1)
            if not frame.endswith(b"\n") or len(frame) > MAX_FRAME:
                raise ValueError("native frame is incomplete or oversized")
            request = json.loads(frame)
            if request == {"op": "describe"}:
                collector = self.server.collector
                metadata = Path(self.server.server_address).stat()
                with collector.lock:
                    os.fsync(collector.fd)
                    issued_ns = time.monotonic_ns()
                    response = {
                        "schema": "ecpa-native-semantic-collector-status/v1",
                        "issued_monotonic_ns": issued_ns,
                        "spec_sha256": collector.spec_sha256,
                        "collector_source_sha256": collector.source_sha256,
                        "collector": collector.collector_identity,
                        "socket_identity": [metadata.st_dev, metadata.st_ino],
                        "socket_path": str(self.server.server_address),
                        "archive_identity": collector.archive_identity,
                        "classification": "CPU_REVIEW_CANDIDATE_ONLY",
                        "formal_producer_registration": False,
                        "failed": collector.failed,
                        "acknowledged_prefix": {
                            "next_sequence": collector.sequence,
                            "last_event_sha256": collector.previous,
                        },
                    }
                self.wfile.write(canonical(response) + b"\n")
                return
            if set(request) != {"raw_base64"}:
                raise ValueError("native frame carries unsupported fields")
            raw = base64.b64decode(request["raw_base64"], validate=True)
            response = self.server.collector.accept(peer, raw)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            response = {
                "schema": "ecpa-native-semantic-collector-reject/v1",
                "reason": str(exc),
                "formal_producer_registration": False,
            }
        self.wfile.write(canonical(response) + b"\n")


def submit_fact(
    path: Path,
    fact: dict,
    *,
    expected_collector: dict,
    expected_socket_identity: tuple[int, int],
    expected_collector_source_sha256: str,
    expected_archive_identity: dict,
) -> dict:
    metadata = path.stat()
    if (
        metadata.st_dev,
        metadata.st_ino,
    ) != expected_socket_identity or not stat.S_ISSOCK(metadata.st_mode):
        raise ValueError("native collector socket changed")
    raw = canonical(fact)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(5)
        channel.connect(str(path))
        pid, uid, _ = struct.unpack(
            "3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if uid != os.getuid() or identity(pid) != expected_collector:
            raise ValueError("ACK kernel peer is not the frozen authority process")
        channel.sendall(
            canonical({"raw_base64": base64.b64encode(raw).decode()}) + b"\n"
        )
        response = b""
        while not response.endswith(b"\n"):
            chunk = channel.recv(MAX_FRAME + 1)
            if not chunk or len(response) + len(chunk) > MAX_FRAME:
                raise ValueError("native ACK is incomplete or oversized")
            response += chunk
    ack = json.loads(response)
    if (
        ack.get("schema") != ACK
        or ack.get("raw_sha256") != sha(raw)
        or ack.get("collector") != expected_collector
        or ack.get("collector_source_sha256") != expected_collector_source_sha256
        or ack.get("archive_identity") != expected_archive_identity
        or ack.get("formal_producer_registration") is not False
        or ack.get("ack_sha256")
        != sha(canonical({k: v for k, v in ack.items() if k != "ack_sha256"}))
    ):
        raise ValueError(
            "native collector rejected fact or ACK digest differs: "
            + str(ack.get("reason"))
        )
    return ack


def probe_collector(
    path: Path,
    *,
    expected_collector: dict,
    expected_socket_identity: tuple[int, int],
    expected_collector_source_sha256: str,
    expected_spec_sha256: str,
    expected_archive_identity: dict,
) -> dict:
    """Read-only live owner seal; caller freezes identities from owned launch."""
    meta = path.stat()
    if (meta.st_dev, meta.st_ino) != expected_socket_identity or not stat.S_ISSOCK(
        meta.st_mode
    ):
        raise ValueError("collector status socket identity changed")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(5)
        channel.connect(str(path))
        pid, uid, _ = struct.unpack(
            "3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if uid != os.getuid() or identity(pid) != expected_collector:
            raise ValueError("collector status peer is not the frozen owner")
        channel.sendall(canonical({"op": "describe"}) + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = channel.recv(MAX_FRAME + 1)
            if not chunk or len(data) + len(chunk) > MAX_FRAME:
                raise ValueError("collector status is incomplete or oversized")
            data += chunk
    status = json.loads(data)
    if (
        status.get("schema") != "ecpa-native-semantic-collector-status/v1"
        or status.get("collector") != expected_collector
        or status.get("socket_identity") != list(expected_socket_identity)
        or status.get("collector_source_sha256") != expected_collector_source_sha256
        or status.get("spec_sha256") != expected_spec_sha256
        or status.get("archive_identity") != expected_archive_identity
        or status.get("failed") is not False
        or status.get("formal_producer_registration") is not False
    ):
        raise ValueError("collector status failed or differs from frozen owner/source")
    return status


def verify_journal(
    spec_path: Path, journal_path: Path, *, acknowledged_prefix: dict | None = None
) -> dict:
    """Read authority-owned raw chain, never a semantic/formal success summary."""
    spec_raw, _ = read_private(spec_path, 0o600)
    spec = json.loads(spec_raw)
    expected_senders = {x["process"]["pid"]: x for x in spec["senders"]}
    expected_phases = {x["external_request_id"]: x for x in spec["phases"]}
    events = [json.loads(line) for line in journal_path.read_bytes().splitlines()]
    if not events or events[0].get("kind") != "collector.start":
        raise ValueError("authority raw chain start is missing")
    start = events[0]
    if (
        start["spec_sha256"] != sha(spec_raw)
        or start["collector_source_sha256"]
        not in {
            sha(Path(__file__).read_bytes()),
            "135a3a9f06f9b2e7afe12a377ae3947a7b96801924fc09b853c836c12d65bf77",
        }
        or spec.get("classification") != "CPU_REVIEW_CANDIDATE_ONLY"
        or spec.get("formal_producer_registration") is not False
    ):
        raise ValueError("authority raw chain source or frozen spec differs")
    previous = "0" * 64
    seen_raw = set()
    projections: dict[Any, Any] = {}
    kv_ranks: dict[Any, Any] = {}
    mappings: dict[Any, Any] = {}
    mapping_stamps: dict[Any, Any] = {}
    chronology_complete = (
        start.get("intake_predicate_schema")
        == (
            "ecpa-native-semantic-intake/v3"
            if spec["schema"] == KV_SPEC_V2
            else "ecpa-native-semantic-intake/v2"
        )
        and type(start.get("started_monotonic_ns")) is int
    )
    last_accepted_ns = start.get("started_monotonic_ns")
    ranks: dict[Any, Any] = {}
    upload_stamps: dict[Any, Any] = {}
    counts: dict[Any, Any] = {}
    for sequence, event in enumerate(events):
        if (
            event.get("sequence") != sequence
            or event.get("previous_sha256") != previous
            or event.get("event_sha256")
            != sha(canonical({k: v for k, v in event.items() if k != "event_sha256"}))
        ):
            raise ValueError("authority raw chain is broken")
        previous = event["event_sha256"]
        if sequence == 0:
            continue
        if event.get("kind") != "native.accepted":
            raise ValueError("authority raw chain kind is unsupported")
        raw = base64.b64decode(event["raw_base64"], validate=True)
        fact = json.loads(raw)
        sender = expected_senders.get(event["peer"]["pid"])
        if (
            sender is None
            or event["peer"]["uid"] != sender["peer_uid"]
            or event["peer"]["gid"] != sender["peer_gid"]
            or fact.get("process") != sender["process"]
            or any(sender["process"].get(k) != v for k, v in event["process"].items())
            or event["source_sha256"]
            != {x["path"]: x["sha256"] for x in sender["source_files"]}
            or event["archive_identity"] != start["archive_identity"]
            or canonical(fact) != raw
            or sha(raw) != event["raw_sha256"]
            or sha(raw) in seen_raw
            or fact.get("plan_id") != spec["native_plan_id"]
            or fact.get("launch_id") != spec["native_launch_id"]
        ):
            raise ValueError("authority raw source/peer/request scope differs")
        seen_raw.add(sha(raw))
        accepted_ns = event.get("accepted_monotonic_ns")
        if accepted_ns is not None:
            start_ns = start.get("started_monotonic_ns")
            if (
                type(start_ns) is not int
                or type(accepted_ns) is not int
                or accepted_ns < start_ns
                or (last_accepted_ns is not None and accepted_ns < last_accepted_ns)
            ):
                raise ValueError("native accept clock violates collector sequence")
            last_accepted_ns = accepted_ns
        else:
            chronology_complete = False
        fact, phase, update = native_fact_predicate(
            raw,
            sender,
            spec,
            accepted_monotonic_ns=accepted_ns,
            mappings=mappings,
            mapping_stamps=mapping_stamps,
            upload_stamps=upload_stamps,
            upload_counts=counts,
        )
        runtime = fact["runtime_request_id"]
        if update[0] == "mapping":
            mappings[runtime] = phase
            mapping_stamps[runtime] = fact["monotonic_ns"]
        elif update[0] == "projection":
            projections.setdefault((runtime, update[1]), []).append(fact)
            counts[("projection", runtime, update[1])] = 1
        else:
            rank = update[1]
            if any(
                projection_upload_match(p, fact)
                for p in projections.get((runtime, rank), [])
            ):
                kv_ranks.setdefault(runtime, set()).add(rank)
            ranks.setdefault(runtime, set()).add(rank)
            upload_stamps[(runtime, rank)] = fact["observed_after_barrier_monotonic_ns"]
            counts[(runtime, rank)] = counts.get((runtime, rank), 0) + 1
        if phase != event["phase"] or runtime != event["runtime_request_id"]:
            raise ValueError("authority raw challenge join differs")
    complete = sum(ranks.get(runtime) == {0, 1} for runtime in mappings)
    kv_complete = sum(kv_ranks.get(runtime) == {0, 1} for runtime in mappings)
    if acknowledged_prefix is not None and acknowledged_prefix != {
        "next_sequence": len(events),
        "last_event_sha256": previous,
    }:
        raise ValueError(
            "authority raw tail is not the authenticated durable ACK prefix"
        )
    return {
        "classification": "CPU_REVIEW_CANDIDATE_ONLY",
        "source_authenticated_native_envelopes": len(events) - 1,
        "native_API_mappings": len(mappings),
        "observed_two_rank_raw_phases": complete,
        "complete_two_rank_transport_phases": complete
        if acknowledged_prefix is not None and chronology_complete
        else None,
        "native_chronology_status": "verified"
        if chronology_complete
        else "unverified-missing-accept-clock",
        "runtime_collector_source_sha256": start["collector_source_sha256"],
        "analysis_source_sha256": sha(Path(__file__).read_bytes()),
        "durable_ACK_status": "verified-authenticated-prefix"
        if acknowledged_prefix is not None
        else "unverified",
        "expected_phase_count": len(expected_phases),
        "formal_producer_registration": False,
        "formal_result": False,
        "complete_two_rank_KV_projection_upload_phases": kv_complete
        if spec.get("schema") in {KV_SPEC, KV_SPEC_V2}
        and acknowledged_prefix is not None
        and chronology_complete
        else None,
        "KV_projection_scope": PROJECTION_SCOPE
        if spec.get("schema") in {KV_SPEC, KV_SPEC_V2}
        else None,
        "GPU_semantic_qualification": None,
        "full_objective_satisfied": None,
        "journal_sha256": sha(journal_path.read_bytes()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--journal", required=True, type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_journal(args.spec, args.journal), sort_keys=True))
        return
    if args.socket is None:
        parser.error("--socket is required for collector service")
    collector = Collector(args.spec, args.journal)
    with Server(args.socket, collector) as server:
        meta = args.socket.stat()
        print(
            json.dumps(
                {
                    "collector": collector.collector_identity,
                    "socket_identity": [meta.st_dev, meta.st_ino],
                    "socket_path": str(args.socket),
                    "collector_source_sha256": sha(Path(__file__).read_bytes()),
                    "spec_sha256": collector.spec_sha256,
                    "archive_identity": collector.archive_identity,
                }
            ),
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
