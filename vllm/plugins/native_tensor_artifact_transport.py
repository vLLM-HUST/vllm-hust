"""Private selected-KV artifact transport candidate; no formal/GPU promotion."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import stat
import time
from pathlib import Path

if __package__ and __package__.startswith("vllm."):
    from . import native_semantic_collector as native
    from . import native_tensor_runtime_contract as runtime_contract
    from . import native_tensor_witness as witness
else:
    from experiments.exposure_gate import native_semantic_collector as native
    from experiments.native_kv_tensor_candidate import runtime_contract, witness

POLICY = "ecpa-selected-KV-private-artifact-policy/v1"
RUNTIME_POLICY = "ecpa-selected-KV-private-artifact-runtime-policy/v1"
RUNTIME_ENVELOPE = "ecpa-selected-KV-private-artifact-runtime-envelope/v1"
ENVELOPE = "ecpa-selected-KV-private-artifact-envelope/v1"
MAX_ARTIFACT = 64 * 1024 * 1024
NAME = re.compile(r"^[a-f0-9]{64}\.json$")
PLAN_FIELDS = {
    "semantic_anchor",
    "physical_anchor",
    "source_block_ids",
    "destination_block_ids",
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def directory(path, expected=None):
    """Traverse without symlinks; pin an owned private directory inode."""
    path = Path(path)
    require(path.is_absolute(), "artifact directory must be absolute")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        info = os.fstat(fd)
        require(
            info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
            "artifact directory is not owned and private",
        )
        actual = [info.st_dev, info.st_ino]
        require(
            expected is None or actual == expected, "artifact directory inode changed"
        )
        return fd, actual
    except BaseException:
        os.close(fd)
        raise


def write_new(dir_fd, name, raw):
    require(
        NAME.fullmatch(name) and 0 < len(raw) <= MAX_ARTIFACT,
        "artifact basename/size is unsupported",
    )
    fd = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd
    )
    try:
        data = memoryview(raw)
        while data:
            count = os.write(fd, data)
            if count <= 0:
                raise OSError("artifact short write")
            data = data[count:]
        os.fsync(fd)
        info = os.fstat(fd)
    finally:
        os.close(fd)
    os.fsync(dir_fd)
    return {
        "basename": name,
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "sha256": native.sha(raw),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def descriptor_predicate(artifact):
    require(
        set(artifact)
        == {"basename", "device", "inode", "size", "sha256", "mtime_ns", "ctime_ns"},
        "artifact descriptor fields differ",
    )
    require(
        isinstance(artifact["basename"], str)
        and NAME.fullmatch(artifact["basename"])
        and all(
            type(artifact[k]) is int and artifact[k] >= 0
            for k in ["device", "inode", "size", "mtime_ns", "ctime_ns"]
        )
        and 0 < artifact["size"] <= MAX_ARTIFACT,
        "artifact descriptor is invalid",
    )


def read_blob(dir_fd, artifact, *, with_descriptor=False):
    descriptor_predicate(artifact)
    fd = os.open(artifact["basename"], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    try:
        before = os.fstat(fd)
        require(
            stat.S_ISREG(before.st_mode)
            and stat.S_IMODE(before.st_mode) == 0o600
            and before.st_uid == os.getuid()
            and before.st_nlink == 1,
            "artifact is not an owned private unaliased regular file",
        )
        require(
            all(
                getattr(before, attr) == artifact[key]
                for attr, key in [
                    ("st_dev", "device"),
                    ("st_ino", "inode"),
                    ("st_size", "size"),
                    ("st_mtime_ns", "mtime_ns"),
                    ("st_ctime_ns", "ctime_ns"),
                ]
            ),
            "artifact inode/clock/size changed",
        )
        chunks, count = [], 0
        while chunk := os.read(fd, min(65536, MAX_ARTIFACT + 1 - count)):
            count += len(chunk)
            require(count <= MAX_ARTIFACT, "artifact grew beyond bound")
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        require(
            all(
                getattr(before, key) == getattr(after, key)
                for key in [
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_uid",
                    "st_gid",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                ]
            )
            and len(raw) == artifact["size"]
            and native.sha(raw) == artifact["sha256"],
            "artifact bytes changed",
        )
        observed = {
            "basename": artifact["basename"],
            "device": after.st_dev,
            "inode": after.st_ino,
            "size": after.st_size,
            "sha256": native.sha(raw),
            "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
        }
        return (raw, observed) if with_descriptor else raw
    finally:
        os.close(fd)


def transaction(fact):
    """Read actual source plan; no owner-created phase or synthetic transaction ID."""
    binding = fact["binding"]
    plan = binding.get("compression_plan")
    require(
        isinstance(plan, dict) and set(plan) == PLAN_FIELDS,
        "actual CompressionRequest.plan is missing",
    )
    require(
        all(
            type(plan[k]) is int and plan[k] > 0
            for k in ["semantic_anchor", "physical_anchor"]
        )
        and plan["semantic_anchor"] == binding["semantic_num_tokens"]
        and plan["physical_anchor"] == len(fact["keep_indices"])
        and plan["physical_anchor"] < plan["semantic_anchor"],
        "actual plan semantic/physical anchors differ",
    )
    factor = binding.get("cache_blocks_per_scheduler_block")
    require(
        type(factor) is int and factor > 0, "actual scheduler/cache expansion missing"
    )
    for side in ["source", "destination"]:
        blocks = plan[side + "_block_ids"]
        require(
            isinstance(blocks, list)
            and blocks
            and all(type(x) is int and x >= 0 for x in blocks)
            and len(set(blocks)) == len(blocks),
            "actual plan blocks are aliased",
        )
        expanded = [b * factor + i for b in blocks for i in range(factor)]
        require(
            expanded == fact[side + "_blocks"], "actual plan/device allocation differs"
        )
    return native.sha(
        native.canonical(
            {
                "runtime_request_id": binding["runtime_request_id"],
                "compression_plan": plan,
            }
        )
    )


def verify_cpu_source(fact):
    witness.verify(fact)
    require(
        all(
            view[stage]["device"] == "cpu" and view[stage]["npu_format"] is None
            for view in fact["cache_views"].values()
            for stage in ["before", "after"]
        ),
        "CPU contract cannot publish actual NPU readback under CPU classification",
    )


def tensor_predicate(fact, envelope, policy, sender, maps, mapping_stamps, now, spec):
    is_runtime = policy["schema"] == RUNTIME_POLICY
    if is_runtime:
        runtime_contract.verify(fact, sender["process"])
    else:
        verify_cpu_source(fact)
    binding = fact["binding"]
    descriptor_predicate(envelope["artifact"])
    require(
        set(envelope)
        == {
            "schema",
            "process",
            "plan_id",
            "launch_id",
            "monotonic_ns",
            "runtime_request_id",
            "transaction_id",
            "artifact",
            "transport_policy_sha256",
        }
        and envelope["schema"] == (RUNTIME_ENVELOPE if is_runtime else ENVELOPE),
        "tensor envelope fields differ",
    )
    runtime = binding["runtime_request_id"]
    require(
        sender["process"]["role"] == "worker"
        and type(sender["process"]["ordinal"]) is int
        and type(sender["process"]["process_epoch"]) is int
        and envelope["process"] == sender["process"]
        and binding["rank"] == sender["process"]["ordinal"]
        and binding["tensor_parallel_size"] == 2
        and runtime == envelope["runtime_request_id"]
        and runtime in maps,
        "tensor artifact lacks actual worker/request/API mapping",
    )
    phase = maps[runtime]
    requirement = policy["requirements"][phase["phase_id"]]
    require(
        binding["expected_layer_indices"] == requirement["layer_indices"]
        and binding["layer_index"] in requirement["layer_indices"]
        and binding["compression_plan"]["semantic_anchor"]
        == requirement["semantic_anchor"]
        and binding["compression_plan"]["physical_anchor"]
        == requirement["physical_anchor"]
        and envelope["transaction_id"] == transaction(fact),
        "tensor layer/actual plan transaction differs from frozen requirement",
    )
    stamp = envelope["monotonic_ns"]
    require(
        type(stamp) is int
        and mapping_stamps[runtime]
        <= fact["source_before_gather_ns"]
        <= fact["destination_readback_completed_ns"]
        <= stamp
        <= now
        and stamp >= spec["frozen_monotonic_ns"]
        and now - stamp <= spec["max_event_age_ns"]
        and envelope["plan_id"] == spec["native_plan_id"]
        and envelope["launch_id"] == spec["native_launch_id"],
        "tensor actual mapping/materialization/accept chronology differs",
    )
    return phase


def policy_predicate(policy, spec, spec_sha256):
    require(
        set(policy)
        == {
            "schema",
            "classification",
            "formal_registration",
            "native_spec_sha256",
            "worker_directories",
            "requirements",
        }
        and policy["schema"] in {POLICY, RUNTIME_POLICY}
        and policy["classification"]
        == (
            runtime_contract.CLASSIFICATION
            if policy["schema"] == RUNTIME_POLICY
            else "CPU_SOURCE_CANDIDATE_ONLY"
        )
        and policy["formal_registration"] is False
        and policy["native_spec_sha256"] == spec_sha256
        and spec["classification"] == "CPU_REVIEW_CANDIDATE_ONLY"
        and spec["formal_producer_registration"] is False,
        "artifact policy scope/native registration differs",
    )
    workers = {
        str(s["process"]["pid"]): s
        for s in spec["senders"]
        if s["process"]["role"] == "worker"
    }
    require(
        len(workers) == 2
        and {s["process"]["ordinal"] for s in workers.values()} == {0, 1}
        and set(policy["worker_directories"]) == set(workers),
        "artifact directory registration must name actual two workers",
    )
    required = {
        native.sha(Path(native.__file__).read_bytes()),
        native.sha(Path(__file__).read_bytes()),
        native.sha(Path(witness.__file__).read_bytes()),
    }
    if policy["schema"] == RUNTIME_POLICY:
        required.add(native.sha(Path(runtime_contract.__file__).read_bytes()))
    for sender in workers.values():
        require(
            required <= {s["sha256"] for s in sender["source_files"]},
            "tensor worker collector/transport/predicate source closure missing",
        )
    for item in policy["worker_directories"].values():
        require(
            set(item) == {"path", "identity"}
            and isinstance(item["path"], str)
            and Path(item["path"]).is_absolute()
            and isinstance(item["identity"], list)
            and len(item["identity"]) == 2
            and all(type(x) is int and x >= 0 for x in item["identity"]),
            "artifact directory registration fields differ",
        )
    require(
        set(policy["requirements"]) == {p["phase_id"] for p in spec["phases"]},
        "tensor phase preregistration missing",
    )
    for req in policy["requirements"].values():
        require(
            set(req) == {"layer_indices", "semantic_anchor", "physical_anchor"}
            and isinstance(req["layer_indices"], list)
            and req["layer_indices"]
            and all(type(x) is int and x >= 0 for x in req["layer_indices"])
            and len(set(req["layer_indices"])) == len(req["layer_indices"])
            and type(req["physical_anchor"]) is int
            and type(req["semantic_anchor"]) is int
            and 0 < req["physical_anchor"] < req["semantic_anchor"],
            "tensor requirement inventory/anchors unsupported",
        )


class TensorArtifactCollector(native.Collector):
    """Opt-in wrapper. Existing generic collector/source/raw formats are unchanged."""

    def __init__(self, spec_path, journal_path, policy_path):
        raw, _ = native.read_private(Path(policy_path), 0o600)
        self.policy = json.loads(raw)
        spec_raw, _ = native.read_private(Path(spec_path), 0o600)
        policy_predicate(self.policy, json.loads(spec_raw), native.sha(spec_raw))
        super().__init__(Path(spec_path), Path(journal_path))
        self.policy_sha256 = native.sha(raw)
        self.artifact_dirs = {}
        for pid, item in self.policy["worker_directories"].items():
            self.artifact_dirs[pid] = directory(item["path"], item["identity"])[0]
        self.archive_dirfd, _ = directory(Path(journal_path).parent)
        self.tensor_seen = set()
        self.source_sha256 = native.sha(Path(__file__).read_bytes())
        self._append(
            {
                "kind": "tensor.transport.start",
                "policy_sha256": self.policy_sha256,
                "transport_source_sha256": self.source_sha256,
                "witness_source_sha256": native.sha(
                    Path(witness.__file__).read_bytes()
                ),
                "formal_registration": False,
                **(
                    {
                        "runtime_contract_source_sha256": native.sha(
                            Path(runtime_contract.__file__).read_bytes()
                        )
                    }
                    if self.policy["schema"] == RUNTIME_POLICY
                    else {}
                ),
            }
        )
        os.fsync(self.archive_dirfd)

    def accept(self, peer, raw):
        envelope = json.loads(raw)
        require(isinstance(envelope, dict), "native envelope is not an object")
        if envelope.get("schema") not in {ENVELOPE, RUNTIME_ENVELOPE}:
            return super().accept(peer, raw)
        with self.lock:
            require(not self.failed, "collector persistence failed")
            pid, uid, gid = peer
            sender = self.senders.get(pid)
            require(
                sender is not None
                and sender["process"]["role"] == "worker"
                and sender["peer_uid"] == uid == os.getuid()
                and sender["peer_gid"] == gid,
                "kernel peer is not an artifact worker source",
            )
            require(
                all(
                    sender["process"].get(k) == v
                    for k, v in native.identity(pid).items()
                )
                and native.sha(Path(f"/proc/{pid}/cmdline").read_bytes())
                == sender["cmdline_sha256"],
                "artifact kernel peer epoch/argv differs",
            )
            sources = native.source_digests(sender["source_files"])
            require(
                native.canonical(envelope) == raw
                and len(raw) <= native.MAX_FRAME
                and envelope["transport_policy_sha256"] == self.policy_sha256,
                "artifact envelope bytes/policy differ",
            )
            current_fd, _ = directory(
                **{
                    "path": self.policy["worker_directories"][str(pid)]["path"],
                    "expected": self.policy["worker_directories"][str(pid)]["identity"],
                }
            )
            os.close(current_fd)
            payload, source_artifact = read_blob(
                self.artifact_dirs[str(pid)], envelope["artifact"], with_descriptor=True
            )
            fact = json.loads(payload)
            require(
                native.canonical(fact) == payload, "artifact payload is noncanonical"
            )
            accepted = time.monotonic_ns()
            phase = tensor_predicate(
                fact,
                envelope,
                self.policy,
                sender,
                self.maps,
                self.mapping_stamps,
                accepted,
                self.spec,
            )
            key = (
                envelope["runtime_request_id"],
                envelope["transaction_id"],
                fact["binding"]["rank"],
                fact["binding"]["layer_index"],
            )
            require(
                key not in self.tensor_seen, "tensor transaction/rank/layer is replayed"
            )
            try:
                archived = write_new(
                    self.archive_dirfd,
                    envelope["artifact"]["sha256"] + ".json",
                    payload,
                )
                # Re-read archived bytes independently before publishing the ACK event.
                require(
                    read_blob(self.archive_dirfd, archived) == payload,
                    "collector artifact readback differs",
                )
                # Late producer mutations cannot be silently accepted during intake.
                require(
                    read_blob(self.artifact_dirs[str(pid)], envelope["artifact"])
                    == payload,
                    "producer artifact changed before ACK",
                )
                event = self._append(
                    {
                        "kind": "tensor.artifact.accepted",
                        "accepted_monotonic_ns": accepted,
                        "raw_base64": base64.b64encode(raw).decode(),
                        "raw_sha256": native.sha(raw),
                        "artifact": archived,
                        "source_artifact": source_artifact,
                        "peer": {"pid": pid, "uid": uid, "gid": gid},
                        "source_sha256": sources,
                        "phase": phase,
                        "process": native.identity(pid),
                        "runtime_request_id": envelope["runtime_request_id"],
                        "archive_identity": self.archive_identity,
                    }
                )
            except OSError:
                self.failed = True
                raise
            self.tensor_seen.add(key)
            ack = {
                "schema": native.ACK,
                "raw_sha256": native.sha(raw),
                "event_sha256": event["event_sha256"],
                "sequence": event["sequence"],
                "phase": phase,
                "collector": self.collector_identity,
                "collector_source_sha256": self.source_sha256,
                "archive_identity": self.archive_identity,
                "formal_producer_registration": False,
                "classification": self.policy["classification"],
            }
            ack["ack_sha256"] = native.sha(native.canonical(ack))
            return ack


def publish_candidate(
    fact,
    *,
    directory_path,
    directory_identity,
    policy_sha256,
    scope,
    socket_path,
    receiver,
    runtime=False,
):
    actual_scope = scope()
    if runtime:
        runtime_contract.verify(fact, actual_scope["process"])
    else:
        verify_cpu_source(fact)
    require(
        actual_scope["process"]["pid"] == os.getpid()
        and actual_scope["process"]["role"] == "worker",
        "artifact publisher must be the actual source worker",
    )
    raw = native.canonical(fact)
    fd, _ = directory(directory_path, directory_identity)
    try:
        artifact = write_new(fd, native.sha(raw) + ".json", raw)
    finally:
        os.close(fd)
    envelope = {
        "schema": RUNTIME_ENVELOPE if runtime else ENVELOPE,
        **actual_scope,
        "monotonic_ns": time.monotonic_ns(),
        "runtime_request_id": fact["binding"]["runtime_request_id"],
        "transaction_id": transaction(fact),
        "artifact": artifact,
        "transport_policy_sha256": policy_sha256,
    }
    # Only this bounded envelope enters the existing framed Unix socket.
    require(
        len(
            native.canonical(
                {"raw_base64": base64.b64encode(native.canonical(envelope)).decode()}
            )
        )
        + 1
        <= native.MAX_FRAME,
        "tensor envelope exceeds native socket bound",
    )
    return native.submit_fact(Path(socket_path), envelope, **receiver)


def verify_archive(spec_path, policy_path, journal_path, *, acknowledged_prefix=None):
    """Portable bytes oracle. An ACK prefix must come from authenticated live status.

    This function reads no current native PIDs or mutable sender paths. Archived
    file inodes are not portable; the original identities remain recorded, while
    the archived payload's exact bytes/digest are independently checked.
    """
    spec_raw = Path(spec_path).read_bytes()
    policy_raw = Path(policy_path).read_bytes()
    spec, policy = json.loads(spec_raw), json.loads(policy_raw)
    require(
        policy["schema"] in {POLICY, RUNTIME_POLICY}
        and policy["classification"]
        in {"CPU_SOURCE_CANDIDATE_ONLY", runtime_contract.CLASSIFICATION}
        and policy["formal_registration"] is False
        and policy["native_spec_sha256"] == native.sha(spec_raw),
        "portable artifact policy scope/source differs",
    )
    policy_predicate(policy, spec, native.sha(spec_raw))
    senders = {s["process"]["pid"]: s for s in spec["senders"]}
    events = [json.loads(line) for line in Path(journal_path).read_bytes().splitlines()]
    require(
        len(events) >= 2
        and events[0]["kind"] == "collector.start"
        and events[0]["spec_sha256"] == native.sha(spec_raw)
        and events[0]["collector_source_sha256"]
        == native.sha(Path(native.__file__).read_bytes())
        and events[1]["kind"] == "tensor.transport.start"
        and events[1]["policy_sha256"] == native.sha(policy_raw)
        and events[1]["transport_source_sha256"]
        == native.sha(Path(__file__).read_bytes())
        and events[1]["witness_source_sha256"]
        == native.sha(Path(witness.__file__).read_bytes()),
        "portable collector/transport/source start differs",
    )
    if policy["schema"] == RUNTIME_POLICY:
        require(
            events[1].get("runtime_contract_source_sha256")
            == native.sha(Path(runtime_contract.__file__).read_bytes()),
            "runtime contract source start differs",
        )
    maps, mapping_stamps, upload_stamps, upload_counts = {}, {}, {}, {}
    seen, tensor_seen, completed = set(), set(), {}
    previous = "0" * 64
    last_clock = events[0]["started_monotonic_ns"]
    for sequence, event in enumerate(events):
        require(
            event["sequence"] == sequence
            and event["previous_sha256"] == previous
            and event["event_sha256"]
            == native.sha(
                native.canonical(
                    {k: v for k, v in event.items() if k != "event_sha256"}
                )
            ),
            "portable authority chain prefix differs",
        )
        previous = event["event_sha256"]
        if sequence < 2:
            continue
        require(
            event["kind"] in {"native.accepted", "tensor.artifact.accepted"},
            "portable authority event kind unsupported",
        )
        raw = base64.b64decode(event["raw_base64"], validate=True)
        digest = native.sha(raw)
        require(
            digest == event["raw_sha256"] and digest not in seen,
            "portable native bytes are changed or replayed",
        )
        seen.add(digest)
        sender = senders.get(event["peer"]["pid"])
        require(
            sender is not None
            and event["peer"]["uid"] == sender["peer_uid"]
            and event["peer"]["gid"] == sender["peer_gid"]
            and event["source_sha256"]
            == {s["path"]: s["sha256"] for s in sender["source_files"]}
            and event["process"]
            == {
                k: sender["process"][k]
                for k in ["pid", "start_ticks", "start_identity", "boot_id", "session"]
            }
            and event["archive_identity"] == events[0]["archive_identity"],
            "portable sender source/kernel binding differs",
        )
        accepted = event["accepted_monotonic_ns"]
        require(
            type(accepted) is int and accepted >= last_clock,
            "portable collector accept chronology differs",
        )
        last_clock = accepted
        if event["kind"] == "native.accepted":
            fact, phase, update = native.native_fact_predicate(
                raw,
                sender,
                spec,
                accepted_monotonic_ns=accepted,
                mappings=maps,
                mapping_stamps=mapping_stamps,
                upload_stamps=upload_stamps,
                upload_counts=upload_counts,
            )
            runtime = fact["runtime_request_id"]
            if update[0] == "mapping":
                maps[runtime], mapping_stamps[runtime] = phase, fact["monotonic_ns"]
            elif update[0] == "upload":
                upload_stamps[(runtime, update[1])] = fact[
                    "observed_after_barrier_monotonic_ns"
                ]
                upload_counts[(runtime, update[1])] = (
                    upload_counts.get((runtime, update[1]), 0) + 1
                )
            else:
                upload_counts[("projection", runtime, update[1])] = 1
        else:
            envelope = json.loads(raw)
            require(
                native.canonical(envelope) == raw
                and envelope["transport_policy_sha256"] == native.sha(policy_raw),
                "portable tensor envelope policy differs",
            )
            artifact = event["artifact"]
            descriptor_predicate(artifact)
            require(
                event["source_artifact"] == envelope["artifact"],
                "portable observed source artifact inode differs",
            )
            require(
                NAME.fullmatch(artifact["basename"])
                and artifact["sha256"] == envelope["artifact"]["sha256"],
                "portable artifact digest descriptor differs",
            )
            path = Path(journal_path).parent / artifact["basename"]
            require(
                not path.is_symlink() and path.is_file(),
                "portable artifact path is not regular",
            )
            payload = path.read_bytes()
            require(
                0 < len(payload) <= MAX_ARTIFACT
                and len(payload) == artifact["size"] == envelope["artifact"]["size"]
                and native.sha(payload) == artifact["sha256"],
                "portable archived blob changed",
            )
            fact = json.loads(payload)
            require(
                native.canonical(fact) == payload,
                "portable tensor payload is noncanonical",
            )
            phase = tensor_predicate(
                fact, envelope, policy, sender, maps, mapping_stamps, accepted, spec
            )
            key = (
                fact["binding"]["runtime_request_id"],
                envelope["transaction_id"],
                fact["binding"]["rank"],
                fact["binding"]["layer_index"],
            )
            require(key not in tensor_seen, "portable tensor rank/layer replay")
            tensor_seen.add(key)
            completed.setdefault(key[:2], set()).add(key[2:])
        require(
            event["phase"] == phase
            and event["runtime_request_id"] == fact["runtime_request_id"]
            if event["kind"] == "native.accepted"
            else event["phase"] == phase
            and event["runtime_request_id"] == fact["binding"]["runtime_request_id"],
            "portable phase/runtime join differs",
        )
    if acknowledged_prefix is not None:
        require(
            acknowledged_prefix
            == {"next_sequence": len(events), "last_event_sha256": previous},
            "authenticated ACK prefix differs from archived chain",
        )
    complete = []
    for (runtime, tx), observed in completed.items():
        phase = maps[runtime]
        expected = {
            (rank, layer)
            for rank in phase["expected_ranks"]
            for layer in policy["requirements"][phase["phase_id"]]["layer_indices"]
        }
        if observed == expected and acknowledged_prefix is not None:
            complete.append(
                {
                    "phase_id": phase["phase_id"],
                    "runtime_request_id": runtime,
                    "transaction_id": tx,
                }
            )
    return {
        "schema": "ecpa-selected-KV-artifact-portable-analysis/v1",
        "classification": policy["classification"],
        "complete_transactions": complete,
        "durable_artifact_join_complete": bool(complete),
        "acknowledged_prefix_status": "verified"
        if acknowledged_prefix is not None
        else "unknown",
        "GPU_qualification": None,
        "attention_consumption": None,
        "full_model_quality": None,
        "formal_registration": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    collector = TensorArtifactCollector(args.spec, args.journal, args.policy)
    with native.Server(args.socket, collector) as server:
        metadata = args.socket.stat()
        print(
            json.dumps(
                {
                    "collector": collector.collector_identity,
                    "socket_identity": [metadata.st_dev, metadata.st_ino],
                    "socket_path": str(args.socket),
                    "collector_source_sha256": collector.source_sha256,
                    "spec_sha256": collector.spec_sha256,
                    "policy_sha256": collector.policy_sha256,
                    "archive_identity": collector.archive_identity,
                }
            ),
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
