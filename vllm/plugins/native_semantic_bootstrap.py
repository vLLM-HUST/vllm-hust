# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Source-owned opt-in callback bootstrap for the native candidate observer.

A context may be published after native process handoff but before HTTP probes.
Absent context is non-producing warmup, never qualification. Registered source
facts remain unchanged; only their transport and authenticated ACK are handled.
"""

from __future__ import annotations

import os
from pathlib import Path

from .native_semantic_collector import (
    canonical,
    read_private,
    sha,
    source_digests,
    submit_fact,
)

BOOTSTRAP = "ecpa-native-semantic-bootstrap/v1"
ENV = "ECPA_NATIVE_SEMANTIC_BOOTSTRAP"


def callback_from_context(path: Path):
    import json

    raw, metadata = read_private(path, 0o600)
    value = json.loads(raw)
    required = {
        "schema",
        "classification",
        "formal_producer_registration",
        "socket_path",
        "socket_identity",
        "collector",
        "collector_source_sha256",
        "archive_identity",
        "spec_path",
        "spec_sha256",
        "source_files",
    }
    if (
        set(value) != required
        or value["schema"] != BOOTSTRAP
        or value["classification"] != "CPU_REVIEW_CANDIDATE_ONLY"
        or value["formal_producer_registration"] is not False
    ):
        raise ValueError("native bootstrap context is unsupported or promoted")
    spec_raw, _ = read_private(Path(value["spec_path"]), 0o600)
    if sha(spec_raw) != value["spec_sha256"]:
        raise ValueError("native bootstrap spec differs from freeze")
    spec = json.loads(spec_raw)
    peers = {x["process"]["pid"]: x for x in spec["senders"]}
    if os.getpid() not in peers:
        raise ValueError("native bootstrap current process is not registered")
    own_sender = peers[os.getpid()]
    source_digests(value["source_files"])
    source_digests(own_sender["source_files"])
    frozen_identity = (metadata.st_dev, metadata.st_ino)
    frozen_sha = sha(raw)

    def callback(fact):
        current, info = read_private(path, 0o600)
        if (info.st_dev, info.st_ino) != frozen_identity or sha(current) != frozen_sha:
            raise ValueError("native bootstrap context changed after registration")
        source_digests(value["source_files"])
        if fact.get("process") != own_sender["process"]:
            raise ValueError("native fact process differs from own source registration")
        # The fact is actual native output. Do not add nonce, role or IDs here.
        ack = submit_fact(
            Path(value["socket_path"]),
            fact,
            expected_collector=value["collector"],
            expected_socket_identity=tuple(value["socket_identity"]),
            expected_collector_source_sha256=value["collector_source_sha256"],
            expected_archive_identity=value["archive_identity"],
        )
        if ack["phase"] not in spec["phases"]:
            raise ValueError("native ACK is not a frozen challenge phase")
        return ack

    return callback


def ensure_registered(register_observer) -> bool:
    path = os.environ.get(ENV)
    if not path:
        return False
    context = Path(path)
    if not context.is_absolute():
        raise ValueError("native bootstrap context path is not absolute")
    # Native warmup precedes authority peer registration. No context means no
    # receipts; the qualification checker must reject any later missing proof.
    if not context.exists():
        return False
    register_observer(callback_from_context(context))
    return True


def write_context(
    path: Path, *, spec_path: Path, ready: dict, source_files: list
) -> str:
    """Launcher helper; readiness must already be independently peer/source checked."""
    value = {
        "schema": BOOTSTRAP,
        "classification": "CPU_REVIEW_CANDIDATE_ONLY",
        "formal_producer_registration": False,
        "socket_path": ready["socket_path"],
        "socket_identity": ready["socket_identity"],
        "collector": ready["collector"],
        "collector_source_sha256": ready["collector_source_sha256"],
        "archive_identity": ready["archive_identity"],
        "spec_path": str(spec_path),
        "spec_sha256": ready["spec_sha256"],
        "source_files": source_files,
    }
    raw = canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        data = memoryview(raw)
        while data:
            n = os.write(fd, data)
            if n <= 0:
                raise OSError("native bootstrap write failed")
            data = data[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha(raw)
