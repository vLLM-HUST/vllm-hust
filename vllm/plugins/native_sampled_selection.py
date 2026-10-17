# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Source-side H4 sampling only; never native fact/ACK or completion evidence.

Selected runtime IDs come only from the actual API mapping accepted by the
original collector. No HTTP-ID-derived runtime guess or collector fact drop.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
from pathlib import Path

from . import native_semantic_collector as native

ENV = "ECPA_H4_NATIVE_SAMPLED_SELECTION"
_selector = None


def enabled():
    return os.environ.get(ENV) == "1"


def require(value, message):
    if not value:
        raise ValueError(message)


class SampledSelection:
    def __init__(self, value, spec, current):
        self.value, self.spec, self.current = value, spec, current
        self.selected = {p["external_request_id"] for p in spec["phases"]}
        require(
            len(spec["phases"]) == len(self.selected) == 32,
            "H4 source selector requires exactly32 selected phases",
        )
        own = [
            sender["process"]
            for sender in spec["senders"]
            if sender["process"]["pid"] == os.getpid()
        ]
        require(len(own) == 1, "selector actual source is not preregistered")
        self.own_process = own[0]
        self.cache = {}

    def _actual_mappings(self):
        self.current()  # original private context inode/bytes/source checks
        value = self.value
        path = Path(value["socket_path"])
        parent, info = path.parent.stat(), path.lstat()
        require(
            stat.S_IMODE(parent.st_mode) == 0o700 and parent.st_uid == os.getuid(),
            "sample selector collector directory is not private",
        )
        require(
            stat.S_ISSOCK(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o600
            and info.st_uid == os.getuid()
            and [info.st_dev, info.st_ino] == value["socket_identity"],
            "sample selector collector socket changed",
        )
        require(
            native.identity(value["collector"]["pid"]) == value["collector"],
            "sample selector collector epoch changed",
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(5)
            stream.connect(str(path))
            pid, uid, _ = struct.unpack(
                "3i", stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            require(
                uid == os.getuid() and native.identity(pid) == value["collector"],
                "sample selector endpoint is not the actual collector",
            )
            stream.sendall(b'{"op":"describe"}\n')
            raw = b""
            while not raw.endswith(b"\n"):
                part = stream.recv(native.MAX_FRAME + 1)
                require(
                    part and len(raw) + len(part) <= native.MAX_FRAME,
                    "sample selector status incomplete or oversized",
                )
                raw += part
        status = json.loads(raw)
        require(
            status["schema"] == "ecpa-h4-sampled-concurrent-native-collector-status/v1"
            and status["collector"] == value["collector"]
            and status["spec_sha256"] == value["spec_sha256"]
            and status["collector_source_sha256"] == value["collector_source_sha256"]
            and status["socket_identity"] == value["socket_identity"]
            and status["archive_identity"] == value["archive_identity"]
            and status["failed"] is False,
            "sample selector authenticated collector status differs",
        )
        mappings = status["source_drain"]["runtimes"]
        require(
            type(mappings) is dict and len(mappings) <= 32,
            "sample selector mapping inventory differs",
        )
        for runtime, state in mappings.items():
            require(
                type(runtime) is str
                and runtime
                and state["phase"] in self.spec["phases"],
                "sample selector did not originate in original accepted API mapping",
            )
        return mappings

    def runtime(self, runtime_request_id):
        require(
            type(runtime_request_id) is str and runtime_request_id,
            "actual source runtime ID required for sampling",
        )
        self.current()
        if runtime_request_id not in self.cache:
            self.cache[runtime_request_id] = (
                runtime_request_id in self._actual_mappings()
            )
        # This Boolean gates instrumentation only. It is never an ACK or a
        # VerifiedNativeRequest. The original full verifier remains mandatory.
        return self.cache[runtime_request_id]

    def fact(self, fact):
        self.current()
        require(fact["process"] == self.own_process, "selection source process differs")
        if fact["process"]["role"] == "api-server":
            require(
                fact["schema"] == "ecpa-native-request-id-mapping/v1",
                "only actual API mapping may select source requests",
            )
            return fact["external_request_id"] in self.selected
        require(fact["process"]["role"] == "worker", "unexpected native source role")
        return self.runtime(fact["runtime_request_id"])


def register(value, spec, current):
    global _selector
    if not enabled():
        return None
    require(_selector is None, "sample selector already registered")
    _selector = SampledSelection(value, spec, current)
    return _selector


def runtime_request_selected(runtime_request_id):
    if not enabled():
        return True
    if _selector is None:
        from .native_block_table_observer import register_observer
        from .native_semantic_bootstrap import ensure_registered

        require(
            ensure_registered(register_observer),
            "H4 sample instrumentation has no actual frozen source context",
        )
    selector = _selector
    if selector is None:
        raise ValueError("H4 sample selector not source-registered")
    return selector.runtime(runtime_request_id)
