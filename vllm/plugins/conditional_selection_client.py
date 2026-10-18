"""Actual authenticated original-RID control lookup, before sampled fact creation.

No transport error is a negative selection. Wrong/missing/replayed peer, context
or unknown original RID fails closed, never masquerades as an unselected call.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import struct
import threading
from pathlib import Path

_LOCK = threading.RLock()
_STATE = None
_SEQUENCE = dict(api=0, scheduler=0)
SCHEMA = "ecpa-h4-conditional-original-rid-selection/v1"


def require(value, why):
    if not value:
        raise ValueError(why)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def kernel(pid):
    row = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    require(row[0] != "Z", "selection authority/participant dead")
    return dict(
        pid=pid,
        start_ticks=int(row[19]),
        session=int(row[3]),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )


def _context():
    global _STATE
    path = Path(os.environ["ECPA_H4_CONDITIONAL_SELECTION_CONTEXT"])
    require(
        path.is_absolute() and path.resolve() == path, "canonical selection context"
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(
            stat.S_ISREG(before.st_mode)
            and stat.S_IMODE(before.st_mode) == 0o600
            and before.st_uid == os.geteuid(),
            "owned private selection context required",
        )
        raw = os.read(fd, 131073)
        after = os.fstat(fd)
        require(
            all(
                getattr(before, k) == getattr(after, k)
                for k in (
                    "st_dev",
                    "st_ino",
                    "st_uid",
                    "st_mode",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
            and len(raw) <= 131072,
            "selection context changed/oversized",
        )
    finally:
        os.close(fd)
    require(
        (path.lstat().st_dev, path.lstat().st_ino) == (before.st_dev, before.st_ino),
        "context name replaced",
    )
    if _STATE is None:
        _STATE = raw
    require(raw == _STATE, "original selection context changed")
    value = json.loads(raw)
    require(
        value["schema"] == SCHEMA
        and value["plan_id"] == os.environ["VLLM_ECPA_PLAN_ID"]
        and value["launch_id"] == os.environ["VLLM_ECPA_LAUNCH_ID"],
        "wrong actual Plan/launch context",
    )
    return value


def exchange(role, operation, payload):
    with _LOCK:
        context = _context()
        require(
            kernel(os.getpid()) == context["kernels"][role],
            "caller not enrolled original actual kernel",
        )
        path = Path(context["socket_path"])
        before = path.lstat()
        require(
            stat.S_ISSOCK(before.st_mode)
            and before.st_uid == os.geteuid()
            and [before.st_dev, before.st_ino] == context["socket_identity"],
            "actual control socket replaced",
        )
        binding = {
            k: context[k]
            for k in ("schema", "start_id", "plan_id", "launch_id", "nonce")
        }
        sequence = _SEQUENCE[role] + 1
        packet = dict(
            binding, role=role, sequence=sequence, operation=operation, payload=payload
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(10)
            connection.connect(str(path))
            pid, uid, _ = struct.unpack(
                "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            require(
                uid == context["uid"]
                and pid == context["authority_kernel"]["pid"]
                and kernel(pid) == context["authority_kernel"],
                "wrong authority exact kernel",
            )
            connection.sendall(canonical(packet))
            raw = connection.recv(131073)
        require(len(raw) <= 131072, "selection response oversized")
        value = json.loads(raw)
        require(
            canonical(value) == raw
            and set(value) == set(binding) | {"sequence", "result", "request_sha256"}
            and all(value[k] == v for k, v in binding.items())
            and value["sequence"] == sequence
            and value["request_sha256"]
            == hashlib.sha256(canonical(packet)).hexdigest(),
            "selection response replay/binding mismatch",
        )
        after = path.lstat()
        require(
            (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino),
            "control socket changed across lookup",
        )
        _SEQUENCE[role] = sequence
        return value["result"], dict(
            role=role, sequence=sequence, request_sha256=value["request_sha256"]
        )


def publish_mapping(request):
    result, reference = exchange(
        "api",
        "mapping",
        dict(
            external_request_id=request.external_req_id,
            runtime_request_id=request.request_id,
        ),
    )
    require(
        set(result) == {"selected"} and type(result["selected"]) is bool,
        "invalid selection response",
    )
    return result["selected"], reference


def selected_context(kind, context):
    values = context.requests if kind == "batch" else context.candidates
    rids = [r.request_id for r in values]
    result, reference = exchange("scheduler", "query", dict(runtime_request_ids=rids))
    require(
        set(result) == {"selected"}
        and type(result["selected"]) is list
        and len(result["selected"]) == len(set(result["selected"]))
        and set(result["selected"]) <= set(rids),
        "invalid scheduler selection",
    )
    return bool(result["selected"]), reference
