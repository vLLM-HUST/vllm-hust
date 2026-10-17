# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in source-owned private tensor artifact bootstrap; absent means no proof."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING or (__package__ and __package__.startswith("vllm.")):
    from . import native_semantic_collector as native
    from . import native_tensor_artifact_transport as transport
else:
    from experiments.exposure_gate import native_semantic_collector as native
    from experiments.native_kv_tensor_candidate import artifact_transport as transport

SCHEMA = "ecpa-selected-KV-private-artifact-bootstrap/v1"
RUNTIME_SCHEMA = "ecpa-selected-KV-private-artifact-runtime-bootstrap/v1"
ENV = "ECPA_NATIVE_TENSOR_ARTIFACT_BOOTSTRAP"


class _SampleSelectableCallback(Protocol):
    native_request_selected: Callable[[str], bool]


def callback_from_context(path, *, actual_scope=None):
    path = Path(path)
    raw, info = native.read_private(path, 0o600)
    value = json.loads(raw)
    transport.require(
        set(value)
        == {
            "schema",
            "classification",
            "formal_registration",
            "spec_path",
            "spec_sha256",
            "policy_path",
            "policy_sha256",
            "ready",
            "source_files",
        }
        and value["schema"] in {SCHEMA, RUNTIME_SCHEMA}
        and value["classification"]
        == (
            transport.runtime_contract.CLASSIFICATION
            if value["schema"] == RUNTIME_SCHEMA
            else "CPU_SOURCE_CANDIDATE_ONLY"
        )
        and value["formal_registration"] is False,
        "tensor bootstrap scope is unsupported",
    )
    spec_raw, _ = native.read_private(Path(value["spec_path"]), 0o600)
    policy_raw, _ = native.read_private(Path(value["policy_path"]), 0o600)
    transport.require(
        native.sha(spec_raw) == value["spec_sha256"]
        and native.sha(policy_raw) == value["policy_sha256"],
        "tensor bootstrap frozen inputs changed",
    )
    spec, policy = json.loads(spec_raw), json.loads(policy_raw)
    is_runtime = policy["schema"] == transport.RUNTIME_POLICY
    transport.require(
        (value["schema"] == RUNTIME_SCHEMA) == is_runtime
        and value["classification"] == policy["classification"],
        "runtime bootstrap/policy contract mismatch",
    )
    senders = {s["process"]["pid"]: s for s in spec["senders"]}
    sender = senders.get(os.getpid())
    transport.require(
        sender is not None
        and sender["process"]["role"] == "worker"
        and all(
            sender["process"][k] == v for k, v in native.identity(os.getpid()).items()
        ),
        "tensor bootstrap current kernel worker is unregistered",
    )
    assert sender is not None
    own = policy["worker_directories"][str(os.getpid())]
    fd, _ = transport.directory(own["path"], own["identity"])
    os.close(fd)
    native.source_digests(value["source_files"])
    native.source_digests(sender["source_files"])
    if actual_scope is None:
        from vllm.plugins.native_block_table_observer import _scope

        def actual_scope():
            return _scope("worker")

    if os.environ.get("ECPA_NATIVE_ATTENTION_OPERATOR_OBSERVER") == "1":
        import torch
        import torch_npu

        from .native_attention_operator_observer import (
            register_source_candidate,
            registered_materialization_inventory,
        )
        from .native_block_table_observer import _scope, emit_native_fact

        inventory = None
        if is_runtime:
            # Full existing transport predicate checks the raw private scope,
            # source closure, TP2 directories, all phases and layer anchors.
            transport.policy_predicate(policy, spec, native.sha(spec_raw))
            inventory = registered_materialization_inventory(
                spec_raw, policy_raw, sender["process"]
            )
        register_source_candidate(
            lambda fact: emit_native_fact(lambda: fact),
            scope=lambda: _scope("worker"),
            current_stream=cast(Any, torch).npu.current_stream,
            get_format=torch_npu.get_npu_format,
            materialization_inventory=inventory,
        )
    inode, digest = (info.st_dev, info.st_ino), native.sha(raw)
    ready = value["ready"]

    def callback(fact):
        current, meta = native.read_private(path, 0o600)
        transport.require(
            (meta.st_dev, meta.st_ino) == inode and native.sha(current) == digest,
            "tensor bootstrap context changed",
        )
        native.source_digests(value["source_files"])
        native.source_digests(sender["source_files"])
        transport.require(
            native.read_private(Path(value["spec_path"]), 0o600)[0] == spec_raw
            and native.read_private(Path(value["policy_path"]), 0o600)[0] == policy_raw,
            "tensor bootstrap preregistration changed",
        )

        def bound_scope():
            scope = actual_scope()
            transport.require(
                scope["process"] == sender["process"],
                "actual native tensor worker role changed",
            )
            return scope

        published = (
            transport.runtime_contract.from_actual_native_readback(
                fact, bound_scope()["process"]
            )
            if is_runtime
            else fact
        )
        from .native_phase_drain import acknowledged_publication

        ack = acknowledged_publication(
            published["binding"]["runtime_request_id"],
            lambda: transport.publish_candidate(
                published,
                directory_path=own["path"],
                directory_identity=own["identity"],
                policy_sha256=value["policy_sha256"],
                scope=bound_scope,
                socket_path=ready["socket_path"],
                runtime=is_runtime,
                receiver={
                    "expected_collector": ready["collector"],
                    "expected_socket_identity": tuple(ready["socket_identity"]),
                    "expected_collector_source_sha256": ready[
                        "collector_source_sha256"
                    ],
                    "expected_archive_identity": ready["archive_identity"],
                },
            ),
        )
        if is_runtime:
            from .native_attention_operator_observer import (
                remember_acknowledged_materialization,
            )

            remember_acknowledged_materialization(published, ack)
        return ack

    from .native_sampled_selection import runtime_request_selected

    cast(_SampleSelectableCallback, callback).native_request_selected = (
        runtime_request_selected
    )
    return callback


def ensure_registered(register):
    raw = os.environ.get(ENV)
    if not raw:
        return False
    path = Path(raw)
    transport.require(path.is_absolute(), "tensor bootstrap path must be absolute")
    if not path.exists():
        return False
    register(callback_from_context(path))
    return True


def write_context(path, *, spec_path, policy_path, ready, source_files):
    """Launcher prepares input only; readiness must be authenticated independently."""
    policy = json.loads(Path(policy_path).read_bytes())
    is_runtime = policy["schema"] == transport.RUNTIME_POLICY
    value = {
        "schema": RUNTIME_SCHEMA if is_runtime else SCHEMA,
        "classification": policy["classification"],
        "formal_registration": False,
        "spec_path": str(spec_path),
        "spec_sha256": native.sha(Path(spec_path).read_bytes()),
        "policy_path": str(policy_path),
        "policy_sha256": native.sha(Path(policy_path).read_bytes()),
        "ready": ready,
        "source_files": source_files,
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        raw = memoryview(native.canonical(value))
        while raw:
            count = os.write(fd, raw)
            if count <= 0:
                raise OSError("tensor bootstrap short write")
            raw = raw[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    parent, _ = transport.directory(Path(path).parent)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
