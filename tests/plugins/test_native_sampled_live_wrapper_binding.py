# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual bootstrap ABI composition; CPU fixtures, never native qualification."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.plugins import native_block_table_observer as block
from vllm.plugins import native_observer_rebind as rebind
from vllm.plugins import native_phase_drain as drain
from vllm.plugins import native_sampled_selection as selection
from vllm.plugins import native_semantic_bootstrap as semantic
from vllm.plugins import native_tensor_artifact_bootstrap as tensor


def private_json(path, value):
    path.write_bytes(semantic.canonical(value))
    path.chmod(0o600)
    return path


@pytest.fixture
def original_cpu_contexts(tmp_path):
    tmp_path.chmod(0o700)
    process = dict(semantic_source_kernel(), role="worker", ordinal=0)
    source_path = Path(__file__).resolve()
    source_files = [
        {
            "path": str(source_path),
            "sha256": semantic.sha(source_path.read_bytes()),
        }
    ]
    spec = {
        "senders": [{"process": process, "source_files": source_files}],
        "phases": [],
    }
    spec_path = private_json(tmp_path / "spec.json", spec)
    ready = dict(
        socket_path=str(tmp_path / "unused.sock"),
        socket_identity=[1, 2],
        collector=semantic_source_kernel(),
        spec_sha256=semantic.sha(spec_path.read_bytes()),
        collector_source_sha256="CPU-only",
        archive_identity={},
    )
    native_path = tmp_path / "semantic-context.json"
    semantic.write_context(
        native_path, spec_path=spec_path, ready=ready, source_files=source_files
    )
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir(mode=0o700)
    stat = worker_dir.stat()
    policy = dict(
        schema=tensor.transport.POLICY,
        classification="CPU_SOURCE_CANDIDATE_ONLY",
        worker_directories={
            str(os.getpid()): dict(
                path=str(worker_dir), identity=[stat.st_dev, stat.st_ino]
            )
        },
    )
    policy_path = private_json(tmp_path / "policy.json", policy)
    tensor_path = tmp_path / "tensor-context.json"
    tensor.write_context(
        tensor_path,
        spec_path=spec_path,
        policy_path=policy_path,
        ready=ready,
        source_files=source_files,
    )
    return native_path, tensor_path


def semantic_source_kernel():
    return selection.native.identity(os.getpid())


@pytest.mark.parametrize("channel", ["semantic", "tensor"])
@pytest.mark.parametrize("install_router", [False, True])
def test_final_callback_keeps_selection_contract(
    original_cpu_contexts, monkeypatch, channel, install_router
):
    native_path, tensor_path = original_cpu_contexts
    selector = SimpleNamespace(fact=lambda fact: False)
    monkeypatch.setattr(selection, "register", lambda *args: selector)
    installed = []

    def transport_wrapper(kind, path, callback, builder):
        # Deliberately plain wrapper, as real Router.add returns. It does not
        # inherit callback attributes; this fixture mints no native facts/ACK.
        def routed(fact):
            return callback(fact)

        installed.append((kind, routed))
        return routed

    monkeypatch.setattr(rebind, "install", transport_wrapper)
    if channel == "semantic":
        callback = semantic.callback_from_context(
            native_path, _install_rebind=install_router
        )
        assert callback.h4_sample_selection is selector
        monkeypatch.setattr(block, "_observer", None)
        calls = []

        def no_drain_for_unsampled(original):
            def wrapped(fact):
                calls.append(fact)
                raise AssertionError("unsampled fact reached ACK/drain tracking")

            return wrapped

        monkeypatch.setattr(drain, "callback_with_drain", no_drain_for_unsampled)
        block.register_observer(callback)
        assert block._observer({"CPU-unsampled": True}) is None
        assert calls == []
    else:
        callback = tensor.callback_from_context(
            tensor_path,
            actual_scope=lambda: {},
            _install_rebind=install_router,
            _initialize_attention=False,
        )
        assert callback.native_request_selected is selection.runtime_request_selected
    assert len(installed) == int(install_router)
    if install_router:
        assert installed[0] == (channel, callback)
