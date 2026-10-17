# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Source-state CPU fixtures only; no tensor/device/model/qualification claim."""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PLUGINS = Path(__file__).resolve().parents[2] / "vllm/plugins"


def load_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, PLUGINS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def observer():
    return load_file(
        "capacity_observer_fixture", "native_attention_operator_observer.py"
    )


def fixture_registration(phases=60, layers=28):
    worker = dict(
        host="CPU-only",
        role="worker",
        ordinal=0,
        pid=100,
        start_identity="pid:100:start_ticks:99",
        process_epoch=1,
        assignment_source="host",
        start_ticks=99,
        session=100,
        boot_id="CPU-only",
    )
    phase_rows = [
        dict(
            phase_id=f"p-{i:03d}",
            client_request_id=f"r-{i:03d}",
            external_request_id=f"cmpl-r-{i:03d}-0",
            challenge_nonce=f"nonce-{i:03d}",
            expected_ranks=[0, 1],
        )
        for i in range(phases)
    ]
    spec = dict(
        schema="ecpa-native-kv-semantic-collector-prereg/v2",
        classification="CPU_REVIEW_CANDIDATE_ONLY",
        formal_producer_registration=False,
        senders=[dict(process=worker, source_files=[])],
        phases=phase_rows,
        kv_requirements={
            p["phase_id"]: dict(semantic_anchor=1195, physical_anchor=1024)
            for p in phase_rows
        },
    )
    policy = dict(
        schema="ecpa-selected-KV-private-artifact-runtime-policy/v1",
        classification="REAL_RUNTIME_UNQUALIFIED",
        formal_registration=False,
        requirements={
            p["phase_id"]: dict(
                layer_indices=list(range(layers)),
                semantic_anchor=1195,
                physical_anchor=1024,
            )
            for p in phase_rows
        },
    )
    return worker, spec, policy


def encode_registration(observer, spec, policy):
    spec_raw = observer.canonical(spec)
    policy = copy.deepcopy(policy)
    policy["native_spec_sha256"] = observer.sha(spec_raw)
    return spec_raw, observer.canonical(policy)


def register(observer, worker, inventory=None):
    observer.register_source_candidate(
        lambda fact: None,
        scope=lambda: {"process": worker},
        current_stream=lambda device: None,
        get_format=lambda tensor: 0,
        materialization_inventory=inventory,
    )


def remembered_fact(observer, worker, phase, layer, *, layers=28):
    fact = dict(
        native_producer=worker,
        classification="REAL_RUNTIME_UNQUALIFIED",
        binding=dict(
            runtime_request_id=phase["client_request_id"],
            layer_name=f"model.layers.{layer}.self_attn",
            rank=0,
            layer_index=layer,
            expected_layer_indices=list(range(layers)),
        ),
    )
    ack = dict(
        schema="ecpa-native-semantic-collector-ack/v1",
        formal_producer_registration=False,
        phase=copy.deepcopy(phase),
    )
    ack["ack_sha256"] = observer.sha(observer.canonical(ack))
    return fact, ack


def test_registered_sixty_by_twenty_eight_retains_all_history(observer):
    worker, spec, policy = fixture_registration()
    spec_raw, policy_raw = encode_registration(observer, spec, policy)
    token = observer.registered_materialization_inventory(spec_raw, policy_raw, worker)
    assert token.capacity == 1680  # per worker, not duplicated by TP2 rank count
    register(observer, worker, token)
    first = None
    for phase in spec["phases"]:
        for layer in range(28):
            fact, ack = remembered_fact(observer, worker, phase, layer)
            observer.remember_acknowledged_materialization(fact, ack)
            if first is None:
                first = copy.deepcopy(fact), copy.deepcopy(ack)
    assert len(observer._materialized) == len(observer._registered_phase_layers) == 1680
    original = observer._materialized[("r-000", "model.layers.0.self_attn")]
    assert original["fact"] == first[0]
    assert original["ack"] == first[1]
    assert original["fact_sha256"] == observer.sha(observer.canonical(first[0]))
    fact, ack = remembered_fact(observer, worker, spec["phases"][0], 0)
    with pytest.raises(ValueError, match="repeated"):
        observer.remember_acknowledged_materialization(fact, ack)
    assert len(observer._materialized) == 1680


def test_undeclared_legacy_registration_remains_bounded_at_sixty_four(observer):
    worker, spec, _ = fixture_registration(phases=65, layers=1)
    register(observer, worker)
    for phase in spec["phases"][:64]:
        observer.remember_acknowledged_materialization(
            *remembered_fact(observer, worker, phase, 0, layers=1)
        )
    with pytest.raises(ValueError, match="bounded fresh"):
        observer.remember_acknowledged_materialization(
            *remembered_fact(observer, worker, spec["phases"][64], 0, layers=1)
        )
    assert len(observer._materialized) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        "phase-missing",
        "phase-duplicate",
        "rank-bool",
        "rank-missing",
        "layer-missing",
        "layer-duplicate",
        "layer-bool",
        "anchor-mismatch",
        "spec-hash",
        "sender-missing",
        "sender-repeated",
    ],
)
def test_registration_inventory_must_exactly_cover_original_phases_and_layers(
    observer, mutation
):
    worker, spec, policy = fixture_registration(phases=3)
    if mutation == "phase-missing":
        del policy["requirements"]["p-001"]
    elif mutation == "phase-duplicate":
        spec["phases"][1] = copy.deepcopy(spec["phases"][0])
    elif mutation == "rank-bool":
        spec["phases"][0]["expected_ranks"] = [0, True]
    elif mutation == "rank-missing":
        spec["phases"][0]["expected_ranks"] = [0]
    elif mutation == "layer-missing":
        policy["requirements"]["p-001"]["layer_indices"] = []
    elif mutation == "layer-duplicate":
        policy["requirements"]["p-001"]["layer_indices"] = [0, 0]
    elif mutation == "layer-bool":
        policy["requirements"]["p-001"]["layer_indices"] = [True]
    elif mutation == "anchor-mismatch":
        policy["requirements"]["p-001"]["physical_anchor"] = 1
    elif mutation == "sender-missing":
        spec["senders"] = []
    elif mutation == "sender-repeated":
        spec["senders"] *= 2
    raw, policy_raw = encode_registration(observer, spec, policy)
    if mutation == "spec-hash":
        parsed = json.loads(policy_raw)
        parsed["native_spec_sha256"] = "0" * 64
        policy_raw = observer.canonical(parsed)
    with pytest.raises((ValueError, KeyError)):
        observer.registered_materialization_inventory(raw, policy_raw, worker)


@pytest.mark.parametrize("value", [1680, {"capacity": 1680}])
def test_owner_supplied_capacity_is_not_a_registered_token(observer, value):
    worker, _, _ = fixture_registration()
    with pytest.raises(ValueError, match="source-derived inventory"):
        register(observer, worker, value)


@pytest.mark.parametrize(
    "mutation", ["phase", "rank", "layer", "layer-list", "ack", "token", "forged-token"]
)
def test_real_ack_and_exact_registered_request_layer_checks_remain(observer, mutation):
    worker, spec, policy = fixture_registration(phases=3)
    token = observer.registered_materialization_inventory(
        *encode_registration(observer, spec, policy), worker
    )
    register(observer, worker, token)
    fact, ack = remembered_fact(observer, worker, spec["phases"][0], 0)
    if mutation == "phase":
        ack["phase"]["external_request_id"] = "different-runtime-mapping"
        ack["ack_sha256"] = observer.sha(
            observer.canonical({k: v for k, v in ack.items() if k != "ack_sha256"})
        )
    elif mutation == "rank":
        fact["binding"]["rank"] = 1
    elif mutation == "layer":
        fact["binding"]["layer_index"] = 28
    elif mutation == "layer-list":
        fact["binding"]["expected_layer_indices"] = [0]
    elif mutation == "ack":
        ack["ack_sha256"] = "0" * 64
    else:
        token.capacity += 1
        if mutation == "forged-token":
            token.sha256 = observer.sha(
                observer.canonical(
                    {
                        "phases": token.phases,
                        "process": token.process,
                        "rank": token.rank,
                        "capacity": token.capacity,
                    }
                )
            )
    with pytest.raises(ValueError):
        observer.remember_acknowledged_materialization(fact, ack)
    assert not observer._materialized


def test_bootstrap_passes_its_private_checked_spec_policy_to_observer(
    observer, monkeypatch, tmp_path
):
    worker, spec, policy = fixture_registration()
    spec_raw, policy_raw = encode_registration(observer, spec, policy)
    policy = json.loads(policy_raw)
    own = tmp_path / "worker"
    own.mkdir(mode=0o700)
    policy["worker_directories"] = {
        str(os.getpid()): dict(path=str(own), identity=[1, 2])
    }
    worker["pid"] = os.getpid()
    spec["senders"][0]["process"] = worker
    spec_raw, policy_raw = encode_registration(observer, spec, policy)
    spec_path, policy_path, context_path = [
        tmp_path / n for n in ("spec", "policy", "context")
    ]
    spec_path.write_bytes(spec_raw)
    policy_path.write_bytes(policy_raw)
    value = dict(
        schema="ecpa-selected-KV-private-artifact-runtime-bootstrap/v1",
        classification="REAL_RUNTIME_UNQUALIFIED",
        formal_registration=False,
        spec_path=str(spec_path),
        spec_sha256=observer.sha(spec_raw),
        policy_path=str(policy_path),
        policy_sha256=observer.sha(policy_raw),
        ready={},
        source_files=[],
    )
    context_path.write_bytes(observer.canonical(value))
    package = "vllm.capacity_fixture"
    parent = ModuleType(package)
    parent.__path__ = [str(PLUGINS)]
    monkeypatch.setitem(sys.modules, package, parent)
    native = SimpleNamespace(
        read_private=lambda path, mode: (Path(path).read_bytes(), Path(path).stat()),
        sha=observer.sha,
        source_digests=lambda files: None,
        identity=lambda pid: {
            k: worker[k] for k in ("pid", "start_ticks", "session", "boot_id")
        },
    )
    validations = []
    transport = SimpleNamespace(
        require=observer.require,
        RUNTIME_POLICY=policy["schema"],
        runtime_contract=SimpleNamespace(CLASSIFICATION="REAL_RUNTIME_UNQUALIFIED"),
        directory=lambda path, expected: (os.open(path, os.O_RDONLY), None),
        policy_predicate=lambda p, s, h: validations.append((p, s, h)),
    )
    block = SimpleNamespace(
        _scope=lambda role: {"process": worker}, emit_native_fact=lambda factory: None
    )
    for name, module in {
        "native_semantic_collector": native,
        "native_tensor_artifact_transport": transport,
        "native_attention_operator_observer": observer,
        "native_block_table_observer": block,
    }.items():
        monkeypatch.setitem(sys.modules, package + "." + name, module)
        setattr(parent, name, module)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device: None)),
    )
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(get_npu_format=lambda x: 0)
    )
    monkeypatch.setenv("ECPA_NATIVE_ATTENTION_OPERATOR_OBSERVER", "1")
    bootstrap = load_file(package + ".bootstrap", "native_tensor_artifact_bootstrap.py")
    callback = bootstrap.callback_from_context(
        context_path, actual_scope=lambda: {"process": worker}
    )
    assert callable(callback)
    assert len(validations) == 1
    assert validations[0][2] == observer.sha(spec_raw)
    assert observer._materialization_inventory.capacity == 1680
    assert observer._materialization_inventory.process == worker
