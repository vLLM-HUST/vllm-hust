# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core import kv_materialization
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.kv_materialization import (
    KV_MATERIALIZATION_RUNTIME_CONTROL_KEY,
    register_kv_materialization_runtime_observer,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.skip_global_cleanup


def _control(
    decision: str,
    *,
    prompt_tokens: int = 20,
    target_reuse_tokens: int = 0,
    tail_salt: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "observed_decision": decision,
        "effective_decision": decision,
        "control_path": "test",
        "cache_salt": "kvmat:test",
        "decision_supported": True,
        "support_tier": "native_runtime_action",
        "fallback_reason": None,
        "target_reuse_tokens": target_reuse_tokens,
        "target_tail_tokens": prompt_tokens - target_reuse_tokens,
        "segmented_tail_cache_salt": tail_salt,
        "requires_segmented_materialization": decision == "partial_reuse",
    }


def _request(control: dict[str, object], prompt_tokens: int = 20) -> Request:
    params = SamplingParams.from_optional(
        max_tokens=1,
        extra_args={KV_MATERIALIZATION_RUNTIME_CONTROL_KEY: control},
    )
    return Request("request", list(range(prompt_tokens)), params, None)


def _manager(hash_block_size: int = 4) -> KVCacheManager:
    manager = object.__new__(KVCacheManager)
    manager.hash_block_size = hash_block_size
    return manager


def test_request_parses_typed_materialization_metadata() -> None:
    request = _request(_control("full_reuse", target_reuse_tokens=16))

    assert request.kv_materialization_runtime_control is not None
    assert request.kv_materialization_runtime_control.effective_decision == (
        "full_reuse"
    )
    assert request.cache_salt == "kvmat:test"


def test_request_rejects_conflicting_prompt_cache_salt() -> None:
    params = SamplingParams.from_optional(
        max_tokens=1,
        extra_args={
            KV_MATERIALIZATION_RUNTIME_CONTROL_KEY: _control(
                "full_reuse", target_reuse_tokens=16
            )
        },
    )

    with pytest.raises(ValueError, match="cache_salt conflicts"):
        Request(
            "request",
            list(range(20)),
            params,
            None,
            cache_salt="caller-owned",
        )


def test_invalid_materialization_metadata_fails_closed() -> None:
    control = _control("recompute")
    control["target_reuse_tokens"] = 4
    control["target_tail_tokens"] = 16

    with pytest.raises(ValueError, match="recompute requires"):
        _request(control)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("effective_decision", [])],
)
def test_malformed_control_types_fail_closed(field: str, value: object) -> None:
    control = _control("recompute")
    control[field] = value

    with pytest.raises(ValueError):
        _request(control)


def test_recompute_disables_lookup_and_commit() -> None:
    request = _request(_control("recompute"))
    request.block_hashes = [b"0", b"1", b"2"]
    manager = _manager()

    assert manager._get_lookup_block_hashes(request) == []
    assert manager._get_cacheable_num_tokens(request, 20) == 0


def test_partial_reuse_uses_one_aligned_boundary() -> None:
    request = _request(
        _control(
            "partial_reuse",
            target_reuse_tokens=9,
            tail_salt="kvmat:segment:test",
        )
    )
    request.block_hashes = [b"0", b"1", b"2"]
    manager = _manager()

    assert manager._get_lookup_block_hashes(request) == [b"0", b"1"]
    assert manager._get_cacheable_num_tokens(request, 20) == 8
    assert request.kv_materialization_runtime_control is not None
    assert request.kv_materialization_runtime_control.reuse_boundary(4) == 8


def test_segmented_tail_hash_isolated_from_prefix() -> None:
    def stable_hash(value: object) -> bytes:
        return hashlib.sha256(repr(value).encode()).digest()

    init_none_hash(stable_hash)
    block_hasher = get_request_block_hasher(2, stable_hash)
    control = _control(
        "partial_reuse",
        prompt_tokens=8,
        target_reuse_tokens=4,
        tail_salt="kvmat:segment:test",
    )
    params = SamplingParams.from_optional(
        max_tokens=1,
        extra_args={KV_MATERIALIZATION_RUNTIME_CONTROL_KEY: control},
    )
    request_a = Request(
        "a", [1, 2, 3, 4, 9, 10, 11, 12], params, None, block_hasher=block_hasher
    )
    request_b = Request(
        "b", [5, 6, 7, 8, 9, 10, 11, 12], params, None, block_hasher=block_hasher
    )

    assert request_a.block_hashes[:2] != request_b.block_hashes[:2]
    assert request_a.block_hashes[2:] == request_b.block_hashes[2:]


def test_runtime_observer_registration_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kv_materialization, "_runtime_observers", {})
    events: list[dict[str, object]] = []

    def observer(event: object) -> None:
        events.append(dict(event))  # type: ignore[arg-type]

    register_kv_materialization_runtime_observer("example", observer)
    register_kv_materialization_runtime_observer("example", observer)
    kv_materialization.emit_kv_materialization_runtime_event({"event": "lookup"})

    assert events == [{"event": "lookup"}]
