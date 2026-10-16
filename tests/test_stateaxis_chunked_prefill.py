# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.stateaxis_chunked_prefill import (
    MOD_ID,
    apply_chunked_prefill_limit,
    parse_chunked_prefill_experiment,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    """This pure contract test never initializes a device allocator."""

    return False


def config(chunk_tokens: int = 1024) -> dict[str, object]:
    return {
        "experiment_mode": True,
        "stateaxis_mod": {
            "mod_id": MOD_ID,
            "version": "0.3.0",
            "manifest_sha256": "0" * 64,
            "performance_qualified": False,
        },
        "stateaxis_chunked_prefill": {
            "chunk_tokens": chunk_tokens,
            "contention_only": True,
            "inplace_continuation": True,
        },
    }


def parse(value: dict[str, object]):
    return parse_chunked_prefill_experiment(
        value,
        device_type="npu",
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
    )


def test_default_off() -> None:
    assert (
        parse_chunked_prefill_experiment(
            {},
            device_type="npu",
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,
        )
        is None
    )


def test_contention_applies_bounded_chunk() -> None:
    experiment = parse(config())
    assert experiment is not None
    decision = apply_chunked_prefill_limit(
        experiment,
        2495,
        active_decode=True,
    )
    assert decision.num_new_tokens == 1024
    assert decision.chunks_applied == 1
    assert decision.tokens_deferred == 1471


def test_no_contention_bypasses() -> None:
    experiment = parse(config())
    assert experiment is not None
    decision = apply_chunked_prefill_limit(
        experiment,
        2495,
        active_decode=False,
    )
    assert decision.num_new_tokens == 2495
    assert decision.bypassed_no_contention == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(experiment_mode=False), "experiment_mode"),
        (
            lambda value: value["stateaxis_mod"].update(mod_id="wrong"),
            "matching StateAxis MOD binding",
        ),
        (
            lambda value: value["stateaxis_mod"].update(manifest_sha256="bad"),
            "SHA-256 manifest binding",
        ),
        (
            lambda value: value["stateaxis_chunked_prefill"].update(chunk_tokens=0),
            "chunk_tokens",
        ),
    ],
)
def test_invalid_contract_fails_closed(mutation, match: str) -> None:
    value = config()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        parse(value)
