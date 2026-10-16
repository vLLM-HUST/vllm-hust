# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.stateaxis_length_aware_queue import (
    MOD_ID,
    QueueItem,
    parse_length_aware_queue_experiment,
    plan_length_aware_order,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def config() -> dict[str, object]:
    return {
        "experiment_mode": True,
        "stateaxis_mod": {
            "mod_id": MOD_ID,
            "version": "0.2.0",
            "manifest_sha256": "0" * 64,
            "performance_qualified": False,
        },
        "stateaxis_length_aware_queue": {
            "enabled": True,
            "telemetry_enabled": True,
            "authority_enabled": True,
            "scope": "cold-prefill-no-state",
            "short_max_tokens": 128,
            "medium_max_tokens": 512,
            "aging_ms": 250,
            "max_queued_requests": 64,
            "max_queued_tokens": 1_048_576,
            "atomic_full_plan_reorder": True,
            "trace_identity_aligned": True,
            "stateful_or_decode_fallback": True,
            "monotonic_clock_fail_closed": True,
            "fail_closed": True,
        },
    }


def experiment():
    parsed = parse_length_aware_queue_experiment(config(), device_type="npu")
    assert parsed is not None
    return parsed


def item(name: str, tokens: int, arrival: float, eligible: bool = True):
    return QueueItem(name, tokens, arrival, eligible)


def test_default_off() -> None:
    assert parse_length_aware_queue_experiment({}, device_type="npu") is None


def test_orders_unaged_cold_prefills_by_bounded_length_class() -> None:
    plan = plan_length_aware_order(
        experiment(),
        [item("long", 900, 10.0), item("short", 64, 10.1), item("medium", 300, 10.2)],
        now=10.2,
    )
    assert plan.request_ids == ("short", "medium", "long")
    assert plan.reordered is True
    assert plan.fallback_reason is None


def test_aged_request_is_promoted_ahead_of_shorter_new_request() -> None:
    plan = plan_length_aware_order(
        experiment(),
        [item("new-short", 64, 10.9), item("aged-long", 900, 10.0)],
        now=11.0,
    )
    assert plan.request_ids == ("aged-long", "new-short")
    assert plan.aged_promotions == 1


@pytest.mark.parametrize(
    ("items", "now", "reason"),
    [
        ([item("stateful", 64, 10.0, False)], 10.0, "ineligible_request"),
        ([item("future", 64, 11.0)], 10.0, "clock_regression"),
        ([item(str(index), 1, 10.0) for index in range(65)], 10.0, "request_bound"),
    ],
)
def test_bounded_fail_closed_fallback(items, now: float, reason: str) -> None:
    plan = plan_length_aware_order(experiment(), items, now=now)
    assert plan.reordered is False
    assert plan.fallback_reason == reason


def test_wrong_identity_fails_closed() -> None:
    value = config()
    value["stateaxis_mod"]["mod_id"] = "wrong"  # type: ignore[index]
    with pytest.raises(ValueError, match="matching StateAxis MOD binding"):
        parse_length_aware_queue_experiment(value, device_type="npu")
