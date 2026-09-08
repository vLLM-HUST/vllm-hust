# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.entrypoints.openai.responses.store import BoundedResponseStore


@dataclass
class _Response:
    status: str


def test_store_expires_terminal_entries_and_cleans_linked_state() -> None:
    now = [10.0]
    evicted: list[str] = []
    store = BoundedResponseStore[_Response](
        max_entries=2,
        ttl_seconds=5.0,
        on_evict=evicted.append,
        clock=lambda: now[0],
    )
    store["resp_1"] = _Response(status="completed")

    now[0] = 16.0

    assert store.get("resp_1") is None
    assert evicted == ["resp_1"]


def test_store_evicts_least_recently_used_terminal_entry() -> None:
    store = BoundedResponseStore[_Response](max_entries=2, ttl_seconds=60.0)
    store["resp_1"] = _Response(status="completed")
    store["resp_2"] = _Response(status="completed")
    assert store.get("resp_1") is not None

    store["resp_3"] = _Response(status="completed")

    assert store.get("resp_2") is None
    assert store.get("resp_1") is not None
    assert store.get("resp_3") is not None


def test_store_never_evicts_an_active_background_response() -> None:
    store = BoundedResponseStore[_Response](max_entries=1, ttl_seconds=1.0)
    store["active"] = _Response(status="in_progress")
    store["completed"] = _Response(status="completed")

    assert store.get("active") is not None
    assert store.get("completed") is None
