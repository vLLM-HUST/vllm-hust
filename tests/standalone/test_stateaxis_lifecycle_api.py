# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real API/client/utility methods with CPU queues replacing the wire.

Run in the owned CPU container with installed vLLM dependencies. No model,
worker bootstrap or NPU is used; ASGI requests do not open a network listener.
"""

import asyncio
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI

from vllm.entrypoints.serve.stateaxis.api_router import attach_router
from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware
from vllm.v1.core.stateaxis_lifecycle import LifecycleObserver
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    DPLBAsyncMPClient,
    _process_utility_output,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def client_with_cpu_wire(observers):
    client = AsyncMPClient.__new__(AsyncMPClient)
    client.core_engines = [rank.to_bytes(2, "little") for rank in range(len(observers))]
    client.core_engine = client.core_engines[0]
    client.client_index = 0
    client.utility_results = {}
    client.encoder = MsgpackEncoder()
    client._ensure_output_queue_task = lambda: None
    calls = []

    async def send(message, engine, args):
        assert message[0] == EngineCoreRequestType.UTILITY.value
        decoded = MsgpackDecoder().decode(message[1:])
        calls.append((engine, decoded[2], decoded[3]))
        core = EngineCoreProc.__new__(EngineCoreProc)
        core.stateaxis_lifecycle = observers[int.from_bytes(engine, "little")]
        core.shutdown_state = EngineShutdownState.RUNNING
        core.output_queue = Queue()
        core._handle_client_request(EngineCoreRequestType.UTILITY, decoded)
        index, outputs = core.output_queue.get_nowait()
        assert index == client.client_index
        _process_utility_output(outputs.utility_output, client.utility_results)

    client._send_input_message = send
    return client, calls


class LifecycleAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_real_utility_off_and_observe_round_trip(self):
        observer = LifecycleObserver()
        observer.admitted(SimpleNamespace(request_id="cpu-only"))
        client, calls = client_with_cpu_wire([None, observer])
        result = await client.get_stateaxis_lifecycle_snapshots_async()
        self.assertFalse(result["cross_rank_atomic"])
        self.assertEqual(result["scope"], "client-managed-engine-cores")
        self.assertIsNone(result["engines"][0]["snapshot"])
        snapshot = result["engines"][1]["snapshot"]
        self.assertEqual(snapshot["active_requests"], 1)
        self.assertFalse(snapshot["observed_lifetimes_closed"])
        self.assertEqual([e["engine_id"] for e in result["engines"]], ["0000", "0100"])
        snapshot["events"].clear()
        self.assertEqual(len(observer.events), 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            all(c[1:] == ("get_stateaxis_lifecycle_snapshot", []) for c in calls)
        )
        self.assertEqual(client.utility_results, {})

    async def test_dp_keeps_both_results_when_second_finishes_first(self):
        client = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
        client.core_engines = [b"\x00\x00", b"\x01\x00"]
        second_done = asyncio.Event()

        async def call(method, *, engine):
            if engine == client.core_engines[0]:
                await second_done.wait()
            else:
                second_done.set()
            return engine.hex()

        client._call_utility_async = call
        result = await client.get_stateaxis_lifecycle_snapshots_async()
        self.assertEqual([e["snapshot"] for e in result["engines"]], ["0000", "0100"])

    async def test_membership_change_rejects_complete_snapshot(self):
        client, _ = client_with_cpu_wire([None])

        async def call(*args, **kwargs):
            client.core_engines.append(b"\x01\x00")

        client._call_utility_async = call
        with self.assertRaisesRegex(RuntimeError, "membership changed"):
            await client.get_stateaxis_lifecycle_snapshots_async()

    async def test_rank_failure_is_not_a_partial_success(self):
        client, _ = client_with_cpu_wire([None, None])
        client._call_utility_async = AsyncMock(side_effect=RuntimeError("rank failed"))
        with self.assertRaisesRegex(RuntimeError, "rank failed"):
            await client.get_stateaxis_lifecycle_snapshots_async()

    async def test_shrink_regrow_same_rank_ids_rejects_stale_membership(self):
        client, _ = client_with_cpu_wire([None, None])

        async def call(*args, **kwargs):
            client.core_engines = client.core_engines[:1]
            client.core_engines.append(b"\x01\x00")

        client._call_utility_async = call
        with self.assertRaisesRegex(RuntimeError, "membership changed"):
            await client.get_stateaxis_lifecycle_snapshots_async()

    async def test_empty_or_duplicate_engine_set_sends_nothing(self):
        client, calls = client_with_cpu_wire([None])
        for engines in ([], [b"\x00\x00", b"\x00\x00"]):
            client.core_engines = engines
            with self.assertRaisesRegex(RuntimeError, "invalid.*engine set"):
                await client.get_stateaxis_lifecycle_snapshots_async()
        self.assertEqual(calls, [])

    async def test_disabled_route_is_absent(self):
        for config in (None, object(), {}, {"stateaxis_lifecycle": {"mode": "off"}}):
            app = FastAPI()
            attach_router(app, config)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://fixture"
            ) as http:
                response = await http.get("/v1/stateaxis/lifecycle")
                self.assertEqual(response.status_code, 404)

    async def test_http_route_uses_existing_auth_and_fixed_core_method(self):
        client, calls = client_with_cpu_wire([LifecycleObserver()])
        frontend = AsyncLLM.__new__(AsyncLLM)
        frontend.engine_core = client
        self.addCleanup(lambda: delattr(frontend, "engine_core"))
        app = FastAPI()
        app.state.engine_client = frontend
        attach_router(app, {"stateaxis_lifecycle": {"mode": "observe"}})
        # Literal test credential, unrelated to the environment or any service.
        app.add_middleware(AuthenticationMiddleware, tokens=["cpu-fixture-token"])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture"
        ) as http:
            denied = await http.get("/v1/stateaxis/lifecycle")
            self.assertEqual(denied.status_code, 401)
            self.assertEqual(calls, [])
            result = await http.get(
                "/v1/stateaxis/lifecycle",
                headers={"Authorization": "Bearer cpu-fixture-token"},
            )
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["engines"][0]["snapshot"]["trace_lossless"])
        self.assertEqual(len(calls), 1)

    async def test_unsupported_frontend_is_explicit(self):
        app = FastAPI()
        app.state.engine_client = object()
        attach_router(app, {"stateaxis_lifecycle": {"mode": "observe"}})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture"
        ) as http:
            result = await http.get("/v1/stateaxis/lifecycle")
        self.assertEqual(result.status_code, 501)


if __name__ == "__main__":
    unittest.main()
