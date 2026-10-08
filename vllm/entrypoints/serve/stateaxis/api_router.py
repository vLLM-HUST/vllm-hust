# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in evidence collection over the existing EngineCore utility channel."""

from fastapi import APIRouter, FastAPI, HTTPException, Request

from vllm.v1.engine.async_llm import AsyncLLM

router = APIRouter()


@router.get("/v1/stateaxis/lifecycle", include_in_schema=False)
async def lifecycle_snapshot(request: Request):
    client = request.app.state.engine_client
    if not isinstance(client, AsyncLLM):
        raise HTTPException(501, "StateAxis snapshots require the V1 async engine")
    return await client.engine_core.get_stateaxis_lifecycle_snapshots_async()


def attach_router(app: FastAPI, additional_config: object) -> None:
    if not isinstance(additional_config, dict):
        return
    config = additional_config.get("stateaxis_lifecycle")
    if isinstance(config, dict) and config.get("mode") == "observe":
        # /v1 uses the existing API authentication; no general-purpose RPC route.
        app.include_router(router)
