# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import hashlib
import time
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

from .s1_s5_observer import state as s1_s5_observer_state

logger = init_logger(__name__)

router = APIRouter()

_QBI_SCHEDULER_MECHANISM = "scheduler_batching"
_QBI_SCHEDULER_FIELDS = {"max_num_scheduled_tokens", "max_num_running_reqs"}
_QBI_TRANSITION_FIELDS = {
    "scheduler_batching": _QBI_SCHEDULER_FIELDS,
    "prefill_admission_guard": {"scheduler_reserve_full_isl"},
    "dynamic_speculative_depth": {"num_speculative_tokens"},
}
_QBI_PREFIX_SESSION_FIELDS = (
    "planned_reuse_ratio",
    "prefix_tokens",
    "session_invocation_count",
    "kv_bytes",
    "namespace",
    "persistence_mode",
    "arrival_gap_ms",
)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _qbi_utility(request: Request):
    client = engine_client(request)
    utility = getattr(client, "call_utility_async", None)
    if not callable(utility):
        utility = getattr(
            getattr(client, "engine_core", None), "call_utility_async", None
        )
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    return utility


async def _qbi_json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {error}",
        ) from error
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Request body must be a JSON object",
        )
    return body


def _qbi_transition_mechanism(body: dict[str, Any]) -> str:
    mechanism_name = body.get("mechanism_name")
    if not isinstance(mechanism_name, str) or mechanism_name not in _QBI_TRANSITION_FIELDS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Supported runtime transitions are scheduler_batching, "
                "prefill_admission_guard, and dynamic_speculative_depth"
            ),
        )
    return mechanism_name


def _qbi_transition_config(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    mechanism_name = _qbi_transition_mechanism(body)
    expected_fields = _QBI_TRANSITION_FIELDS[mechanism_name]
    config = body.get("config")
    if not isinstance(config, dict) or set(config) != expected_fields:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Runtime transition config has unexpected or missing fields",
        )
    if mechanism_name == "prefill_admission_guard":
        value = config["scheduler_reserve_full_isl"]
        if not isinstance(value, bool):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="scheduler_reserve_full_isl must be a boolean",
            )
        return mechanism_name, {"scheduler_reserve_full_isl": value}
    try:
        normalized: dict[str, Any] = {}
        for field in sorted(expected_fields):
            value = config[field]
            if isinstance(value, bool):
                raise TypeError
            normalized[field] = int(value)
        return mechanism_name, normalized
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Runtime transition values must be integers",
        ) from error


def _qbi_scheduler_config(body: dict[str, Any]) -> dict[str, int]:
    if body.get("mechanism_name") != _QBI_SCHEDULER_MECHANISM:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Only scheduler_batching is supported",
        )
    config = body.get("config")
    if not isinstance(config, dict) or set(config) != _QBI_SCHEDULER_FIELDS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Scheduler config has unexpected or missing fields",
        )
    try:
        normalized: dict[str, int] = {}
        for field in sorted(_QBI_SCHEDULER_FIELDS):
            value = config[field]
            if isinstance(value, bool):
                raise TypeError
            normalized[field] = int(value)
        return normalized
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Scheduler config values must be integers",
        ) from error


async def _qbi_call(request: Request, method: str, *args: Any) -> JSONResponse:
    try:
        result = await _qbi_utility(request)(method, *args)
    except (RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Engine returned a non-object runtime transition receipt",
        )
    return JSONResponse(content=result)


@router.post("/collective_rpc")
async def collective_rpc(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e
    method = body.get("method")
    if method is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'method' in request body",
        )
    # For security reason, only serialized string args/kwargs are passed.
    # User-defined `method` is responsible for deserialization if needed.
    args: list[str] = body.get("args", [])
    kwargs: dict[str, str] = body.get("kwargs", {})
    timeout: float | None = body.get("timeout")
    results = await engine_client(raw_request).collective_rpc(
        method=method, timeout=timeout, args=tuple(args), kwargs=kwargs
    )
    if results is None:
        return Response(status_code=200)
    response: list[Any] = []
    for result in results:
        if result is None or isinstance(result, dict | list):
            response.append(result)
        else:
            response.append(str(result))
    return JSONResponse(content={"results": response})


@router.get("/qbi/causal-pre-decision/state")
async def causal_pre_decision_state(raw_request: Request, after_sequence: int = 0):
    """Read bounded scheduler projections captured before each real decision."""

    utility = _qbi_utility(raw_request)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    try:
        result = await utility("get_causal_pre_decision_state", after_sequence)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Engine returned a non-object causal pre-decision state",
        )
    return JSONResponse(content=result)


@router.get("/qbi/native-recapture/scope")
async def native_recapture_scope(raw_request: Request, after_sequence: int = 0):
    """Read prompt-free scheduler receipts from the isolated control surface."""

    utility = _qbi_utility(raw_request)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    try:
        result = await utility("get_native_recapture_scope_state", after_sequence)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Engine returned a non-object native recapture state",
        )
    return JSONResponse(content=result)


@router.get("/qbi/native-recapture/ngram")
async def native_recapture_ngram(raw_request: Request, after_sequence: int = 0):
    """Read the dedicated worker-side N-gram observer readiness surface."""

    utility = _qbi_utility(raw_request)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    try:
        result = await utility("get_native_ngram_scope_state", after_sequence)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Engine returned a non-object native N-gram state",
        )
    return JSONResponse(content=result)


@router.post("/qbi/native-recapture/pre-action")
async def native_recapture_pre_action(raw_request: Request):
    """Render one text chat prompt and snapshot N-gram inputs before send."""
    route_started_ns = time.perf_counter_ns()
    body = await _qbi_json_body(raw_request)
    required = {
        "provider_request_id",
        "source_request_id",
        "source_receipt_sha256",
        "checkpoint_sha256",
        "config_support_range",
        "action_config",
        "payload_sha256",
        "payload",
    }
    if set(body) != required:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="N-gram pre-action body has unexpected or missing fields",
        )
    payload = _qbi_validate_ngram_prompt_payload(body["payload"])
    payload_sha256 = _qbi_json_sha256(payload)
    if body["payload_sha256"] != payload_sha256:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail="N-gram pre-action payload hash differs",
        )

    tokenization = getattr(
        raw_request.app.state, "openai_serving_tokenization", None
    )
    if tokenization is None or not callable(
        getattr(tokenization, "create_tokenize", None)
    ):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="OpenAI serving tokenization is unavailable",
        )
    try:
        from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest

        tokenize_request = TokenizeChatRequest.model_validate(payload)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"N-gram pre-action chat payload is invalid: {error}",
        ) from error

    tokenization_started_ns = time.perf_counter_ns()
    tokenized = await tokenization.create_tokenize(tokenize_request, raw_request)
    tokenization_elapsed_ns = time.perf_counter_ns() - tokenization_started_ns
    token_ids = getattr(tokenized, "tokens", None)
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token) is not int or token < 0 for token in token_ids)
    ):
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value,
            detail="OpenAI tokenization did not return exact text prompt token IDs",
        )
    token_ids_sha256 = _qbi_json_sha256(token_ids)
    tokenizer = tokenization.renderer.get_tokenizer()
    tokenizer_identity_sha256 = _qbi_json_sha256(
        {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        },
        allow_default=True,
    )
    chat_template_sha256 = _qbi_json_sha256(
        {
            "template": (
                tokenize_request.chat_template
                or tokenization.chat_template
                or getattr(tokenizer, "chat_template", None)
            ),
            "content_format": str(tokenization.chat_template_content_format),
            "server_defaults": tokenization.default_chat_template_kwargs,
            "request_kwargs": tokenize_request.chat_template_kwargs,
        },
        allow_default=True,
    )

    engine_started_ns = time.perf_counter_ns()
    try:
        result = await _qbi_utility(raw_request)(
            "get_native_ngram_pre_action_state",
            body["provider_request_id"],
            body["source_request_id"],
            body["source_receipt_sha256"],
            body["checkpoint_sha256"],
            body["config_support_range"],
            body["action_config"],
            token_ids,
            payload_sha256,
            token_ids_sha256,
            tokenizer_identity_sha256,
            chat_template_sha256,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=str(error),
        ) from error
    engine_snapshot_rpc_elapsed_ns = time.perf_counter_ns() - engine_started_ns
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Engine returned a non-object N-gram pre-action state",
        )
    return JSONResponse(
        content={
            **result,
            "tokenization_elapsed_ns": tokenization_elapsed_ns,
            "engine_snapshot_rpc_elapsed_ns": engine_snapshot_rpc_elapsed_ns,
            "pre_action_route_elapsed_ns": time.perf_counter_ns()
            - route_started_ns,
        }
    )


@router.get("/qbi/repetition-detection/state")
async def repetition_detection_state(
    raw_request: Request, after_sequence: int = 0
):
    """Read provider-bound events captured immediately before check_stop."""
    utility = _qbi_utility(raw_request)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    try:
        result = await utility("get_repetition_detection_state", after_sequence)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Repetition detector state is not an object",
        )
    return JSONResponse(content=result)


@router.get("/qbi/priority-scheduling/state")
async def qbi_priority_scheduling_state(
    raw_request: Request, after_sequence: int = 0
):
    """Read request-bound queue competition captured before scheduling."""

    utility = _qbi_utility(raw_request)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    try:
        result = await utility("get_priority_scheduling_state", after_sequence)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Priority scheduling state is not an object",
        )
    return JSONResponse(content=result)


@router.get("/qbi/dynamic-suffix/state")
async def dynamic_suffix_state(raw_request: Request, after_sequence: int = 0):
    """Read completed native suffix proposal/acceptance windows."""

    utility = _qbi_utility(raw_request)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    try:
        result = await utility("get_suffix_acceptance_state", after_sequence)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Suffix acceptance state is not an object",
        )
    return JSONResponse(content=result)


@router.get("/qbi/suffix-preverification/state")
async def suffix_preverification_state(raw_request: Request, after_sequence: int = 0):
    """Read worker candidate state captured before suffix verifier admission."""
    utility = _qbi_utility(raw_request)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    try:
        result = await utility("get_native_suffix_proposal_state", after_sequence)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Suffix pre-verification state is not an object",
        )
    return JSONResponse(content=result)


@router.post("/qbi/suffix-preverification/reset-cache")
async def reset_suffix_preverification_cache(raw_request: Request):
    """Reset the opt-in local Suffix cache for matched TRAIN arms."""
    utility = _qbi_utility(raw_request)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    try:
        result = await utility("reset_native_suffix_cache")
    except (TypeError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Suffix cache reset result is not an object",
        )
    return JSONResponse(content=result)


async def _qbi_s1_s5_observer_state(
    mechanism: str, after_sequence: int
) -> JSONResponse:
    try:
        result = s1_s5_observer_state(mechanism, after_sequence)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    return JSONResponse(content=result)


@router.get("/qbi/tokens-prompt/state")
async def tokens_prompt_state(raw_request: Request, after_sequence: int = 0):
    return await _qbi_s1_s5_observer_state(
        "native_tokens_prompt_prepared_input", after_sequence
    )


@router.post('/qbi/tokens-prompt/prepare')
async def qbi_tokens_prompt_prepare(raw_request: Request):
    from .tokens_prompt_prepare import CACHE
    from .s1_s5_observer import _provider_renderer
    body = await _qbi_json_body(raw_request)
    renderer = _provider_renderer(raw_request.app.state.openai_serving_chat)
    tokenizer = getattr(renderer, 'tokenizer', None)
    if tokenizer is None:
        raise HTTPException(status_code=503, detail='provider tokenizer unavailable')
    if body.get('reset') is True:
        return {'cleared': CACHE.clear()}
    runtime = await _qbi_utility(raw_request)('get_runtime_transition_state')
    receipt = await CACHE.prepare(body, tokenizer)
    receipt['engine_boot_id'] = runtime['boot_id']
    receipt['config_epoch'] = runtime['config_epoch']
    return receipt


@router.get("/qbi/token-budget/state")
async def token_budget_state(raw_request: Request, after_sequence: int = 0):
    return await _qbi_s1_s5_observer_state(
        "token_budget_output_control", after_sequence
    )


@router.get("/qbi/native-output-stream/state")
async def native_output_stream_state(raw_request: Request, after_sequence: int = 0):
    return await _qbi_s1_s5_observer_state(
        "native_output_stream_batching", after_sequence
    )


@router.get("/qbi/native-output-stream/carrier-receipts")
async def native_output_stream_carrier_receipts(
    raw_request: Request, after_sequence: int = 0, decision_id: str | None = None
):
    from .s1_s5_observer import stream_delivery_state

    try:
        return JSONResponse(content=stream_delivery_state(after_sequence, decision_id))
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(error)) from error


@router.get("/qbi/suffix-speculation/provider-shape/state")
async def suffix_provider_shape_state(
    raw_request: Request, after_sequence: int = 0
):
    """Read the provider-observed request shape for Suffix pre-action rows.

    S4's Suffix mechanism needs a provider-observed ``available_tool_count``;
    this exposes the same opt-in provider snapshot the S1/S5 mechanisms use.
    """
    return await _qbi_s1_s5_observer_state(
        "native_suffix_speculation", after_sequence
    )


@router.get("/qbi/repetition-request/state")
async def repetition_request_state(raw_request: Request, after_sequence: int = 0):
    return await _qbi_s1_s5_observer_state(
        "repetition_detection_termination", after_sequence
    )


@router.get("/qbi/runtime-transition/state")
async def qbi_runtime_transition_state(raw_request: Request):
    """Return the live EngineCore epoch and scheduler-limit identity."""

    return await _qbi_call(raw_request, "get_runtime_transition_state")


@router.get("/qbi/prefix-cache/policy-receipts")
async def qbi_prefix_cache_policy_receipts(raw_request: Request):
    """Drain bounded prompt-free session APC policy receipts."""

    return await _qbi_call(raw_request, "get_prefix_cache_policy_receipts")


@router.get("/qbi/dynamic-mtp/state")
async def qbi_dynamic_mtp_state(raw_request: Request, after_sequence: int = 0):
    """Read scheduler/model-runner joined dynamic-MTP receipts."""

    return await _qbi_call(raw_request, "get_dynamic_mtp_state", int(after_sequence))


@router.get("/qbi/reasoning-budget/state")
async def qbi_reasoning_budget_state(raw_request: Request, after_sequence: int = 0):
    """Read live sampler receipts for bounded reasoning requests."""

    return await _qbi_call(
        raw_request, "get_reasoning_budget_state", int(after_sequence)
    )

@router.get("/qbi/batch-scope/state")
async def qbi_batch_scope_state(raw_request: Request, after_sequence: int = 0):
    """Read the exact-boot batch proposal channel and its effective receipts."""

    return await _qbi_call(raw_request, "get_batch_scope_state", after_sequence)


@router.post("/qbi/batch-scope/propose")
async def qbi_batch_scope_propose(raw_request: Request):
    """Bind one budget proposal to an exact engine boot, epoch and request set."""

    body = await _qbi_json_body(raw_request)
    expected_fields = {
        "expected_boot_id",
        "expected_config_epoch",
        "batch_proposal_id",
        "member_request_ids",
        "token_budget",
        "member_match",
        "require_prefill_decode_overlap",
        "target_prefill_member_id",
        "peer_decode_member_id",
    }
    if set(body) != expected_fields:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Batch-scope proposal has unexpected or missing fields",
        )
    expected_boot_id = body["expected_boot_id"]
    batch_proposal_id = body["batch_proposal_id"]
    if not isinstance(expected_boot_id, str) or not expected_boot_id.strip():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="expected_boot_id must be non-empty text",
        )
    if not isinstance(batch_proposal_id, str) or not batch_proposal_id.strip():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="batch_proposal_id must be non-empty text",
        )
    expected_config_epoch = body["expected_config_epoch"]
    token_budget = body["token_budget"]
    if (
        isinstance(expected_config_epoch, bool)
        or not isinstance(expected_config_epoch, int)
        or expected_config_epoch < 0
    ):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="expected_config_epoch must be a non-negative integer",
        )
    if (
        isinstance(token_budget, bool)
        or not isinstance(token_budget, int)
        or token_budget < 0
    ):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="token_budget must be a non-negative integer",
        )
    member_request_ids = body["member_request_ids"]
    if (
        not isinstance(member_request_ids, list)
        or not member_request_ids
        or any(
            not isinstance(value, str) or not value.strip()
            for value in member_request_ids
        )
        or len(set(member_request_ids)) != len(member_request_ids)
    ):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="member_request_ids must be a non-empty list of unique non-empty ids",
        )
    member_match = body["member_match"]
    if not isinstance(member_match, str) or member_match not in {
        "exact",
        "client_alias",
    }:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="member_match must be 'exact' or 'client_alias'",
        )
    require_overlap = body["require_prefill_decode_overlap"]
    if not isinstance(require_overlap, bool):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="require_prefill_decode_overlap must be a boolean",
        )
    target_prefill_member_id = body["target_prefill_member_id"]
    peer_decode_member_id = body["peer_decode_member_id"]
    for field_name, value in (
        ("target_prefill_member_id", target_prefill_member_id),
        ("peer_decode_member_id", peer_decode_member_id),
    ):
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
            or value not in member_request_ids
        ):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"{field_name} must identify one of member_request_ids",
            )
    if require_overlap and (
        target_prefill_member_id is None
        or peer_decode_member_id is None
        or target_prefill_member_id == peer_decode_member_id
    ):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="overlap-gated proposals require distinct target and peer members",
        )
    return await _qbi_call(
        raw_request,
        "propose_batch_scope_budget",
        expected_boot_id,
        expected_config_epoch,
        batch_proposal_id,
        member_request_ids,
        token_budget,
        member_match,
        require_overlap,
        target_prefill_member_id,
        peer_decode_member_id,
    )


@router.post("/qbi/batch-scope/cancel")
async def qbi_batch_scope_cancel(raw_request: Request):
    """Remove an unmatched proposal when its caller will not send that batch."""

    body = await _qbi_json_body(raw_request)
    if set(body) != {"expected_boot_id", "batch_proposal_id"}:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Batch-scope cancellation has unexpected or missing fields",
        )
    expected_boot_id = body["expected_boot_id"]
    batch_proposal_id = body["batch_proposal_id"]
    if not isinstance(expected_boot_id, str) or not expected_boot_id.strip():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="expected_boot_id must be non-empty text",
        )
    if not isinstance(batch_proposal_id, str) or not batch_proposal_id.strip():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="batch_proposal_id must be non-empty text",
        )
    return await _qbi_call(
        raw_request,
        "cancel_batch_scope_proposal",
        expected_boot_id,
        batch_proposal_id,
    )


@router.post("/qbi/online-selector/benchmark")
async def qbi_online_selector_benchmark(raw_request: Request):
    """Run a no-generation benchmark inside the live EngineCore process."""

    body = await _qbi_json_body(raw_request)
    run_tag = body.get("run_tag")
    iterations = body.get("iterations")
    if not isinstance(run_tag, str) or not run_tag:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="run_tag must be non-empty text",
        )
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="iterations must be an integer",
        )
    return await _qbi_call(
        raw_request,
        "run_qbi_online_selector_benchmark",
        run_tag,
        iterations,
    )


@router.post("/qbi/schema/pre-action-prepare")
async def qbi_schema_pre_action_prepare(raw_request: Request):
    """Compile and snapshot one request-bound Schema without model work."""

    body = await _qbi_json_body(raw_request)
    schema_document = body.get("schema_document")
    tools = body.get("tools", [])
    if not isinstance(schema_document, dict) or not schema_document:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="schema_document must be a non-empty object",
        )
    if not isinstance(tools, list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="tools must be a list",
        )
    try:
        requested_max_tokens = int(body["requested_max_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="requested_max_tokens must be an integer",
        ) from error
    request_id = body.get("request_id")
    if request_id is not None and (
        not isinstance(request_id, str) or not request_id.strip()
    ):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="request_id must be non-empty text",
        )
    require_compile_cache_state = body.get("require_compile_cache_state", False)
    if not isinstance(require_compile_cache_state, bool):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="require_compile_cache_state must be a boolean",
        )
    return await _qbi_call(
        raw_request,
        "prepare_schema_pre_action",
        schema_document,
        len(tools),
        requested_max_tokens,
        request_id,
        require_compile_cache_state,
    )


@router.post("/qbi/schema/pre-action-state")
async def qbi_schema_pre_action_state(raw_request: Request):
    """Read schema facts before opening a completion request.

    The schema is supplied to this zero-model observer endpoint.  Compiler
    state is returned only when EngineCore has a request-bound pre-action
    snapshot; it is never inferred from the schema or a later mask event.
    """

    body = await _qbi_json_body(raw_request)
    schema_document = body.get("schema_document")
    tools = body.get("tools", [])
    if not isinstance(schema_document, dict) or not schema_document:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="schema_document must be a non-empty object")
    if not isinstance(tools, list):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="tools must be a list")
    try:
        requested_max_tokens = int(body["requested_max_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="requested_max_tokens must be an integer") from error
    request_sha256 = _qbi_request_sha256(body.get("request_sha256"))
    if request_sha256 is None:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="request_sha256 is required")
    request_id = body.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="request_id must be non-empty text")
    require_compile_cache_state = body.get("require_compile_cache_state", False)
    if not isinstance(require_compile_cache_state, bool):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="require_compile_cache_state must be a boolean")
    return await _qbi_call(
        raw_request,
        "get_schema_pre_action_state",
        schema_document,
        len(tools),
        requested_max_tokens,
        request_sha256,
        request_id,
        require_compile_cache_state,
    )


@router.get("/qbi/schema/pre-action-state")
async def qbi_schema_pre_action_capability(raw_request: Request):
    """Expose the zero-call Schema route capability for a GET gate."""

    return await _qbi_call(raw_request, "get_schema_pre_action_capability")


@router.get("/qbi/prefix-cache/pre-action-state")
async def qbi_prefix_cache_pre_action_state(
    raw_request: Request,
    request_id: str | None = None,
    provider_request_id: str | None = None,
    request_sha256: str | None = None,
    session_plan: str | None = None,
    planned_reuse_ratio: float | None = None,
    prefix_tokens: int | None = None,
    session_invocation_count: int | None = None,
    kv_bytes: int | None = None,
    namespace: str | None = None,
    persistence_mode: str | None = None,
    arrival_gap_ms: float | None = None,
):
    """Read a provider-bound KV/session snapshot before admission.

    GET is kept prompt-free, but it is no longer an aggregate-only capability
    probe when the caller supplies a request-bound plan.  With no plan the
    same EngineCore observer is called explicitly and returns ``NOT_READY``.
    """

    normalized_request_id = _qbi_prefix_request_id(
        request_id, provider_request_id
    )
    return await _qbi_call(
        raw_request,
        "get_prefix_cache_pre_action_state",
        _qbi_prefix_session_plan_from_query(
            session_plan=session_plan,
            planned_reuse_ratio=planned_reuse_ratio,
            prefix_tokens=prefix_tokens,
            session_invocation_count=session_invocation_count,
            kv_bytes=kv_bytes,
            namespace=namespace,
            persistence_mode=persistence_mode,
            arrival_gap_ms=arrival_gap_ms,
        ),
        normalized_request_id,
        _qbi_request_sha256(request_sha256),
    )


@router.post("/qbi/prefix-cache/pre-action-state")
async def qbi_prefix_cache_pre_action_state_post(raw_request: Request):
    """Join caller session plan to provider-owned KV state before admission."""

    body = await _qbi_json_body(raw_request)
    session_plan = body.get("session_plan")
    if session_plan is not None and not isinstance(session_plan, dict):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="session_plan must be an object")
    request_id = _qbi_prefix_request_id(
        body.get("request_id"), body.get("provider_request_id")
    )
    request_sha256 = _qbi_request_sha256(body.get("request_sha256"))
    return await _qbi_call(
        raw_request,
        "get_prefix_cache_pre_action_state",
        session_plan,
        request_id,
        request_sha256,
    )


@router.get("/qbi/prefix-cache/runtime-receipts")
async def qbi_prefix_cache_runtime_receipts(raw_request: Request):
    """Drain request-bound lookup/admit/residency/eviction receipts."""
    return await _qbi_call(raw_request, "get_prefix_cache_policy_receipts")


@router.post("/qbi/runtime-transition/stage")
async def qbi_runtime_transition_stage(raw_request: Request):
    """Stage one bounded scheduler-limit change without accepting mixed epochs."""

    body = await _qbi_json_body(raw_request)
    mechanism_name, config = _qbi_transition_config(body)
    return await _qbi_call(
        raw_request,
        "stage_runtime_transition",
        mechanism_name,
        config,
    )


@router.post("/qbi/runtime-transition/commit")
async def qbi_runtime_transition_commit(raw_request: Request):
    """Commit a staged change after the previous scheduler epoch drains."""

    body = await _qbi_json_body(raw_request)
    mechanism_name = _qbi_transition_mechanism(body)
    try:
        next_epoch = int(body["next_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="next_epoch must be an integer",
        ) from error
    return await _qbi_call(
        raw_request,
        "commit_staged_runtime_transition",
        mechanism_name,
        next_epoch,
    )


@router.post("/qbi/runtime-transition/abort")
async def qbi_runtime_transition_abort(raw_request: Request):
    """Abort an uncommitted scheduler transition and release pending requests."""

    body = await _qbi_json_body(raw_request)
    mechanism_name = _qbi_transition_mechanism(body)
    return await _qbi_call(
        raw_request,
        "abort_staged_runtime_transition",
        mechanism_name,
    )


@router.post("/qbi/runtime-transition/verify")
async def qbi_runtime_transition_verify(raw_request: Request):
    """Return the EngineCore effectiveness receipt for a committed epoch."""

    body = await _qbi_json_body(raw_request)
    mechanism_name, config = _qbi_transition_config(body)
    try:
        epoch = int(body["epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="epoch must be an integer",
        ) from error
    return await _qbi_call(
        raw_request,
        "verify_runtime_transition",
        mechanism_name,
        config,
        epoch,
    )


@router.post("/qbi/runtime-transition/rollback")
async def qbi_runtime_transition_rollback(raw_request: Request):
    """Restore the staged transition's previous limits at an idle boundary."""

    body = await _qbi_json_body(raw_request)
    mechanism_name = _qbi_transition_mechanism(body)
    prepared = body.get("prepared")
    if not isinstance(prepared, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="prepared transition receipt must be an object",
        )
    try:
        previous_epoch = int(body["previous_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="previous_epoch must be an integer",
        ) from error
    return await _qbi_call(
        raw_request,
        "rollback_runtime_transition",
        mechanism_name,
        prepared,
        previous_epoch,
    )


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
