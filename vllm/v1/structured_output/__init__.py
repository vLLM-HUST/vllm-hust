# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import itertools
import json
import multiprocessing
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")


logger = init_logger(__name__)
_QBI_SCHEMA_RECEIPT_WINDOW = max(int(os.getenv("QBI_SCHEMA_RECEIPT_WINDOW", "8192")), 1)
_QBI_SOURCE_IDENTITY_CACHE: dict[str, tuple[str, str]] = {}


def _qbi_source_identity(path: object) -> tuple[str, str]:
    raw_path = str(path or "").strip()
    if not raw_path:
        return "unknown", "unknown"
    source_file = os.path.realpath(raw_path)
    cached = _QBI_SOURCE_IDENTITY_CACHE.get(source_file)
    if cached is not None:
        return cached
    try:
        with open(source_file, "rb") as source:
            source_sha256 = hashlib.sha256(source.read()).hexdigest()
    except OSError:
        source_sha256 = "unavailable"
    identity = (source_file, source_sha256)
    _QBI_SOURCE_IDENTITY_CACHE[source_file] = identity
    return identity


_QBI_MANAGER_SOURCE_FILE, _QBI_MANAGER_SOURCE_SHA256 = _qbi_source_identity(__file__)


def _remember_qbi_request(
    request_id: str,
    seen: set[str],
    order: deque[str],
) -> bool:
    if request_id in seen:
        return False
    seen.add(request_id)
    order.append(request_id)
    if len(order) > _QBI_SCHEMA_RECEIPT_WINDOW:
        seen.remove(order.popleft())
    return True


def _emit_qbi_schema_event(phase: str, request_id: str, **fields: object) -> None:
    receipt_path = os.getenv("QBI_SCHEMA_EFFECTIVE_EVENT_LOG")
    if not receipt_path:
        return
    event = {
        "schema": "qbi.schema-effective-event.v1",
        "phase": phase,
        "request_id": request_id,
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        **fields,
    }
    payload = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(receipt_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: StructuredOutputBackend | None = None
        self._qbi_backend_name: str | None = None
        self.reasoner: ReasoningParser | None = None
        self.vllm_config = vllm_config
        # Receipt-only telemetry for mechanism admission.  These sets bound
        # logging to one event per request while keeping the normal structured
        # output execution path unchanged.
        self._qbi_compiled_request_ids: set[str] = set()
        self._qbi_bitmask_filled_request_ids: set[str] = set()
        self._qbi_compiled_request_order: deque[str] = deque()
        self._qbi_bitmask_filled_request_order: deque[str] = deque()
        self._qbi_pre_action_snapshots: dict[str, dict[str, object]] = {}
        self._qbi_pre_action_snapshot_order: deque[str] = deque()
        # Request-bound compile-cache instrumentation.  This is deliberately
        # a manager/compiler wrapper receipt: it never observes logits,
        # bitmask application, or response latency.
        self._qbi_compile_cache_lock = threading.Lock()
        self._qbi_compile_cache_query_count = 0
        self._qbi_compile_cache_keys: set[str] = set()
        self._qbi_compile_cache_key_order: deque[str] = deque()

        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING transition to
        # happen at different times on different TP ranks,
        # breaking the determinism assumption that external_launcher relies on.
        self._use_async_grammar_compilation = (
            vllm_config.parallel_config.distributed_executor_backend
            != "external_launcher"
        )

        self._grammar_bitmask: torch.Tensor | None = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of
            # CPUs * 5, which is way too high since these tasks are CPU-bound,
            # not I/O bound. We also know we would never dominate CPU usage
            # with just grammar compilation, so we set it to half the number
            # of CPUs.
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.tokenizer = cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
            reasoning_parser_plugin = (
                self.vllm_config.structured_outputs_config.reasoning_parser_plugin
            )
            if reasoning_parser_plugin and len(reasoning_parser_plugin) > 3:
                ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

            reasoning_parser = (
                self.vllm_config.structured_outputs_config.reasoning_parser
            )
            if reasoning_parser:
                reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )
                self.reasoner = reasoner_cls(tokenizer=self.tokenizer)

        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def _ensure_backend(self, backend: str) -> StructuredOutputBackend:
        """Initialize the configured backend once and return the live instance."""

        if not isinstance(backend, str) or not backend or backend == "auto":
            raise ValueError(
                "request-bound Schema precompile requires a resolved backend"
            )
        if self.backend is not None:
            if self._qbi_backend_name == backend:
                return self.backend
            raise ValueError(
                "structured output backend changed within one EngineCore: "
                f"{type(self.backend).__name__} -> {backend}"
            )
        vocab_size = self.vllm_config.model_config.get_vocab_size()
        if backend == "xgrammar":
            self.backend = XgrammarBackend(
                self.vllm_config,
                tokenizer=self.tokenizer,
                vocab_size=vocab_size,
            )
        elif backend == "guidance":
            self.backend = GuidanceBackend(
                self.vllm_config,
                tokenizer=self.tokenizer,
                vocab_size=vocab_size,
            )
        elif backend == "outlines":
            from vllm.v1.structured_output.backend_outlines import OutlinesBackend

            self.backend = OutlinesBackend(
                self.vllm_config,
                tokenizer=self.tokenizer,
                vocab_size=vocab_size,
            )
        elif backend == "lm-format-enforcer":
            from vllm.v1.structured_output.backend_lm_format_enforcer import (
                LMFormatEnforcerBackend,
            )

            self.backend = LMFormatEnforcerBackend(
                self.vllm_config,
                tokenizer=self.tokenizer,
                vocab_size=vocab_size,
            )
        else:
            raise ValueError(f"Unsupported structured output backend: {backend}")
        self._qbi_backend_name = backend
        return self.backend

    def grammar_init(self, request: "Request") -> None:
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert (
                request.sampling_params is not None
                and request.sampling_params.structured_outputs is not None
            )

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        assert request.sampling_params is not None
        backend = request.sampling_params.structured_outputs._backend
        self._ensure_backend(backend)

        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            grammar = self._create_grammar(request)  # type: ignore[assignment]
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    @staticmethod
    def _pre_action_request_sha256(
        schema_document: dict[str, object],
        tool_count: int,
        requested_max_tokens: int,
    ) -> str:
        payload = json.dumps(
            {
                "schema_document": schema_document,
                "tool_count": tool_count,
                "requested_max_tokens": requested_max_tokens,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _remember_pre_action_snapshot(
        self, request_id: str, state: dict[str, object]
    ) -> None:
        if request_id not in self._qbi_pre_action_snapshots:
            self._qbi_pre_action_snapshot_order.append(request_id)
        self._qbi_pre_action_snapshots[request_id] = state
        while len(self._qbi_pre_action_snapshot_order) > _QBI_SCHEMA_RECEIPT_WINDOW:
            expired = self._qbi_pre_action_snapshot_order.popleft()
            self._qbi_pre_action_snapshots.pop(expired, None)

    def prepare_pre_action_snapshot(
        self,
        schema_document: dict[str, object],
        tool_count: int,
        requested_max_tokens: int,
        *,
        request_id: str,
        engine_boot_id: str,
        config_epoch: int,
        clock_id: str,
        observed_at_ns: int,
        decision_at_ns: int,
        runtime_snapshot: dict[str, object] | None = None,
        require_compile_cache_state: bool = False,
    ) -> dict[str, object]:
        """Compile one request-bound schema and retain its provider snapshot.

        Backend selection reuses ``SamplingParams.verify`` so ``auto`` follows
        the same runtime selection path as a real request.  The compiled
        grammar and mask facts are then extracted from the live backend object;
        no schema or configured value is promoted to a compiler measurement.
        """

        if not isinstance(schema_document, dict) or not schema_document:
            raise ValueError("schema_document is required for pre-action precompile")
        if type(tool_count) is not int or tool_count < 0:
            raise ValueError("tool_count must be a non-negative integer")
        if type(requested_max_tokens) is not int or requested_max_tokens < 0:
            raise ValueError("requested_max_tokens must be a non-negative integer")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id is required for pre-action precompile")
        if not isinstance(engine_boot_id, str) or not engine_boot_id.strip():
            raise ValueError("engine_boot_id is required for pre-action precompile")
        if type(config_epoch) is not int or config_epoch < 0:
            raise ValueError("config_epoch is required for pre-action precompile")
        request_sha256 = self._pre_action_request_sha256(
            schema_document, tool_count, requested_max_tokens
        )
        existing = self._qbi_pre_action_snapshots.get(request_id)
        if existing is not None:
            binding = existing.get("binding", {})
            if not isinstance(binding, dict):
                raise ValueError("cached Schema pre-action binding is malformed")
            if (
                binding.get("request_sha256") != request_sha256
                or binding.get("engine_boot_id") != engine_boot_id
                or binding.get("config_epoch") != config_epoch
            ):
                raise ValueError(
                    "request-bound Schema pre-action identity differs from cached snapshot"
                )
            if require_compile_cache_state:
                cache = existing.get("compile_cache_evidence")
                if not isinstance(cache, dict) or cache.get("status") != "observed":
                    raise ValueError(
                        "request-bound Schema compile-cache state is missing"
                    )
            return dict(existing)

        from vllm.sampling_params import SamplingParams, StructuredOutputsParams
        from vllm.v1.structured_output.pre_action_observer import (
            compiler_snapshot_from_backend,
            observe_provider_bound_schema_pre_action,
        )
        from vllm.v1.structured_output.request import get_structured_output_key

        params = SamplingParams.from_optional(
            max_tokens=requested_max_tokens,
            structured_outputs=StructuredOutputsParams(json=dict(schema_document)),
        )
        params.verify(
            self.vllm_config.model_config,
            self.vllm_config.speculative_config,
            self.vllm_config.structured_outputs_config,
            self.tokenizer,
        )
        if params.structured_outputs is None:
            raise ValueError("Schema precompile did not create structured output params")
        backend_name = params.structured_outputs._backend
        if not isinstance(backend_name, str) or backend_name == "auto":
            raise ValueError("Schema precompile did not resolve a backend")
        self._ensure_backend(backend_name)
        request_type, grammar_spec = get_structured_output_key(
            params.structured_outputs
        )
        grammar = self._compile_grammar(
            request_type, grammar_spec, request_id=request_id
        )
        compiler_snapshot = compiler_snapshot_from_backend(
            backend=self.backend,
            grammar=grammar,
            request_id=request_id,
            engine_boot_id=engine_boot_id,
            config_epoch=config_epoch,
        )
        state = observe_provider_bound_schema_pre_action(
            schema_document=schema_document,
            tool_count=tool_count,
            requested_max_tokens=requested_max_tokens,
            request_id=request_id,
            engine_boot_id=engine_boot_id,
            config_epoch=config_epoch,
            clock_id=clock_id,
            observed_at_ns=observed_at_ns,
            decision_at_ns=decision_at_ns,
            compiler_snapshot=compiler_snapshot,
            runtime_snapshot=runtime_snapshot,
            require_compile_cache_state=require_compile_cache_state,
        )
        self._remember_pre_action_snapshot(request_id, state)
        return dict(state)

    def get_pre_action_state(
        self,
        schema_document: dict[str, object],
        tool_count: int,
        requested_max_tokens: int,
        *,
        request_id: str | None = None,
        engine_boot_id: str | None = None,
        config_epoch: int | None = None,
        clock_id: str | None = None,
        observed_at_ns: int | None = None,
        decision_at_ns: int | None = None,
        provider_snapshot: dict[str, object] | None = None,
        runtime_snapshot: dict[str, object] | None = None,
        require_compile_cache_state: bool = False,
    ) -> dict[str, object]:
        """Expose caller-visible schema facts before a completion request.

        Compiler state is returned only from an explicit request-bound
        precompile snapshot or a backend-provided source-bound snapshot.  A
        missing snapshot remains ``NOT_READY``; this method never uses schema
        structure, configured limits, or a post-action mask event as a proxy.
        """

        from vllm.v1.structured_output.pre_action_observer import (
            observe_provider_bound_schema_pre_action,
            observe_schema_pre_action,
        )

        if any(
            value is not None
            for value in (
                request_id,
                engine_boot_id,
                config_epoch,
                clock_id,
                observed_at_ns,
                decision_at_ns,
                provider_snapshot,
            )
        ):
            snapshot = provider_snapshot
            if snapshot is None and request_id is not None:
                cached = self._qbi_pre_action_snapshots.get(request_id)
                if cached is not None:
                    binding = cached.get("binding", {})
                    if not isinstance(binding, dict):
                        raise ValueError("cached Schema pre-action binding is malformed")
                    request_sha256 = self._pre_action_request_sha256(
                        schema_document, tool_count, requested_max_tokens
                    )
                    if (
                        binding.get("request_sha256") != request_sha256
                        or binding.get("engine_boot_id") != engine_boot_id
                        or binding.get("config_epoch") != config_epoch
                    ):
                        raise ValueError(
                            "request-bound Schema pre-action identity differs from cached snapshot"
                        )
                    if require_compile_cache_state:
                        cache = cached.get("compile_cache_evidence")
                        if not isinstance(cache, dict) or cache.get("status") != "observed":
                            raise ValueError(
                                "request-bound Schema compile-cache state is missing"
                            )
                    return dict(cached)
            if snapshot is None and self.backend is not None:
                provider_method = getattr(self.backend, "pre_action_snapshot", None)
                if callable(provider_method):
                    candidate = provider_method(
                        schema_document, tool_count, requested_max_tokens
                    )
                    if isinstance(candidate, dict):
                        snapshot = candidate
            return observe_provider_bound_schema_pre_action(
                schema_document=schema_document,
                tool_count=tool_count,
                requested_max_tokens=requested_max_tokens,
                request_id=request_id,
                engine_boot_id=engine_boot_id,
                config_epoch=config_epoch,
                clock_id=clock_id,
                observed_at_ns=observed_at_ns,
                decision_at_ns=decision_at_ns,
                compiler_snapshot=snapshot,
                runtime_snapshot=runtime_snapshot,
                require_compile_cache_state=require_compile_cache_state,
            )

        return observe_schema_pre_action(
            schema_document=schema_document,
            tool_count=tool_count,
            requested_max_tokens=requested_max_tokens,
            backend=type(self.backend).__name__ if self.backend is not None else None,
        )

    def get_predictor_pre_action_state(
        self,
        schema_document: dict[str, object],
        tool_count: int,
        requested_max_tokens: int,
        *,
        request_sha256: str,
        request_id: str | None,
        engine_boot_id: str,
        config_epoch: int,
        clock_id: str,
        runtime_snapshot: dict[str, object] | None = None,
        require_compile_cache_state: bool = False,
    ) -> dict[str, object]:
        """Read only cache state before prepare; never reuse a compiled snapshot.

        ``prepare_pre_action_snapshot`` compiles the request grammar and returns
        post-prepare measurement. This method is the predictor boundary: it only
        reads backend existence and the provider's cache-ready/size/limit view.
        Exact cache hit remains unknown until the compiler query executes.
        """
        from vllm.v1.structured_output.pre_action_observer import (
            observe_schema_predictor_pre_action_state,
        )

        backend_initialized = self.backend is not None
        cache_state = (
            self._qbi_backend_cache_state(self.backend)
            if self.backend is not None
            else None
        )
        # Timestamp after the read-only provider/cache snapshot so it cannot
        # appear earlier than the state it binds.
        now = time.perf_counter_ns()
        return observe_schema_predictor_pre_action_state(
            schema_document=schema_document,
            tool_count=tool_count,
            requested_max_tokens=requested_max_tokens,
            request_sha256=request_sha256,
            request_id=request_id,
            engine_boot_id=engine_boot_id,
            config_epoch=config_epoch,
            clock_id=clock_id,
            observed_at_ns=now,
            decision_at_ns=now,
            backend_initialized_before_prepare=backend_initialized,
            compile_cache_state_before_prepare=cache_state,
            runtime_snapshot=runtime_snapshot,
            require_compile_cache_state=require_compile_cache_state,
        )

    def _compile_grammar(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        *,
        request_id: str | None,
    ) -> StructuredOutputGrammar:
        assert self.backend is not None
        grammar_key = self._qbi_canonical_grammar_key(
            self.backend, request_type, grammar_spec
        )
        with self._qbi_compile_cache_lock:
            self._qbi_compile_cache_query_count += 1
            query_sequence = self._qbi_compile_cache_query_count
            preexisting_key = grammar_key in self._qbi_compile_cache_keys
        cache_before = self._qbi_backend_cache_state(self.backend)
        stats_before = self._qbi_backend_cache_stats(self.backend)
        grammar = self.backend.compile_grammar(request_type, grammar_spec)
        cache_after = self._qbi_backend_cache_state(self.backend)
        stats_after = self._qbi_backend_cache_stats(self.backend)
        cache_evidence = self._qbi_compile_cache_evidence(
            grammar=grammar,
            grammar_key=grammar_key,
            request_id=request_id,
            query_sequence=query_sequence,
            preexisting_key=preexisting_key,
            cache_before=cache_before,
            cache_after=cache_after,
            stats_before=stats_before,
            stats_after=stats_after,
        )
        try:
            setattr(grammar, "compile_cache_evidence", cache_evidence)
        except (AttributeError, TypeError):
            # Provider snapshots remain fail-closed if an external backend
            # returns an immutable grammar object without an evidence slot.
            pass
        with self._qbi_compile_cache_lock:
            if grammar_key not in self._qbi_compile_cache_keys:
                self._qbi_compile_cache_keys.add(grammar_key)
                self._qbi_compile_cache_key_order.append(grammar_key)
            while len(self._qbi_compile_cache_key_order) > _QBI_SCHEMA_RECEIPT_WINDOW:
                self._qbi_compile_cache_keys.discard(
                    self._qbi_compile_cache_key_order.popleft()
                )
        if request_id is not None and _remember_qbi_request(
            request_id,
            self._qbi_compiled_request_ids,
            self._qbi_compiled_request_order,
        ):
            key_sha256 = hashlib.sha256(
                f"{request_type}\0{grammar_spec}".encode()
            ).hexdigest()
            backend_module = sys.modules.get(type(self.backend).__module__)
            backend_source_file, backend_source_sha256 = _qbi_source_identity(
                getattr(backend_module, "__file__", None)
            )
            logger.info(
                "QBI_SCHEMA_EVENT phase=grammar_compiled request_id=%s "
                "backend=%s request_type=%s key_sha256=%s "
                "source_file=%s source_sha256=%s "
                "backend_source_file=%s backend_source_sha256=%s",
                request_id,
                type(self.backend).__name__,
                request_type,
                key_sha256,
                _QBI_MANAGER_SOURCE_FILE,
                _QBI_MANAGER_SOURCE_SHA256,
                backend_source_file,
                backend_source_sha256,
            )
            _emit_qbi_schema_event(
                "grammar_compiled",
                request_id,
                backend=type(self.backend).__name__,
                request_type=str(request_type),
                key_sha256=key_sha256,
                source_file=_QBI_MANAGER_SOURCE_FILE,
                source_sha256=_QBI_MANAGER_SOURCE_SHA256,
                backend_source_file=backend_source_file,
                backend_source_sha256=backend_source_sha256,
            )
        return grammar

    @staticmethod
    def _qbi_canonical_grammar_key(
        backend: StructuredOutputBackend,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
    ) -> str:
        payload = json.dumps(
            {
                "backend": type(backend).__name__,
                "request_type": getattr(request_type, "name", str(request_type)),
                "grammar_spec": grammar_spec,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _qbi_backend_cache_state(backend: StructuredOutputBackend) -> dict[str, object]:
        method = getattr(backend, "compiled_grammar_cache_state", None)
        try:
            raw = method() if callable(method) else None
        except (TypeError, RuntimeError, ValueError):
            raw = None
        state = dict(raw) if isinstance(raw, Mapping) else {}
        ready = state.get("cache_ready")
        size = state.get("cache_size_bytes")
        limit = state.get("cache_limit_bytes")
        return {
            "cache_ready": ready if type(ready) is bool else None,
            "cache_size_bytes": size if type(size) is int and size >= 0 else None,
            "cache_limit_bytes": limit if type(limit) is int and limit >= 0 else None,
            "cache_ready_source_field": str(
                state.get("cache_ready_source_field", "structured_output_backend.compiler.cache_enabled")
            ),
            "cache_size_source_field": str(
                state.get("cache_size_source_field", state.get("source_field", "structured_output_backend.compiler.cache_state"))
            ),
            "cache_limit_source_field": str(
                state.get("cache_limit_source_field", "structured_output_backend.compiler.cache_limit_bytes")
            ),
            "source_field": str(
                state.get(
                    "source_field",
                    "structured_output_backend.compiler.cache_state",
                )
            ),
        }

    @staticmethod
    def _qbi_backend_cache_stats(
        backend: StructuredOutputBackend,
    ) -> tuple[int, int] | None:
        method = getattr(backend, "compiled_grammar_cache_stats", None)
        if not callable(method):
            return None
        try:
            raw = method()
        except (TypeError, RuntimeError, ValueError):
            return None
        if isinstance(raw, Mapping):
            hits, total = raw.get("hits"), raw.get("total")
        else:
            hits, total = getattr(raw, "hits", None), getattr(raw, "total", None)
        if (
            isinstance(hits, bool)
            or not isinstance(hits, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or hits < 0
            or total < 0
            or hits > total
        ):
            return None
        return hits, total

    @classmethod
    def _qbi_compile_cache_evidence(
        cls,
        *,
        grammar: StructuredOutputGrammar,
        grammar_key: str,
        request_id: str | None,
        query_sequence: int,
        preexisting_key: bool,
        cache_before: Mapping[str, object],
        cache_after: Mapping[str, object],
        stats_before: tuple[int, int] | None,
        stats_after: tuple[int, int] | None,
    ) -> dict[str, object]:
        authoritative = False
        hit_status = "unknown"
        hit_value: bool | None = None
        queries: int | None = query_sequence
        hits: int | None = None
        ratio: float | None = None
        if stats_before is not None and stats_after is not None:
            delta_hits = stats_after[0] - stats_before[0]
            delta_queries = stats_after[1] - stats_before[1]
            if delta_queries >= 0 and 0 <= delta_hits <= delta_queries and delta_queries > 0:
                authoritative = True
                queries = delta_queries
                hits = delta_hits
                ratio = delta_hits / float(delta_queries)
                hit_value = delta_hits > 0
                hit_status = "hit" if hit_value else "miss"

        backend_evidence = getattr(grammar, "compile_cache_evidence", None)
        if not authoritative and isinstance(backend_evidence, Mapping):
            backend_hit = backend_evidence.get("compile_cache_hit")
            backend_queries = backend_evidence.get("compile_cache_queries")
            backend_hits = backend_evidence.get("compile_cache_hits")
            backend_ratio = backend_evidence.get("compile_cache_hit_ratio")
            if (
                type(backend_hit) is bool
                and type(backend_queries) is int
                and backend_queries > 0
                and type(backend_hits) is int
                and 0 <= backend_hits <= backend_queries
                and isinstance(backend_ratio, (int, float))
                and not isinstance(backend_ratio, bool)
            ):
                authoritative = True
                hit_value = backend_hit
                hit_status = "hit" if backend_hit else "miss"
                queries = backend_queries
                hits = backend_hits
                ratio = float(backend_ratio)

        before_size = cache_before.get("cache_size_bytes")
        after_size = cache_after.get("cache_size_bytes")
        size_delta = (
            int(after_size) - int(before_size)
            if type(before_size) is int and type(after_size) is int
            else None
        )
        ready = cache_after.get("cache_ready")
        if type(ready) is not bool:
            ready = cache_before.get("cache_ready")
        evidence: dict[str, object] = {
            "status": "observed",
            "compile_cache_hit": hit_value,
            "compile_cache_hit_status": hit_status,
            "compile_cache_queries": queries,
            "compile_cache_query_count": query_sequence,
            "compile_cache_hits": hits,
            "compile_cache_hit_ratio": ratio,
            "preexisting_compile_key": preexisting_key,
            "preexisting_compile_key_source": "manager_compile_registry",
            "cache_ready": ready if type(ready) is bool else None,
            "cache_ready_status": "observed" if type(ready) is bool else "unknown",
            "cache_size_before_bytes": before_size,
            "cache_size_after_bytes": after_size,
            "cache_size_delta_bytes": size_delta,
            "cache_limit_bytes": cache_after.get("cache_limit_bytes")
            or cache_before.get("cache_limit_bytes"),
            "canonical_grammar_key_sha256": grammar_key,
            "request_id": request_id,
            "authority": (
                "backend_cache_stats"
                if authoritative
                else "manager_compile_wrapper+backend_cache_size"
            ),
            "source_field": str(
                cache_after.get(
                    "source_field",
                    "structured_output_backend.compiler.cache_state",
                )
            ),
        }
        return evidence

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key
        return self._compile_grammar(
            request_type, grammar_spec, request_id=request.request_id
        )

    def _fill_bitmasks(
        self, batch: Iterable[tuple[StructuredOutputGrammar, int, bool]]
    ) -> None:
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool]]
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
            )

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar
                apply_bitmask = self.should_fill_bitmask(request)

                state_advancements = 0
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                for token in itertools.chain(req_tokens, (-1,)):
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    if token == -1:
                        # Stop advancing the grammar once we hit a padding token.
                        apply_bitmask = False
                    if apply_bitmask and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        assert accepted, (token, req_id, scheduled_spec_decode_tokens)
                        state_advancements += 1
                    cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        for req_id in structured_output_request_ids:
            if _remember_qbi_request(
                req_id,
                self._qbi_bitmask_filled_request_ids,
                self._qbi_bitmask_filled_request_order,
            ):
                backend_module = (
                    sys.modules.get(type(self.backend).__module__)
                    if self.backend is not None
                    else None
                )
                backend_source_file, backend_source_sha256 = _qbi_source_identity(
                    getattr(backend_module, "__file__", None)
                )
                logger.info(
                    "QBI_SCHEMA_EVENT phase=bitmask_filled request_id=%s "
                    "backend=%s rows=%s source_file=%s source_sha256=%s "
                    "backend_source_file=%s backend_source_sha256=%s",
                    req_id,
                    type(self.backend).__name__ if self.backend is not None else "none",
                    cumulative_index,
                    _QBI_MANAGER_SOURCE_FILE,
                    _QBI_MANAGER_SOURCE_SHA256,
                    backend_source_file,
                    backend_source_sha256,
                )
                _emit_qbi_schema_event(
                    "bitmask_filled",
                    req_id,
                    backend=(
                        type(self.backend).__name__
                        if self.backend is not None
                        else "none"
                    ),
                    rows=cumulative_index,
                    source_file=_QBI_MANAGER_SOURCE_FILE,
                    source_sha256=_QBI_MANAGER_SOURCE_SHA256,
                    backend_source_file=backend_source_file,
                    backend_source_sha256=backend_source_sha256,
                )

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def should_fill_bitmask(self, request: "Request") -> bool:
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        if self.reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            if request.structured_output_request.reasoning_ended is None:
                # This should be removed here, but since `openai_gptoss`
                # is an independent code path, it is kept for now.
                # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
                # it can be removed.
                request.structured_output_request.reasoning_ended = (
                    self.reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        return True

    def should_advance(self, request: "Request") -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        if self.reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if self.reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            # Reasoning just ended, so we shouldn't advance til
            # next pass
            structured_req.reasoning_ended = True

        return False

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
        self.backend = None
        self._qbi_backend_name = None
        self._qbi_pre_action_snapshots.clear()
        self._qbi_pre_action_snapshot_order.clear()
