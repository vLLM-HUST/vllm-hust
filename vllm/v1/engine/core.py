# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json
import os
import queue
import signal
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager, nullcontext
from enum import IntEnum
from functools import partial
from inspect import isclass, signature
from logging import DEBUG
from multiprocessing.queues import Queue
from typing import Any, TypeVar, cast

import msgspec
import zmq

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils import numa_utils
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    EEPNotificationType,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.tensor_ipc import TensorIpcReceiver
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    SignalCallback,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import compute_iteration_details
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def _native_model_result_context(scheduler, scheduler_output):
    if os.environ.get("ANISE_LIFECYCLE_OBSERVATION", "0") != "1":
        return nullcontext(None)
    from vllm_quality_bounded_inference.native_lifecycle_observation import (
        native_model_result,
    )

    return native_model_result(scheduler, scheduler_output)


def _prepare_native_worker_plan(core, scheduler_output):
    if os.environ.get("ANISE_LIFECYCLE_OBSERVATION", "0") != "1":
        return
    from vllm_quality_bounded_inference.worker_execution_observation import (
        prepare_worker_plan,
    )

    prepare_worker_plan(core, scheduler_output)

HANDSHAKE_TIMEOUT_MINS = 5
_QBI_MECHANISM_RECEIPT_LIMIT = 4096

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if not vllm_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self._eep_scale_up_before_kv_init()

        # Setup KV Caches and update CacheConfig after profiling.
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
            vllm_config
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_ec_consumer = (
            vllm_config.ec_transfer_config is None
            or vllm_config.ec_transfer_config.is_ec_consumer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling
        self._runtime_control_boot_id = uuid.uuid4().hex
        self._qbi_dynamic_mtp_receipts: deque[dict[str, Any]] = deque(
            maxlen=_QBI_MECHANISM_RECEIPT_LIMIT
        )
        self._qbi_reasoning_budget_receipts: deque[dict[str, Any]] = deque(
            maxlen=_QBI_MECHANISM_RECEIPT_LIMIT
        )
        self._qbi_dynamic_mtp_sequence = 0
        self._qbi_reasoning_budget_sequence = 0
        self._qbi_dynamic_mtp_dropped_receipts = 0
        self._qbi_reasoning_budget_dropped_receipts = 0
        self._qbi_pending_dynamic_mtp_receipt: dict[str, Any] | None = None

        set_scheduler_boot_id = getattr(
            self.scheduler, "set_runtime_control_boot_id", None
        )
        if callable(set_scheduler_boot_id):
            set_scheduler_boot_id(self._runtime_control_boot_id)

        self.aborts_queue = queue.Queue[list[str]]()

        self._idle_state_callbacks: list[Callable] = []

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

        # Bind the opt-in lifecycle observer only after the native Core and
        # scheduler are fully initialized. This keeps the default runtime
        # unchanged while making the QBI utility calls available to the API
        # handler in the same EngineCore process that owns the observations.
        if os.environ.get("ANISE_LIFECYCLE_OBSERVATION", "0") == "1":
            from vllm_quality_bounded_inference.native_core_observation import (
                bind_core_observation,
            )

            bind_core_observation(self)

    @instrument(span_name="Prepare model")
    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                # NOTE(yongji): should already be set
                # during _eep_scale_up_before_kv_init
                assert self.available_gpu_memory_for_kv_cache > 0
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = vllm_config.model_config.max_model_len

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )

        vllm_config.validate_block_size()

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        compile_time = vllm_config.compilation_config.compilation_time
        encoder_compile_time = vllm_config.compilation_config.encoder_compilation_time
        if encoder_compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s — language_model: %.2f s, "
                "encoder: %.2f s)",
                elapsed,
                compile_time + encoder_compile_time,
                compile_time,
                encoder_compile_time,
            )
        elif compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s)",
                elapsed,
                compile_time,
            )
        else:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took %.2f s",
                elapsed,
            )
        return scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def _schema_runtime_snapshot(self) -> dict[str, object]:
        """Read live scheduler/KV state for a Schema pre-action receipt."""

        snapshot: dict[str, object] = {}
        sources: dict[str, str] = {}
        running = getattr(self.scheduler, "running", None)
        if running is not None:
            try:
                snapshot["active_request_count"] = len(running)
                sources["active_request_count"] = "engine_core.scheduler.running"
            except TypeError:
                pass

        kv_manager = getattr(self.scheduler, "kv_cache_manager", None)
        pressure = getattr(kv_manager, "usage", None)
        if (
            isinstance(pressure, (int, float))
            and not isinstance(pressure, bool)
            and 0.0 <= float(pressure) <= 1.0
        ):
            snapshot["runtime_pressure_ratio"] = float(pressure)
            sources["runtime_pressure_ratio"] = (
                "engine_core.scheduler.kv_cache_manager.usage"
            )
        snapshot["source_fields"] = sources
        return snapshot

    def prepare_schema_pre_action(
        self,
        schema_document: dict[str, object],
        tool_count: int,
        requested_max_tokens: int,
        request_id: str | None = None,
        require_compile_cache_state: bool = False,
    ) -> dict[str, object]:
        """Precompile one request-bound Schema grammar without model work."""

        limits = self.scheduler.get_runtime_scheduler_limits()
        now = time.perf_counter_ns()
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id is required for Schema precompile")
        return self.structured_output_manager.prepare_pre_action_snapshot(
            schema_document,
            tool_count,
            requested_max_tokens,
            request_id=request_id,
            engine_boot_id=self._runtime_control_boot_id,
            config_epoch=limits["config_epoch"],
            clock_id="engine-core:" + self._runtime_control_boot_id,
            observed_at_ns=now,
            decision_at_ns=now,
            runtime_snapshot=self._schema_runtime_snapshot(),
            require_compile_cache_state=require_compile_cache_state,
        )

    def get_schema_pre_action_state(
        self,
        schema_document: dict[str, object],
        tool_count: int,
        requested_max_tokens: int,
        request_sha256: str,
        request_id: str | None = None,
        require_compile_cache_state: bool = False,
    ) -> dict[str, object]:
        """Return provider-bound schema facts before request admission."""

        manager = self.structured_output_manager
        backend_initialized_before_prepare = manager.backend is not None
        limits = self.scheduler.get_runtime_scheduler_limits()
        state = manager.get_predictor_pre_action_state(
            schema_document,
            tool_count,
            requested_max_tokens,
            request_sha256=request_sha256,
            request_id=request_id,
            engine_boot_id=self._runtime_control_boot_id,
            config_epoch=limits["config_epoch"],
            clock_id="engine-core:" + self._runtime_control_boot_id,
            runtime_snapshot=self._schema_runtime_snapshot(),
            require_compile_cache_state=require_compile_cache_state,
        )
        # This fresh observation has its own clock/binding even when the
        # compiler snapshot above came from an earlier request cache entry.
        state["backend_initialization_observation"] = {
            "schema": "qbi.schema-backend-before-prepare.v1",
            "status": "observed",
            "backend_initialized_before_prepare": backend_initialized_before_prepare,
            "source_field": "engine_core.structured_output_manager.backend",
            "request_id": state.get("binding", {}).get("request_id"),
            "schema_document_sha256": hashlib.sha256(
                json.dumps(schema_document, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "engine_boot_id": self._runtime_control_boot_id,
            "config_epoch": limits["config_epoch"],
            "clock_id": state.get("binding", {}).get("clock_id"),
            "observed_at_ns": state.get("binding", {}).get("observed_at_ns"),
            "boundary": "readonly_getter_before_native_prepare",
        }
        return state

    def get_schema_pre_action_capability(self) -> dict[str, object]:
        """Return a zero-call route capability descriptor for GET gates."""

        limits = self.scheduler.get_runtime_scheduler_limits()
        return {
            "schema": "qbi.schema-pre-action-capability.v1",
            "route": "/qbi/schema/pre-action-state",
            "methods": ["GET", "POST"],
            "precompile_route": "/qbi/schema/pre-action-prepare",
            "provider_bound": True,
            "request_specific_post_required": True,
            "engine_boot_id": self._runtime_control_boot_id,
            "config_epoch": limits["config_epoch"],
            "status": "READY_FOR_REQUEST_BOUND_POST",
            "send_allowed": False,
            "model_calls": 0,
            "remote": False,
        }

    def get_prefix_cache_pre_action_state(
        self,
        session_plan: dict[str, object] | None = None,
        request_id: str | None = None,
        request_sha256: str | None = None,
    ) -> dict[str, object]:
        """Return a provider-bound KV/session snapshot before admission."""

        from vllm.v1.structured_output.pre_action_observer import (
            observe_prefix_pre_action,
        )

        manager = self.scheduler.kv_cache_manager
        pool = manager.block_pool
        # Block zero is vLLM's null block.  A non-positive usable capacity is
        # an unavailable provider fact, not a capacity of zero to be imputed
        # into the K feature contract.
        raw_total_blocks = getattr(pool, "num_gpu_blocks", None)
        total_blocks = (
            int(raw_total_blocks) - 1
            if type(raw_total_blocks) is int and int(raw_total_blocks) > 1
            else None
        )
        free_blocks = int(pool.get_num_free_blocks())
        pressure = float(pool.get_usage())
        capacity = (free_blocks / total_blocks) if total_blocks is not None else None
        cached_hash_map = getattr(pool, "cached_block_hash_to_block", None)
        cached_hash_count = (
            len(cached_hash_map)
            if cached_hash_map is not None and hasattr(cached_hash_map, "__len__")
            else None
        )
        limits = self.scheduler.get_runtime_scheduler_limits()
        now = time.perf_counter_ns()
        runtime_state = manager.get_qbi_prefix_cache_pre_action_state()
        snapshot = {
            "cache_usage_ratio": pressure,
            "cache_pressure_ratio": pressure,
            "cache_capacity_ratio": capacity,
            "free_gpu_blocks": free_blocks,
            "capacity_blocks": total_blocks,
            "cached_block_hash_count": cached_hash_count,
            "prefix_caching_enabled": bool(manager.enable_caching),
            "prefix_caching_hash_algo": getattr(
                self.vllm_config.cache_config, "prefix_caching_hash_algo", None
            ),
            "engine_boot_id": self._runtime_control_boot_id,
            "config_epoch": limits["config_epoch"],
            "runtime_receipt_contract": runtime_state["runtime_receipt_contract"],
            "resident_blocks": runtime_state.get("cached_block_hash_count"),
        }
        state = observe_prefix_pre_action(
            engine_snapshot=snapshot,
            session_plan=session_plan,
            request_id=request_id,
            engine_boot_id=self._runtime_control_boot_id,
            config_epoch=limits["config_epoch"],
            clock_id="engine-core:" + self._runtime_control_boot_id,
            observed_at_ns=now,
            decision_at_ns=now,
        )
        if (
            request_sha256 is not None
            and request_sha256 != state["binding"]["request_sha256"]
        ):
            raise ValueError("Prefix request identity differs from provider binding")
        state["scheduler_limits"] = limits
        state["prefix_caching_hash_algo"] = snapshot["prefix_caching_hash_algo"]
        return state

    def get_prefix_cache_policy_receipts(self) -> dict[str, Any]:
        """Drain request-bound lookup/admit/residency/eviction receipts."""
        state = self.scheduler.kv_cache_manager.take_qbi_prefix_cache_policy_receipts()
        limits = self.scheduler.get_runtime_scheduler_limits()
        return {
            **state,
            "engine_boot_id": self._runtime_control_boot_id,
            "config_epoch": limits["config_epoch"],
            "prefix_caching_enabled": bool(
                self.vllm_config.cache_config.enable_prefix_caching
            ),
            "prefix_caching_hash_algo": (
                self.vllm_config.cache_config.prefix_caching_hash_algo
            ),
            "scheduler_limits": limits,
        }

    def get_runtime_transition_state(self) -> dict[str, Any]:
        limits = self.scheduler.get_runtime_scheduler_limits()
        return {
            "boot_id": self._runtime_control_boot_id,
            "config_epoch": limits["config_epoch"],
            "unfinished_requests": (
                self.scheduler.get_num_old_epoch_unfinished_requests()
            ),
            "pending_epoch_requests": limits["pending_epoch_request_count"],
            "transition_staged": bool(limits["transition_staged"]),
            "scheduler_limits": limits,
        }

    def get_batch_scope_state(self, after_sequence: int = 0) -> dict[str, Any]:
        """Read batch proposals and effective receipts bound to this engine epoch."""

        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("batch-scope cursor must be an integer")
        if after_sequence < 0:
            raise ValueError("batch-scope cursor must be non-negative")
        limits = self.scheduler.get_runtime_scheduler_limits()
        return {
            **self.scheduler.get_batch_scope_state(after_sequence),
            "engine_boot_id": self._runtime_control_boot_id,
            "config_epoch": limits["config_epoch"],
            "scheduler_limits": limits,
        }

    def propose_batch_scope_budget(
        self,
        expected_boot_id: str,
        expected_config_epoch: int,
        batch_proposal_id: str,
        member_request_ids: list[str],
        token_budget: int,
        member_match: str = "client_alias",
        require_prefill_decode_overlap: bool = False,
        target_prefill_member_id: str | None = None,
        peer_decode_member_id: str | None = None,
    ) -> dict[str, Any]:
        """Accept a proposal only against the exact live boot and config epoch."""

        if not isinstance(expected_boot_id, str) or not expected_boot_id:
            raise ValueError("expected boot_id must be non-empty text")
        if isinstance(expected_config_epoch, bool) or not isinstance(
            expected_config_epoch, int
        ):
            raise ValueError("expected config_epoch must be an integer")
        state = self.get_runtime_transition_state()
        if expected_boot_id != state["boot_id"]:
            raise ValueError("batch-scope proposal boot_id is stale")
        if expected_config_epoch != state["config_epoch"]:
            raise ValueError("batch-scope proposal config_epoch is stale")
        proposal = self.scheduler.propose_batch_scope_budget(
            expected_config_epoch=expected_config_epoch,
            batch_proposal_id=batch_proposal_id,
            member_request_ids=member_request_ids,
            token_budget=token_budget,
            member_match=member_match,
            require_prefill_decode_overlap=require_prefill_decode_overlap,
            target_prefill_member_id=target_prefill_member_id,
            peer_decode_member_id=peer_decode_member_id,
        )
        return {
            "schema": "qbi.batch-scope-proposal-acceptance.v1",
            "accepted": True,
            "engine_boot_id": state["boot_id"],
            "config_epoch": state["config_epoch"],
            "proposal": proposal,
            "effect_status": "pending_until_exact_schedule_confirmation",
        }

    def cancel_batch_scope_proposal(
        self, expected_boot_id: str, batch_proposal_id: str
    ) -> dict[str, Any]:
        if not isinstance(expected_boot_id, str) or not expected_boot_id:
            raise ValueError("expected boot_id must be non-empty text")
        state = self.get_runtime_transition_state()
        if expected_boot_id != state["boot_id"]:
            raise ValueError("batch-scope cancellation boot_id is stale")
        result = self.scheduler.cancel_batch_scope_proposal(batch_proposal_id)
        return {
            **result,
            "engine_boot_id": state["boot_id"],
            "config_epoch": state["config_epoch"],
        }

    def run_qbi_online_selector_benchmark(
        self, run_tag: str, iterations: int
    ) -> dict[str, Any]:
        """Time a read-only shadow pass through the formal selector."""
        import hashlib
        import json
        import struct
        from pathlib import Path

        from vllm_quality_bounded_inference.controller import InvocationContext
        from vllm_quality_bounded_inference.live_policy import LIVE_POLICY_PROFILES
        from vllm_quality_bounded_inference.mechanism_registry import (
            RUNTIME_MECHANISM_BINDINGS,
            list_runtime_mechanism_configurations,
        )
        from vllm_quality_bounded_inference.online_fast_path import (
            DECISION_RECORD_BYTES,
            CompiledLinearHead,
            CompiledMaskedFeatureEncoder,
            FixedDecisionJournal,
            NativeSchedulerFeatureReader,
            benchmark_percentiles,
        )
        from vllm_quality_bounded_inference.scope_policy import (
            ConsequenceEstimate,
            select_registered_configuration,
        )

        if (
            not isinstance(run_tag, str)
            or not run_tag
            or len(run_tag) > 40
            or any(not (char.isalnum() or char in "-_") for char in run_tag)
        ):
            raise ValueError(
                "run_tag must contain at most 40 letters, digits, '-' or '_'"
            )
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or not 100 <= iterations <= 20000
        ):
            raise ValueError("iterations must be an integer between 100 and 20000")

        artifact_path_text = os.environ.get("ANISE_QBI_FAST_SELECTOR_ARTIFACT")
        expected_sha256 = os.environ.get("ANISE_QBI_FAST_SELECTOR_ARTIFACT_SHA256")
        if not artifact_path_text or not expected_sha256:
            raise ValueError("compiled TRAIN head path and SHA256 must be configured")
        artifact_bytes = Path(artifact_path_text).read_bytes()
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_sha256 != expected_sha256:
            raise ValueError("compiled TRAIN head file hash mismatch")
        artifact = json.loads(artifact_bytes)
        if (
            artifact.get("fit_mode") != "train_only_benchmark_compile"
            or artifact.get("deployment_authorized") is not False
            or artifact.get("calibration") is not None
        ):
            raise ValueError(
                "shadow benchmark requires the existing uncalibrated TRAIN bundle"
            )

        boot_id = self._runtime_control_boot_id
        journal_dir = Path(
            os.environ.get("ANISE_QBI_FAST_SELECTOR_JOURNAL_DIR", "/tmp")
        )
        if not journal_dir.is_dir():
            raise ValueError("selector journal directory must already exist")
        policy = LIVE_POLICY_PROFILES["strict"]
        profile_order = (
            ("token_budget_output_control", "S1"),
            ("native_output_stream_batching", "S5"),
        )
        warmup = 128
        results: dict[str, Any] = {}
        for mechanism, stage_label in profile_order:
            profile = artifact.get("mechanisms", {}).get(mechanism)
            if not isinstance(profile, dict):
                raise ValueError(f"compiled TRAIN bundle misses {mechanism}")
            if (
                profile.get("activation_authorized") is not False
                or profile.get("decision_scope_support") != "invocation"
            ):
                raise ValueError(f"{mechanism} is not an invocation-scoped shadow head")

            encoder = CompiledMaskedFeatureEncoder(profile["encoder"])
            head = CompiledLinearHead(
                (tuple(profile["head_weights"]),),
                (float(profile["head_bias"]),),
            )
            if encoder.width != head.feature_width:
                raise ValueError(f"compiled encoder/head width differs for {mechanism}")
            reader = NativeSchedulerFeatureReader(
                profile["feature_names"],
                profile["request_plan_values"],
                request_plan_mask=0,
            )
            live_field_mask = sum(
                1 << index
                for index, name in enumerate(profile["feature_names"])
                if name
                in {"active_request_count", "queue_depth", "memory_pressure_ratio"}
            )
            configurations = list_runtime_mechanism_configurations(
                decision_scope="invocation"
            )
            config_ids = {item.name: index for index, item in enumerate(configurations)}
            default_name = next(item.name for item in configurations if item.is_default)
            default_id = config_ids[default_name]
            estimates = {
                default_name: ConsequenceEstimate(
                    estimated_cost=0.0,
                    predicted_quality_loss=0.0,
                    predicted_violation_probability=0.0,
                )
            }
            opportunity_configurations = frozenset()
            capability_mechanisms = frozenset(RUNTIME_MECHANISM_BINDINGS.values())
            journal_path = journal_dir / (
                f"online-selector-shadow-{boot_id}-{run_tag}-{stage_label}.bin"
            )
            journal = FixedDecisionJournal(
                journal_path,
                initial_records=1 << (iterations + warmup).bit_length(),
            )
            vector = [0.0] * encoder.width
            score = [0.0]

            def select_current_scope(
                decision_id: str,
                *,
                _estimates=estimates,
                _opportunity_configurations=opportunity_configurations,
                _capability_mechanisms=capability_mechanisms,
                _default_name=default_name,
            ):
                context = InvocationContext(
                    quality_sensitivity=1.0,
                    latency_pressure=0.0,
                    calibration_drift=0.0,
                    snapshot_id=decision_id,
                )
                return select_registered_configuration(
                    policy,
                    context,
                    estimates=_estimates,
                    opportunity_configurations=_opportunity_configurations,
                    capability_mechanisms=_capability_mechanisms,
                    artifact_hash=str(artifact["bundle_sha256"]),
                    current_configuration=_default_name,
                    minimum_hold_satisfied=True,
                    include_traces=False,
                )

            def one_decision(
                sequence: int,
                *,
                record: bool,
                _mechanism=mechanism,
                _stage_label=stage_label,
                _reader=reader,
                _live_field_mask=live_field_mask,
                _encoder=encoder,
                _vector=vector,
                _head=head,
                _score=score,
                _journal=journal,
                _config_ids=config_ids,
                _default_id=default_id,
                _select_current_scope=select_current_scope,
            ):
                epoch = int(self.scheduler._runtime_config_epoch)
                state = _reader.read(self.scheduler)
                if state.observed_mask & _live_field_mask != _live_field_mask:
                    raise RuntimeError(
                        f"live scheduler fields unavailable for {_mechanism}"
                    )
                if not _encoder.encode_state_into(
                    state, _vector
                ) or not _head.evaluate_into(_vector, _score):
                    raise RuntimeError(
                        f"compiled TRAIN head rejected live state for {_mechanism}"
                    )
                decision_id = f"{boot_id}:{run_tag}:{epoch}:{_stage_label}:{sequence}"
                selected = _select_current_scope(decision_id)
                decision = selected.decision
                candidate_mask = (1 << len(decision.assessments)) - 1
                feasible_mask = sum(
                    1 << index
                    for index, assessment in enumerate(decision.assessments)
                    if assessment.feasible
                )
                if record:
                    _journal.append(
                        decision_sequence=sequence,
                        boot_high=int(boot_id.replace("-", "")[:16], 16),
                        boot_low=int(boot_id.replace("-", "")[16:], 16),
                        decision_epoch=epoch,
                        request_sequence=0,
                        candidate_mask=candidate_mask,
                        feasible_mask=feasible_mask,
                        proposed_config_id=_config_ids[
                            decision.selected_mechanism.name
                        ],
                        effective_config_id=_default_id,
                        effective_epoch=epoch,
                        decision_status=1 if decision.used_fallback else 0,
                        action_status=2,
                        fallback_code=1 if decision.used_fallback else 0,
                    )
                return None

            for index in range(warmup):
                one_decision(index + 1, record=False)
            timings_ns: list[int] = []
            for index in range(iterations):
                started = time.perf_counter_ns()
                one_decision(index + 1, record=True)
                timings_ns.append(time.perf_counter_ns() - started)

            component_samples = min(1000, iterations)
            state_ns: list[int] = []
            encode_head_ns: list[int] = []
            selection_ns: list[int] = []
            observed_masks: list[int] = []
            for index in range(component_samples):
                started = time.perf_counter_ns()
                state = reader.read(self.scheduler)
                state_ns.append(time.perf_counter_ns() - started)
                observed_masks.append(state.observed_mask)
                started = time.perf_counter_ns()
                if not encoder.encode_state_into(
                    state, vector
                ) or not head.evaluate_into(vector, score):
                    raise RuntimeError(
                        f"compiled TRAIN head rejected component sample for {mechanism}"
                    )
                encode_head_ns.append(time.perf_counter_ns() - started)
                started = time.perf_counter_ns()
                select_current_scope(f"{boot_id}:component:{stage_label}:{index}")
                selection_ns.append(time.perf_counter_ns() - started)

            selection_profile = []
            if os.environ.get("ANISE_QBI_PROFILE_SELECTOR") == "1":
                import cProfile

                from vllm_quality_bounded_inference import controller, scope_policy

                constructor_names = {
                    cls.__init__.__code__: cls.__name__ + ".__init__"
                    for cls in (
                        controller.InvocationContext,
                        controller.MechanismCandidate,
                        controller.MechanismAssessment,
                        controller.MechanismDecision,
                        scope_policy.ScopeSelectionResult,
                    )
                }
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    for index in range(component_samples):
                        select_current_scope(
                            f"{boot_id}:profile:{stage_label}:{index}"
                        )
                finally:
                    profiler.disable()
                for entry in sorted(
                    profiler.getstats(), key=lambda item: item.inlinetime,
                    reverse=True,
                )[:20]:
                    code = entry.code
                    selection_profile.append({
                        "function": constructor_names.get(
                            code, getattr(code, "co_name", str(code))
                        ),
                        "file": getattr(code, "co_filename", "builtin"),
                        "line": getattr(code, "co_firstlineno", 0),
                        "calls": entry.callcount,
                        "self_ms": entry.inlinetime * 1000,
                        "cumulative_ms": entry.totaltime * 1000,
                    })

            journal.flush()
            decision_count = journal.count
            journal.close()
            raw_records = journal_path.read_bytes()
            records_valid = len(raw_records) >= decision_count * DECISION_RECORD_BYTES
            fallback_count = 0
            for index in range(decision_count):
                record = struct.unpack_from(
                    "<QQQQQQQQHHBBH", raw_records, index * DECISION_RECORD_BYTES
                )
                if (
                    record[0] != index + 1
                    or record[7] != record[3]
                    or record[9] != default_id
                    or record[11] != 2
                ):
                    records_valid = False
                    break
                fallback_count += int(record[10] == 1)
            if not records_valid or decision_count != iterations:
                raise RuntimeError(
                    f"shadow decision journal validation failed for {mechanism}"
                )

            p50_us, p95_us = benchmark_percentiles(timings_ns)

            def percentiles(samples: list[int]) -> dict[str, float]:
                p50, p95 = benchmark_percentiles(samples)
                return {"p50_us": p50, "p95_us": p95}

            results[stage_label] = {
                "mechanism": mechanism,
                "decision_scope": "invocation",
                "head": profile["head_name"],
                "head_is_joint_shared_lifecycle_mechanism": True,
                "head_prediction_authoritative": False,
                "feature_width": encoder.width,
                "live_scheduler_fields": [
                    name
                    for name in profile["feature_names"]
                    if name
                    in {"active_request_count", "queue_depth", "memory_pressure_ratio"}
                ],
                "missing_request_features": len(profile["feature_names"])
                - live_field_mask.bit_count(),
                "runtime_candidates": [item.name for item in configurations],
                "selected_configuration": default_name,
                "eligible_nondefault_candidates": 0,
                "warmup_decisions": warmup,
                "timed_decisions": iterations,
                "decision_records": decision_count,
                "decision_record_bytes": DECISION_RECORD_BYTES,
                "journal_associations_valid": records_valid,
                "action_submissions": 0,
                "effective_configuration_changed": False,
                "fallback_decisions": fallback_count,
                "observed_feature_masks": sorted(set(observed_masks)),
                "online_path_including_scheduler_read_encode_head_"
                "formal_ranking_journal": {
                    "p50_us": p50_us,
                    "p95_us": p95_us,
                },
                "components": {
                    "scheduler_state_read": percentiles(state_ns),
                    "compiled_encoder_plus_head": percentiles(encode_head_ns),
                    "scope_policy_and_select_for_scope": percentiles(selection_ns),
                },
                "journal_path": str(journal_path),
                "selection_profile_separate_from_timed_pass": selection_profile,
            }

        return {
            "schema": "anise.online-selector-cost-shadow.v1",
            "mode": (
                "shadow_only; no action submission; no config mutation; "
                "no model requests"
            ),
            "engine_boot_id": boot_id,
            "train_head_bundle_sha256": artifact.get("bundle_sha256"),
            "artifact_file_sha256": artifact_sha256,
            "calibration_present": False,
            "deployment_authorized": False,
            "selector": (
                "scope_policy.select_registered_configuration -> "
                "controller.select_for_scope"
            ),
            "http_round_trip_in_timed_path": False,
            "profile_results": results,
        }

    def get_native_recapture_scope_state(
        self, after_sequence: int = 0
    ) -> dict[str, Any]:
        """Join scheduler scope receipts to the live EngineCore identity."""

        state = self.scheduler.get_native_recapture_scope_state(after_sequence)
        return {
            **state,
            "engine_boot_id": self._runtime_control_boot_id,
            "scheduler_limits": self.scheduler.get_runtime_scheduler_limits(),
        }

    def get_priority_scheduling_state(self, after_sequence: int = 0) -> dict[str, Any]:
        """Join request-priority receipts to the live EngineCore identity."""

        state = self.scheduler.get_priority_scheduling_state(after_sequence)
        return {
            **state,
            "engine_boot_id": self._runtime_control_boot_id,
            "queue_policy": self.scheduler.policy.value,
            "scheduler_limits": self.scheduler.get_runtime_scheduler_limits(),
        }

    @staticmethod
    def _qbi_receipts_after(
        receipts: deque[dict[str, Any]], after_sequence: int
    ) -> list[dict[str, Any]]:
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
        return [row for row in receipts if int(row["sequence"]) > after_sequence]

    def get_dynamic_mtp_state(self, after_sequence: int = 0) -> dict[str, Any]:
        """Read prompt-free scheduler/model-runner dynamic-MTP receipts."""

        return {
            "schema": "qbi.dynamic-qwen35-mtp-engine-state.v1",
            "engine_boot_id": self._runtime_control_boot_id,
            "capacity": _QBI_MECHANISM_RECEIPT_LIMIT,
            "latest_sequence": self._qbi_dynamic_mtp_sequence,
            "dropped_receipts": self._qbi_dynamic_mtp_dropped_receipts,
            "receipts": self._qbi_receipts_after(
                self._qbi_dynamic_mtp_receipts, after_sequence
            ),
        }

    def get_reasoning_budget_state(self, after_sequence: int = 0) -> dict[str, Any]:
        """Read prompt-free live sampler receipts for reasoning budgets."""

        return {
            "schema": "qbi.reasoning-token-budget-engine-state.v1",
            "engine_boot_id": self._runtime_control_boot_id,
            "capacity": _QBI_MECHANISM_RECEIPT_LIMIT,
            "latest_sequence": self._qbi_reasoning_budget_sequence,
            "dropped_receipts": self._qbi_reasoning_budget_dropped_receipts,
            "receipts": self._qbi_receipts_after(
                self._qbi_reasoning_budget_receipts, after_sequence
            ),
        }

    def _record_qbi_model_runner_telemetry(
        self, scheduler_output: SchedulerOutput, model_output: ModelRunnerOutput
    ) -> None:
        mtp_state = getattr(model_output, "qbi_dynamic_mtp_receipt", None)
        if isinstance(mtp_state, dict):
            if len(self._qbi_dynamic_mtp_receipts) == _QBI_MECHANISM_RECEIPT_LIMIT:
                self._qbi_dynamic_mtp_dropped_receipts += 1
            self._qbi_dynamic_mtp_sequence += 1
            sequence = self._qbi_dynamic_mtp_sequence
            scheduler_members = list(scheduler_output.num_scheduled_tokens.keys())
            configured_schedule = (
                self.vllm_config.speculative_config.num_speculative_tokens_per_batch_size
                if self.vllm_config.speculative_config is not None
                else None
            )
            schedule_payload = {
                "runtime_max_speculative_tokens": (
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config is not None
                    else 0
                ),
                "num_speculative_tokens_per_batch_size": configured_schedule,
            }
            schedule_sha256 = hashlib.sha256(
                json.dumps(
                    schedule_payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            batch_identity = {
                "engine_boot_id": self._runtime_control_boot_id,
                "sequence": sequence,
                "member_request_ids": scheduler_members,
                "total_num_scheduled_tokens": (
                    scheduler_output.total_num_scheduled_tokens
                ),
            }
            batch_proposal_id = hashlib.sha256(
                json.dumps(
                    batch_identity, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            receipt = {
                **mtp_state,
                "schema": "qbi.dynamic-qwen35-mtp-engine-receipt.v1",
                "sequence": sequence,
                "batch_proposal_id": batch_proposal_id,
                "engine_boot_id": self._runtime_control_boot_id,
                "member_request_ids": scheduler_members,
                "schedule_sha256": schedule_sha256,
                "scheduler_effective_k_proven": (
                    mtp_state.get("scheduler_selected_num_spec_tokens")
                    == scheduler_output.num_spec_tokens_to_schedule
                ),
                "model_runner_effective_k_proven": (
                    mtp_state.get("model_runner_consumed_num_spec_tokens")
                    == scheduler_output.num_spec_tokens_to_schedule
                    and mtp_state.get("member_request_ids") == scheduler_members
                ),
                "proposed_token_count": None,
                "draft_receipt_complete": False,
                "draft_receipt_collision": False,
            }
            if self._qbi_pending_dynamic_mtp_receipt is not None:
                self._qbi_pending_dynamic_mtp_receipt["draft_receipt_collision"] = True
                receipt["draft_receipt_collision"] = True
            self._qbi_dynamic_mtp_receipts.append(receipt)
            self._qbi_pending_dynamic_mtp_receipt = receipt

        raw_reasoning = getattr(model_output, "qbi_reasoning_budget_receipts", [])
        if not isinstance(raw_reasoning, list):
            return
        for sampler_state in raw_reasoning:
            if not isinstance(sampler_state, dict):
                continue
            if len(self._qbi_reasoning_budget_receipts) == _QBI_MECHANISM_RECEIPT_LIMIT:
                self._qbi_reasoning_budget_dropped_receipts += 1
            self._qbi_reasoning_budget_sequence += 1
            budget_binding_proven = bool(
                sampler_state.get("sampler_state_tracked")
                and sampler_state.get("initialized_start_token_ids")
                and sampler_state.get("initialized_end_token_ids")
                and sampler_state.get("requested_thinking_token_budget")
                == sampler_state.get("effective_thinking_token_budget")
            )
            budget_enforced_proven = bool(
                budget_binding_proven
                and sampler_state.get("reasoning_end_kind") == "forced_budget"
                and sampler_state.get("forced_end_token_applied")
            )
            receipt = {
                **sampler_state,
                "schema": "qbi.reasoning-token-budget-engine-receipt.v1",
                "sequence": self._qbi_reasoning_budget_sequence,
                "engine_boot_id": self._runtime_control_boot_id,
                "transport_parameter_applied_proven": bool(
                    sampler_state.get("sampler_state_tracked")
                ),
                "runtime_budget_binding_proven": budget_binding_proven,
                "runtime_budget_effective_proven": budget_enforced_proven,
                "runtime_budget_enforced_proven": budget_enforced_proven,
                "natural_end_before_budget": bool(
                    budget_binding_proven
                    and sampler_state.get("reasoning_end_kind") == "natural_end"
                    and not sampler_state.get("forced_end_token_applied")
                ),
            }
            self._qbi_reasoning_budget_receipts.append(receipt)

    def _finalize_qbi_dynamic_mtp_draft_receipt(self, draft_token_ids: Any) -> None:
        receipt = self._qbi_pending_dynamic_mtp_receipt
        if receipt is None:
            return
        rows = getattr(draft_token_ids, "draft_token_ids", None)
        req_ids = getattr(draft_token_ids, "req_ids", None)
        if rows is None:
            proposed_token_count = 0
            member_identity_match = (
                receipt["model_runner_consumed_num_spec_tokens"] == 0
            )
        else:
            proposed_token_count = sum(len(row) for row in rows)
            member_identity_match = list(req_ids or []) == receipt["member_request_ids"]
        receipt["proposed_token_count"] = proposed_token_count
        receipt["draft_member_identity_match"] = member_identity_match
        receipt["proposal_count_within_bound_proven"] = proposed_token_count <= (
            int(receipt["model_runner_consumed_num_spec_tokens"])
            * len(receipt["member_request_ids"])
        )
        receipt["draft_receipt_complete"] = True
        self._qbi_pending_dynamic_mtp_receipt = None

    def get_native_ngram_pre_action_state(
        self,
        provider_request_id: str,
        source_request_id: str,
        source_receipt_sha256: str,
        checkpoint_sha256: str,
        config_support_range: str,
        action_config: dict[str, Any],
        engine_token_ids: list[int],
        payload_sha256: str,
        token_ids_sha256: str,
        tokenizer_identity_sha256: str,
        chat_template_sha256: str,
    ) -> dict[str, Any]:
        """Build a request-bound N-gram profile before generation is submitted."""
        from vllm_quality_bounded_inference.mechanism_pre_action_collectors import (
            ngram_pre_action_profile,
        )

        def require_sha256(value: Any, name: str) -> str:
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            return value

        for value, name in (
            (provider_request_id, "provider_request_id"),
            (source_request_id, "source_request_id"),
            (config_support_range, "config_support_range"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        source_receipt_sha256 = require_sha256(
            source_receipt_sha256, "source_receipt_sha256"
        )
        checkpoint_sha256 = require_sha256(checkpoint_sha256, "checkpoint_sha256")
        payload_sha256 = require_sha256(payload_sha256, "payload_sha256")
        token_ids_sha256 = require_sha256(token_ids_sha256, "token_ids_sha256")
        tokenizer_identity_sha256 = require_sha256(
            tokenizer_identity_sha256, "tokenizer_identity_sha256"
        )
        chat_template_sha256 = require_sha256(
            chat_template_sha256, "chat_template_sha256"
        )
        if os.environ.get("QBI_CHECKPOINT_SHA256") != checkpoint_sha256:
            raise ValueError("N-gram checkpoint identity differs from service binding")
        if (
            os.environ.get("QBI_NATIVE_NGRAM_CONFIG_SUPPORT_RANGE")
            != config_support_range
        ):
            raise ValueError("N-gram config support differs from service binding")
        if not isinstance(action_config, dict) or set(action_config) != {
            "method",
            "num_speculative_tokens",
            "prompt_lookup_min",
            "prompt_lookup_max",
        }:
            raise ValueError("N-gram action config has unexpected or missing fields")
        if action_config.get("method") != "ngram":
            raise ValueError("resident proposer is not N-gram")
        for field in (
            "num_speculative_tokens",
            "prompt_lookup_min",
            "prompt_lookup_max",
        ):
            value = action_config[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"N-gram action config {field} is invalid")
        if action_config["num_speculative_tokens"] <= 0:
            raise ValueError("N-gram action depth must be positive")
        if action_config["prompt_lookup_min"] < 1 or action_config[
            "prompt_lookup_max"
        ] < action_config["prompt_lookup_min"]:
            raise ValueError("N-gram lookup bounds are invalid")
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None or speculative_config.method != "ngram":
            raise ValueError("resident proposer is not N-gram")
        if (
            action_config["num_speculative_tokens"]
            != speculative_config.num_speculative_tokens
            or action_config["prompt_lookup_min"]
            != speculative_config.prompt_lookup_min
            or action_config["prompt_lookup_max"]
            != speculative_config.prompt_lookup_max
        ):
            raise ValueError("N-gram action config differs from startup proposer")
        if (
            not isinstance(engine_token_ids, list)
            or not engine_token_ids
            or any(type(token) is not int or token < 0 for token in engine_token_ids)
        ):
            raise ValueError("engine_token_ids must be exact non-negative token IDs")
        computed_token_ids_sha256 = hashlib.sha256(
            json.dumps(
                engine_token_ids,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if computed_token_ids_sha256 != token_ids_sha256:
            raise ValueError("token_ids_sha256 differs from exact engine token IDs")

        runtime_state = self.get_runtime_transition_state()
        limits = runtime_state["scheduler_limits"]
        startup_depth = limits["startup_num_speculative_tokens"]
        current_depth = limits["num_speculative_tokens"]
        if (
            limits.get("speculative_method") != "ngram"
            or action_config["num_speculative_tokens"] != startup_depth
            or current_depth not in {0, startup_depth}
            or runtime_state["transition_staged"]
            or runtime_state["pending_epoch_requests"] != 0
        ):
            raise RuntimeError("N-gram runtime epoch is not complete and stable")

        observed_at_ns = time.monotonic_ns()
        scope_state = self.scheduler.get_native_recapture_scope_state()
        observation = scope_state.get("pre_action_observation", {})
        engine_state = observation.get("engine", {})
        active_request_count = engine_state.get("active_request_count")
        queue_depth = engine_state.get("queue_depth")
        runtime_pressure_ratio = engine_state.get("kv_pressure_ratio")
        if type(active_request_count) is not int or active_request_count < 0:
            raise RuntimeError("scheduler active-request state is unavailable")
        if type(queue_depth) is not int or queue_depth < 0:
            raise RuntimeError("scheduler queue state is unavailable")
        if (
            isinstance(runtime_pressure_ratio, bool)
            or not isinstance(runtime_pressure_ratio, (int, float))
            or not 0 <= float(runtime_pressure_ratio) <= 1
        ):
            raise RuntimeError("scheduler KV pressure is unavailable")
        profile = ngram_pre_action_profile(
            engine_token_ids, engine_token_ids, action_config
        )
        decision_at_ns = time.monotonic_ns()
        boot_id = self._runtime_control_boot_id
        return {
            "schema": "qbi.native-ngram-pre-action-state.v2",
            "status": "READY",
            "scope": "invocation",
            "provider_request_id": provider_request_id,
            "source_request_id": source_request_id,
            "source_receipt_sha256": source_receipt_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "config_support_range": config_support_range,
            "engine_boot_id": boot_id,
            "config_epoch": runtime_state["config_epoch"],
            "clock_id": "engine-core-monotonic:" + boot_id,
            "observed_at_ns": observed_at_ns,
            "decision_at_ns": decision_at_ns,
            "decision_id": provider_request_id,
            "epoch_complete": True,
            "payload_sha256": payload_sha256,
            "token_ids_sha256": token_ids_sha256,
            "tokenizer_identity_sha256": tokenizer_identity_sha256,
            "chat_template_sha256": chat_template_sha256,
            "pre_action_observation": {
                "schema": "anise-native-ngram-pre-action-observation-v2",
                "engine": {
                    **profile,
                    "engine_token_ids": engine_token_ids,
                    "lookup_token_ids": engine_token_ids,
                    "active_request_count": active_request_count,
                    "queue_depth": queue_depth,
                    "runtime_pressure_ratio": float(runtime_pressure_ratio),
                },
            },
        }

    def get_causal_pre_decision_state(self, after_sequence: int = 0) -> dict[str, Any]:
        """Join pre-decision scheduler projections to the live EngineCore."""

        state = self.scheduler.get_causal_pre_decision_state(after_sequence)
        return {
            **state,
            "engine_boot_id": self._runtime_control_boot_id,
        }

    def get_native_ngram_scope_state(self, after_sequence: int = 0) -> dict[str, Any]:
        """Join worker-local pre-lookup receipts to this EngineCore.

        N-gram lookup is executed by the model runner, after Scheduler has
        produced a batch.  The scheduler-only surface therefore cannot be the
        source of these fields.  This read-only RPC joins every worker's
        bounded state and keeps the worker states visible for rank auditing.
        """
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("native N-gram sequence cursor must be an integer")
        if after_sequence < 0:
            raise ValueError("native N-gram sequence cursor must be non-negative")
        try:
            worker_states = self.model_executor.collective_rpc(
                "get_native_ngram_scope_state", args=(after_sequence,)
            )
        except Exception as error:
            return {
                "schema": "qbi.native-ngram-worker-state.v1",
                "status": "NOT_READY",
                "observer_enabled": False,
                "latest_sequence": 0,
                "oldest_available_sequence": 0,
                "receipts": [],
                "worker_states": [],
                "scope": "worker_pre_lookup",
                "reason": (
                    f"worker observer RPC unavailable: {type(error).__name__}: {error}"
                ),
                "required_fields": [
                    "current_input_token_suffix",
                    "lookup_token_corpus",
                    "draft_config",
                    "clock_id",
                    "config_epoch",
                ],
                "forbidden_substitutes": [
                    "request_id",
                    "prompt_token_id_hash",
                    "scheduled_token_count",
                    "accepted_token_count",
                ],
            }
        if isinstance(worker_states, dict):
            worker_states = [worker_states]
        if not isinstance(worker_states, list) or not all(
            isinstance(state, dict) for state in worker_states
        ):
            raise RuntimeError("worker observer RPC returned a non-object state")
        receipts = [
            receipt
            for state in worker_states
            for receipt in state.get("receipts", [])
            if isinstance(receipt, dict)
        ]
        latest = max(
            [int(state.get("latest_sequence", 0)) for state in worker_states] or [0]
        )
        oldest_values = [
            int(state.get("oldest_available_sequence", 0))
            for state in worker_states
            if int(state.get("oldest_available_sequence", 0)) > 0
        ]
        enabled = bool(worker_states) and all(
            state.get("status") == "READY" and state.get("observer_enabled") is True
            for state in worker_states
        )
        return {
            "schema": "qbi.native-ngram-engine-state.v1",
            "status": "READY" if enabled else "NOT_READY",
            "observer_enabled": enabled,
            "engine_boot_id": self._runtime_control_boot_id,
            "scope": "worker_pre_lookup",
            "latest_sequence": latest,
            "oldest_available_sequence": min(oldest_values) if oldest_values else 0,
            "receipts": receipts,
            "worker_states": worker_states,
            "cursor_semantics": (
                "worker-local sequence cursors are preserved in worker_states"
            ),
            "model_calls": 0,
            "remote_calls": 0,
            "training": False,
        }

    def get_repetition_detection_state(self, after_sequence: int = 0) -> dict[str, Any]:
        """Join provider pre-stop events to the current EngineCore identity."""
        state = self.scheduler.get_repetition_detection_state(after_sequence)
        return {
            **state,
            "engine_boot_id": self._runtime_control_boot_id,
            "config_epoch": self.scheduler.get_runtime_scheduler_limits().get(
                "config_epoch", 0
            ),
        }

    def get_suffix_acceptance_state(self, after_sequence: int = 0) -> dict[str, Any]:
        """Join completed suffix windows to the current EngineCore identity."""

        state = self.scheduler.get_suffix_acceptance_state(after_sequence)
        return {
            **state,
            "engine_boot_id": self._runtime_control_boot_id,
            "config_epoch": self.scheduler.get_runtime_scheduler_limits().get(
                "config_epoch", 0
            ),
        }

    def get_native_suffix_proposal_state(
        self, after_sequence: int = 0
    ) -> dict[str, Any]:
        """Join worker-local S4 features captured before suffix verification."""
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("native Suffix sequence cursor must be an integer")
        if after_sequence < 0:
            raise ValueError("native Suffix sequence cursor must be non-negative")
        try:
            worker_states = self.model_executor.collective_rpc(
                "get_native_suffix_proposal_state", args=(after_sequence,)
            )
        except Exception as error:
            return {
                "schema": "qbi.native-suffix-worker-state.v1",
                "status": "NOT_READY",
                "observer_enabled": False,
                "latest_sequence": 0,
                "oldest_available_sequence": 0,
                "dropped_receipts": 0,
                "receipts": [],
                "worker_states": [],
                "scope": "s4_pre_verification",
                "reason": (
                    f"worker observer RPC unavailable: {type(error).__name__}: {error}"
                ),
            }
        states = [state for state in worker_states if isinstance(state, dict)]
        receipts = [
            receipt
            for state in states
            for receipt in state.get("receipts", [])
            if isinstance(receipt, dict)
        ]
        return {
            "schema": "qbi.native-suffix-worker-state.v1",
            "status": (
                "READY"
                if states and all(state.get("status") == "READY" for state in states)
                else "NOT_READY"
            ),
            "observer_enabled": bool(states)
            and all(state.get("observer_enabled") is True for state in states),
            "latest_sequence": max(
                (int(state.get("latest_sequence", 0)) for state in states), default=0
            ),
            "oldest_available_sequence": min(
                (
                    int(state.get("oldest_available_sequence", 0))
                    for state in states
                    if int(state.get("oldest_available_sequence", 0)) > 0
                ),
                default=0,
            ),
            "dropped_receipts": sum(
                int(state.get("dropped_receipts", 0)) for state in states
            ),
            "receipts": receipts,
            "worker_states": states,
            "scope": "s4_pre_verification",
            "engine_boot_id": self._runtime_control_boot_id,
            "config_epoch": self.scheduler.get_runtime_scheduler_limits().get(
                "config_epoch", 0
            ),
        }

    def reset_native_suffix_cache(self) -> dict[str, Any]:
        """Reset all worker-local Suffix trees between controlled TRAIN arms."""
        state = self.get_runtime_transition_state()
        if state["unfinished_requests"] != 0 or state["transition_staged"]:
            raise RuntimeError(
                "cannot reset Suffix history while the engine has unfinished requests"
            )
        worker_results = self.model_executor.collective_rpc(
            "reset_native_suffix_cache"
        )
        return {
            "schema": "qbi.native-suffix-cache-reset.v1",
            "status": "RESET",
            "engine_boot_id": self._runtime_control_boot_id,
            "config_epoch": state["config_epoch"],
            "unfinished_requests": state["unfinished_requests"],
            "worker_results": worker_results,
        }

    def prepare_runtime_transition(
        self, mechanism_name: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        if mechanism_name == "scheduler_batching":
            expected = {"max_num_scheduled_tokens", "max_num_running_reqs"}
            if set(config) != expected:
                raise ValueError(
                    "scheduler transition requires exactly the registered fields"
                )
            return self.scheduler.prepare_runtime_scheduler_limits(
                int(config["max_num_scheduled_tokens"]),
                int(config["max_num_running_reqs"]),
            )
        if mechanism_name == "prefill_admission_guard":
            if set(config) != {"scheduler_reserve_full_isl"} or not isinstance(
                config["scheduler_reserve_full_isl"], bool
            ):
                raise ValueError(
                    "prefill admission transition requires one boolean registered field"
                )
            return self.scheduler.prepare_runtime_prefill_admission_guard(
                config["scheduler_reserve_full_isl"]
            )
        if mechanism_name == "dynamic_speculative_depth":
            if set(config) != {"num_speculative_tokens"}:
                raise ValueError(
                    "speculative-depth transition requires one registered field"
                )
            return self.scheduler.prepare_runtime_speculative_depth(
                int(config["num_speculative_tokens"])
            )
        raise ValueError(f"runtime transition is not implemented for {mechanism_name}")

    def stage_runtime_transition(
        self, mechanism_name: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        if mechanism_name == "scheduler_batching":
            expected = {"max_num_scheduled_tokens", "max_num_running_reqs"}
            if set(config) != expected:
                raise ValueError(
                    "scheduler transition requires exactly the registered fields"
                )
            return self.scheduler.stage_runtime_scheduler_limits(
                int(config["max_num_scheduled_tokens"]),
                int(config["max_num_running_reqs"]),
            )
        if mechanism_name == "prefill_admission_guard":
            if set(config) != {"scheduler_reserve_full_isl"} or not isinstance(
                config["scheduler_reserve_full_isl"], bool
            ):
                raise ValueError(
                    "prefill admission transition requires one boolean registered field"
                )
            return self.scheduler.stage_runtime_prefill_admission_guard(
                config["scheduler_reserve_full_isl"]
            )
        if mechanism_name == "dynamic_speculative_depth":
            if set(config) != {"num_speculative_tokens"}:
                raise ValueError(
                    "speculative-depth transition requires one registered field"
                )
            return self.scheduler.stage_runtime_speculative_depth(
                int(config["num_speculative_tokens"])
            )
        raise ValueError(
            f"staged runtime transition is not implemented for {mechanism_name}"
        )

    def commit_staged_runtime_transition(
        self, mechanism_name: str, next_epoch: int
    ) -> dict[str, Any]:
        if mechanism_name == "scheduler_batching":
            return self.scheduler.commit_staged_runtime_scheduler_limits(next_epoch)
        if mechanism_name == "prefill_admission_guard":
            return self.scheduler.commit_staged_runtime_prefill_admission_guard(
                next_epoch
            )
        if mechanism_name == "dynamic_speculative_depth":
            return self.scheduler.commit_staged_runtime_speculative_depth(next_epoch)
        raise ValueError(
            f"staged runtime transition is not implemented for {mechanism_name}"
        )

    def abort_staged_runtime_transition(self, mechanism_name: str) -> dict[str, Any]:
        if mechanism_name == "scheduler_batching":
            return self.scheduler.abort_staged_runtime_scheduler_limits()
        if mechanism_name == "prefill_admission_guard":
            return self.scheduler.abort_staged_runtime_prefill_admission_guard()
        if mechanism_name == "dynamic_speculative_depth":
            return self.scheduler.abort_staged_runtime_speculative_depth()
        raise ValueError(
            f"staged runtime transition is not implemented for {mechanism_name}"
        )

    def commit_runtime_transition(
        self, mechanism_name: str, prepared: dict[str, Any], next_epoch: int
    ) -> dict[str, Any]:
        if mechanism_name == "scheduler_batching":
            return self.scheduler.commit_runtime_scheduler_limits(prepared, next_epoch)
        if mechanism_name == "prefill_admission_guard":
            return self.scheduler.commit_runtime_prefill_admission_guard(
                prepared, next_epoch
            )
        if mechanism_name == "dynamic_speculative_depth":
            return self.scheduler.commit_runtime_speculative_depth(prepared, next_epoch)
        raise ValueError(f"runtime transition is not implemented for {mechanism_name}")

    def verify_runtime_transition(
        self, mechanism_name: str, config: dict[str, Any], epoch: int
    ) -> dict[str, Any]:
        state = self.get_runtime_transition_state()
        limits = state["scheduler_limits"]
        if mechanism_name == "scheduler_batching":
            if state["config_epoch"] != epoch or any(
                int(limits[field]) != int(value) for field, value in config.items()
            ):
                raise RuntimeError("scheduler transition did not become effective")
            receipt_field = "scheduler_profile_receipt"
            profile = "scheduler_batching"
        elif mechanism_name == "prefill_admission_guard":
            if set(config) != {"scheduler_reserve_full_isl"} or not isinstance(
                config["scheduler_reserve_full_isl"], bool
            ):
                raise ValueError("prefill admission verification config is invalid")
            if (
                state["config_epoch"] != epoch
                or bool(limits["scheduler_reserve_full_isl"])
                is not config["scheduler_reserve_full_isl"]
            ):
                raise RuntimeError(
                    "prefill admission transition did not become effective"
                )
            receipt_field = "prefill_admission_guard_receipt"
            profile = "prefill_admission_guard"
        elif mechanism_name == "dynamic_speculative_depth":
            if set(config) != {"num_speculative_tokens"}:
                raise ValueError("speculative-depth verification config is invalid")
            resident_method = limits.get("speculative_method")
            if (
                state["config_epoch"] != epoch
                or int(limits["num_speculative_tokens"])
                != int(config["num_speculative_tokens"])
                or resident_method not in {"suffix", "ngram", "ngram_gpu"}
                or (
                    resident_method == "suffix"
                    and int(limits["speculative_method_is_suffix"]) != 1
                )
            ):
                raise RuntimeError(
                    "speculative-depth transition did not become effective"
                )
            receipt_field = "dynamic_speculative_depth_receipt"
            profile = "dynamic_speculative_depth"
        else:
            raise ValueError(
                f"runtime transition is not implemented for {mechanism_name}"
            )
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "effective_runtime_profile": profile,
            "engine_config_hash": config_hash,
            receipt_field: {
                "boot_id": self._runtime_control_boot_id,
                "config_epoch": epoch,
                "limits": limits,
                **(
                    {"resident_speculative_method": resident_method}
                    if mechanism_name == "dynamic_speculative_depth"
                    else {}
                ),
            },
        }

    def rollback_runtime_transition(
        self,
        mechanism_name: str,
        prepared: dict[str, Any],
        previous_epoch: int,
    ) -> dict[str, Any]:
        if mechanism_name == "scheduler_batching":
            proof = self.scheduler.rollback_runtime_scheduler_limits(prepared)
        elif mechanism_name == "prefill_admission_guard":
            proof = self.scheduler.rollback_runtime_prefill_admission_guard(prepared)
        elif mechanism_name == "dynamic_speculative_depth":
            proof = self.scheduler.rollback_runtime_speculative_depth(prepared)
        else:
            raise ValueError(
                f"runtime transition is not implemented for {mechanism_name}"
            )
        return {
            **proof,
            "last_known_good_epoch": previous_epoch,
            "rollback_epoch_receipt": True,
            "boot_id": self._runtime_control_boot_id,
        }

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    @contextmanager
    def log_iteration_details(self, scheduler_output: SchedulerOutput):
        if not self.vllm_config.observability_config.enable_logging_iteration_details:
            yield
            return
        self._iteration_index = getattr(self, "_iteration_index", 0)
        iteration_details = compute_iteration_details(scheduler_output)
        before = time.monotonic()
        yield
        logger.info(
            "".join(
                [
                    "Iteration(",
                    str(self._iteration_index),
                    "): ",
                    str(iteration_details.num_ctx_requests),
                    " context requests, ",
                    str(iteration_details.num_ctx_tokens),
                    " context tokens, ",
                    str(iteration_details.num_generation_requests),
                    " generation requests, ",
                    str(iteration_details.num_generation_tokens),
                    " generation tokens, iteration elapsed time: ",
                    format((time.monotonic() - before) * 1000, ".2f"),
                    " ms",
                ]
            )
        )
        self._iteration_index += 1

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        _prepare_native_worker_plan(self, scheduler_output)
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        # Measure the device work of exactly this scheduler output.  The window
        # covers waiting for the execution future and, when the executor did not
        # sample, the sampling call, so the recorded value is step device time
        # rather than scheduler CPU time.
        qbi_exec_start_ns = time.perf_counter_ns()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
            _native_model_result_context(self.scheduler, scheduler_output) as _anise_model_result,
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)
            if _anise_model_result is not None:
                _anise_model_result.received(model_output)
        qbi_exec_elapsed_ns = time.perf_counter_ns() - qbi_exec_start_ns
        self.scheduler.record_qbi_model_execution_sample(
            scheduler_output,
            elapsed_ns=qbi_exec_elapsed_ns,
            observed_at_ns=time.perf_counter_ns(),
        )

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        self._record_qbi_model_runner_telemetry(scheduler_output, model_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            self._finalize_qbi_dynamic_mtp_draft_receipt(draft_token_ids)
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """

        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            _prepare_native_worker_plan(self, scheduler_output)
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
            _native_model_result_context(self.scheduler, scheduler_output) as _anise_model_result,
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")
            if _anise_model_result is not None:
                _anise_model_result.received(model_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        self._record_qbi_model_runner_telemetry(scheduler_output, model_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # If we are doing speculative decoding with structured output,
            # we need to get the draft token ids from the prior step before
            # we can compute the grammar bitmask for the deferred request.
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
                # Update the draft token ids in the scheduler output to
                # filter out the invalid spec tokens, which will be padded
                # with -1 and skipped by the grammar bitmask computation.
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.model_executor.profile(is_start, profile_prefix)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        Clears both the scheduler's cache manager and the GPU model runner's cache.
        """
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the encoder cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # Reset the scheduler's encoder cache manager (logical state)
        self.scheduler.reset_encoder_cache()
        # Reset the GPU model runner's encoder cache (physical storage)
        self.model_executor.reset_encoder_cache()

    def _reset_caches(self, reset_running_requests=True) -> None:
        self.reset_prefix_cache(reset_running_requests=reset_running_requests)
        self.reset_mm_cache()
        self.reset_encoder_cache()

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")
        if mode == "wait":
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if clear_cache:
            self._reset_caches()

        return None

    def resume_scheduler(self) -> None:
        """Resume the scheduler and flush any requests queued while paused."""
        self.scheduler.set_pause_state(PauseState.UNPAUSED)

    def is_scheduler_paused(self) -> bool:
        """Return whether the scheduler is in any pause state."""
        return self.scheduler.pause_state != PauseState.UNPAUSED

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
        """Put the engine to sleep at the specified level.

        Args:
            level: Sleep level.
                - Level 0: Pause scheduling only. Requests are still accepted
                           but not processed. No GPU memory changes.
                - Level 1: Offload model weights to CPU, discard KV cache.
                - Level 2: Discard all GPU memory.
            mode: Pause mode - how to deal with any existing requests, see
                documentation of pause_scheduler method.
        """

        # Pause scheduler before sleeping.
        clear_prefix_cache = level >= 1
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        if level < 1:
            return pause_future

        # Level 1+: Delegate to executor for GPU memory management
        model_executor = self.model_executor
        if pause_future is None:
            model_executor.sleep(level)
            return None

        future = Future[Any]()

        def pause_complete(f: Future):
            try:
                f.result()  # propagate any exception
                future.set_result(model_executor.sleep(level))
            except Exception as e:
                future.set_exception(e)

        logger.info("Waiting for in-flight requests to complete before sleeping...")
        pause_future.add_done_callback(pause_complete)
        return future

    def wake_up(self, tags: list[str] | None = None):
        """Wake up the engine from sleep.

        Args:
            tags: Tags to wake up. Use ["scheduling"] for level 0 wake up.
        """
        if tags is not None and "scheduling" in tags:
            # Remove "scheduling" from tags if there are other tags to process.
            tags = [t for t in tags if t != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # Resume scheduling (applies to all levels)
        self.resume_scheduler()

    def is_sleeping(self) -> bool:
        """Check if engine is sleeping at any level."""
        return self.is_scheduler_paused() or self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave

    def _eep_scale_up_before_kv_init(self):
        raise NotImplementedError

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,
        vllm_config: VllmConfig | None = None,
    ):
        raise NotImplementedError


class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    addresses: EngineZmqAddresses

    @instrument(span_name="EngineCoreProc init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
        *,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING

        # Receiver for tensor IPC
        self.tensor_ipc_receiver: TensorIpcReceiver | None = None
        if tensor_queue is not None:
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)
            logger.info("Using tensor IPC queue for multimodal tensor sharing")

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            self.publish_dp_lb_stats = internal_dp_balancing

            self.addresses = addresses
            self.process_input_queue_block = True
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                self._eep_send_engine_core_notification(
                    EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,
                    vllm_config=vllm_config,
                )
            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            # We only need to handshake with one party.
            with handshake as addresses:
                yield addresses
        else:
            # We need to handshake with rank 0 front-end and our colocated frontend.
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                # 1. Obtain DP Coordinator zmq address and DP process group address
                #    (addresses).
                # 2. Add front-end input/output addresses from colocated front-end
                #    (client_addresses).
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
            }
            # Include config hash for DP configuration validation
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            decorate_logs()
            if parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(process_title)

            if data_parallel and vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum, frame):
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

        raise SystemExit

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        # If no model execution happened but there are waiting requests
        # (e.g., WAITING_FOR_REMOTE_KVS), yield the GIL briefly to allow
        # background threads (like NIXL handshake) to make progress.
        # Without this, the tight polling loop can starve background threads.
        if not model_executed and self.scheduler.has_unfinished_requests():
            time.sleep(0.001)

        return model_executed

    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.vllm_config.shutdown_timeout

            logger.info("Shutdown initiated (timeout=%d)", shutdown_timeout)

            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info("Aborting %d requests", num_requests)
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
            else:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "Draining %d in-flight requests (timeout=%ds)",
                        num_requests,
                        shutdown_timeout,
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        if not self.has_work():
            logger.info("Shutdown complete")
            return False

        return True

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            output = UtilityOutput(call_id)
            # Lazily look-up utility method so that failure will be handled/returned.
            get_result = lambda: (
                (method := getattr(self, method_name))
                and method(*self._convert_msgspec_args(method, args))
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    def _reject_add_in_shutdown(self, request: Request) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.info("Rejecting request %s (server shutting down)", request.request_id)
        self._send_abort_outputs_to_client([request.request_id], request.client_index)
        return True

    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.warning("Rejecting utility call %s (server shutting down)", method_name)
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        return True

    @staticmethod
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        try:
            result = get_result()
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            logger.exception("Invocation of %s method failed", name)
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding with optional tensor IPC receiver.
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            ready_response = EngineCoreReadyResponse(
                max_model_len=self.vllm_config.model_config.max_model_len,
                num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
                dp_stats_address=self.frontend_stats_publish_address,
            )
            ready_payload = msgspec.msgpack.encode(ready_response)
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(ready_payload)
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # NOTE(yongji): ignore READY message sent by DP coordinator
                    # that is used to notify newly started engines
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self._send_error_outputs_to_client([request.request_id], request.client_index)

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")

        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:
            if clear_cache:
                engine._reset_caches()
            future.set_result(None)

        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if not self.has_work():
            if clear_cache:
                self._reset_caches()
            return None

        future = Future[Any]()
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
        return future

    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason
    ) -> None:
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            for req_id in req_ids
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        self.output_queue.put_nowait((client_index, eco))

    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    def _send_error_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)

    def _send_abort_outputs(self, aborted_reqs: list[tuple[str, int]]) -> None:
        # TODO(nick) this will be moved inside the scheduler
        if aborted_reqs:
            # Map client_index to list of request_ids that belong to that client.
            by_client = defaultdict[int, set[str]](set)
            for req_id, client_index in aborted_reqs:
                by_client[client_index].add(req_id)
            for client_index, req_ids in by_client.items():
                self._send_abort_outputs_to_client(list(req_ids), client_index)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state: ElasticEPScalingState | None = None

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
            tensor_queue=tensor_queue,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        parallel_config = vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        local_dp_rank = parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
        self.dp_group, self.dp_store = dp_group, dp_store

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        super().add_request(request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif (
                not self.engines_running
                and self.scheduler.pause_state == PauseState.UNPAUSED
            ):
                self.engines_running = True
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    def resume_scheduler(self):
        super().resume_scheduler()
        if (
            self.has_coordinator
            and not self.engines_running
            and self.scheduler.has_unfinished_requests()
        ):
            # Wake up other DP engines.
            self.output_queue.put_nowait(
                (-1, EngineCoreOutputs(start_wave=self.current_wave))
            )

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    if self.eep_scaling_state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

        raise SystemExit

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from copy import deepcopy

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        new_parallel_config = deepcopy(self.vllm_config.parallel_config)
        old_dp_size = new_parallel_config.data_parallel_size
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            new_parallel_config.data_parallel_rank = (
                reconfig_request.new_data_parallel_rank
            )
        new_parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        new_parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        new_parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        new_parallel_config._coord_store_port = reconfig_request.coord_store_port

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
        is_shutdown = (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=new_parallel_config,
            worker_type="removing" if is_shutdown else "existing",
            scale_type="scale_down" if is_scale_down else "scale_up",
            reconfig_request=reconfig_request,
        )
        self.process_input_queue_block = False
        logger.info(
            "[Elastic EP] Received reconfiguration request and starting scaling up/down"
        )

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,
        vllm_config: VllmConfig | None = None,
    ):
        """
        Send notifications to EngineCoreClient, which can then forward
        the notifications to other engine core processes. It is used for:
        1) In scale up: new core engines to notify existing core engines
           that they are ready;
        2) In scale down: removing core engines to notify EngineCoreClient
           so EngineCoreClient can release their ray placement groups;
        3) Both scale up/down: to notify EngineCoreClient that existing
           core engines have already switched to the new parallel setup.
        """
        if vllm_config is None:
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        else:
            dp_rank = vllm_config.parallel_config.data_parallel_rank
        notification_data = (notification_type.value, dp_rank)
        outputs = EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID,
                result=UtilityResult(notification_data),
            )
        )
        outputs.engine_index = self.engine_index

        if hasattr(self, "output_thread") and self.output_thread.is_alive():
            self.output_queue.put_nowait((0, outputs))
        else:
            encoder = MsgpackEncoder()
            with (
                zmq.Context() as ctx,
                make_zmq_socket(
                    ctx, self.addresses.outputs[0], zmq.PUSH, linger=4000
                ) as socket,
            ):
                socket.send_multipart(encoder.encode(outputs))

    def eep_handle_engine_core_notification(
        self, notification_type: str | EEPNotificationType
    ):
        """
        Handle notification received from EngineCoreClient
        (forwarded from new core engines).
        """
        assert self.eep_scaling_state is not None
        if isinstance(notification_type, str):
            notification_type = EEPNotificationType(notification_type)
        self.eep_scaling_state.handle_notification(notification_type)

    def _eep_scale_up_before_kv_init(self):
        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=self.vllm_config.parallel_config,
            worker_type="new",
            scale_type="scale_up",
            reconfig_request=None,
        )
        self.eep_scaling_state.run_pre_kv_init_states()
        self.process_input_queue_block = False


class EngineCoreActorMixin:
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # Initialize tracer for distributed tracing if configured.
        maybe_init_worker_tracer(
            instrumenting_module_name="vllm.engine_core",
            process_kind="engine_core",
            process_name=f"DPEngineCoreActor_DP{dp_rank}",
        )

        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_index = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()  # type: ignore[attr-defined]


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """Used for MoE model data parallel cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_rank = dp_rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """Used for non-MoE and/or non-DP cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size_local = 1
        vllm_config.parallel_config.data_parallel_rank = 0

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
