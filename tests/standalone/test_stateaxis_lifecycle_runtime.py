# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU runtime integration, without pytest/conftest, weights or workers.

Usage: .venv/bin/python tests/standalone/test_stateaxis_lifecycle_runtime.py \
    --model-config /absolute/local/model-config-directory -v

Scheduler, AsyncScheduler, Request, KV managers and EngineCore methods are real
imports. EngineCoreProc.__new__ skips worker/bootstrap construction; the fake
executor supplies CPU-only results to unchanged step/queue/utility methods.
This does not qualify device execution, connector transport, or Core bootstrap.
The test process must run in the main-owned CPU/no-device environment.
"""

import argparse
import json
import os
import sys
import unittest
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from queue import Queue

# A local config is required, and a missing dependency/config must fail rather
# than fetch anything or silently skip runtime integration.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

BLOCK_SIZE = 16
NUM_BLOCKS = 64


def model_output(output):
    ids = list(output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: index for index, rid in enumerate(ids)},
        sampled_token_ids=[[3] for _ in ids],
    )


class ObservedFuture(Future):
    def __init__(self, executor, value):
        super().__init__()
        self.executor = executor
        self.set_result(value)

    def result(self, timeout=None):
        self.executor.calls.append("result")
        callback = self.executor.before_result
        self.executor.before_result = None
        if callback is not None:
            callback()
        return super().result(timeout=timeout)


class CPUModelExecutor:
    """No model: deterministic token results and observable original waits."""

    supported_tasks = ("generate",)

    def __init__(self):
        self.calls = []
        self.scheduled = []
        self.pending_sampling = deque()
        self.before_result = None

    def execute_model(self, output, non_block=False):
        if non_block is not True:
            raise AssertionError("Core unexpectedly requested blocking execution")
        self.calls.append("execute")
        self.scheduled.append(output)
        value = model_output(output)
        if output.total_num_scheduled_tokens:
            self.pending_sampling.append(value)
        return ObservedFuture(self, value)

    def sample_tokens(self, grammar_output, non_block=False):
        if grammar_output is not None:
            raise AssertionError("These fixtures do not use structured output")
        self.calls.append("sample")
        value = self.pending_sampling.popleft()
        return ObservedFuture(self, value) if non_block else value


def make_core(scheduler):
    # Do not call __init__: it bootstraps real workers and KV tensor allocation.
    # No production methods are copied, AST-extracted, patched or overridden.
    core = EngineCoreProc.__new__(EngineCoreProc)
    core.vllm_config = scheduler.vllm_config
    core.scheduler = scheduler
    core.stateaxis_lifecycle = scheduler.stateaxis_lifecycle
    core.model_executor = CPUModelExecutor()
    core.structured_output_manager = scheduler.structured_output_manager
    core.aborts_queue = Queue()
    core.output_queue = Queue()
    core.batch_queue = deque()
    core.batch_queue_size = 2
    core.is_ec_consumer = True
    core.is_pooling_model = False
    core.async_scheduling = isinstance(scheduler, AsyncScheduler)
    core.check_for_draft_tokens = False
    core.shutdown_state = EngineShutdownState.RUNNING
    return core


class LifecycleRuntimeTests(unittest.TestCase):
    model_path = None

    @classmethod
    def setUpClass(cls):
        if cls.model_path is None:
            raise RuntimeError("Run this file with --model-config LOCAL_DIRECTORY")
        init_none_hash(sha256)

    def scheduler(self, observe=False, asynchronous=False, opaque=False):
        model = ModelConfig(
            model=str(self.model_path),
            trust_remote_code=False,
            dtype="float32",
            seed=42,
            max_model_len=128,
            enforce_eager=True,
            skip_tokenizer_init=True,
        )
        config = VllmConfig(
            model_config=model,
            device_config=DeviceConfig(device="cpu"),
            parallel_config=ParallelConfig(distributed_executor_backend="uni"),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4,
                max_num_batched_tokens=32,
                max_model_len=128,
                enable_chunked_prefill=True,
                async_scheduling=asynchronous,
                is_encoder_decoder=model.is_encoder_decoder,
                watermark=0.0,
            ),
            cache_config=CacheConfig(
                block_size=BLOCK_SIZE,
                cache_dtype="auto",
                enable_prefix_caching=False,
            ),
            additional_config=(
                {"stateaxis_lifecycle": {"mode": "observe", "max_events": 4096}}
                if observe
                else {}
            ),
        )
        if opaque:

            class OpaqueConfig:
                def compute_hash(self):
                    return "stateaxis-runtime-opaque-test"

            # This is the real documented SupportsHash alternative to dict.
            config.additional_config = OpaqueConfig()
        config.cache_config.num_gpu_blocks = NUM_BLOCKS
        kv_config = KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        )
        register_all_kvcache_specs(config)
        cls = AsyncScheduler if asynchronous else Scheduler
        result = cls(
            vllm_config=config,
            kv_cache_config=kv_config,
            structured_output_manager=StructuredOutputManager(config),
            block_size=BLOCK_SIZE,
            log_stats=False,
        )
        self.addCleanup(result.shutdown)
        self.addCleanup(result.structured_output_manager.clear_backend)
        return result

    def request(self, rid="r", max_tokens=2, prompt_tokens=8):
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, 2)
        return Request(
            request_id=rid,
            prompt_token_ids=[1] * prompt_tokens,
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )

    def assert_retired(self, scheduler):
        self.assertEqual(scheduler.requests, {})
        self.assertEqual(
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), NUM_BLOCKS - 1
        )
        observer = scheduler.stateaxis_lifecycle
        if observer is not None:
            snapshot = observer.snapshot()
            self.assertTrue(snapshot["trace_lossless"])
            for key in (
                "active_requests",
                "uncommitted_steps",
                "submitted_uncommitted_steps",
                "completed_uncommitted_steps",
                "deferred_block_groups",
            ):
                self.assertEqual(snapshot[key], 0, key)

    def ordinary_run(self, observe, asynchronous):
        scheduler = self.scheduler(observe=observe, asynchronous=asynchronous)
        core = make_core(scheduler)
        self.assertIsInstance(core, EngineCore)
        request = self.request(max_tokens=2)
        core.add_request(request)
        fingerprint = []
        for _ in range(8):
            if not scheduler.has_requests():
                break
            outputs, executed = core.step()
            output = core.model_executor.scheduled[-1]
            fingerprint.append(
                (
                    dict(output.num_scheduled_tokens),
                    executed,
                    [
                        (item.request_id, list(item.new_token_ids), item.finish_reason)
                        for client in outputs.values()
                        for item in client.outputs
                    ],
                )
            )
        self.assertFalse(scheduler.has_requests())
        self.assertEqual(list(request.output_token_ids), [3, 3])
        self.assert_retired(scheduler)
        return fingerprint

    def test_scheduler_off_on_same_real_schedule_and_outputs(self):
        self.assertEqual(
            self.ordinary_run(False, False), self.ordinary_run(True, False)
        )

    def test_async_scheduler_off_on_same_real_schedule_and_outputs(self):
        self.assertEqual(self.ordinary_run(False, True), self.ordinary_run(True, True))

    def test_opaque_additional_config_preserves_disabled_scheduler(self):
        scheduler = self.scheduler(opaque=True)
        self.assertIsNone(scheduler.stateaxis_lifecycle)
        core = make_core(scheduler)
        core.add_request(self.request(max_tokens=1))
        outputs, executed = core.step()
        self.assertTrue(executed)
        self.assertEqual(outputs[0].outputs[0].new_token_ids, [3])
        self.assert_retired(scheduler)

    def test_actual_async_queue_submits_twice_before_first_wait(self):
        scheduler = self.scheduler(observe=True, asynchronous=True)
        core = make_core(scheduler)
        request = self.request(max_tokens=2)
        core.add_request(request)
        self.assertEqual(core.step_with_batch_queue(), (None, True))
        self.assertNotIn("result", core.model_executor.calls)
        self.assertEqual(
            scheduler.stateaxis_lifecycle.snapshot()["submitted_uncommitted_steps"], 1
        )
        core.step_with_batch_queue()
        first_wait = core.model_executor.calls.index("result")
        self.assertEqual(core.model_executor.calls[:first_wait].count("execute"), 2)
        for _ in range(8):
            if not scheduler.has_requests() and not core.batch_queue:
                break
            core.step_with_batch_queue()
        self.assertFalse(core.batch_queue)
        self.assertFalse(scheduler.has_requests())
        self.assertEqual(list(request.output_token_ids), [3, 3])
        self.assert_retired(scheduler)

    def test_abort_during_result_precedes_real_scheduler_update(self):
        for observe in (False, True):
            with self.subTest(observe=observe):
                scheduler = self.scheduler(observe=observe)
                core = make_core(scheduler)
                request = self.request(max_tokens=4)
                core.add_request(request)
                core.model_executor.before_result = (
                    lambda core=core: core.aborts_queue.put(["r"])
                )
                outputs, executed = core.step()
                self.assertTrue(executed)
                self.assertFalse(any(client.outputs for client in outputs.values()))
                self.assertEqual(list(request.output_token_ids), [])
                self.assertEqual(request.status, RequestStatus.FINISHED_ABORTED)
                # The second input-queue abort must remain idempotent.
                core.abort_requests(["r"])
                self.assert_retired(scheduler)

    def test_real_deferred_fence_preserves_external_shared_reference(self):
        for observe in (False, True):
            with self.subTest(observe=observe):
                scheduler = self.scheduler(observe=observe, asynchronous=True)
                # Exercise the real fence mechanism without pretending to test a
                # remote KV consumer's enablement or transport implementation.
                scheduler.defer_block_free = True
                request = self.request(max_tokens=4)
                scheduler.add_request(request)
                first = scheduler.schedule()
                second = scheduler.schedule()
                self.assertEqual(scheduler.sched_step_seq, 2)
                observer = scheduler.stateaxis_lifecycle
                if observer is not None:
                    observer.execution_submitted(first)
                    observer.execution_submitted(second)
                pool = scheduler.kv_cache_manager.block_pool
                blocks = scheduler.kv_cache_manager.get_blocks("r").blocks[0]
                shared = blocks[0]
                pool.touch([shared])
                self.assertEqual(shared.ref_cnt, 2)
                scheduler.finish_requests(["r"], RequestStatus.FINISHED_ABORTED)
                self.assertNotIn("r", scheduler.requests)
                self.assertEqual(len(scheduler.deferred_frees), 1)
                self.assertEqual(shared.ref_cnt, 2)
                scheduler.update_from_output(first, model_output(first))
                self.assertEqual(scheduler.processed_step_seq, 1)
                self.assertEqual(shared.ref_cnt, 2)
                scheduler.update_from_output(second, model_output(second))
                self.assertEqual(scheduler.processed_step_seq, 2)
                self.assertFalse(scheduler.deferred_frees)
                self.assertEqual(shared.ref_cnt, 1)
                self.assertEqual(pool.get_num_free_blocks(), NUM_BLOCKS - 2)
                pool.free_blocks([shared])
                self.assert_retired(scheduler)

    def test_actual_utility_dispatch_returns_detached_snapshot(self):
        scheduler = self.scheduler(observe=True)
        core = make_core(scheduler)
        core.add_request(self.request(max_tokens=1))
        core.step()
        core._handle_client_request(
            EngineCoreRequestType.UTILITY,
            (7, 41, "get_stateaxis_lifecycle_snapshot", ()),
        )
        client, output = core.output_queue.get_nowait()
        self.assertEqual(client, 7)
        utility = output.utility_output
        self.assertEqual(utility.call_id, 41)
        self.assertIsNone(utility.failure_message)
        # Exercise the real utility wrapper and JSON-safe detached evidence;
        # this is not a socket or frontend transport test.
        snapshot = utility.result.result
        original = json.dumps(core.get_stateaxis_lifecycle_snapshot(), sort_keys=True)
        snapshot["events"].clear()
        self.assertEqual(
            json.dumps(core.get_stateaxis_lifecycle_snapshot(), sort_keys=True),
            original,
        )
        self.assert_retired(scheduler)

    def test_connector_delays_detachment_until_send_ack(self):
        class DelayedConnector:
            def request_finished(self, request, block_ids):
                return True, None

            def update_connector_output(self, output):
                pass

            def shutdown(self):
                pass

        for observe in (False, True):
            with self.subTest(observe=observe):
                scheduler = self.scheduler(observe=observe)
                core = make_core(scheduler)
                request = self.request(max_tokens=4)
                core.add_request(request)
                core.step()
                pool = scheduler.kv_cache_manager.block_pool
                free_before = pool.get_num_free_blocks()
                # Transport is a fixture; the actual scheduler finish/ACK path
                # and KV allocator are exercised without network or workers.
                scheduler.connector = DelayedConnector()
                scheduler.finish_requests(["r"], RequestStatus.FINISHED_ABORTED)
                self.assertIn("r", scheduler.requests)
                self.assertEqual(pool.get_num_free_blocks(), free_before)
                if observe:
                    self.assertEqual(
                        scheduler.stateaxis_lifecycle.snapshot()["active_requests"], 1
                    )
                scheduler._update_from_kv_xfer_finished(
                    KVConnectorOutput(finished_sending={"r"})
                )
                self.assert_retired(scheduler)

    def test_failed_original_future_keeps_unresolved_observation(self):
        scheduler = self.scheduler(observe=True)
        core = make_core(scheduler)
        core.add_request(self.request(max_tokens=4))

        def fail():
            raise RuntimeError("injected executor result failure")

        core.model_executor.before_result = fail
        with self.assertRaisesRegex(RuntimeError, "injected executor result failure"):
            core.step()
        snapshot = core.get_stateaxis_lifecycle_snapshot()
        self.assertEqual(snapshot["submitted_uncommitted_steps"], 1)
        self.assertEqual(snapshot["completed_uncommitted_steps"], 0)
        self.assertFalse(snapshot["observed_lifetimes_closed"])
        self.assertIn("r", scheduler.requests)
        self.assertLess(
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), NUM_BLOCKS - 1
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    args, unittest_args = parser.parse_known_args()
    if not args.model_config.is_absolute():
        parser.error("--model-config must be an absolute local directory")
    model_path = args.model_config.resolve(strict=True)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        parser.error("--model-config must contain a local config.json")
    LifecycleRuntimeTests.model_path = model_path
    unittest.main(argv=[sys.argv[0], *unittest_args])
