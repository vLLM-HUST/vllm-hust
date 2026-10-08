# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free tests, runnable with unittest without importing vLLM/Torch.

Lifecycle logic is loaded unchanged. Focused retirement tests execute actual
scheduler/manager method ASTs with a small reference-count fixture. These are
not full Scheduler, EngineCore, worker, or hardware acceptance tests.
"""

import ast
import importlib.util
import json
import sys
import unittest
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "vllm/v1/core"
SPEC = importlib.util.spec_from_file_location(
    "stateaxis_lifecycle_under_test", CORE / "stateaxis_lifecycle.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
LifecycleObserver = MODULE.LifecycleObserver


def load_methods(path, class_name, names, namespace=None):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(methods) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    scope = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return {name: scope[name] for name in names}


class Block:
    def __init__(self, block_id, references=1):
        self.block_id = block_id
        self.ref_cnt = references


class Pool:
    def __init__(self):
        self.returned = []

    def free_blocks(self, blocks):
        for block in blocks:
            block.ref_cnt -= 1
            self.returned.append(block.block_id)


class RetirementFixture:
    def __init__(self, observer, blocks):
        self.stateaxis_lifecycle = observer
        self.defer_block_free = True
        self.sched_step_seq = 2
        self.processed_step_seq = 0
        self.deferred_frees = deque()
        self.blocks = blocks
        self.pool = Pool()
        self.kv_cache_manager = SimpleNamespace(
            pop_blocks_for_free=lambda request: self.blocks,
            block_pool=self.pool,
        )


for name, method in load_methods(
    CORE / "sched/scheduler.py",
    "Scheduler",
    {"_free_request_blocks", "_drain_deferred_frees"},
).items():
    setattr(RetirementFixture, name, method)


class EngineFixture:
    def __init__(self, structured=False):
        self.stateaxis_lifecycle = LifecycleObserver()
        self.request = SimpleNamespace(request_id="r")
        self.stateaxis_lifecycle.admitted(self.request)
        self.outputs = deque(
            [
                SimpleNamespace(
                    num_scheduled_tokens={"r": 1},
                    total_num_scheduled_tokens=1,
                    pending_structured_output_tokens=False,
                ),
                SimpleNamespace(
                    num_scheduled_tokens={"r": 1},
                    total_num_scheduled_tokens=1,
                    pending_structured_output_tokens=structured,
                ),
            ]
        )
        self.calls = []
        self.wait_observations = []
        self.batch_queue = deque()
        self.batch_queue_size = 2
        self.is_ec_consumer = True
        self.is_pooling_model = False
        self.check_for_draft_tokens = False
        self.model_executor = SimpleNamespace(
            execute_model=self.execute,
            sample_tokens=self.sample,
        )
        self.scheduler = SimpleNamespace(
            has_requests=lambda: bool(self.outputs),
            schedule=self.schedule,
            get_grammar_bitmask=lambda output: None,
            update_from_output=self.update,
        )

    def _should_throttle_prefills(self):
        return False

    def log_error_detail(self, output):
        return nullcontext()

    def log_iteration_details(self, output):
        return nullcontext()

    def _process_aborts_queue(self):
        self.calls.append("aborts")

    def schedule(self, throttle):
        output = self.outputs.popleft()
        self.stateaxis_lifecycle.scheduled(output, {"r": self.request})
        return output

    def execute(self, output, non_block):
        assert non_block
        self.calls.append("execute")
        return SimpleNamespace(result=self.result)

    def sample(self, grammar, non_block=True):
        self.calls.append("sample")
        return SimpleNamespace(result=self.result)

    def result(self):
        self.calls.append("wait")
        self.wait_observations.append(
            (
                len(self.batch_queue),
                self.stateaxis_lifecycle.snapshot()["submitted_uncommitted_steps"],
            )
        )
        return object()

    def update(self, output, result):
        self.calls.append("update")
        self.stateaxis_lifecycle.execution_completed(output)
        self.stateaxis_lifecycle.committed(output)
        return {}


for name, method in load_methods(
    ROOT / "vllm/v1/engine/core.py",
    "EngineCore",
    {"step", "step_with_batch_queue"},
    {"cast": cast, "Future": Future, "ModelRunnerOutput": Any},
).items():
    setattr(EngineFixture, name, method)


class LifecycleTests(unittest.TestCase):
    def request(self, name="r"):
        return SimpleNamespace(request_id=name, last_sched_seq=2)

    def output(self, observer, request):
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})
        observer.scheduled(output, {request.request_id: request})
        observer.execution_submitted(output)
        return output

    def test_disabled_and_invalid_config(self):
        self.assertIsNone(LifecycleObserver.from_config(None))
        self.assertIsNone(LifecycleObserver.from_config({"mode": "off"}))
        for config in [
            {"mode": "enforce"},
            {"typo": 1},
            True,
            {"mode": "observe", "max_events": True},
            {"mode": "observe", "max_events": 0},
        ]:
            with self.assertRaises(ValueError):
                LifecycleObserver.from_config(config)

    def test_opaque_additional_config_remains_disabled(self):
        class OpaqueConfig:
            def compute_hash(self):
                return "opaque"

        self.assertIsNone(LifecycleObserver.from_additional_config(OpaqueConfig()))
        self.assertIsNone(LifecycleObserver.from_additional_config({"unrelated": 1}))
        self.assertIsInstance(
            LifecycleObserver.from_additional_config(
                {
                    "stateaxis_lifecycle": {"mode": "observe"},
                }
            ),
            LifecycleObserver,
        )

    def test_completion_requires_actual_submission(self):
        observer = LifecycleObserver()
        request = self.request()
        observer.admitted(request)
        output = SimpleNamespace(num_scheduled_tokens={"r": 1})
        observer.scheduled(output, {"r": request})
        with self.assertRaises(RuntimeError):
            observer.execution_completed(output)
        self.assertEqual(observer.snapshot()["uncommitted_steps"], 1)
        observer.execution_submitted(output)
        observer.execution_completed(output)
        observer.committed(output)
        self.assertEqual(observer.snapshot()["uncommitted_steps"], 0)

    def test_cancel_then_id_reuse_does_not_reassign_old_completion(self):
        observer = LifecycleObserver()
        old = self.request()
        observer.admitted(old)
        output = self.output(observer, old)
        old_token = observer.token(old)
        observer.retired(old)
        new = self.request()
        observer.admitted(new)
        self.assertNotEqual(old_token.generation, observer.token(new).generation)
        observer.execution_completed(output)
        event = observer.events[-1]
        self.assertEqual(event["tokens"], (old_token,))
        self.assertEqual(event["epoch_current"], (False,))
        observer.committed(output)
        self.assertEqual(observer.token(new).execution_epoch, 0)

    def test_preemption_keeps_generation_changes_episode(self):
        observer = LifecycleObserver()
        request = self.request()
        observer.admitted(request)
        old = observer.token(request)
        output = self.output(observer, request)
        observer.preempted(request)
        new = observer.token(request)
        self.assertEqual(old.generation, new.generation)
        self.assertEqual(new.execution_epoch, 1)
        observer.execution_completed(output)
        self.assertEqual(observer.events[-1]["epoch_current"], (False,))

    def test_future_failure_or_interrupted_update_leaves_pending(self):
        observer = LifecycleObserver()
        request = self.request()
        observer.admitted(request)
        output = self.output(observer, request)
        state = observer.snapshot()
        self.assertEqual(state["submitted_uncommitted_steps"], 1)
        self.assertEqual(state["completed_uncommitted_steps"], 0)
        observer.execution_completed(output)
        state = observer.snapshot()
        self.assertEqual(state["completed_uncommitted_steps"], 1)
        self.assertEqual(state["uncommitted_steps"], 1)

    def test_trace_overflow_is_explicit_and_json_safe(self):
        observer = LifecycleObserver(2)
        request = self.request()
        observer.admitted(request)
        self.output(observer, request)
        state = json.loads(json.dumps(observer.snapshot()))
        self.assertFalse(state["trace_lossless"])
        self.assertEqual(state["dropped_events"], 1)
        self.assertEqual(state["events"][0]["sequence"], 2)
        self.assertFalse(state["governance_enforced"])

    def test_actual_deferred_fence_preserves_shared_reference(self):
        observer = LifecycleObserver()
        request = self.request()
        observer.admitted(request)
        blocks = [Block(10), Block(11, 2)]
        fixture = RetirementFixture(observer, blocks)
        fixture._free_request_blocks(request)
        observer.retired(request)
        self.assertEqual(fixture.pool.returned, [])
        self.assertEqual(observer.snapshot()["deferred_block_groups"], 1)
        fixture.processed_step_seq = 1
        fixture._drain_deferred_frees()
        self.assertEqual(fixture.pool.returned, [])
        fixture.processed_step_seq = 2
        fixture._drain_deferred_frees()
        self.assertEqual(fixture.pool.returned, [11, 10])
        self.assertEqual(blocks[1].ref_cnt, 1)
        self.assertEqual(observer.events[-1]["blocks"], ((10, 0), (11, 1)))
        self.assertEqual(observer.snapshot()["deferred_block_groups"], 0)

    def test_actual_deferred_path_identical_when_disabled(self):
        blocks = [Block(10), Block(11)]
        fixture = RetirementFixture(None, blocks)
        fixture._free_request_blocks(self.request())
        fixture.processed_step_seq = 2
        fixture._drain_deferred_frees()
        self.assertEqual(fixture.pool.returned, [11, 10])
        self.assertEqual([b.ref_cnt for b in blocks], [0, 0])

    def test_actual_return_failure_keeps_observer_unresolved(self):
        observer = LifecycleObserver()
        request = self.request()
        observer.admitted(request)
        fixture = RetirementFixture(observer, [Block(10)])
        fixture._free_request_blocks(request)

        def fail(blocks):
            raise OSError("injected return failure")

        fixture.pool.free_blocks = fail
        fixture.processed_step_seq = 2
        with self.assertRaisesRegex(OSError, "injected"):
            fixture._drain_deferred_frees()
        self.assertEqual(observer.snapshot()["deferred_block_groups"], 1)
        self.assertNotIn("kv_refs_returned", [e["kind"] for e in observer.events])

    def test_actual_engine_fills_batch_queue_without_added_wait(self):
        engine = EngineFixture()
        self.assertEqual(engine.step_with_batch_queue(), (None, True))
        self.assertNotIn("wait", engine.calls)
        engine.step_with_batch_queue()
        self.assertEqual(engine.calls.count("wait"), 1)
        self.assertEqual(engine.wait_observations, [(1, 2)])
        self.assertLess(engine.calls.index("aborts"), engine.calls.index("update"))

    def test_actual_engine_counts_structured_submission_not_in_queue(self):
        engine = EngineFixture(structured=True)
        engine.step_with_batch_queue()
        engine.step_with_batch_queue()
        self.assertEqual(engine.wait_observations, [(0, 2)])
        self.assertEqual(len(engine.batch_queue), 1)
        engine.step_with_batch_queue()
        self.assertEqual(engine.calls.count("wait"), 2)
        self.assertEqual(engine.stateaxis_lifecycle.snapshot()["uncommitted_steps"], 0)

    def test_actual_engine_simple_step_does_not_add_wait(self):
        engine = EngineFixture()
        engine.step()
        self.assertEqual(engine.calls, ["execute", "wait", "aborts", "update"])

    def test_actual_pool_return_consumes_iterator_once_and_preserves_sharing(self):
        method = load_methods(CORE / "block_pool.py", "BlockPool", {"free_blocks"})[
            "free_blocks"
        ]
        observer = LifecycleObserver()
        shared = SimpleNamespace(block_id=1, ref_cnt=2, is_null=False, block_hash=None)
        free = SimpleNamespace(block_id=2, ref_cnt=1, is_null=False, block_hash=None)
        null = SimpleNamespace(block_id=0, ref_cnt=1, is_null=True, block_hash=None)
        queued = []
        fixture = SimpleNamespace(
            lifecycle_observer=observer,
            free_block_queue=SimpleNamespace(
                prepend_n=lambda blocks: queued.extend(blocks),
                append_n=lambda blocks: queued.extend(blocks),
            ),
        )
        method(fixture, iter([shared, free, null]))
        self.assertEqual([b.block_id for b in queued], [2])
        self.assertEqual(shared.ref_cnt, 1)
        self.assertEqual(observer.events[-1]["uncached"], (2,))

    def test_actual_streaming_add_does_not_create_new_generation(self):
        method = load_methods(
            CORE / "sched/scheduler.py",
            "Scheduler",
            {"add_request"},
            {
                "StreamingUpdate": SimpleNamespace(
                    from_request=lambda request: request
                ),
                "RequestStatus": SimpleNamespace(WAITING_FOR_STREAMING_REQ="stream"),
                "deque": deque,
            },
        )["add_request"]
        observer = LifecycleObserver()
        updates = []
        fixture = SimpleNamespace(
            requests={},
            stateaxis_lifecycle=observer,
            connector=None,
            log_stats=False,
            _enqueue_waiting_request=lambda request: None,
            _update_request_as_session=lambda old, update: updates.append(update),
        )
        old = SimpleNamespace(request_id="r", resumable=False, status="stream")
        method(fixture, old)
        token = observer.token(old)
        update = SimpleNamespace(request_id="r")
        method(fixture, update)
        self.assertEqual(observer.token(old), token)
        self.assertEqual(updates, [update])
        self.assertEqual(observer.snapshot()["active_requests"], 1)


if __name__ == "__main__":
    unittest.main()
