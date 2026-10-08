# StateAxis lifecycle observation

This experimental integration retains vLLM's scheduler, async batch queue,
model runner and physical KV block pool. It adds no execution RPC, future
callback, device synchronization, or second allocator. It is disabled by default.

Enable the strict diagnostic mode through `VllmConfig.additional_config`:

```json
{"stateaxis_lifecycle": {"mode": "observe", "max_events": 4096}}
```

Absent configuration, `mode: off`, or an opaque `SupportsHash` additional
configuration leaves the observer disabled. Other modes and malformed observer
configuration are rejected. Internal observation identity/order violations raise
errors in this diagnostic mode; it is not a production fault-isolated telemetry
service. No enforcement mode is implemented.

`EngineCore.get_stateaxis_lifecycle_snapshot()` returns a JSON-compatible
snapshot through the existing utility-call mechanism, or `None` when disabled.
Use it outside the timed interval. Events stay in a bounded in-memory ring;
overflow increments `dropped_events` and makes `trace_lossless` false. Full
acceptance cannot use a lossy trace. There is no automatic file writer.

The integration records:

- New-request generation, preserving generation across streaming input updates;
  preemption advances a separate execution epoch.
- Scheduled output identity, successful nonblocking execution submission,
  original output completion consumption and successful scheduler update.
- Actual KV allocation results, block reference changes and reusable-pool queue
  updates, including shared and null blocks.
- Logical request finish, request detachment, deferred blocks and fence-based
  reference return.

`epoch_current` compares the complete request-generation/execution-epoch token.
It does not establish token validity or prove that the scheduler accepted or
discarded a particular output. Completed scheduler updates do not establish
client delivery. The snapshot retains unresolved scheduled/submitted work and
deferred groups after failures; these counts do not prove device occupancy.
`trace_lossless` only means no ring events were dropped.
`observed_lifetimes_closed` only describes tracked requests, steps and deferred
groups, not physical HBM deallocation or global resource closure.

Object references are retained for identity checking only while tracked work is
unresolved. In particular, exceptions may retain these references until engine
teardown. Event payloads at the integration points are scalar snapshots.

This is the first integration slice, not state-governance enforcement. It does
not implement cross-type budgets, dependency invalidation, per-request attribution
for every SWA/Mamba local block transition, or a Rust contract bridge. It has no
serving-performance qualification. Those require separate full-runtime tests,
real-device validation and matched overhead measurements.

The standalone test file executes the observer and selected production method
ASTs against CPU fixtures without importing Torch or initializing an accelerator:

```bash
.venv/bin/python -B tests/standalone/test_stateaxis_lifecycle.py -v
```

These tests do not replace imported Scheduler/EngineCore/Ascend integration tests.

`tests/standalone/test_stateaxis_lifecycle_runtime.py --model-config ABSOLUTE_DIR`
adds real Scheduler/AsyncScheduler construction, KV management, and EngineCore
step, async queue and utility dispatch. It uses a local OPT metadata fixture,
CPU eager configuration, a deterministic executor and a connector transport
fixture. EngineCore's worker/bootstrap constructor is explicitly bypassed.
This CPU diagnostic configuration is not a serving/performance candidate.
The suite covers OFF/ON equivalence, cancellation, deferred shared references,
send-completion ACK, unresolved failed futures and detached utility snapshots.
Ascend graphs, mixed KV families, workers and transport require separate gates.

The observer evidence API is `GET /v1/stateaxis/lifecycle`. It is registered only
when `additional_config.stateaxis_lifecycle.mode` is `observe` and uses the
existing `/v1` authentication middleware. It accepts no RPC method or engine ID.
Read it outside measured HTTP workloads; it copies the bounded event buffers and
does not pause or drain the scheduler, reset evidence, or authorize state actions.

`AsyncMPClient.get_stateaxis_lifecycle_snapshots_async()` reads each managed
EngineCore using the existing utility transport. The general DP utility returns
only the first rank result, so this method preserves every result with its engine
identity. A changed membership, including elastic shrink/regrow with the same
rank IDs, rejects the read. The envelope explicitly states client-managed scope
and `cross_rank_atomic=false`; it is not a distributed consistent snapshot or
proof about engines owned by other API clients.

`tests/standalone/test_stateaxis_lifecycle_api.py` exercises real API, client,
msgpack request encoding, EngineCore utility dispatch and result delivery with
ASGI and CPU queues replacing the wire. It covers authentication, OFF route
absence, all-rank results, failures and membership changes. It does not start
model workers or qualify real network/IPC transport or serving performance.
