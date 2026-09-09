# KV materialization decision plugin

This package adds a vLLM connector plugin that chooses between loading a
complete CPU-resident prefix and recomputing that prefix. It is intentionally
outside the vLLM scheduler, cache hashing, and connector implementation: the
plugin subclasses `SimpleCPUOffloadConnector` through the public connector
module-path configuration.

## Modes

Set `kv_connector_module_path` to
`kv_materialization_plugin.connector.DynamicSimpleCPUOffloadConnector` and
select a mode in `kv_connector_extra_config`:

- `load`: force the native CPU KV load path (the default and cold-start
  fallback).
- `recompute`: force prefix recomputation.
- `dynamic`: compare the median recent end-to-end cost of loading and
  recomputing. A tie chooses recompute deterministically.

为保持兼容，历史估计器仍是默认值。要启用新的二元公式决策，将
`dynamic_predictor` 设为 `formula`，并提供由强制模式 telemetry 生成的成本
模型：

```json
{
  "materialization_mode": "dynamic",
  "dynamic_predictor": "formula",
  "cost_model_path": "/absolute/path/to/cost_model.json",
  "fallback_mode": "load",
  "min_copy_samples": 3,
  "min_recompute_samples": 3,
  "max_observation_age_ms": 5000,
  "sample_window_size": 32,
  "prefill_batch_size": 1,
  "kv_bytes_per_block": 0
}
```

公式模式比较以下两式：

`W_copy + t_load_fixed + D_KV / B_H2D_eff`

和

`W_compute + t_recompute_fixed + F_prefill / P_prefill_eff`。
`D_KV` 是包含全部 KV cache tensor 的实际对齐页面字节数；当
`kv_bytes_per_block` 为零时，connector 从运行时 `KVCacheConfig` 推导该值，
显式填写的正数会覆盖推导结果。`F_prefill` 是完整模型前向的估计值，包含
已有设备前缀带来的 Attention 工作，而不是只计算 KV projection。加载的成本
模型会检查校准范围，绝不会静默外推。

Dynamic mode is conservative about data quality: only measurements for the
exact same token/block-size buckets are reused. Missing, stale, or invalid
measurements use the configured fallback. Every decision records the selected
branch, reason, predictions, observation counts/ages, measured service/wait
components, and completion cost in the connector audit log.

The dynamic confidence gate requires both branches to have fresh samples and
requires each sample to carry an explicit phase wait measurement. A missing
old calibration file therefore falls back to `load` with reason
`insufficient_observation_confidence`; it is never treated as a zero wait.

M1 dynamic estimates are valid only when no other materialization request is
active. If a second request overlaps an unfinished load or recompute attempt,
the plugin records `unsupported_concurrent_context` and uses the configured
fallback (`load` in the M1 configuration). Forced modes remain available for
matched baselines. Maintaining copy-byte or recompute-token backlog across
requests is intentionally outside the M1 scope.

The audit field `queue_wait_ms` is measured separately from service time. Its
scope is the plugin's admission-to-first-service-start interval: for load it
is scheduler decision to worker copy submission, and for recompute it is
scheduler decision to the first compute step. This is an observed runtime
admission queue, not a claim about an opaque device driver's internal queue.
`extra_wait_ms` remains a residual between scheduler-observed total time and
worker service time, and may still include dispatch, process communication,
and metadata return. The decision estimator uses the explicit phase wait
field and the service field; it does not mistake the residual for queue time.

Recompute timing follows a request across chunked-prefill steps and completes
only after all CPU-hit prefix tokens have actually been recomputed. Worker
service time sums model-execution steps; gaps between steps remain part of the
scheduler-observed end-to-end cost.

Example extra configuration:

```json
{
  "materialization_mode": "dynamic",
  "fallback_mode": "load",
  "min_copy_samples": 3,
  "min_recompute_samples": 3,
  "max_observation_age_ms": 5000,
  "sample_window_size": 32,
  "kv_bytes_per_block": 0
}
```

公式校准需要在多个命中规模上分别运行强制 `load` 和 `recompute`，并对每个
规模重复采样。计时记录由 connector 的 audit/telemetry 导出；拟合时只将
service 分量分别对 KV 字节数和完整 prefill FLOPs 拟合。队列等待不写入冻结
模型：公式决策使用已完成观测的新鲜中位数；没有有效队列估计时必须显式回退。
校准样本和 holdout 对比样本不能混用。

拟合命令如下：

```bash
PYTHONPATH=upstream/vllm-hust/plugins/kv_materialization/src \
upstream/vllm-hust/.venv/bin/python \
experiments/scripts/kv_materialization/fit_cost_model.py \
  --telemetry /path/to/forced-load/telemetry.json \
              /path/to/forced-recompute/telemetry.json \
  --output /path/to/cost_model.json \
  --calibration-id qwen25-0.5b-cann90-<date>
```

命令还会写出 `.diagnostics.json`，其中包含样本数、规模范围、拟合得到的固定
项和 service 项、残差以及 R²。除非两条路径都至少有三个样本且至少覆盖两个
不同工作规模，否则该校准不可用于在线决策。

The package has no install-time dependency on a PyPI vLLM release. Install it
in the same environment as the checked-out `vllm-hust` tree, or add its
`src/` directory to `PYTHONPATH` as the parent repository's experiment runner
does.

Use the existing vLLM-HUST runtime environment for tests; do not create a
second virtual environment inside this plugin directory. From the parent
repository root, run:

```bash
python -m pytest -q upstream/vllm-hust/plugins/kv_materialization/tests
```
