#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT}
VLLM_HUST_BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-$WORKSPACE_ROOT/vllm-hust-benchmark}
VLLM_HUST_WEBSITE_REPO=${VLLM_HUST_WEBSITE_REPO:-$WORKSPACE_ROOT/vllm-hust-website}

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-$(printf '%s' "${GITHUB_SHA:-local}" | cut -c1-8)}
RESULT_ROOT=${RESULT_ROOT:-$VLLM_HUST_REPO/.benchmarks/ci/$RUN_ID}
RAW_RESULT_FILE=${RAW_RESULT_FILE:-$RESULT_ROOT/raw_benchmark.json}
SUBMISSIONS_ROOT=${SUBMISSIONS_ROOT:-$VLLM_HUST_BENCHMARK_REPO/.ci-submissions/$RUN_ID}
SUBMISSION_DIR=${SUBMISSION_DIR:-$SUBMISSIONS_ROOT/$RUN_ID}
AGGREGATE_OUTPUT_DIR=${AGGREGATE_OUTPUT_DIR:-$VLLM_HUST_BENCHMARK_REPO/.ci-leaderboard/$RUN_ID}
SERVER_LOG=${SERVER_LOG:-$RESULT_ROOT/server.log}
BENCH_SCENARIO=${BENCH_SCENARIO:-random-online}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-}
BENCH_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-}
ALLOW_RANDOM_HF_PUBLISH=${ALLOW_RANDOM_HF_PUBLISH:-0}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PARAMETERS=${MODEL_PARAMETERS:-0.5B}
MODEL_PRECISION=${MODEL_PRECISION:-BF16}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
DTYPE=${DTYPE:-bfloat16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-256}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-8}
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-64}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-16}
BENCH_RANDOM_BATCH_SIZE=${BENCH_RANDOM_BATCH_SIZE:-1}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-4}
BENCH_INPUT_LEN=${BENCH_INPUT_LEN:-}
BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-}
HARDWARE_VENDOR=${HARDWARE_VENDOR:-Huawei}
HARDWARE_CHIP_MODEL=${HARDWARE_CHIP_MODEL:-Ascend-910B2}
CHIP_COUNT=${CHIP_COUNT:-1}
NODE_COUNT=${NODE_COUNT:-1}
PUBLISH_TO_HF=${PUBLISH_TO_HF:-0}
HF_REPO_ID=${HF_REPO_ID:-}

# Avoid implicit fallback to /root when Actions runtime injects a root HOME.
HOME=${HOME:-/home/shuhao}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache}
XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-$HOME/.config}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}
VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-$XDG_CONFIG_HOME/vllm}

export HOME XDG_CACHE_HOME XDG_CONFIG_HOME VLLM_CACHE_ROOT VLLM_CONFIG_ROOT

server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" || true
    wait "$server_pid" || true
  fi
}

trap cleanup EXIT

mirror_artifacts_to_result_root() {
  local result_submissions_dir="$RESULT_ROOT/submissions"
  local result_leaderboard_dir="$RESULT_ROOT/leaderboard-data"

  mkdir -p "$result_submissions_dir" "$result_leaderboard_dir"

  if [[ -d "$SUBMISSION_DIR" ]]; then
    rm -rf "$result_submissions_dir/$RUN_ID"
    cp -a "$SUBMISSION_DIR" "$result_submissions_dir/$RUN_ID"
  fi

  if [[ -d "$AGGREGATE_OUTPUT_DIR" ]]; then
    rm -rf "$result_leaderboard_dir"
    cp -a "$AGGREGATE_OUTPUT_DIR" "$result_leaderboard_dir"
  fi
}

retry_command() {
  local max_attempts=$1
  local delay_seconds=$2
  shift 2

  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi

    if [[ "$attempt" -ge "$max_attempts" ]]; then
      echo "Command failed after ${attempt} attempt(s): $*" >&2
      return 1
    fi

    echo "Attempt ${attempt}/${max_attempts} failed for: $*" >&2
    echo "Retrying in ${delay_seconds}s..." >&2
    sleep "$delay_seconds"
    attempt=$((attempt + 1))
  done
}

ensure_hf_commit_api_compat() {
  python - "$VLLM_HUST_BENCHMARK_REPO/src/vllm_hust_benchmark/integration.py" <<'PY'
import inspect
import sys
from pathlib import Path

from huggingface_hub import HfApi

integration_path = Path(sys.argv[1])
if not integration_path.is_file():
  print(f"benchmark integration file not found: {integration_path}", file=sys.stderr)
  raise SystemExit(1)

signature = inspect.signature(HfApi.create_commit)
if "branch" in signature.parameters:
  print("huggingface_hub supports create_commit(branch=...); no compatibility patch needed")
  raise SystemExit(0)

source = integration_path.read_text(encoding="utf-8")
needle = """        api.create_commit(
            repo_id=repo_id,
            repo_type=\"dataset\",
            branch=branch,
            operations=operations,
            commit_message=commit_message,
        )
"""
replacement = """        api.create_commit(
            repo_id=repo_id,
            repo_type=\"dataset\",
            revision=branch,
            operations=operations,
            commit_message=commit_message,
        )
"""

if needle not in source:
  if "revision=branch" in source:
    print("benchmark integration already uses revision=branch")
    raise SystemExit(0)
  print("unable to locate create_commit(branch=...) block for compatibility patch", file=sys.stderr)
  raise SystemExit(1)

integration_path.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
print("patched benchmark integration to use create_commit(revision=...) for compatibility")
PY
}

mkdir -p "$RESULT_ROOT" "$SUBMISSIONS_ROOT" "$AGGREGATE_OUTPUT_DIR"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"

select_usable_ascend_device() {
  # Respect explicit pinning from the caller/runner.
  if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    echo "Using preconfigured ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
    return 0
  fi

  local candidate_csv=${ASCEND_DEVICE_CANDIDATES:-0,1,2,3,4,5,6,7}
  IFS=',' read -r -a candidates <<< "$candidate_csv"

  for device_id in "${candidates[@]}"; do
    if python - "$device_id" <<'PY' >/dev/null 2>&1
import sys

import torch

device_id = int(sys.argv[1])
torch.npu.set_device(device_id)
PY
    then
      export ASCEND_RT_VISIBLE_DEVICES="$device_id"
      echo "Selected usable Ascend device: $ASCEND_RT_VISIBLE_DEVICES"
      return 0
    fi
  done

  echo "Failed to find a usable Ascend device among candidates: $candidate_csv" >&2
  return 1
}

select_usable_ascend_device

echo "== Ascend benchmark CI =="
echo "workspace root: $WORKSPACE_ROOT"
echo "run id: $RUN_ID"
echo "result root: $RESULT_ROOT"
echo "benchmark scenario: $BENCH_SCENARIO"
echo "publish to hf: $PUBLISH_TO_HF"
echo "ascend visible devices: ${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"

case "$BENCH_SCENARIO" in
  random-online)
    EFFECTIVE_DATASET_NAME="random"
    EFFECTIVE_DATASET_PATH=""
    EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-$BENCH_RANDOM_INPUT_LEN}
    EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-$BENCH_RANDOM_OUTPUT_LEN}
    EFFECTIVE_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-$VLLM_HUST_REPO/.github/workflows/data/random-online-ci-constraints.json}
    bench_args=(
      --backend vllm
      --endpoint /v1/completions
      --dataset-name random
      --random-input-len "$BENCH_RANDOM_INPUT_LEN"
      --random-output-len "$BENCH_RANDOM_OUTPUT_LEN"
      --random-batch-size "$BENCH_RANDOM_BATCH_SIZE"
      --num-prompts "$BENCH_NUM_PROMPTS"
      --request-rate "$BENCH_REQUEST_RATE"
      --max-concurrency "$BENCH_MAX_CONCURRENCY"
    )
    ;;
  sharegpt-online)
    if [[ -z "$BENCH_DATASET_PATH" ]]; then
      echo "BENCH_DATASET_PATH is required for sharegpt-online" >&2
      exit 2
    fi
    if [[ -z "$BENCH_CONSTRAINTS_FILE" ]]; then
      echo "BENCH_CONSTRAINTS_FILE is required for sharegpt-online" >&2
      exit 2
    fi
    EFFECTIVE_DATASET_NAME="sharegpt"
    EFFECTIVE_DATASET_PATH="$BENCH_DATASET_PATH"
    EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-1024}
    EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-256}
    EFFECTIVE_CONSTRAINTS_FILE="$BENCH_CONSTRAINTS_FILE"
    bench_args=(
      --backend vllm
      --endpoint /v1/completions
      --dataset-name sharegpt
      --dataset-path "$BENCH_DATASET_PATH"
      --num-prompts "$BENCH_NUM_PROMPTS"
      --request-rate "$BENCH_REQUEST_RATE"
      --max-concurrency "$BENCH_MAX_CONCURRENCY"
    )
    ;;
  *)
    echo "Unsupported BENCH_SCENARIO: $BENCH_SCENARIO" >&2
    exit 2
    ;;
esac

if [[ "$PUBLISH_TO_HF" == "1" && "$BENCH_SCENARIO" == "random-online" && "$ALLOW_RANDOM_HF_PUBLISH" != "1" ]]; then
  echo "Refusing to publish random-online CI preview to HF without ALLOW_RANDOM_HF_PUBLISH=1" >&2
  exit 2
fi

if [[ ! -f "$EFFECTIVE_CONSTRAINTS_FILE" ]]; then
  echo "constraints file not found: $EFFECTIVE_CONSTRAINTS_FILE" >&2
  exit 2
fi

vllm serve "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enforce-eager >"$SERVER_LOG" 2>&1 &
server_pid=$!

for attempt in $(seq 1 120); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null; then
    break
  fi

  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited before becoming ready"
    cat "$SERVER_LOG"
    exit 1
  fi

  if [[ "$attempt" -eq 120 ]]; then
    echo "Timed out waiting for vLLM server to become ready"
    cat "$SERVER_LOG"
    exit 1
  fi

  sleep 2
done

vllm bench serve \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  "${bench_args[@]}" \
  --save-result \
  --result-dir "$RESULT_ROOT" \
  --result-filename "$(basename "$RAW_RESULT_FILE")"

ENGINE_VERSION=$(python - <<'PY'
import vllm
print(vllm.__version__)
PY
)

python -m vllm_hust_benchmark.cli submit \
  "$BENCH_SCENARIO" \
  --benchmark-result-file "$RAW_RESULT_FILE" \
  --constraints-file "$EFFECTIVE_CONSTRAINTS_FILE" \
  --run-id "$RUN_ID" \
  --engine vllm-hust \
  --engine-version "$ENGINE_VERSION" \
  --model-name "$MODEL_NAME" \
  --model-parameters "$MODEL_PARAMETERS" \
  --model-precision "$MODEL_PRECISION" \
  --hardware-vendor "$HARDWARE_VENDOR" \
  --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
  --chip-count "$CHIP_COUNT" \
  --node-count "$NODE_COUNT" \
  --submitter "${GITHUB_ACTOR:-ci}" \
  --data-source "vllm-hust-ci-$BENCH_SCENARIO" \
  --input-length "$EFFECTIVE_INPUT_LEN" \
  --output-length "$EFFECTIVE_OUTPUT_LEN" \
  --concurrent-requests "$BENCH_MAX_CONCURRENCY" \
  --submissions-dir "$SUBMISSIONS_ROOT"

# Normalize schema-sensitive fields before website aggregation/HF sync.
# For non-long-context scenarios, benchmark exporters may emit 0 here,
# but leaderboard schema requires null or an integer >= 1.
python - "$SUBMISSION_DIR/run_leaderboard.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
constraints_metrics = payload.setdefault("constraints", {}).setdefault("metrics", {})

if constraints_metrics.get("long_context_length") == 0:
  constraints_metrics["long_context_length"] = None

path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

if [[ "$PUBLISH_TO_HF" == "1" ]]; then
  if [[ -z "$HF_REPO_ID" ]]; then
    echo "HF_REPO_ID must be set when PUBLISH_TO_HF=1" >&2
    exit 2
  fi

  ensure_hf_commit_api_compat

  retry_command 3 20 \
    python -m vllm_hust_benchmark.cli sync-submission-to-hf \
      --submission-dir "$SUBMISSION_DIR" \
      --aggregate-output-dir "$AGGREGATE_OUTPUT_DIR" \
      --repo-id "$HF_REPO_ID" \
      --submissions-prefix submissions-auto \
      --commit-message "chore: sync vllm-hust benchmark $RUN_ID (${GITHUB_REF_NAME:-detached}@$(printf '%s' "${GITHUB_SHA:-local}" | cut -c1-8))" \
      --execute
else
  python -m vllm_hust_benchmark.cli publish-website \
    --source-dir "$SUBMISSIONS_ROOT" \
    --output-dir "$AGGREGATE_OUTPUT_DIR" \
    --execute
fi

  mirror_artifacts_to_result_root

echo "RUN_ID=$RUN_ID"
echo "RAW_RESULT_FILE=$RAW_RESULT_FILE"
echo "SUBMISSION_DIR=$SUBMISSION_DIR"
echo "AGGREGATE_OUTPUT_DIR=$AGGREGATE_OUTPUT_DIR"
echo "SERVER_LOG=$SERVER_LOG"
echo "BENCH_SCENARIO=$BENCH_SCENARIO"
echo "EFFECTIVE_CONSTRAINTS_FILE=$EFFECTIVE_CONSTRAINTS_FILE"