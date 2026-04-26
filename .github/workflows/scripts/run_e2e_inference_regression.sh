#!/bin/bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-facebook/opt-125m}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-}
DTYPE=${DTYPE:-float32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
MAX_TOKENS=${MAX_TOKENS:-8}
PROMPT=${PROMPT:-The capital of France is}
CHAT_MESSAGE=${CHAT_MESSAGE:-Tell me one short fact about France.}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-5}
SERVER_LOG=${SERVER_LOG:-/tmp/vllm-e2e-regression.log}

server_pid=""
server_group_pid=""
marker_pid_file=""
marker_pgid_file=""

cleanup() {
  if [[ -n "$server_group_pid" ]] && kill -0 "$server_group_pid" 2>/dev/null; then
    kill -TERM -- "-$server_group_pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$server_group_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL -- "-$server_group_pid" 2>/dev/null || true
  elif [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
  fi

  if [[ -n "$server_pid" ]]; then
    wait "$server_pid" || true
  fi

  if [[ -n "$marker_pid_file" || -n "$marker_pgid_file" ]]; then
    rm -f "$marker_pid_file" "$marker_pgid_file"
  fi
}

start_server() {
  if command -v setsid >/dev/null 2>&1; then
    setsid vllm serve "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enforce-eager >"$SERVER_LOG" 2>&1 &
    server_pid=$!
    server_group_pid=$server_pid
  else
    vllm serve "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enforce-eager >"$SERVER_LOG" 2>&1 &
    server_pid=$!
  fi

  printf '%s\n' "$server_pid" > "$marker_pid_file"
  if [[ -n "$server_group_pid" ]]; then
    printf '%s\n' "$server_group_pid" > "$marker_pgid_file"
  fi
}

allocate_local_port() {
  python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

trap cleanup EXIT

if [[ -z "$PORT" ]]; then
  PORT=$(allocate_local_port)
fi

runtime_root=${VLLM_HUST_CI_RUNTIME_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.ci-runtime}
export HOME="$runtime_root/home"
export XDG_CACHE_HOME="$runtime_root/cache"
export XDG_CONFIG_HOME="$runtime_root/config"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$PIP_CACHE_DIR"

marker_dir="$runtime_root/process-markers"
marker_pid_file="$marker_dir/vllm-server.pid"
marker_pgid_file="$marker_dir/vllm-server.pgid"
mkdir -p "$marker_dir"

echo "Starting vLLM inference regression test for $MODEL_NAME"
echo "Using regression test port $PORT"

start_server

for attempt in $(seq 1 120); do
  if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null; then
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

curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null

completion_response=$(mktemp)
curl -fsS "http://$HOST:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_NAME\", \"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS, \"temperature\": 0}" \
  > "$completion_response"

python - "$completion_response" "$MODEL_NAME" <<'PY'
import json
import sys

response_path, expected_model = sys.argv[1:3]

with open(response_path, encoding="utf-8") as handle:
    payload = json.load(handle)

assert payload.get("model") == expected_model, payload
choices = payload.get("choices")
assert isinstance(choices, list) and choices, payload
text = choices[0].get("text")
assert isinstance(text, str) and text.strip(), payload
usage = payload.get("usage")
assert isinstance(usage, dict) and usage.get("total_tokens", 0) > 0, payload
PY
rm -f "$completion_response"

chat_response=$(mktemp)
curl -fsS "http://$HOST:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"$CHAT_MESSAGE\"}], \"max_tokens\": $MAX_TOKENS, \"temperature\": 0}" \
  > "$chat_response"

python - "$chat_response" "$MODEL_NAME" <<'PY'
import json
import sys

response_path, expected_model = sys.argv[1:3]

with open(response_path, encoding="utf-8") as handle:
    payload = json.load(handle)

assert payload.get("model") == expected_model, payload
choices = payload.get("choices")
assert isinstance(choices, list) and choices, payload
message = choices[0].get("message")
assert isinstance(message, dict), payload
content = message.get("content")
assert isinstance(content, str) and content.strip(), payload
usage = payload.get("usage")
assert isinstance(usage, dict) and usage.get("total_tokens", 0) > 0, payload
PY
rm -f "$chat_response"

vllm bench serve \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dataset-name random \
  --random-input-len 32 \
  --random-output-len 4 \
  --num-prompts "$BENCH_NUM_PROMPTS" \
  --endpoint /v1/chat/completions \
  --backend openai-chat

echo "vLLM inference regression test passed"