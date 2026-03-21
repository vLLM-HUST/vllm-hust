#!/usr/bin/env bash
set -euo pipefail

# Minimal and reproducible Ascend latency benchmark entry.
# This script avoids mixed toolkit runtime by sourcing
# scripts/use_single_ascend_env.sh first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ASCEND_ROOT="${1:-${ASCEND_ROOT:-}}"
if [[ -n "${ASCEND_ROOT}" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/use_single_ascend_env.sh" "${ASCEND_ROOT}"
else
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/use_single_ascend_env.sh"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

cd "${ROOT_DIR}"

python - <<'PY'
import argparse
from vllm.benchmarks.latency import add_cli_args, main

parser = argparse.ArgumentParser()
add_cli_args(parser)

args = parser.parse_args([
    "--model", "sshleifer/tiny-gpt2",
    "--input-len", "32",
    "--output-len", "32",
    "--batch-size", "1",
    "--num-iters-warmup", "1",
    "--num-iters", "3",
    "--dtype", "float16",
    "--gpu-memory-utilization", "0.1",
    "--trust-remote-code",
    "--load-format", "dummy",
    "--enforce-eager",
    "--compilation-config", '{"cudagraph_mode":"NONE"}',
])

main(args)
PY
