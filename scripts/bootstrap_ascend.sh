#!/usr/bin/env bash
set -euo pipefail

# Thin compatibility wrapper.
# Canonical setup and launch flow lives in hust-ascend-manager.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_REF="${1:-${VLLM_HUST_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}}"
MANAGER_REPO="${HUST_ASCEND_MANAGER_REPO:-/home/shuhao/vllm-hust-dev-hub/ascend-runtime-manager}"
MANAGER_MANIFEST="${HUST_ASCEND_MANAGER_MANIFEST:-${MANAGER_REPO}/manifests/euleros-910b.json}"
MANAGER_PYPI_SPEC="${HUST_ASCEND_MANAGER_PYPI_SPEC:-hust-ascend-manager}"

if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

cd "${REPO_ROOT}"

if ! command -v hust-ascend-manager >/dev/null 2>&1; then
  if [[ -f "${MANAGER_REPO}/pyproject.toml" ]]; then
    python -m pip install -e "${MANAGER_REPO}" --no-deps
  else
    python -m pip install --upgrade "${MANAGER_PYPI_SPEC}"
  fi
fi

LAUNCH_ARGS=(
  launch
  "${MODEL_REF}"
  --manifest "${MANAGER_MANIFEST}"
)

if [[ "${HUST_MANAGER_APPLY_SYSTEM:-1}" != "1" ]]; then
  LAUNCH_ARGS+=(--no-apply-system)
fi
if [[ "${HUST_MANAGER_INSTALL_PYTHON_STACK:-1}" == "1" ]]; then
  LAUNCH_ARGS+=(--install-python-stack)
fi
if [[ "${HUST_MANAGER_SKIP_SETUP:-0}" == "1" ]]; then
  LAUNCH_ARGS+=(--skip-setup)
fi
if [[ -n "${VLLM_HUST_HOST:-}" ]]; then
  LAUNCH_ARGS+=(--host "${VLLM_HUST_HOST}")
fi
if [[ -n "${VLLM_HUST_PORT:-}" ]]; then
  LAUNCH_ARGS+=(--port "${VLLM_HUST_PORT}")
fi
if [[ -n "${VLLM_HUST_SERVED_MODEL_NAME:-}" ]]; then
  LAUNCH_ARGS+=(--served-model-name "${VLLM_HUST_SERVED_MODEL_NAME}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  LAUNCH_ARGS+=(-- "${EXTRA_ARGS[@]}")
fi

exec hust-ascend-manager "${LAUNCH_ARGS[@]}"
