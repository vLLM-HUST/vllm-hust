#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/install_local_ascend_plugin.sh [path_to_vllm_ascend_repo]
#
# Default path assumes this multi-root workspace layout:
#   vllm-hust/
#   reference-repos/vllm-ascend/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_HUST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLUGIN_REPO="${1:-${VLLM_HUST_ROOT}/../reference-repos/vllm-ascend}"

if [[ ! -f "${PLUGIN_REPO}/pyproject.toml" ]]; then
  echo "[ERROR] vllm-ascend repo not found: ${PLUGIN_REPO}"
  echo "Provide path manually, e.g.:"
  echo "  scripts/install_local_ascend_plugin.sh /path/to/vllm-ascend"
  exit 1
fi

echo "[INFO] Installing local vllm-ascend plugin from: ${PLUGIN_REPO}"
echo "[INFO] Using lightweight mode: COMPILE_CUSTOM_KERNELS=0, --no-deps"
export COMPILE_CUSTOM_KERNELS="${COMPILE_CUSTOM_KERNELS:-0}"

if ! python -m pip install -e "${PLUGIN_REPO}" --no-build-isolation --no-deps; then
  echo "[WARN] Local editable install failed."
  echo "[WARN] Continue with currently installed vllm-ascend package if present."
fi

echo "[INFO] Checking vLLM platform plugin entry points"
python - <<'PY'
from importlib.metadata import entry_points

eps = entry_points(group="vllm.platform_plugins")
if not eps:
    raise SystemExit("[ERROR] No platform plugins discovered in group vllm.platform_plugins")

print("[INFO] Discovered platform plugins:")
found_ascend = False
for ep in eps:
    print(f"  - {ep.name} -> {ep.value}")
    if ep.name == "ascend":
        found_ascend = True

if not found_ascend:
    raise SystemExit("[ERROR] ascend plugin entry point not found")
PY

echo "[OK] vllm-ascend is installed as a vLLM platform plugin."
echo "[NOTE] Runtime compatibility still requires matching torch/torch_npu/CANN versions."
