# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes


@dataclass(frozen=True)
class HardwareSchedulingDefaults:
    llm_class_max_num_batched_tokens: int
    api_server_max_num_batched_tokens: int
    max_num_seqs: int
    max_cudagraph_capture_size: int


COMPACT_ACCELERATOR_DEFAULTS = HardwareSchedulingDefaults(
    llm_class_max_num_batched_tokens=8192,
    api_server_max_num_batched_tokens=2048,
    max_num_seqs=256,
    max_cudagraph_capture_size=128,
)

BALANCED_ACCELERATOR_DEFAULTS = HardwareSchedulingDefaults(
    llm_class_max_num_batched_tokens=12288,
    api_server_max_num_batched_tokens=4096,
    max_num_seqs=512,
    max_cudagraph_capture_size=256,
)

LARGE_ACCELERATOR_DEFAULTS = HardwareSchedulingDefaults(
    llm_class_max_num_batched_tokens=16384,
    api_server_max_num_batched_tokens=8192,
    max_num_seqs=1024,
    max_cudagraph_capture_size=512,
)


def infer_accelerator_scheduling_defaults(
    device_memory: int,
    device_name: str,
    *,
    is_xpu: bool = False,
    is_data_center_gpu: bool = False,
) -> HardwareSchedulingDefaults:
    device_name = device_name.lower()
    is_a100 = "a100" in device_name

    if device_memory >= 70 * GiB_bytes and not is_a100:
        return LARGE_ACCELERATOR_DEFAULTS

    if device_memory >= 40 * GiB_bytes and not is_a100:
        return BALANCED_ACCELERATOR_DEFAULTS

    if is_xpu and is_data_center_gpu and device_memory >= 32 * GiB_bytes:
        return BALANCED_ACCELERATOR_DEFAULTS

    return COMPACT_ACCELERATOR_DEFAULTS


def get_current_accelerator_scheduling_defaults() -> HardwareSchedulingDefaults:
    try:
        device_memory = current_platform.get_device_total_memory()
        device_name = current_platform.get_device_name()
    except Exception:
        return COMPACT_ACCELERATOR_DEFAULTS

    is_xpu = current_platform.is_xpu()
    is_data_center_gpu = False
    if is_xpu:
        is_data_center_gpu_fn = getattr(current_platform, "is_data_center_gpu", None)
        if callable(is_data_center_gpu_fn):
            is_data_center_gpu = bool(is_data_center_gpu_fn())

    return infer_accelerator_scheduling_defaults(
        device_memory,
        device_name,
        is_xpu=is_xpu,
        is_data_center_gpu=is_data_center_gpu,
    )