# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import types
from importlib.metadata import version
from importlib.util import find_spec

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)


def _has_active_out_of_tree_triton_driver() -> bool:
    """Detect Triton distributions whose backend registry is not upstream-shaped.

    Some out-of-tree platforms, notably Triton-Ascend, expose their active
    driver through ``triton.runtime.driver`` but do not export the
    ``triton.backends.backends`` mapping used by upstream Triton.  Treating
    that layout as "no Triton" replaces every ``@triton.jit`` kernel with a
    plain Python function and makes later ``kernel[grid]`` launches fail.
    """
    if not current_platform.is_out_of_tree():
        return False

    try:
        import triton

        active_driver = triton.runtime.driver.active
        return bool(active_driver and active_driver.is_active())
    except Exception:
        return False


HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)
if HAS_TRITON:
    try:
        from triton.backends import backends

        # It's generally expected that x.driver exists and has
        # an is_active method.
        # The `x.driver and` check adds a small layer of safety.
        active_drivers = [
            x.driver for x in backends.values() if x.driver and x.driver.is_active()
        ]

        # Check if we're in a distributed environment where CUDA_VISIBLE_DEVICES
        # or HIP_VISIBLE_DEVICES might be temporarily empty (e.g., Ray sets it to ""
        # during actor init)
        visible_devices_env = (
            "HIP_VISIBLE_DEVICES"
            if current_platform.is_rocm()
            else "CUDA_VISIBLE_DEVICES"
        )
        visible_devices = os.environ.get(visible_devices_env)
        is_distributed_env = (
            visible_devices is not None and len(visible_devices.strip()) == 0
        )

        # Apply lenient driver check for distributed environments
        if is_distributed_env and len(active_drivers) == 0:
            # Allow 0 drivers in distributed environments - they may become
            # active later when CUDA context is properly initialized
            logger.debug(
                "Triton found 0 active drivers in distributed environment. "
                "This is expected during initialization."
            )
        elif not is_distributed_env and len(active_drivers) != 1:
            # Strict check for non-distributed environments
            logger.info(
                "Triton is installed but %d active driver(s) found "
                "(expected 1). Disabling Triton to prevent runtime errors.",
                len(active_drivers),
            )
            HAS_TRITON = False

        # Check Triton CPU
        if "cpu" in version("vllm"):
            if "cpu" in backends:
                HAS_TRITON = True
            else:
                logger.warning(
                    "Triton is installed, but doesn't include CPU backend. "
                    "Disabling Triton."
                )
                HAS_TRITON = False
    except ImportError:
        # This can occur if Triton is partially installed or triton.backends
        # is missing.
        if _has_active_out_of_tree_triton_driver():
            logger.info(
                "Using the active out-of-tree Triton driver because "
                "`triton.backends.backends` is unavailable."
            )
        else:
            logger.warning(
                "Triton is installed, but `triton.backends` could not be imported. "
                "Disabling Triton."
            )
            HAS_TRITON = False
    except Exception as e:
        # Catch any other unexpected errors during the check.
        logger.warning(
            "An unexpected error occurred while checking Triton active drivers:"
            " %s. Disabling Triton.",
            e,
        )
        HAS_TRITON = False

if not HAS_TRITON:
    logger.info(
        "Triton not installed or not compatible; certain GPU-related"
        " functions will not be available."
    )


class TritonPlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton")
        self.__version__ = "3.4.0"
        self.jit = self._dummy_decorator("jit")
        self.autotune = self._dummy_decorator("autotune")
        self.heuristics = self._dummy_decorator("heuristics")
        self.Config = self._dummy_decorator("Config")
        self.cdiv = cdiv
        self.language = TritonLanguagePlaceholder()

    def _dummy_decorator(self, name):
        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

        return decorator


class TritonLanguagePlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton.language")
        self.constexpr = None
        self.dtype = None
        self.int64 = None
        self.int32 = None
        self.tensor = None
        self.exp = None
        self.exp2 = None
        self.log = None
        self.log2 = None
