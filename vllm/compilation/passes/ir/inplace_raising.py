# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict

from torch import fx
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger

from ..vllm_inductor_pass import VllmInductorPass
from .lowering_pass import get_ir_op, overload_or_default

logger = init_logger(__name__)


# TODO for pickling
def f():
    return 0


# TODO avoid pickling pre_grad_pass
class VllmIRInplaceRaisingPass(VllmInductorPass):
    """
    This pass raises maybe_inplace vLLM IR ops to their functional equivalents.
    The maybe_inplace overloads have the same signature as the default overload
    so the pass simply replaces the called overload.
    That makes the graph properly functional.

    This pass operates pre-AOTAutograd,
    so it must handle non-normalized and non-functional IR.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)
        self.patterns = PatternMatcherPass(self.pass_name)
        self.raised_ops: dict[str, int] = defaultdict(f)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        # clear at the beginning instead of end, so that tests can inspect
        self.raised_ops.clear()

        for node in graph.nodes:
            if (ir_op := get_ir_op(node)) is None:
                continue

            op_overload = overload_or_default(node.target)
            overload_name = op_overload._overloadname
            if overload_name != "maybe_inplace":
                assert overload_name == "default", (
                    f"Found overload {overload_name} for op {ir_op.name}, "
                    f"expected maybe_inplace or default"
                )
                continue

            # must have maybe_inplace overload and allow_inplace
            assert ir_op.allow_inplace and ir_op.maybe_inplace is not None

            # Check that activation inputs are not used after this op
            for arg_idx in ir_op.activation_indices:
                arg = node.args[arg_idx]
                assert isinstance(arg, fx.Node), "Activation inputs must be fx.Node"
                for user in arg.users:
                    if user is not node:
                        # TODO only check topologically?
                        logger.warning(
                            "Node %s (input to %s) has another use", arg, node
                        )
                        # TODO raise error, this is undefined behavior, which should not be allowed.
                        #  Users can just use the default overload if they want to keep activation inputs untouched.

                if arg.op == "placeholder":
                    # This node represents a graph input, and maybe_inplace might modify it,
                    # meaning the user does not care about it.
                    # Mark it dirty so downstream passes know it can be modified without affecting correctness.
                    # TODO should we store this in node.meta instead?
                    arg.meta["custom"] = {
                        "is_consumed": True,
                        **arg.meta.get("custom", {}),
                    }
                    logger.debug(
                        "vLLM IR op %s has an activation input that is a graph input",
                        ir_op.name,
                    )

            # Same signature, just replace the overload that's called.
            node.target = ir_op.torch_op
            node.meta["custom"] = {"maybe_inplace": True, **node.meta.get("custom", {})}
            self.raised_ops[ir_op.name] += 1

        count = sum(self.raised_ops.values())
        ops = ",".join(self.raised_ops.keys())
        logger.debug(
            "VllmIRInplaceRaisingPass raised %d vLLM IR nodes for op(s) %s", count, ops
        )
