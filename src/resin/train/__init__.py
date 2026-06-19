__all__ = [
    "build_param_update_program",
    "commit_param_updates",
]

from collections.abc import Iterable, Mapping
from typing import Any

from frozendict import frozendict

import resin.grad as grad_mod
from resin.dsl import dsl
from resin.ir import IrProgramBuilder
from resin.wgpu import WgpuProgram, WgslKernelConfig, build_wgpu_program


def build_param_update_program(
    loss: dsl.View,
    *,
    trainable_params: Iterable[dsl.View],
    learning_rate: float,
    wgsl_kernel_config: WgslKernelConfig | None = None,
) -> tuple[WgpuProgram, frozendict[int, str]]:
    grads = grad_mod.grad(loss)
    lr = dsl.const(learning_rate, dtype=loss.dtype)

    updated_param_sinks: dict[int, str] = {}
    builder = IrProgramBuilder()
    builder.build_sink("loss", loss)

    for param_view in trainable_params:
        node = param_view.node
        if not isinstance(node, dsl.ParamNode):
            raise TypeError(f"trainable param must be a ParamNode, got {type(node)}")
        grad_view = grads.get(node)
        if grad_view is None:
            raise ValueError(f"no gradient for param {id(node):#x}")
        updated = param_view - lr * grad_view
        sink_name = f"updated_param_{id(node)}"
        builder.build_sink(sink_name, updated)
        updated_param_sinks[id(node)] = sink_name

    ir_program = builder.finish()
    wgpu_program = build_wgpu_program(
        ir_program,
        wgsl_kernel_config=wgsl_kernel_config,
    )
    return wgpu_program, frozendict(updated_param_sinks)


def commit_param_updates(
    interp: Any,
    program: WgpuProgram,
    updated_param_sinks: Mapping[int, str],
) -> None:
    for param_id, sink_name in updated_param_sinks.items():
        src_view_index = program.sinks[sink_name]
        src_buffer_index = program.buffer_views[src_view_index].buffer_index
        dst_buffer_index = interp.param_buffer_index(param_id)
        interp.copy_buffer_to_buffer(src_buffer_index, dst_buffer_index)
