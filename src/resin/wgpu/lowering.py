from frozendict import frozendict
from resin.dsl import dsl
from resin.core.accessor import Accessor
from resin.ir.ir import IrBufferView, IrProgram

from .codegen import WgslKernelConfig, dispatch_size_for_kernel, emit_wgsl_for_kernel
from .spec import (
    SCHEMA_VERSION,
    WgpuAccessorSpec,
    WgpuBufferSpec,
    WgpuBufferViewSpec,
    WgpuComputePipelineSpec,
    WgpuDispatch,
    WgpuProgram,
)

__all__ = [
    "build_wgpu_program",
    "param_buffer_index",
]

# build_wgpu_program()
#


def build_wgpu_program(
    program: IrProgram,
    *,
    wgsl_kernel_config: WgslKernelConfig | None = None,
) -> WgpuProgram:
    config = wgsl_kernel_config or WgslKernelConfig()

    buffer_index = {buffer: index for index, buffer in enumerate(program.buffers)}
    buffer_view_index = {
        buffer_view: index for index, buffer_view in enumerate(program.buffer_views)
    }

    buffers = [
        WgpuBufferSpec(
            shape=tuple(int(dim) for dim in buffer.shape),
            stype=buffer.stype,
            init=buffer.init,
            readonly=buffer.readonly,
        )
        for buffer in program.buffers
    ]

    buffer_views = [
        WgpuBufferViewSpec(
            buffer_index=buffer_index[buffer_view.buffer],
            accessor=_accessor_spec(buffer_view),
        )
        for buffer_view in program.buffer_views
    ]

    kernel_to_pipeline_index: dict[int, int] = {}
    pipelines: list[WgpuComputePipelineSpec] = []

    queue = []
    for dispatch in program.queue:
        kernel = dispatch.kernel
        kernel_key = id(kernel)
        if kernel_key not in kernel_to_pipeline_index:
            wgsl = emit_wgsl_for_kernel(kernel, config)
            dispatch_size = dispatch_size_for_kernel(kernel, config)
            kernel_to_pipeline_index[kernel_key] = len(pipelines)
            pipelines.append(
                WgpuComputePipelineSpec(
                    wgsl=wgsl,
                    dispatch_size=(
                        int(dispatch_size[0]),
                        int(dispatch_size[1]),
                        int(dispatch_size[2]),
                    ),
                    num_arg_bindings=len(kernel.arg_accessors),
                )
            )

        queue.append(
            WgpuDispatch(
                pipeline_index=kernel_to_pipeline_index[kernel_key],
                arg_buffer_view_indices=tuple(
                    buffer_view_index[arg] for arg in dispatch.args
                ),
                output_buffer_index=buffer_index[dispatch.output],
            )
        )

    param_buffer_ids = program.param_buffer_ids
    sinks = {
        name: buffer_view_index[buffer_view]
        for name, buffer_view in program.sinks.items()
    }

    return WgpuProgram(
        buffers=tuple(buffers),
        buffer_views=tuple(buffer_views),
        pipelines=tuple(pipelines),
        queue=tuple(queue),
        sinks=frozendict(sinks),
        param_buffer_ids=frozendict(param_buffer_ids),
        schema_version=SCHEMA_VERSION,
    )


def _accessor_spec(buffer_view: IrBufferView) -> WgpuAccessorSpec:
    accessor = buffer_view.accessor
    return WgpuAccessorSpec(
        offset=accessor.offset,
        shape=tuple(int(dim) for dim in accessor.shape),
        pitch=tuple(int(dim) for dim in accessor.pitch),
    )


def param_buffer_index(program: WgpuProgram, param: dsl.ParamNode) -> int:
    return program.param_buffer_ids[id(param)]


