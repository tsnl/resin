from frozendict import frozendict
from resin.core.accessor import is_c_contiguous
from resin.dsl import dsl
from resin.ir.ir import (
    IrBufferView,
    IrKernel,
    IrProgram,
    IrReindexKernel,
    IrReindexViaAccessor,
)

from .codegen import WgslKernelConfig, dispatch_size_for_kernel, emit_wgsl_for_kernel
from .spec import (
    SCHEMA_VERSION,
    WgpuAccessorSpec,
    WgpuBufferSpec,
    WgpuBufferViewSpec,
    WgpuComputePipelineSpec,
    WgpuCopy,
    WgpuDispatch,
    WgpuProgram,
    WgpuQueueOp,
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
            etype=buffer.etype,
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

    queue: list[WgpuQueueOp] = []
    for dispatch in program.queue:
        kernel = dispatch.kernel
        arg_views = tuple(buffer_view_index[arg] for arg in dispatch.args)
        output_idx = buffer_index[dispatch.output]

        match kernel:
            case IrReindexKernel(
                direction="gather",
                addressing=IrReindexViaAccessor(),
            ):
                accessor = kernel.arg_accessors[0]
                if is_c_contiguous(accessor.shape, accessor.pitch):
                    queue.append(
                        WgpuCopy(
                            source_buffer_view_index=arg_views[0],
                            output_buffer_index=output_idx,
                        )
                    )
                    continue
            case _:
                pass

        kernel_key = id(kernel)
        if kernel_key not in kernel_to_pipeline_index:
            kernel_to_pipeline_index[kernel_key] = len(pipelines)
            pipelines.append(_compute_pipeline_for_kernel(kernel, config))

        queue.append(
            WgpuDispatch(
                pipeline_index=kernel_to_pipeline_index[kernel_key],
                arg_buffer_view_indices=arg_views,
                output_buffer_index=output_idx,
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


def _compute_pipeline_for_kernel(
    kernel: IrKernel, config: WgslKernelConfig
) -> WgpuComputePipelineSpec:
    return WgpuComputePipelineSpec(
        wgsl=emit_wgsl_for_kernel(kernel, config),
        dispatch_size=dispatch_size_for_kernel(kernel, config),
        num_arg_bindings=len(kernel.arg_accessors),
        clear_output_before_dispatch=kernel.clear_output_before_dispatch,
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