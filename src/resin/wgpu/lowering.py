from frozendict import frozendict
import resin.dsl as dsl
from resin.ir.ir import IrBufferView, IrKernel, IrProgram

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

    buffers: list[WgpuBufferSpec] = []
    for buffer in program.buffers:
        spec: WgpuBufferSpec = {
            "shape": [int(dim) for dim in buffer.shape],
            "etype": buffer.etype,
            "readonly": buffer.readonly,
        }
        if buffer.init is not None:
            spec["init"] = buffer.init
        buffers.append(spec)

    buffer_views = [
        WgpuBufferViewSpec(
            buffer_index=buffer_index[buffer_view.buffer],
            accessor=_accessor_spec(buffer_view),
        )
        for buffer_view in program.buffer_views
    ]

    kernel_to_pipeline_index: dict[int, int] = {}
    pipelines: list[WgpuComputePipelineSpec] = []

    queue: list[WgpuDispatch] = []
    for dispatch in program.queue:
        kernel = dispatch.kernel
        kernel_key = id(kernel)
        if kernel_key not in kernel_to_pipeline_index:
            kernel_to_pipeline_index[kernel_key] = len(pipelines)
            pipelines.append(_compute_pipeline_for_kernel(kernel, config))

        queue.append(
            {
                "kind": "dispatch",
                "pipeline_index": kernel_to_pipeline_index[kernel_key],
                "arg_buffer_view_indices": [
                    buffer_view_index[arg] for arg in dispatch.args
                ],
                "output_buffer_index": buffer_index[dispatch.output],
            }
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
    return {
        "wgsl": emit_wgsl_for_kernel(kernel, config),
        "dispatch_size": list(dispatch_size_for_kernel(kernel, config)),
        "num_arg_bindings": len(kernel.arg_accessors),
        "clear_output_before_dispatch": kernel.clear_output_before_dispatch,
    }


def _accessor_spec(buffer_view: IrBufferView) -> WgpuAccessorSpec:
    accessor = buffer_view.accessor
    return {
        "offset": accessor.offset,
        "shape": [int(dim) for dim in accessor.shape],
        "pitch": [int(dim) for dim in accessor.pitch],
    }


def param_buffer_index(program: WgpuProgram, param: dsl.ParamNode) -> int:
    return program.param_buffer_ids[id(param)]
