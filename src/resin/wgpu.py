__all__ = [
    "AbstractKernelException",
    "WgpuAccessorSpec",
    "WgpuBufferSpec",
    "WgpuBufferViewSpec",
    "WgpuComputePipelineSpec",
    "WgpuDispatch",
    "WgpuProgram",
    "WgslKernelConfig",
    "build_param_update_program",
    "build_wgpu_program",
    "commit_param_updates",
    "dispatch_size_for_kernel",
    "emit_wgsl_for_kernel",
    "param_buffer_index",
]

import math
import textwrap
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, cast

import msgpack
from frozendict import frozendict

from . import dsl
from .accessor import Accessor, is_c_contiguous
from .ir import (
    IrBufferView,
    IrElementwiseRpnKernel,
    IrKernel,
    IrMatmulKernel,
    IrProgram,
    IrReductionKernel,
    IrScatterKernel,
)
from .rpn import ScalarRpnExpr
from .scalar import (
    BinaryAssocScalarOperator,
    ScalarType,
    spell_stype_in_wgsl,
)

#
# WgpuProgram types (mirrors resin-runtime::program)
#


@dataclass(frozen=True)
class WgpuProgram:
    buffers: tuple[WgpuBufferSpec, ...]
    buffer_views: tuple[WgpuBufferViewSpec, ...]
    pipelines: tuple[WgpuComputePipelineSpec, ...]
    queue: tuple[WgpuDispatch, ...]
    sinks: frozendict[str, int]
    param_buffer_ids: frozendict[int, int] = field(
        default_factory=lambda: frozendict[int, int]()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_buffer_ids": dict(self.param_buffer_ids),
            "sinks": dict(self.sinks),
            "queue": [_dispatch_to_dict(d) for d in self.queue],
            "buffers": [_buffer_to_dict(b) for b in self.buffers],
            "buffer_views": [_buffer_view_to_dict(v) for v in self.buffer_views],
            "pipelines": [_pipeline_to_dict(p) for p in self.pipelines],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WgpuProgram":
        return cls(
            param_buffer_ids=frozendict(payload.get("param_buffer_ids", {})),
            sinks=frozendict(payload["sinks"]),
            queue=tuple(_dispatch_from_dict(d) for d in payload["queue"]),
            buffers=tuple(_buffer_from_dict(b) for b in payload["buffers"]),
            buffer_views=tuple(
                _buffer_view_from_dict(v) for v in payload["buffer_views"]
            ),
            pipelines=tuple(_pipeline_from_dict(p) for p in payload["pipelines"]),
        )

    def to_msgpack(self) -> bytes:
        return cast(bytes, msgpack.packb(self.to_dict(), use_bin_type=True))

    @classmethod
    def from_msgpack(cls, data: bytes) -> "WgpuProgram":
        payload = msgpack.unpackb(data, raw=False)
        return cls.from_dict(payload)


@dataclass(frozen=True)
class WgpuBufferSpec:
    shape: tuple[int, ...]
    stype: ScalarType
    init: bytes | None = None
    readonly: bool = False


@dataclass(frozen=True)
class WgpuBufferViewSpec:
    buffer_index: int
    accessor: WgpuAccessorSpec


@dataclass(frozen=True)
class WgpuAccessorSpec:
    offset: int
    shape: tuple[int, ...]
    pitch: tuple[int, ...]


@dataclass(frozen=True)
class WgpuComputePipelineSpec:
    wgsl: str
    dispatch_size: tuple[int, int, int]
    num_arg_bindings: int
    entry_point: str = "main"


@dataclass(frozen=True)
class WgpuDispatch:
    pipeline_index: int
    arg_buffer_view_indices: tuple[int, ...]
    output_buffer_index: int


def _buffer_to_dict(spec: WgpuBufferSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "shape": list(spec.shape),
        "stype": spec.stype,
        "readonly": spec.readonly,
    }
    if spec.init is not None:
        payload["init"] = spec.init
    return payload


def _buffer_from_dict(payload: dict[str, Any]) -> WgpuBufferSpec:
    init = payload.get("init")
    if isinstance(init, list):
        init = bytes(init)
    return WgpuBufferSpec(
        shape=tuple(payload["shape"]),
        stype=payload["stype"],
        init=init,
        readonly=payload.get("readonly", False),
    )


def _accessor_to_dict(spec: WgpuAccessorSpec) -> dict[str, Any]:
    return {
        "offset": spec.offset,
        "shape": list(spec.shape),
        "pitch": list(spec.pitch),
    }


def _accessor_from_dict(payload: dict[str, Any]) -> WgpuAccessorSpec:
    return WgpuAccessorSpec(
        offset=payload["offset"],
        shape=tuple(payload["shape"]),
        pitch=tuple(payload["pitch"]),
    )


def _buffer_view_to_dict(spec: WgpuBufferViewSpec) -> dict[str, Any]:
    return {
        "buffer_index": spec.buffer_index,
        "accessor": _accessor_to_dict(spec.accessor),
    }


def _buffer_view_from_dict(payload: dict[str, Any]) -> WgpuBufferViewSpec:
    return WgpuBufferViewSpec(
        buffer_index=payload["buffer_index"],
        accessor=_accessor_from_dict(payload["accessor"]),
    )


def _pipeline_to_dict(spec: WgpuComputePipelineSpec) -> dict[str, Any]:
    return {
        "kind": "compute",
        "wgsl": spec.wgsl,
        "entry_point": spec.entry_point,
        "dispatch_size": list(spec.dispatch_size),
        "num_arg_bindings": spec.num_arg_bindings,
    }


def _pipeline_from_dict(payload: dict[str, Any]) -> WgpuComputePipelineSpec:
    if payload["kind"] != "compute":
        raise ValueError(f"unsupported pipeline kind: {payload['kind']!r}")
    return WgpuComputePipelineSpec(
        wgsl=payload["wgsl"],
        entry_point=payload.get("entry_point", "main"),
        dispatch_size=tuple(payload["dispatch_size"]),
        num_arg_bindings=payload["num_arg_bindings"],
    )


def _dispatch_to_dict(spec: WgpuDispatch) -> dict[str, Any]:
    return {
        "pipeline_index": spec.pipeline_index,
        "arg_buffer_view_indices": list(spec.arg_buffer_view_indices),
        "output_buffer_index": spec.output_buffer_index,
    }


def _dispatch_from_dict(payload: dict[str, Any]) -> WgpuDispatch:
    return WgpuDispatch(
        pipeline_index=payload["pipeline_index"],
        arg_buffer_view_indices=tuple(payload["arg_buffer_view_indices"]),
        output_buffer_index=payload["output_buffer_index"],
    )


#
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

    param_buffer_ids = {
        id(node): index for node, index in program.param_buffer_ids.items()
    }
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


def build_param_update_program(
    loss: dsl.View,
    *,
    trainable_params: Iterable[dsl.View],
    learning_rate: float,
    wgsl_kernel_config: WgslKernelConfig | None = None,
) -> tuple[WgpuProgram, frozendict[int, str]]:
    from . import grad as grad_mod
    from .ir import IrProgramBuilder

    grads = grad_mod.grad(loss)
    lr = dsl.const(learning_rate, stype=loss.stype)

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


#
# dispatch_size_for_kernel()
#


def dispatch_size_for_kernel(
    kernel: IrKernel,
    config: WgslKernelConfig,
) -> tuple[int, int, int]:
    match kernel:
        case IrScatterKernel():
            n = math.prod(kernel.arg_accessors[0].shape)
        case _:
            n = math.prod(kernel.shape)
    if n == 0:
        return (0, 1, 1)

    items_per_thread = 1 << config.lg2_items_per_thread
    threads = (n + items_per_thread - 1) // items_per_thread
    workgroups_x = (threads + config.workgroup_size - 1) // config.workgroup_size
    return (workgroups_x, 1, 1)


#
# WGSL emission
#


def emit_wgsl_for_kernel(kernel: IrKernel, config: WgslKernelConfig) -> str:
    w = WgslWriter(enable_f16=(kernel.stype == "f2"))

    match kernel:
        case IrElementwiseRpnKernel():
            _emit_wgsl_for_elementwise_rpn_kernel(w, kernel, config)
        case IrMatmulKernel():
            _emit_wgsl_for_matmul_kernel(w, kernel, config)
        case IrReductionKernel():
            _emit_wgsl_for_reduction_kernel(w, kernel, config)
        case IrScatterKernel():
            _emit_wgsl_for_scatter_kernel(w, kernel, config)
        case _:
            raise AbstractKernelException(f"Unsupported kernel type: {type(kernel)}")

    return w.finish()


class AbstractKernelException(Exception):
    """Raised when a kernel cannot be compiled to WGSL."""


@dataclass(frozen=True, kw_only=True)
class WgslKernelConfig:
    lg2_items_per_thread: int = 3
    workgroup_size: int = 8


def _emit_bindings(w: "WgslWriter", kernel: IrKernel) -> None:
    t = spell_stype_in_wgsl(kernel.stype)

    w.print(
        f"""
        @group(0) @binding(0)
        var<storage, read_write> output: array<{t}>;
        """
    )

    for i in range(len(kernel.arg_accessors)):
        w.print(
            f"""
            @group(1) @binding({i})
            var<storage, read> arg{i}: array<{t}>;
            """
        )


def _define_address_function(w: "WgslWriter", name: str, accessor: Accessor) -> None:
    n = len(accessor.shape)
    if n == 0:
        with w.block(f"fn {name}() -> u32"):
            w.print(f"return {accessor.offset}u;")
        return

    with w.block(
        f"""
        fn {name}(index: array<u32, {n}>) -> u32
        """
    ):
        w.print(f"var acc: u32 = {accessor.offset}u;")
        for i in range(n):
            w.print(
                f"""
                acc += index[{i}] * {accessor.pitch[i]}u;  // dim {i}
                """
            )
        w.print("return acc;")


def _define_cc_index_function(
    w: "WgslWriter", name: str, cc_accessor: Accessor
) -> None:
    assert is_c_contiguous(cc_accessor.shape, cc_accessor.pitch)

    n = len(cc_accessor.shape)
    if n == 0:
        return

    with w.block(
        f"""
        fn {name}(address: u32) -> array<u32, {n}>
        """
    ):
        w.print("var addr: u32 = address;")
        w.print(f"var index: array<u32, {n}>;")
        for i in range(n):
            w.print(
                f"""
                index[{i}] = addr / {cc_accessor.pitch[i]}u;  // dim {i}
                addr = addr % {cc_accessor.pitch[i]}u;
                """
            )
        w.print("return index;")


def _emit_arg_address_functions(w: "WgslWriter", kernel: IrKernel) -> None:
    for i, arg_accessor in enumerate(kernel.arg_accessors):
        _define_address_function(w, f"address_arg{i}", arg_accessor)


def _emit_out_index_function(
    w: "WgslWriter",
    kernel: IrKernel,
    name: str = "index",
) -> None:
    _define_cc_index_function(w, name, Accessor.dense(kernel.shape))


def _arg_address_expr(arg_index: int, accessor: Accessor, index_expr: str) -> str:
    if len(accessor.shape) == 0:
        return f"address_arg{arg_index}()"
    return f"address_arg{arg_index}({index_expr})"


@contextmanager
def _per_output_element(
    w: "WgslWriter",
    kernel: IrKernel,
    config: WgslKernelConfig,
) -> Generator[tuple["WgslWriter", str], None, None]:
    if len(kernel.shape) == 0:
        with w.block(
            """
            @compute @workgroup_size(1)
            fn main(
                @builtin(global_invocation_id) global_id: vec3<u32>
            )
            """
        ):
            with w.block("if (global_id.x > 0u)"):
                w.print("return;")
            yield w, "0u"
        return

    _emit_out_index_function(w, kernel)

    items_per_thread = 1 << config.lg2_items_per_thread

    with w.block(
        f"""
        @compute @workgroup_size({config.workgroup_size})
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>
        )
        """
    ):
        w.print(
            f"""
            let out_address_beg = global_id.x << {config.lg2_items_per_thread}u;
            let out_address_end = out_address_beg + {items_per_thread}u;
            """
        )

        with w.block(
            """
            for (
                var out_address = out_address_beg;
                out_address < out_address_end;
                out_address += 1u
            )
            """
        ):
            with w.block("if (out_address >= arrayLength(&output))"):
                w.print("return;")

            w.print(
                """
                let out_index = index(out_address);
                """
            )

            yield w, "out_address"


#
# Emit WGSL for IrElementwiseRpnKernel
#


def _emit_wgsl_for_elementwise_rpn_kernel(
    w: "WgslWriter",
    kernel: IrElementwiseRpnKernel,
    config: WgslKernelConfig,
) -> None:
    n = len(kernel.arg_accessors)

    _emit_bindings(w, kernel)
    _emit_arg_address_functions(w, kernel)
    _emit_eval_rpn_expr(w, kernel.rpn_expr, n=n, stype=kernel.stype)

    with _per_output_element(w, kernel, config) as (w, out_address_expr):
        arg_values = [
            f"arg{i}[{_arg_address_expr(i, kernel.arg_accessors[i], 'out_index')}]"
            for i in range(n)
        ]
        w.print(
            f"""
            output[{out_address_expr}] = eval_rpn_expr({", ".join(arg_values)});
            """
        )


def _emit_eval_rpn_expr(
    w: "WgslWriter",
    rpn_expr: ScalarRpnExpr,
    *,
    n: int,
    stype: ScalarType,
) -> None:
    t = spell_stype_in_wgsl(stype)

    with w.block(
        f"""
        fn eval_rpn_expr({",".join(f"a{i}: {t}" for i in range(n))}) -> {t}
        """
    ):
        expr_stack = []
        for token in rpn_expr.string:
            match token:
                case int():
                    expr_stack.append(f"a{token}")
                case "neg":
                    operand = expr_stack.pop()
                    expr_stack.append(f"(-{operand})")
                case "exp":
                    operand = expr_stack.pop()
                    expr_stack.append(f"(exp({operand}))")
                case "log":
                    operand = expr_stack.pop()
                    expr_stack.append(f"(log({operand}))")
                case "sqrt":
                    operand = expr_stack.pop()
                    expr_stack.append(f"(sqrt({operand}))")
                case "sin":
                    operand = expr_stack.pop()
                    expr_stack.append(f"(sin({operand}))")
                case "cos":
                    operand = expr_stack.pop()
                    expr_stack.append(f"(cos({operand}))")
                case "not":
                    operand = expr_stack.pop()
                    expr_stack.append(f"(abs(1.0 - {operand}))")
                case "pow":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"pow({lhs}, {rhs})")
                case "mul":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"({lhs} * {rhs})")
                case "div":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"({lhs} / {rhs})")
                case "add":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"({lhs} + {rhs})")
                case "sub":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"({lhs} - {rhs})")
                case "max":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"max({lhs}, {rhs})")
                case "min":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"min({lhs}, {rhs})")
                case "gt":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"select({t}(0), {t}(1), {lhs} > {rhs})")
                case "lt":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"select({t}(0), {t}(1), {lhs} < {rhs})")
                case "ge":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"select({t}(0), {t}(1), {lhs} >= {rhs})")
                case "le":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"select({t}(0), {t}(1), {lhs} <= {rhs})")
                case "eq":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"select({t}(0), {t}(1), {lhs} == {rhs})")
                case "ne":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"select({t}(0), {t}(1), {lhs} != {rhs})")
                case _:
                    raise NotImplementedError(token)

        assert len(expr_stack) == 1
        w.print(f"return {expr_stack[0]};")


#
# Emit WGSL for IrMatmulKernel
#


def _emit_wgsl_for_matmul_kernel(
    w: "WgslWriter",
    kernel: IrMatmulKernel,
    config: WgslKernelConfig,
) -> None:
    t = spell_stype_in_wgsl(kernel.stype)
    rank = len(kernel.shape)
    k = kernel.k

    _emit_bindings(w, kernel)
    _emit_arg_address_functions(w, kernel)

    _define_matmul_arg_index_function(
        w,
        "arg0_index",
        rank=rank,
        k_axis=rank - 1,
        m_axis=rank - 2,
    )
    _define_matmul_arg_index_function(
        w,
        "arg1_index",
        rank=rank,
        k_axis=rank - 2,
        n_axis=rank - 1,
    )

    with _per_output_element(w, kernel, config) as (w, out_address_expr):
        w.print(
            f"""
            var sum: {t} = {t}(0);
            """
        )

        with w.block(f"for (var ki: u32 = 0u; ki < {k}u; ki += 1u)"):
            w.print(
                f"""
                let a_idx = arg0_index(out_index, ki);
                let b_idx = arg1_index(out_index, ki);
                sum += arg0[{_arg_address_expr(0, kernel.arg_accessors[0], "a_idx")}] * arg1[{_arg_address_expr(1, kernel.arg_accessors[1], "b_idx")}];
                """
            )

        w.print(f"output[{out_address_expr}] = sum;")


def _define_matmul_arg_index_function(
    w: "WgslWriter",
    name: str,
    *,
    rank: int,
    k_axis: int,
    m_axis: int | None = None,
    n_axis: int | None = None,
) -> None:
    with w.block(
        f"""
        fn {name}(out_index: array<u32, {rank}>, k: u32) -> array<u32, {rank}>
        """
    ):
        w.print(f"var idx: array<u32, {rank}>;")

        for i in range(rank - 2):
            w.print(f"idx[{i}] = out_index[{i}];")

        if m_axis is not None:
            w.print(f"idx[{m_axis}] = out_index[{m_axis}];")

        if n_axis is not None:
            w.print(f"idx[{n_axis}] = out_index[{n_axis}];")

        w.print(f"idx[{k_axis}] = k;")
        w.print("return idx;")


#
# Emit WGSL for IrScatterKernel
#


def _emit_wgsl_for_scatter_kernel(
    w: "WgslWriter",
    kernel: IrScatterKernel,
    config: WgslKernelConfig,
) -> None:
    source_accessor = kernel.arg_accessors[0]
    source_shape = source_accessor.shape
    rank = len(source_shape)

    _emit_bindings(w, kernel)
    _emit_arg_address_functions(w, kernel)

    if rank == 0:
        with w.block(
            """
            @compute @workgroup_size(1)
            fn main(
                @builtin(global_invocation_id) global_id: vec3<u32>
            )
            """
        ):
            with w.block("if (global_id.x > 0u)"):
                w.print("return;")
            w.print(
                f"""
                output[{kernel.woffset}u] = arg0[{_arg_address_expr(0, source_accessor, "")}];
                """
            )
        return

    _define_cc_index_function(w, "source_index", Accessor.dense(source_shape))
    _define_scatter_out_address_function(w, kernel)

    items_per_thread = 1 << config.lg2_items_per_thread
    source_count = math.prod(source_shape)

    with w.block(
        f"""
        @compute @workgroup_size({config.workgroup_size})
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>
        )
        """
    ):
        w.print(
            f"""
            let src_address_beg = global_id.x << {config.lg2_items_per_thread}u;
            let src_address_end = src_address_beg + {items_per_thread}u;
            """
        )

        with w.block(
            """
            for (
                var src_address = src_address_beg;
                src_address < src_address_end;
                src_address += 1u
            )
            """
        ):
            with w.block(f"if (src_address >= {source_count}u)"):
                w.print("return;")

            w.print(
                f"""
                let src_index = source_index(src_address);
                let out_address = scatter_out_address(src_index);
                output[out_address] = arg0[{_arg_address_expr(0, source_accessor, "src_index")}];
                """
            )


def _define_scatter_out_address_function(
    w: "WgslWriter",
    kernel: IrScatterKernel,
) -> None:
    rank = len(kernel.wpitch)
    with w.block(
        f"""
        fn scatter_out_address(index: array<u32, {rank}>) -> u32
        """
    ):
        w.print(f"var acc: u32 = {kernel.woffset}u;")
        for i in range(rank):
            w.print(f"acc += index[{i}] * {kernel.wpitch[i]}u;")
        w.print("return acc;")


#
# Emit WGSL for IrReductionKernel
#


def _emit_wgsl_for_reduction_kernel(
    w: "WgslWriter",
    kernel: IrReductionKernel,
    config: WgslKernelConfig,
) -> None:
    t = spell_stype_in_wgsl(kernel.stype)
    rank = len(kernel.shape)
    count = kernel.reduced_count
    input_shape = kernel.input_shape

    _emit_bindings(w, kernel)
    _emit_arg_address_functions(w, kernel)

    _define_reduction_input_index_function(
        w,
        "input_index",
        rank=rank,
        input_shape=input_shape,
        axes=kernel.axes,
    )

    with _per_output_element(w, kernel, config) as (w, out_address_expr):
        if count == 0:
            w.print(f"output[{out_address_expr}] = {t}(0);")
            return

        w.print(
            f"""
            var acc: {t} = arg0[{_arg_address_expr(0, kernel.arg_accessors[0], "input_index(out_index, 0u)")}];
            """
        )

        if count > 1:
            with w.block(f"for (var ri: u32 = 1u; ri < {count}u; ri += 1u)"):
                w.print(
                    f"""
                    let v = arg0[{_arg_address_expr(0, kernel.arg_accessors[0], "input_index(out_index, ri)")}];
                    """
                )
                w.print(_reduction_accumulate_wgsl(kernel.operator, "acc", "v"))

        w.print(f"output[{out_address_expr}] = acc;")


def _define_reduction_input_index_function(
    w: "WgslWriter",
    name: str,
    *,
    rank: int,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> None:
    sorted_axes = tuple(sorted(axes))
    non_reduced_axes = tuple(i for i in range(rank) if i not in axes)

    with w.block(
        f"""
        fn {name}(out_index: array<u32, {rank}>, ri: u32) -> array<u32, {rank}>
        """
    ):
        w.print(f"var idx: array<u32, {rank}>;")

        for i in non_reduced_axes:
            w.print(f"idx[{i}] = out_index[{i}];")

        if sorted_axes:
            w.print("var remaining: u32 = ri;")
            for axis in reversed(sorted_axes):
                dim = input_shape[axis]
                w.print(f"idx[{axis}] = remaining % {dim}u;")
                w.print(f"remaining = remaining / {dim}u;")

        w.print("return idx;")


def _reduction_accumulate_wgsl(
    operator: BinaryAssocScalarOperator,
    acc: str,
    value: str,
) -> str:
    match operator:
        case "add":
            return f"{acc} += {value};"
        case "mul":
            return f"{acc} *= {value};"
        case "max":
            return f"{acc} = max({acc}, {value});"
        case "min":
            return f"{acc} = min({acc}, {value});"
        case _:
            raise NotImplementedError(f"{operator=}")


#
# WgslWriter
#


class WgslWriter:
    lines: list[str]

    def __init__(self, *, enable_f16: bool) -> None:
        super().__init__()
        self.lines = []

        if enable_f16:
            self.lines.append("enable f16;")

    def print(self, text: str) -> None:
        self.lines.append(textwrap.dedent(text))

    @contextmanager
    def block(self, prefix: str = ""):
        if prefix:
            self.print(prefix)

        self.print("{")
        yield self
        self.print("}")

    def finish(self) -> str:
        return "\n".join(self.lines)
