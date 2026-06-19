import math
import textwrap
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from resin.core.accessor import Accessor, is_c_contiguous
from resin.core.dtype import (
    BinaryAssocScalarOperator,
    DType,
    dtype_kind,
    dtype_needs_enable_f16,
    spell_dtype_in_wgsl,
)
from resin.ir.rpn import ElementRpnExpr
from resin.ir.ir import (
    IrElementwiseRpnKernel,
    IrKernel,
    IrMatmulKernel,
    IrReductionKernel,
    IrScatterAccumulateKernel,
    IrScatterClobberKernel,
    IrScatterKernel,
)

__all__ = [
    "AbstractKernelException",
    "WgslKernelConfig",
    "WgslTargetFeatures",
    "dispatch_size_for_kernel",
    "emit_wgsl_for_kernel",
]

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
    w = WgslWriter(enable_f16=dtype_needs_enable_f16(kernel.dtype))

    match kernel:
        case IrElementwiseRpnKernel():
            _emit_wgsl_for_elementwise_rpn_kernel(w, kernel, config)
        case IrMatmulKernel():
            _emit_wgsl_for_matmul_kernel(w, kernel, config)
        case IrReductionKernel():
            _emit_wgsl_for_reduction_kernel(w, kernel, config)
        case IrScatterClobberKernel():
            _emit_wgsl_for_scatter_clobber_kernel(w, kernel, config)
        case IrScatterAccumulateKernel():
            _emit_wgsl_for_scatter_accumulate_kernel(w, kernel, config)
        case _:
            raise AbstractKernelException(f"Unsupported kernel type: {type(kernel)}")

    return w.finish()


class AbstractKernelException(Exception):
    """Raised when a kernel cannot be compiled to WGSL."""


@dataclass(frozen=True, kw_only=True)
class WgslTargetFeatures:
    """wgpu adapter/device features available when compiling kernels."""

    shader_float32_atomic: bool = False


@dataclass(frozen=True, kw_only=True)
class WgslKernelConfig:
    lg2_items_per_thread: int = 3
    workgroup_size: int = 8
    target_features: WgslTargetFeatures = field(default_factory=WgslTargetFeatures)


def _emit_bindings(w: "WgslWriter", kernel: IrKernel) -> None:
    t = spell_dtype_in_wgsl(kernel.dtype)

    w.print(
        f"""
        @group(0) @binding(0)
        var<storage, read_write> output: array<{t}>;
        """
    )

    for i in range(len(kernel.arg_accessors)):
        w.print(
            f"""
            @group(0) @binding({i + 1})
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
    _emit_eval_rpn_expr(
        w,
        kernel.rpn_expr,
        n=n,
        dtype=kernel.dtype,
    )

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
    rpn_expr: ElementRpnExpr,
    *,
    n: int,
    dtype: DType,
) -> None:
    t = spell_dtype_in_wgsl(dtype)

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
    t = spell_dtype_in_wgsl(kernel.dtype)
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
# Emit WGSL for scatter kernels
#


def _emit_scatter_clobber_bindings(w: "WgslWriter", kernel: IrScatterKernel) -> None:
    t = spell_dtype_in_wgsl(kernel.dtype)
    w.print(
        f"""
        @group(0) @binding(0)
        var<storage, read_write> output: array<{t}>;
        """
    )
    for i in range(len(kernel.arg_accessors)):
        w.print(
            f"""
            @group(0) @binding({i + 1})
            var<storage, read> arg{i}: array<{t}>;
            """
        )


def _scatter_accumulate_atomic_storage_wgsl(
    output_dtype: DType,
    operator: BinaryAssocScalarOperator,
    features: WgslTargetFeatures,
) -> str:
    if (
        output_dtype == "f4"
        and features.shader_float32_atomic
        and operator == "add"
    ):
        return "atomic<f32>"
    return "atomic<u32>"


def _emit_scatter_accumulate_bindings(
    w: "WgslWriter",
    kernel: IrScatterAccumulateKernel,
    features: WgslTargetFeatures,
) -> None:
    atomic_t = _scatter_accumulate_atomic_storage_wgsl(
        kernel.dtype, kernel.operator, features
    )
    w.print(
        f"""
        @group(0) @binding(0)
        var<storage, read_write> output: array<{atomic_t}>;
        """
    )
    t = spell_dtype_in_wgsl(kernel.dtype)
    for i in range(len(kernel.arg_accessors)):
        w.print(
            f"""
            @group(0) @binding({i + 1})
            var<storage, read> arg{i}: array<{t}>;
            """
        )


def _scatter_accumulate_uses_native_atomic(
    output_dtype: DType,
    operator: BinaryAssocScalarOperator,
    features: WgslTargetFeatures,
) -> bool:
    match output_dtype:
        case "u4":
            return operator in ("add", "max", "min")
        case "f4":
            return features.shader_float32_atomic and operator == "add"
        case _:
            return False


def _scatter_native_accumulate_wgsl(
    operator: BinaryAssocScalarOperator,
    *,
    out_address_expr: str,
    value_expr: str,
) -> str:
    match operator:
        case "add":
            builtin = "atomicAdd"
        case "max":
            builtin = "atomicMax"
        case "min":
            builtin = "atomicMin"
        case _:
            raise AssertionError(f"no native atomic for {operator=}")
    return f"{builtin}(&output[{out_address_expr}], {value_expr});"


def _scatter_cas_accumulate_wgsl(
    operator: BinaryAssocScalarOperator,
    output_dtype: DType,
    *,
    out_address_expr: str,
    value_expr: str,
) -> str:
    if output_dtype not in ("f4", "u4"):
        raise NotImplementedError(
            f"atomic scatter accumulate not supported for {output_dtype=}"
        )

    match operator:
        case "add":
            combine = f"old_val + ({value_expr})"
        case "mul":
            combine = f"old_val * ({value_expr})"
        case "max":
            combine = f"max(old_val, {value_expr})"
        case "min":
            combine = f"min(old_val, {value_expr})"
        case _:
            raise NotImplementedError(f"{operator=}")

    if dtype_kind(output_dtype) == "uint":
        return textwrap.dedent(
            f"""
            {{
                let out_slot = &output[{out_address_expr}];
                loop {{
                    let old_val = atomicLoad(out_slot);
                    let new_val = {combine};
                    let exchanged = atomicCompareExchangeWeak(
                        out_slot,
                        old_val,
                        new_val
                    ).exchanged;
                    if (exchanged) {{
                        break;
                    }}
                }}
            }}
            """
        ).strip()

    wgsl_t = spell_dtype_in_wgsl(output_dtype)
    return textwrap.dedent(
        f"""
        {{
            let out_slot = &output[{out_address_expr}];
            loop {{
                let old_bits = atomicLoad(out_slot);
                let old_val = bitcast<{wgsl_t}>(old_bits);
                let new_val = {combine};
                let new_bits = bitcast<u32>(new_val);
                let exchanged = atomicCompareExchangeWeak(
                    out_slot,
                    old_bits,
                    new_bits
                ).exchanged;
                if (exchanged) {{
                    break;
                }}
            }}
        }}
        """
    ).strip()


def _scatter_accumulate_wgsl(
    operator: BinaryAssocScalarOperator,
    output_dtype: DType,
    features: WgslTargetFeatures,
    *,
    out_address_expr: str,
    value_expr: str,
) -> str:
    if _scatter_accumulate_uses_native_atomic(output_dtype, operator, features):
        return _scatter_native_accumulate_wgsl(
            operator,
            out_address_expr=out_address_expr,
            value_expr=value_expr,
        )
    return _scatter_cas_accumulate_wgsl(
        operator,
        output_dtype,
        out_address_expr=out_address_expr,
        value_expr=value_expr,
    )


def _emit_scatter_clobber_store(
    w: "WgslWriter",
    *,
    out_address_expr: str,
    value_expr: str,
) -> None:
    w.print(f"output[{out_address_expr}] = {value_expr};")


def _emit_scatter_accumulate_store(
    w: "WgslWriter",
    kernel: IrScatterAccumulateKernel,
    config: WgslKernelConfig,
    *,
    out_address_expr: str,
    value_expr: str,
) -> None:
    w.print(
        _scatter_accumulate_wgsl(
            kernel.operator,
            kernel.dtype,
            config.target_features,
            out_address_expr=out_address_expr,
            value_expr=value_expr,
        )
    )


def _emit_wgsl_for_scatter_kernel(
    w: "WgslWriter",
    kernel: IrScatterKernel,
    config: WgslKernelConfig,
    *,
    emit_bindings: Callable[["WgslWriter", IrScatterKernel, WgslKernelConfig], None],
    emit_store: Callable[
        ["WgslWriter", IrScatterKernel, WgslKernelConfig, str, str], None
    ],
) -> None:
    source_accessor = kernel.arg_accessors[0]
    source_shape = source_accessor.shape
    rank = len(source_shape)

    emit_bindings(w, kernel, config)
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
            value_expr = f"arg0[{_arg_address_expr(0, source_accessor, '')}]"
            emit_store(
                w,
                kernel,
                config,
                f"{kernel.woffset}u",
                value_expr,
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

            value_expr = f"arg0[{_arg_address_expr(0, source_accessor, 'src_index')}]"
            w.print(
                """
                let src_index = source_index(src_address);
                let out_address = scatter_out_address(src_index);
                """
            )
            emit_store(w, kernel, config, "out_address", value_expr)


def _emit_wgsl_for_scatter_clobber_kernel(
    w: "WgslWriter",
    kernel: IrScatterClobberKernel,
    config: WgslKernelConfig,
) -> None:
    def emit_bindings(
        w: "WgslWriter",
        kernel: IrScatterKernel,
        config: WgslKernelConfig,
    ) -> None:
        del config
        _emit_scatter_clobber_bindings(w, kernel)

    def emit_store(
        w: "WgslWriter",
        kernel: IrScatterKernel,
        config: WgslKernelConfig,
        out_address_expr: str,
        value_expr: str,
    ) -> None:
        del kernel, config
        _emit_scatter_clobber_store(
            w,
            out_address_expr=out_address_expr,
            value_expr=value_expr,
        )

    _emit_wgsl_for_scatter_kernel(
        w,
        kernel,
        config,
        emit_bindings=emit_bindings,
        emit_store=emit_store,
    )


def _emit_scatter_accumulate_bindings_only(
    w: "WgslWriter",
    kernel: IrScatterKernel,
    config: WgslKernelConfig,
) -> None:
    assert isinstance(kernel, IrScatterAccumulateKernel)
    _emit_scatter_accumulate_bindings(w, kernel, config.target_features)


def _emit_scatter_accumulate_store_only(
    w: "WgslWriter",
    kernel: IrScatterKernel,
    config: WgslKernelConfig,
    out_address_expr: str,
    value_expr: str,
) -> None:
    assert isinstance(kernel, IrScatterAccumulateKernel)
    _emit_scatter_accumulate_store(
        w,
        kernel,
        config,
        out_address_expr=out_address_expr,
        value_expr=value_expr,
    )


def _emit_wgsl_for_scatter_accumulate_kernel(
    w: "WgslWriter",
    kernel: IrScatterAccumulateKernel,
    config: WgslKernelConfig,
) -> None:
    _emit_wgsl_for_scatter_kernel(
        w,
        kernel,
        config,
        emit_bindings=_emit_scatter_accumulate_bindings_only,
        emit_store=_emit_scatter_accumulate_store_only,
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
    t = spell_dtype_in_wgsl(kernel.dtype)
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
