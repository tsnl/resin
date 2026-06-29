import math
import textwrap
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import assert_never

from resin.core.accessor import Accessor, c_contiguous_pitch_for_shape, is_c_contiguous
from resin.dsl.node import RemapGatherInfo, RemapScatterInfo
from resin.core.etype import (
    BinaryAssocElementOperator,
    ElementType,
    etype_needs_enable_f16,
    spell_etype_in_wgsl,
)
from resin.ir.rpn import ElementRpnExpr
from resin.ir.ir import (
    IrElementwiseRpnKernel,
    IrKernel,
    IrMatmulKernel,
    IrPrefixSumKernel,
    IrReductionKernel,
    IrRemapKernel,
)

__all__ = [
    "AbstractKernelException",
    "WgslKernelConfig",
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
        case IrPrefixSumKernel():
            # Single-threaded for correctness on small/medium vectors.
            return (1, 1, 1) if math.prod(kernel.shape) > 0 else (0, 1, 1)
        case IrRemapKernel(info=info) if isinstance(info, RemapScatterInfo):
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
    enable_f16 = etype_needs_enable_f16(kernel.etype)
    w = WgslWriter(enable_f16=enable_f16)

    match kernel:
        case IrElementwiseRpnKernel():
            _emit_wgsl_for_elementwise_rpn_kernel(w, kernel, config)
        case IrMatmulKernel():
            _emit_wgsl_for_matmul_kernel(w, kernel, config)
        case IrReductionKernel():
            _emit_wgsl_for_reduction_kernel(w, kernel, config)
        case IrRemapKernel():
            _emit_wgsl_for_remap_kernel(w, kernel, config)
        case IrPrefixSumKernel():
            _emit_wgsl_for_prefix_sum_kernel(w, kernel)
        case _:
            raise AbstractKernelException(f"Unsupported kernel type: {type(kernel)}")

    return w.finish()


class AbstractKernelException(Exception):
    """Raised when a kernel cannot be compiled to WGSL."""


@dataclass(frozen=True, kw_only=True)
class WgslKernelConfig:
    lg2_items_per_thread: int = 3
    workgroup_size: int = 8


def _arg_etypes_for_kernel(kernel: IrKernel) -> tuple[ElementType, ...]:
    match kernel:
        case IrElementwiseRpnKernel() if kernel.arg_etypes:
            return kernel.arg_etypes
        case IrRemapKernel() | IrPrefixSumKernel():
            return kernel.arg_etypes
        case _:
            return tuple(kernel.etype for _ in kernel.arg_accessors)


def _emit_bindings(w: "WgslWriter", kernel: IrKernel) -> None:
    t = spell_etype_in_wgsl(kernel.etype)
    num_outputs = kernel.num_outputs

    if num_outputs == 1:
        w.print(
            f"""
            @group(0) @binding(0)
            var<storage, read_write> output: array<{t}>;
            """
        )
    else:
        for i in range(num_outputs):
            w.print(
                f"""
                @group(0) @binding({i})
                var<storage, read_write> output{i}: array<{t}>;
                """
            )

    for i, arg_etype in enumerate(_arg_etypes_for_kernel(kernel)):
        arg_t = spell_etype_in_wgsl(arg_etype)
        w.print(
            f"""
            @group(0) @binding({num_outputs + i})
            var<storage, read> arg{i}: array<{arg_t}>;
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
    arg_etypes = _arg_etypes_for_kernel(kernel)
    _emit_eval_rpn_expr(
        w,
        kernel.rpn_expr,
        n=n,
        etype=kernel.etype,
        arg_etypes=arg_etypes,
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
    etype: ElementType,
) -> None:
    t = spell_etype_in_wgsl(etype)
    if arg_etypes is None or len(arg_etypes) != n:
        arg_etypes = tuple(etype for _ in range(n))

    with w.block(
        f"""
        fn eval_rpn_expr({",".join(f"a{i}: {spell_etype_in_wgsl(arg_etypes[i])}" for i in range(n))}) -> {t}
        """
    ):
        expr_stack: list[str] = []
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
                case "floor":
                    operand = expr_stack.pop()
                    expr_stack.append(f"floor({operand})")
                case "ceil":
                    operand = expr_stack.pop()
                    expr_stack.append(f"ceil({operand})")
                case "bitcast":
                    operand = expr_stack.pop()
                    expr_stack.append(f"bitcast<{t}>({operand})")
                case "band":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"({lhs} & {rhs})")
                case "bor":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"({lhs} | {rhs})")
                case "bxor":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"({lhs} ^ {rhs})")
                case "shl":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"({lhs} << {rhs})")
                case "shr":
                    rhs = expr_stack.pop()
                    lhs = expr_stack.pop()
                    expr_stack.append(f"({lhs} >> {rhs})")
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
                    assert_never(token)

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
    t = spell_etype_in_wgsl(kernel.etype)
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
# Emit WGSL for IrRemapKernel
#


def _emit_remap_bindings(
    w: "WgslWriter",
    *,
    output_etype: ElementType,
    arg_etypes: tuple[ElementType, ...],
    operator: BinaryAssocElementOperator | None,
) -> None:
    t = spell_etype_in_wgsl(output_etype)
    if operator is None:
        w.print(
            f"""
            @group(0) @binding(0)
            var<storage, read_write> output: array<{t}>;
            """
        )
    else:
        w.print(
            """
            @group(0) @binding(0)
            var<storage, read_write> output: array<atomic<u32>>;
            """
        )
    for i, arg_etype in enumerate(arg_etypes):
        arg_t = spell_etype_in_wgsl(arg_etype)
        w.print(
            f"""
            @group(0) @binding({i + 1})
            var<storage, read> arg{i}: array<{arg_t}>;
            """
        )


def _scatter_atomic_accumulate_wgsl(
    operator: BinaryAssocElementOperator,
    *,
    out_address_expr: str,
    value_expr: str,
) -> str:
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
            assert_never(operator)

    return textwrap.dedent(
        f"""
        {{
            let out_slot = &output[{out_address_expr}];
            loop {{
                let old_bits = atomicLoad(out_slot);
                let old_val = bitcast<f32>(old_bits);
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


def _emit_remap_store(
    w: "WgslWriter",
    *,
    operator: BinaryAssocElementOperator | None,
    out_address_expr: str,
    value_expr: str,
) -> None:
    if operator is None:
        w.print(f"output[{out_address_expr}] = {value_expr};")
    else:
        w.print(
            _scatter_atomic_accumulate_wgsl(
                operator,
                out_address_expr=out_address_expr,
                value_expr=value_expr,
            )
        )


def _define_out_address_function(
    w: "WgslWriter",
    *,
    name: str,
    woffset: int,
    pitch: tuple[int, ...],
) -> None:
    rank = len(pitch)
    with w.block(
        f"""
        fn {name}(index: array<u32, {rank}>) -> u32
        """
    ):
        w.print(f"var acc: u32 = {woffset}u;")
        for i in range(rank):
            w.print(f"acc += index[{i}] * {pitch[i]}u;")
        w.print("return acc;")


def _define_indices_at_function(
    w: "WgslWriter",
    *,
    source_rank: int,
    out_rank: int,
    indices_accessor: Accessor,
) -> None:
    indices_rank = len(indices_accessor.shape)
    with w.block(
        f"""
        fn indices_at(source_index: array<u32, {source_rank}>) -> array<u32, {out_rank}>
        """
    ):
        w.print(f"var result: array<u32, {out_rank}>;")
        with w.block(f"for (var k: u32 = 0u; k < {out_rank}u; k += 1u)"):
            if indices_rank == 0:
                w.print(f"result[k] = arg1[{_arg_address_expr(1, indices_accessor, '')}];")
            else:
                w.print(f"var idx: array<u32, {indices_rank}>;")
                for d in range(source_rank):
                    w.print(f"idx[{d}] = source_index[{d}];")
                w.print(f"idx[{source_rank}] = k;")
                w.print(
                    f"result[k] = arg1[{_arg_address_expr(1, indices_accessor, 'idx')}];"
                )
        w.print("return result;")


def _emit_wgsl_for_source_driven_remap(
    w: "WgslWriter",
    kernel: IrRemapKernel,
    config: WgslKernelConfig,
    *,
    operator: BinaryAssocElementOperator | None,
    out_address_expr_for_source_index: Callable[[str], str],
    value_expr_for_source_index: Callable[[str], str] | None = None,
) -> None:
    source_accessor = kernel.arg_accessors[0]
    source_shape = source_accessor.shape
    rank = len(source_shape)

    _emit_remap_bindings(
        w,
        output_etype=kernel.etype,
        arg_etypes=kernel.arg_etypes,
        operator=operator,
    )
    _emit_arg_address_functions(w, kernel)

    if value_expr_for_source_index is None:
        value_expr_for_source_index = lambda source_index_expr: (
            f"arg0[{_arg_address_expr(0, source_accessor, source_index_expr)}]"
        )

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
            value_expr = value_expr_for_source_index("")
            _emit_remap_store(
                w,
                operator=operator,
                out_address_expr=out_address_expr_for_source_index(""),
                value_expr=value_expr,
            )
        return

    _define_cc_index_function(w, "source_index", Accessor.dense(source_shape))

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

            w.print("let src_index = source_index(src_address);")
            value_expr = value_expr_for_source_index("src_index")
            _emit_remap_store(
                w,
                operator=operator,
                out_address_expr=out_address_expr_for_source_index("src_index"),
                value_expr=value_expr,
            )


def _emit_wgsl_for_remap_kernel(
    w: "WgslWriter",
    kernel: IrRemapKernel,
    config: WgslKernelConfig,
) -> None:
    match kernel.info:
        case RemapScatterInfo(accessor=accessor, operator=operator) if accessor is not None:
            if accessor.pitch:
                _define_out_address_function(
                    w,
                    name="scatter_out_address",
                    woffset=accessor.offset,
                    pitch=accessor.pitch,
                )

            def out_address_expr(source_index_expr: str) -> str:
                if not kernel.arg_accessors[0].shape:
                    return f"{accessor.offset}u"
                return f"scatter_out_address({source_index_expr})"

            _emit_wgsl_for_source_driven_remap(
                w,
                kernel,
                config,
                operator=operator,
                out_address_expr_for_source_index=out_address_expr,
            )
        case RemapScatterInfo(accessor=None, operator=operator):
            source_accessor, indices_accessor = kernel.arg_accessors
            out_rank = len(kernel.shape)
            out_pitch = c_contiguous_pitch_for_shape(kernel.shape)
            _define_indices_at_function(
                w,
                source_rank=len(source_accessor.shape),
                out_rank=out_rank,
                indices_accessor=indices_accessor,
            )
            _define_out_address_function(
                w,
                name="out_address_from_indices",
                woffset=0,
                pitch=out_pitch,
            )

            def out_address_expr(source_index_expr: str) -> str:
                if not source_accessor.shape:
                    return "out_address_from_indices(indices_at(array<u32, 0>()))"
                return f"out_address_from_indices(indices_at({source_index_expr}))"

            _emit_wgsl_for_source_driven_remap(
                w,
                kernel,
                config,
                operator=operator,
                out_address_expr_for_source_index=out_address_expr,
            )
        case RemapGatherInfo(accessor=None):
            _emit_remap_bindings(
                w,
                output_etype=kernel.etype,
                arg_etypes=kernel.arg_etypes,
                operator=None,
            )
            _emit_arg_address_functions(w, kernel)
            with _per_output_element(w, kernel, config) as (w, out_addr):
                w.print(
                    f"""
                    output[{out_addr}] = arg0[{_arg_address_expr(0, kernel.arg_accessors[0], "out_index")}];
                    """
                )
        case RemapGatherInfo(accessor=accessor) if accessor is not None:
            _, indices_accessor = kernel.arg_accessors
            source_rank = len(kernel.shape)
            out_rank = accessor.rank
            _emit_remap_bindings(
                w,
                output_etype=kernel.etype,
                arg_etypes=kernel.arg_etypes,
                operator=None,
            )
            _emit_arg_address_functions(w, kernel)
            _define_indices_at_function(
                w,
                source_rank=source_rank,
                out_rank=out_rank,
                indices_accessor=indices_accessor,
            )
            _define_out_address_function(
                w,
                name="out_address_from_indices",
                woffset=accessor.offset,
                pitch=accessor.pitch,
            )
            with _per_output_element(w, kernel, config) as (w, out_addr):
                if source_rank == 0:
                    indices_expr = "indices_at(array<u32, 0>())"
                else:
                    indices_expr = "indices_at(out_index)"
                source_address_expr = f"out_address_from_indices({indices_expr})"
                w.print(f"output[{out_addr}] = arg0[{source_address_expr}];")
        case _:
            raise AbstractKernelException(f"Unsupported RemapInfo: {kernel.info!r}")


#
# Emit WGSL for IrReductionKernel
#


def _emit_wgsl_for_reduction_kernel(
    w: "WgslWriter",
    kernel: IrReductionKernel,
    config: WgslKernelConfig,
) -> None:
    t = spell_etype_in_wgsl(kernel.etype)
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
    operator: BinaryAssocElementOperator,
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
            assert_never(operator)


#
# Emit WGSL for IrPrefixSumKernel / IrSortKernel
#


def _emit_wgsl_for_prefix_sum_kernel(
    w: "WgslWriter",
    kernel: IrPrefixSumKernel,
) -> None:
    _emit_bindings(w, kernel)
    n = math.prod(kernel.shape)
    t = spell_etype_in_wgsl(kernel.etype)
    inclusive = kernel.inclusive
    with w.block("@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>)"):
        w.print("if (gid.x != 0u) { return; }")
        w.print(f"var acc: {t} = {t}(0);")
        with w.block(f"for (var i: u32 = 0u; i < {n}u; i++)"):
            if inclusive:
                w.print("acc = acc + arg0[i];")
                w.print("output[i] = acc;")
            else:
                w.print("output[i] = acc;")
                w.print("acc = acc + arg0[i];")




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
