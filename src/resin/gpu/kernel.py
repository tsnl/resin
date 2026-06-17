__all__ = [
    "Kernel",
]

import textwrap
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from .accessor import Accessor
from .rpn import ScalarRpnExpr
from .scalar import (
    ScalarType,
    spell_stype_in_wgsl,
)
from .shape import c_contiguous_pitch_for_shape, is_c_contiguous

#
# Kernel
#


@dataclass(frozen=True, kw_only=True)
class Kernel(ABC):
    """
    Kernels are GPU programs that can be dispatched for evaluation.

    They can be reified as WGSL for dispatch or analyzed and replaced during
    optimizations like kernel fusion.
    """

    arg_accessors: tuple[Accessor, ...]
    stype: ScalarType
    shape: tuple[int, ...]

    @abstractmethod
    def emit_wgsl(self) -> str:
        """
        Renders the shader into a WGSL string.
        """


@dataclass(frozen=True, kw_only=True)
class ElementwiseRpnKernel(Kernel):
    """
    Accepts a scalar expression in reverse polish notation and several bound argument
    buffers, all of the same shape and stype. Emits a single WGSL kernel for evaluating
    the expression.
    """

    rpn_expr: ScalarRpnExpr
    lg2_items_per_thread: int = 3

    def __post_init__(self):
        assert all(x.shape == self.shape for x in self.arg_accessors)

    def emit_wgsl(self) -> str:
        t = spell_stype_in_wgsl(self.stype)
        n = len(self.arg_accessors)

        # Create a WgslWriter for emitting the shader:
        w = WgslWriter(
            enable_f16=(self.stype == "f2"),
        )

        # Setting up bindings:
        # DEF: output at @group(0) @binding(0)
        # DEF: arg{i} at @group(1) @binding({i})
        w.print(
            f"""
            @group(0) @binding(0)
            var<storage, read_write> output: array<{t}>;
            """
        )
        for i in range(n):
            w.print(
                f"""
                @group(1) @binding({i})
                var<storage, read> arg{i}: array<{t}>;
                """,
            )

        # Define a function for evaluating the RPN expression:
        # DEF: eval_rpn_expr()
        with w.block(
            f"""
            fn eval_rpn_expr({",".join(f"a{i}: {t}" for i in range(n))}) -> {t}
            """
        ):
            expr_stack = []
            for token in self.rpn_expr.string:
                match token:
                    # Operand:
                    case int():
                        expr_stack.append(f"a{token}")
                    # Unary operator:
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
                    # Binary operator:
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
                    # TODO: binary compare operators
                    case _:
                        raise NotImplementedError()

            assert len(expr_stack) == 1
            w.print(f"return {expr_stack[0]};")

        # Define `address()` function for each argument:
        # DEF: address_arg{i}
        for i, arg_accessor in enumerate(self.arg_accessors):
            w.define_address_function(f"address_arg{i}", arg_accessor)

        # Define `index()` function just the output:
        # DEF: index()
        w.define_cc_index_function(
            "index",
            Accessor(
                offset=0,
                shape=self.shape,
                pitch=c_contiguous_pitch_for_shape(self.shape),
            ),
        )

        # Define the entry point for the kernel:
        # DEF: main()
        with w.block(
            f"""
            @compute @workgroup_size({self.lg2_items_per_thread})
            fn main(
                @builtin(global_invocation_id) global_id: vec3<u32>
            )
            """
        ):
            # Determine which output buffer address range we'll write to in this thread.
            # These are indices into the flattened C-contiguous output tensor.
            w.print(
                f"""
                let out_address_beg = global_id.x << {self.lg2_items_per_thread};
                let out_address_end = out_address_beg + {1 << self.lg2_items_per_thread};
                """
            )

            # Iterate over the output addresses we're responsible for, compute the
            # corresponding input addresses, load the inputs, and call the RPN function.
            with w.block(
                """
                for (
                    var out_address = out_address_beg;
                    out_address < out_address_end;
                    out_address += 1
                )
                """
            ):
                with w.block("if (out_address >= arrayLength(&output))"):
                    # Early out for threads that are out of bounds of the output buffer.
                    w.print("return;")

                # Map the output address to an index.
                # We can resolve these indices to addresses in sparse argument views.
                w.print(
                    """
                    let out_index = index(out_address);
                    """
                )

                # For each argument, compute the corresponding address for this output
                # index. Load the value.
                # TODO: more efficient to fetch all values for a specific argument array
                # at once.
                arg_addresses = [f"address_arg{i}(out_index)" for i in range(n)]
                arg_values = [f"arg{i}[{arg_addresses[i]}]" for i in range(n)]

                # Evaluate the RPN expression for this output element and write it to
                # the output buffer.
                w.print(
                    f"""
                    output[out_address] = eval_rpn_expr({", ".join(arg_values)});
                    """
                )

        return w.finish()


class MatmulKernel(Kernel):
    """
    A kernel for performing matrix multiplication on 2D tensors.

    arg0 is the left-hand-side matrix, arg1 is the right-hand-side matrix, and output
    is the result. The kernel assumes that all matrices are in row-major order.
    """

    def emit_wgsl(self) -> str:
        raise NotImplementedError()


#
# WgslWriter
#


class WgslWriter:
    lines: list[str]

    def __init__(
        self,
        *,
        enable_f16: bool,
    ) -> None:
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

    def define_address_function(self, name: str, accessor: Accessor):
        n = len(accessor.shape)

        with self.block(
            f"""
            fn {name}(index: array<u32, {n}>) -> u32
            """
        ):
            self.print(f"var acc: u32 = {accessor.offset};")
            for i in range(n):
                self.print(
                    f"""
                    acc += index[{i}] * {accessor.pitch[i]};  // dim {i}
                    """
                )
            self.print("return acc;")

    def define_cc_index_function(self, name: str, cc_accessor: Accessor):
        assert is_c_contiguous(cc_accessor.shape, cc_accessor.pitch)

        n = len(cc_accessor.shape)

        with self.block(
            f"""
            fn {name}(address: u32) -> array<u32, {n}>
            """
        ):
            self.print(f"var index: array<u32, {n}>;")
            for i in range(n):
                self.print(
                    f"""
                    index[{i}] = address / {cc_accessor.pitch[i]};  // dim {i}
                    address = address % {cc_accessor.pitch[i]};
                    """
                )
            self.print("return index;")


def address(index: tuple[int, ...], pitch: tuple[int, ...]) -> int:
    return sum(i * p for i, p in zip(index, pitch))


def index(
    address: int,
    shape: tuple[int, ...],
    pitch: tuple[int, ...],
) -> tuple[int, ...]:
    if not is_c_contiguous(shape, pitch):
        raise ValueError()

    index = []
    for p in pitch:
        i, address = divmod(address, p)
        index.append(i)
    return tuple(index)


def iter_index(shape: tuple[int, ...]) -> Generator[tuple[int, ...], None, None]:
    if not shape:
        yield ()
        return

    for i in range(shape[0]):
        for sub_index in iter_index(shape[1:]):
            yield (i,) + sub_index
