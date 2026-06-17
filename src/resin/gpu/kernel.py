__all__ = [
    "Kernel",
]

import math
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from .accessor import Accessor
from .rpn import ScalarRpnExpr
from .scalar import (
    ScalarType,
    spell_stype_in_wgsl,
)
from .shape import is_c_contiguous

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
    lg2_items_per_thread: int = 3
    workgroup_size: int = 8

    @abstractmethod
    def emit_wgsl(self) -> str:
        """
        Renders the shader into a WGSL string.
        """

    def dispatch_size(self) -> tuple[int, int, int]:
        """
        Returns `(workgroups_x, workgroups_y, workgroups_z)` for dispatching this kernel.

        Each thread handles ``2 ** lg2_items_per_thread`` output elements, and each
        workgroup contains ``workgroup_size`` threads along x.
        """
        n = math.prod(self.shape)
        if n == 0:
            return (0, 1, 1)

        items_per_thread = 1 << self.lg2_items_per_thread
        threads = (n + items_per_thread - 1) // items_per_thread
        workgroups_x = (threads + self.workgroup_size - 1) // self.workgroup_size
        return (workgroups_x, 1, 1)

    def _make_wgsl_writer(self) -> "WgslWriter":
        # Create a WgslWriter for emitting the shader:
        return WgslWriter(enable_f16=(self.stype == "f2"))

    def _emit_bindings(self, w: "WgslWriter") -> None:
        t = spell_stype_in_wgsl(self.stype)

        # Setting up bindings:
        # DEF: output at @group(0) @binding(0)
        w.print(
            f"""
            @group(0) @binding(0)
            var<storage, read_write> output: array<{t}>;
            """
        )

        # DEF: arg{i} at @group(1) @binding({i})
        for i in range(len(self.arg_accessors)):
            w.print(
                f"""
                @group(1) @binding({i})
                var<storage, read> arg{i}: array<{t}>;
                """
            )

    def _emit_arg_address_functions(self, w: "WgslWriter") -> None:
        # Define `address()` function for each argument:
        # DEF: address_arg{i}
        for i, arg_accessor in enumerate(self.arg_accessors):
            w.define_address_function(f"address_arg{i}", arg_accessor)

    def _emit_out_index_function(self, w: "WgslWriter", name: str = "index") -> None:
        # Define `index()` function for the output:
        # DEF: index()
        w.define_cc_index_function(name, Accessor.dense(self.shape))

    @contextmanager
    def _per_output_element(self, w: "WgslWriter") -> Iterator["WgslWriter"]:
        """
        Emit the compute entry point and loop over output elements handled by each
        thread. Yields while emitting the body for a single output element.

        Within the yielded block, the following WGSL names are in scope:

        DEF: out_address — flat C-contiguous address of the output element being written
        DEF: out_index — multidimensional output index decoded from `out_address`

        Also emits:

        DEF: main()
        DEF: index()
        """
        self._emit_out_index_function(w)

        items_per_thread = 1 << self.lg2_items_per_thread

        # Define the entry point for the kernel:
        # DEF: main()
        with w.block(
            f"""
            @compute @workgroup_size({self.workgroup_size})
            fn main(
                @builtin(global_invocation_id) global_id: vec3<u32>
            )
            """
        ):
            # Determine which output buffer address range we'll write to in this thread.
            # These are indices into the flattened C-contiguous output tensor.
            w.print(
                f"""
                let out_address_beg = global_id.x << {self.lg2_items_per_thread}u;
                let out_address_end = out_address_beg + {items_per_thread}u;
                """
            )

            # Iterate over the output addresses we're responsible for.
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
                    # Early out for threads that are out of bounds of the output buffer.
                    w.print("return;")

                # Map the output address to an index.
                # We can resolve these indices to addresses in sparse argument views.
                w.print(
                    """
                    let out_index = index(out_address);
                    """
                )

                yield w


@dataclass(frozen=True, kw_only=True)
class ElementwiseRpnKernel(Kernel):
    """
    Accepts a scalar expression in reverse polish notation and several bound argument
    buffers, all of the same shape and stype. Emits a single WGSL kernel for evaluating
    the expression.
    """

    rpn_expr: ScalarRpnExpr

    def __post_init__(self):
        assert all(x.shape == self.shape for x in self.arg_accessors)

    def emit_wgsl(self) -> str:
        t = spell_stype_in_wgsl(self.stype)
        n = len(self.arg_accessors)

        # Create a WgslWriter for emitting the shader:
        w = self._make_wgsl_writer()

        # Emit bindings and address functions for the arguments:
        # DEF: output at @group(0) @binding(0)
        # DEF: arg{i} at @group(1) @binding({i})
        # DEF: address_arg{i}()
        self._emit_bindings(w)
        self._emit_arg_address_functions(w)

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

        with self._per_output_element(w):
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


@dataclass(frozen=True, kw_only=True)
class MatmulKernel(Kernel):
    """
    Batched matrix multiply with sparse argument accessors.

    After shape joining, arg0 has shape ``batch + (m, k)``, arg1 has
    ``batch + (k, n)``, and output is ``batch + (m, n)``. Each output element is
    ``sum_k arg0[batch, m, k] * arg1[batch, k, n]``, with indices mapped through the
    argument accessors.
    """

    def __post_init__(self):
        assert len(self.arg_accessors) == 2
        a0, a1 = self.arg_accessors
        assert len(a0.shape) >= 2 and len(a1.shape) >= 2
        assert a0.shape[-1] == a1.shape[-2]
        assert a0.shape[:-2] == a1.shape[:-2]
        assert len(a0.shape) == len(a1.shape) == len(self.shape)
        assert self.shape == a0.shape[:-1] + (a1.shape[-1],)

    @property
    def k(self) -> int:
        return self.arg_accessors[0].shape[-1]

    def emit_wgsl(self) -> str:
        t = spell_stype_in_wgsl(self.stype)
        rank = len(self.shape)
        k = self.k

        # Create a WgslWriter for emitting the shader:
        w = self._make_wgsl_writer()

        # Emit bindings and address functions for the arguments:
        # DEF: output at @group(0) @binding(0)
        # DEF: arg{i} at @group(1) @binding({i})
        # DEF: address_arg{i}()
        self._emit_bindings(w)
        self._emit_arg_address_functions(w)

        # Define helpers that map (output index, k) to argument indices:
        # DEF: arg0_index()
        self._define_matmul_arg_index_function(
            w,
            "arg0_index",
            rank=rank,
            k_axis=rank - 1,
            m_axis=rank - 2,
        )
        # DEF: arg1_index()
        self._define_matmul_arg_index_function(
            w,
            "arg1_index",
            rank=rank,
            k_axis=rank - 2,
            n_axis=rank - 1,
        )

        with self._per_output_element(w):
            w.print(
                f"""
                var sum: {t} = {t}(0);
                """
            )

            # For each k along the contraction axis, compute the corresponding
            # argument indices, load the values, and accumulate their product.
            with w.block(f"for (var ki: u32 = 0u; ki < {k}u; ki += 1u)"):
                w.print(
                    """
                    let a_idx = arg0_index(out_index, ki);
                    let b_idx = arg1_index(out_index, ki);
                    sum += arg0[address_arg0(a_idx)] * arg1[address_arg1(b_idx)];
                    """
                )

            # Write the accumulated dot product to the output buffer.
            w.print("output[out_address] = sum;")

        return w.finish()

    def _define_matmul_arg_index_function(
        self,
        w: "WgslWriter",
        name: str,
        *,
        rank: int,
        k_axis: int,
        m_axis: int | None = None,
        n_axis: int | None = None,
    ) -> None:
        """
        Emit a helper that maps an output index and contraction index `k` to an argument
        index. Batch dimensions (all axes before the final two) are copied from the
        output index; the trailing two axes are filled per `m_axis`/`n_axis`/`k_axis`.
        """
        with w.block(
            f"""
            fn {name}(out_index: array<u32, {rank}>, k: u32) -> array<u32, {rank}>
            """
        ):
            w.print(f"var idx: array<u32, {rank}>;")

            # Copy batch dimensions from the output index:
            for i in range(rank - 2):
                w.print(f"idx[{i}] = out_index[{i}];")

            # M-axis:
            if m_axis is not None:
                w.print(f"idx[{m_axis}] = out_index[{m_axis}];")

            # N-axis:
            if n_axis is not None:
                w.print(f"idx[{n_axis}] = out_index[{n_axis}];")

            # K-axis:
            w.print(f"idx[{k_axis}] = k;")

            # Done
            w.print("return idx;")


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
