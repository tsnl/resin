__all__ = [
    "AbstractKernelException",
    "GpuInterp",
    "GpuInterpBuilder",
    "WgslBuilder",
]

import math
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import wgpu

from .accessor import Accessor, is_c_contiguous
from .ir import (
    IrBuffer,
    IrElementwiseRpnKernel,
    IrKernel,
    IrMatmulKernel,
    IrProgram,
)
from .rpn import ScalarRpnExpr
from .scalar import ScalarType, spell_stype_in_wgsl, stype_nbytes


class AbstractKernelException(Exception):
    """Raised when a kernel cannot be compiled to WGSL."""


@dataclass(frozen=True, kw_only=True)
class GpuInterp:
    device: wgpu.GPUDevice
    program: IrProgram
    buffers: tuple[wgpu.GPUBuffer, ...]
    kernels: tuple[wgpu.GPUComputePipeline, ...]


class GpuInterpBuilder:
    device: wgpu.GPUDevice
    buffer_memo: dict[IrBuffer, wgpu.GPUBuffer]
    kernel_memo: dict[IrKernel, wgpu.GPUComputePipeline]
    wgsl_builder: WgslBuilder

    def __init__(
        self, device: wgpu.GPUDevice, *, wgsl_builder: WgslBuilder | None = None
    ):
        super().__init__()
        self.device = device
        self.buffer_memo = {}
        self.kernel_memo = {}
        self.wgsl_builder = wgsl_builder or WgslBuilder()

    def _build_buffer(self, buffer: IrBuffer) -> wgpu.GPUBuffer:
        if wgpu_buffer := self.buffer_memo.get(buffer):
            return wgpu_buffer

        wgpu_buffer = self.device.create_buffer(
            size=math.prod(buffer.shape) * stype_nbytes(buffer.stype),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            mapped_at_creation=bool(buffer.init),
        )

        if buffer.init is not None:
            self.device.queue.write_buffer(wgpu_buffer, 0, buffer.init)
            wgpu_buffer.unmap()

        self.buffer_memo[buffer] = wgpu_buffer
        return wgpu_buffer

    def _build_kernel(self, kernel: IrKernel) -> wgpu.GPUComputePipeline:
        if wgpu_kernel := self.kernel_memo.get(kernel):
            return wgpu_kernel

        wgsl = self.wgsl_builder.build(kernel)
        shader_module = self.device.create_shader_module(code=wgsl)
        wgpu_kernel = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader_module, "entry_point": "main"},
        )

        self.kernel_memo[kernel] = wgpu_kernel
        return wgpu_kernel


@dataclass(frozen=True, kw_only=True)
class WgslBuilder:
    lg2_items_per_thread: int = 3
    workgroup_size: int = 8

    def build(self, kernel: IrKernel) -> str:
        match kernel:
            case IrElementwiseRpnKernel():
                return self._emit_elementwise_rpn(kernel)
            case IrMatmulKernel():
                return self._emit_matmul(kernel)
            case _:
                raise AbstractKernelException(
                    f"Unsupported kernel type: {type(kernel)}"
                )

    def dispatch_size(self, kernel: IrKernel) -> tuple[int, int, int]:
        n = math.prod(kernel.shape)
        if n == 0:
            return (0, 1, 1)

        items_per_thread = 1 << self.lg2_items_per_thread
        threads = (n + items_per_thread - 1) // items_per_thread
        workgroups_x = (threads + self.workgroup_size - 1) // self.workgroup_size
        return (workgroups_x, 1, 1)

    def _make_wgsl_writer(self, stype: ScalarType) -> "_WgslWriter":
        return _WgslWriter(enable_f16=(stype == "f2"))

    def _emit_bindings(self, w: "_WgslWriter", kernel: IrKernel) -> None:
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

    def _emit_arg_address_functions(self, w: "_WgslWriter", kernel: IrKernel) -> None:
        for i, arg_accessor in enumerate(kernel.arg_accessors):
            w.define_address_function(f"address_arg{i}", arg_accessor)

    def _emit_out_index_function(
        self,
        w: "_WgslWriter",
        kernel: IrKernel,
        name: str = "index",
    ) -> None:
        w.define_cc_index_function(name, Accessor.dense(kernel.shape))

    @contextmanager
    def _per_output_element(
        self, w: "_WgslWriter", kernel: IrKernel
    ) -> Iterator["_WgslWriter"]:
        self._emit_out_index_function(w, kernel)

        items_per_thread = 1 << self.lg2_items_per_thread

        with w.block(
            f"""
            @compute @workgroup_size({self.workgroup_size})
            fn main(
                @builtin(global_invocation_id) global_id: vec3<u32>
            )
            """
        ):
            w.print(
                f"""
                let out_address_beg = global_id.x << {self.lg2_items_per_thread}u;
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

                yield w

    def _emit_elementwise_rpn(self, kernel: IrElementwiseRpnKernel) -> str:
        n = len(kernel.arg_accessors)

        w = self._make_wgsl_writer(kernel.stype)
        self._emit_bindings(w, kernel)
        self._emit_arg_address_functions(w, kernel)
        self._emit_eval_rpn_expr(w, kernel.rpn_expr, n=n, stype=kernel.stype)

        with self._per_output_element(w, kernel):
            arg_addresses = [f"address_arg{i}(out_index)" for i in range(n)]
            arg_values = [f"arg{i}[{arg_addresses[i]}]" for i in range(n)]
            w.print(
                f"""
                output[out_address] = eval_rpn_expr({", ".join(arg_values)});
                """
            )

        return w.finish()

    def _emit_eval_rpn_expr(
        self,
        w: "_WgslWriter",
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
                    case _:
                        raise NotImplementedError()

            assert len(expr_stack) == 1
            w.print(f"return {expr_stack[0]};")

    def _emit_matmul(self, kernel: IrMatmulKernel) -> str:
        t = spell_stype_in_wgsl(kernel.stype)
        rank = len(kernel.shape)
        k = kernel.k

        w = self._make_wgsl_writer(kernel.stype)
        self._emit_bindings(w, kernel)
        self._emit_arg_address_functions(w, kernel)

        self._define_matmul_arg_index_function(
            w,
            "arg0_index",
            rank=rank,
            k_axis=rank - 1,
            m_axis=rank - 2,
        )
        self._define_matmul_arg_index_function(
            w,
            "arg1_index",
            rank=rank,
            k_axis=rank - 2,
            n_axis=rank - 1,
        )

        with self._per_output_element(w, kernel):
            w.print(
                f"""
                var sum: {t} = {t}(0);
                """
            )

            with w.block(f"for (var ki: u32 = 0u; ki < {k}u; ki += 1u)"):
                w.print(
                    """
                    let a_idx = arg0_index(out_index, ki);
                    let b_idx = arg1_index(out_index, ki);
                    sum += arg0[address_arg0(a_idx)] * arg1[address_arg1(b_idx)];
                    """
                )

            w.print("output[out_address] = sum;")

        return w.finish()

    def _define_matmul_arg_index_function(
        self,
        w: "_WgslWriter",
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


class _WgslWriter:
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
