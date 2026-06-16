import math
from dataclasses import dataclass
from typing import cast

from frozendict import frozendict

from . import front
from .kernel import (
    ElementwiseBinaryKernel,
    ElementwiseUnaryKernel,
    Kernel,
    MatmulKernel,
)
from .pytree import marshall_pytensor
from .scalar import (
    BinaryCompareOperator,
    BinaryScalarOperator,
    ScalarType,
    UnaryScalarOperator,
)

#
# Program
#


@dataclass(frozen=True, kw_only=True)
class Program:
    sinks: frozendict[str, BufferView]
    queue: tuple[Dispatch, ...]
    buffers: tuple[Buffer, ...]
    buffer_views: tuple[BufferView, ...]


@dataclass(frozen=True, kw_only=True, eq=False)
class Buffer:
    shape: tuple[int, ...]
    stype: ScalarType
    init: bytes | None = None
    readonly: bool


@dataclass(frozen=True, kw_only=True)
class BufferView:
    buffer: Buffer
    offset: int
    shape: tuple[int, ...]
    pitch: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class Dispatch:
    kernel: Kernel
    args: tuple[BufferView, ...]
    output: Buffer


#
# ProgramBuilder
#


class ProgramBuilder:
    buffer_memo: dict[front.Node, Buffer]
    buffer_view_memo: dict[front.View, BufferView]
    queue: list[Dispatch]
    sinks: dict[str, BufferView]

    def __init__(self):
        super().__init__()

        self.buffer_memo = {}
        self.buffer_view_memo = {}
        self.queue = []
        self.sinks = {}

    def finish(self) -> Program:
        return Program(
            sinks=frozendict(self.sinks),
            queue=tuple(self.queue),
            buffers=tuple(self.buffer_memo.values()),
            buffer_views=tuple(self.buffer_view_memo.values()),
        )

    def build_sink(self, name: str, view: front.View):
        if name in self.sinks:
            raise ValueError(f"Sink name conflict: {name}")

        # TODO: Verify acyclicity front graph

        self.sinks[name] = self._build_view(view)

    def _build_view(self, view: front.View) -> BufferView:
        if bv := self.buffer_view_memo.get(view):
            return bv

        buffer = self._build_node(view.node)

        bv = BufferView(
            buffer=buffer,
            offset=view.offset,
            shape=view.shape,
            pitch=view.pitch,
        )
        self.buffer_view_memo[view] = bv

        return bv

    def _build_node(self, node: front.Node) -> Buffer:
        # First check memoization.
        if b := self.buffer_memo.get(node):
            return b

        # Allocate output buffer.
        output_buffer = self._allocate_buffer_for_node(node)

        # Compute input buffer views.
        input_buffer_views = tuple(self._build_view(view) for view in node.args)

        # Generate kernel for this node.
        # If no kernel is needed (e.g. for ConstNode), this will return None, and the
        # KernelDispatch will be a no-op.
        kernel = self._build_kernel_for_node(node)

        # Record the kernel dispatch using the above output, input, and kernel iff the
        # kernel is not None.
        if kernel:
            dispatch = Dispatch(
                kernel=kernel,
                args=input_buffer_views,
                output=output_buffer,
            )
            self.queue.append(dispatch)

        # Memoize and return the output buffer.
        self.buffer_memo[node] = output_buffer
        return output_buffer

    def _allocate_buffer_for_node(self, node: front.Node) -> Buffer:
        # ConstNode is the only special case, everything else is just an uninitialized
        # writable buffer.
        is_const_node = isinstance(node, front.ConstNode)
        return Buffer(
            shape=node.shape,
            stype=node.stype,
            init=(
                marshall_pytensor(node.value, stype=node.stype)
                if is_const_node
                else None
            ),
            readonly=is_const_node,
        )

    def _build_kernel_for_node(self, node: front.Node) -> Kernel | None:
        match node:
            case front.ConstNode():
                return None  # ConstNode is a special case that doesn't require a kernel
            case front.ElementwiseNode():
                return self._build_kernel_for_elementwise_node(node)
            case front.MatmulNode():
                return self._build_kernel_for_matmul_node(node)
            case _:
                raise NotImplementedError(f"Unsupported node type: {type(node)}")

    def _build_kernel_for_elementwise_node(self, node: front.ElementwiseNode) -> Kernel:
        assert len(node.args) in (1, 2)

        n = math.prod(node.shape)

        match len(node.args):
            case 1:
                return ElementwiseUnaryKernel(
                    selected_uop=cast(
                        UnaryScalarOperator,
                        node.operator,
                    ),
                    n=n,
                    t=node.stype,
                )
            case 2:
                return ElementwiseBinaryKernel(
                    selected_bop=cast(
                        BinaryScalarOperator | BinaryCompareOperator,
                        node.operator,
                    ),
                    n=n,
                    t=node.stype,
                )
            case _:
                raise NotImplementedError()

    def _build_kernel_for_matmul_node(self, node: front.MatmulNode) -> Kernel:
        return MatmulKernel(
            m=node.shape[0],
            n=node.shape[1],
            k=node.shape[2],
            t=node.stype,
        )
