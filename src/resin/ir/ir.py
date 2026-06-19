__all__ = [
    "IrBuffer",
    "IrBufferView",
    "IrDispatch",
    "IrElementwiseRpnKernel",
    "IrKernel",
    "IrMatmulKernel",
    "IrProgram",
    "IrProgramBuilder",
    "IrReductionKernel",
    "IrScatterAccumulateKernel",
    "IrScatterClobberKernel",
    "IrScatterKernel",
]

from abc import ABC
from dataclasses import dataclass

from frozendict import frozendict

from resin.core.accessor import Accessor
from resin.core.pytree import marshall_pytensor
from .rpn import ElementRpnExpr
from resin.core.dtype import (
    BinaryAssocScalarOperator,
    BinaryCompareOperator,
    BinaryScalarOperator,
    DType,
    ScalarOperator,
    UnaryScalarOperator,
)
from resin.dsl import dsl

#
# IrProgram
#


@dataclass(frozen=True, kw_only=True)
class IrProgram:
    # Maps Python object id of each ParamNode to its buffer index in `buffers`.
    param_buffer_ids: frozendict[int, int]
    sinks: frozendict[str, IrBufferView]
    queue: tuple[IrDispatch, ...]
    buffers: tuple[IrBuffer, ...]
    buffer_views: tuple[IrBufferView, ...]


@dataclass(frozen=True, kw_only=True, eq=False)
class IrBuffer:
    shape: tuple[int, ...]
    dtype: DType
    init: bytes | None = None
    readonly: bool


@dataclass(frozen=True, kw_only=True)
class IrBufferView:
    buffer: IrBuffer
    accessor: Accessor


@dataclass(frozen=True, kw_only=True)
class IrDispatch:
    kernel: IrKernel
    args: tuple[IrBufferView, ...]
    output: IrBuffer


@dataclass(frozen=True, kw_only=True)
class IrKernel(ABC):
    """Base kernel. ``dtype`` is the output buffer element type.

    All current kernels share one dtype for inputs and output. Tile
    kernels may need per-operand types or epilogue-specific types later.
    """

    arg_accessors: tuple[Accessor, ...]
    dtype: DType
    shape: tuple[int, ...]
    clear_output_before_dispatch: bool = False


@dataclass(frozen=True, kw_only=True)
class IrElementwiseRpnKernel(IrKernel):
    rpn_expr: ElementRpnExpr

    def __post_init__(self):
        assert all(x.shape == self.shape for x in self.arg_accessors)


@dataclass(frozen=True, kw_only=True)
class IrMatmulKernel(IrKernel):
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


@dataclass(frozen=True, kw_only=True)
class IrScatterKernel(IrKernel):
    woffset: int
    wpitch: tuple[int, ...]
    clear_output_before_dispatch: bool = True

    def __post_init__(self):
        assert len(self.arg_accessors) == 1


@dataclass(frozen=True, kw_only=True)
class IrScatterClobberKernel(IrScatterKernel):
    pass


@dataclass(frozen=True, kw_only=True)
class IrScatterAccumulateKernel(IrScatterKernel):
    operator: BinaryAssocScalarOperator


@dataclass(frozen=True, kw_only=True)
class IrReductionKernel(IrKernel):
    operator: BinaryAssocScalarOperator
    axes: tuple[int, ...]

    def __post_init__(self):
        assert len(self.arg_accessors) == 1
        input_shape = self.arg_accessors[0].shape
        assert len(input_shape) == len(self.shape)
        for axis in self.axes:
            assert 0 <= axis < len(self.shape)
            assert self.shape[axis] == 1
        for i, (out_dim, in_dim) in enumerate(zip(self.shape, input_shape)):
            if i in self.axes:
                continue
            assert out_dim == in_dim

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.arg_accessors[0].shape

    @property
    def reduced_count(self) -> int:
        count = 1
        for axis in self.axes:
            count *= self.input_shape[axis]
        return count


#
# IrProgramBuilder
#


class IrProgramBuilder:
    buffer_memo: dict[dsl.Node, IrBuffer]
    buffer_view_memo: dict[dsl.View, IrBufferView]
    queue: list[IrDispatch]
    sinks: dict[str, IrBufferView]

    def __init__(self):
        super().__init__()

        self.buffer_memo = {}
        self.buffer_view_memo = {}
        self.queue = []
        self.sinks = {}

    def finish(self) -> IrProgram:
        param_buffer_ids = frozendict(
            {
                id(node): index
                for index, node in enumerate(self.buffer_memo.keys())
                if isinstance(node, dsl.ParamNode)
            }
        )
        return IrProgram(
            param_buffer_ids=param_buffer_ids,
            sinks=frozendict(self.sinks),
            queue=tuple(self.queue),
            buffers=tuple(self.buffer_memo.values()),
            buffer_views=tuple(self.buffer_view_memo.values()),
        )

    def build_sink(self, name: str, view: dsl.View):
        if name in self.sinks:
            raise ValueError(f"Sink name conflict: {name}")

        self.sinks[name] = self._build_view(view)

    def _build_view(self, view: dsl.View) -> IrBufferView:
        if bv := self.buffer_view_memo.get(view):
            return bv

        buffer = self._build_node(view.node)

        bv = IrBufferView(buffer=buffer, accessor=view.accessor)
        self.buffer_view_memo[view] = bv

        return bv

    def _build_node(self, node: dsl.Node) -> IrBuffer:
        if b := self.buffer_memo.get(node):
            return b

        output_buffer = self._allocate_buffer_for_node(node)
        input_buffer_views = tuple(self._build_view(view) for view in node.args)
        kernel = self._build_kernel_for_node(node)

        if kernel:
            dispatch = IrDispatch(
                kernel=kernel,
                args=input_buffer_views,
                output=output_buffer,
            )
            self.queue.append(dispatch)

        self.buffer_memo[node] = output_buffer
        return output_buffer

    def _allocate_buffer_for_node(self, node: dsl.Node) -> IrBuffer:
        match node:
            case dsl.ConstNode():
                return IrBuffer(
                    shape=node.shape,
                    dtype=node.dtype,
                    init=marshall_pytensor(node.value, dtype=node.dtype),
                    readonly=True,
                )
            case _:
                return IrBuffer(
                    shape=node.shape,
                    dtype=node.dtype,
                    init=None,
                    readonly=False,
                )

    def _build_kernel_for_node(self, node: dsl.Node) -> IrKernel | None:
        match node:
            case dsl.ConstNode() | dsl.ParamNode():
                return None
            case dsl.ElementwiseNode():
                return self._build_kernel_for_elementwise_node(node)
            case dsl.MatmulNode():
                return self._build_kernel_for_matmul_node(node)
            case dsl.ReductionNode():
                return self._build_kernel_for_reduction_node(node)
            case dsl.ScatterNode():
                return self._build_kernel_for_scatter_node(node)
            case _:
                raise NotImplementedError(f"Unsupported node type: {type(node)}")

    def _build_kernel_for_elementwise_node(self, node: dsl.ElementwiseNode) -> IrKernel:
        return IrElementwiseRpnKernel(
            arg_accessors=tuple(view.accessor for view in node.args),
            dtype=node.dtype,
            shape=node.shape,
            rpn_expr=ElementRpnExpr(string=_rpn_string_for_elementwise(node)),
        )

    def _build_kernel_for_matmul_node(self, node: dsl.MatmulNode) -> IrKernel:
        return IrMatmulKernel(
            arg_accessors=tuple(view.accessor for view in node.args),
            dtype=node.dtype,
            shape=node.shape,
        )

    def _build_kernel_for_reduction_node(self, node: dsl.ReductionNode) -> IrKernel:
        return IrReductionKernel(
            arg_accessors=(node.args[0].accessor,),
            dtype=node.dtype,
            shape=node.shape,
            operator=node.operator,
            axes=node.axes,
        )

    def _build_kernel_for_scatter_node(self, node: dsl.ScatterNode) -> IrKernel:
        common = dict(
            arg_accessors=(node.args[0].accessor,),
            dtype=node.dtype,
            shape=node.shape,
            woffset=node.woffset,
            wpitch=node.wpitch,
        )
        if node.operator is None:
            return IrScatterClobberKernel(**common)
        return IrScatterAccumulateKernel(operator=node.operator, **common)


_UNARY_OPS: tuple[UnaryScalarOperator, ...] = (
    "neg",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "not",
)
_BINARY_OPS: tuple[BinaryScalarOperator | BinaryCompareOperator, ...] = (
    "pow",
    "mul",
    "div",
    "add",
    "sub",
    "max",
    "min",
    "eq",
    "ne",
    "gt",
    "lt",
    "ge",
    "le",
)


def _rpn_string_for_elementwise(
    node: dsl.ElementwiseNode,
) -> tuple[int | ScalarOperator, ...]:
    op = node.operator
    if op in _UNARY_OPS:
        return (0, op)
    if op in _BINARY_OPS:
        return (0, 1, op)
    raise NotImplementedError(f"{op=}")
