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
    "IrRemapKernel",
]

from abc import ABC
from dataclasses import dataclass
from typing import Literal

from frozendict import frozendict

from resin.core.accessor import Accessor, c_contiguous_pitch_for_shape
from resin.core.pytree import marshall_pytensor
from .rpn import ElementRpnExpr
from resin.core.etype import (
    BinaryAssocElementOperator,
    BinaryBitwiseOperator,
    BinaryCompareOperator,
    BinaryElementOperator,
    ElementType,
    ElementOperator,
    UnaryElementOperator,
)
import resin.dsl as dsl

#
# IrProgram
#


@dataclass(frozen=True, kw_only=True)
class IrProgram:
    # Maps each param's stable name to its buffer index in `buffers`.
    param_buffers: frozendict[str, int]
    sinks: frozendict[str, IrBufferView]
    queue: tuple[IrDispatch, ...]
    buffers: tuple[IrBuffer, ...]
    buffer_views: tuple[IrBufferView, ...]


@dataclass(frozen=True, kw_only=True, eq=False)
class IrBuffer:
    shape: tuple[int, ...]
    etype: ElementType | str
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
    """Base kernel. ``etype`` is the output buffer element type.

    All current kernels share one etype for inputs and output. Tile
    kernels may need per-operand types or epilogue-specific types later.
    """

    arg_accessors: tuple[Accessor, ...]
    etype: ElementType | str
    shape: tuple[int, ...]
    clear_output_before_dispatch: bool = False


@dataclass(frozen=True, kw_only=True)
class IrElementwiseRpnKernel(IrKernel):
    rpn_expr: ElementRpnExpr
    arg_etypes: tuple[ElementType | str, ...] = ()

    def __post_init__(self):
        assert all(x.shape == self.shape for x in self.arg_accessors)
        if self.arg_etypes and len(self.arg_etypes) != len(self.arg_accessors):
            raise ValueError("arg_etypes length must match arg_accessors")


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


type RemapDirection = Literal["gather", "scatter"]
type RemapKeys = Literal["accessor", "indices"]


@dataclass(frozen=True, kw_only=True)
class IrRemapKernel(IrKernel):
    direction: RemapDirection
    keys: RemapKeys
    operator: BinaryAssocElementOperator | None
    woffset: int
    wpitch: tuple[int, ...]
    arg_etypes: tuple[ElementType | str, ...]
    source_shape: tuple[int, ...] | None = None
    clear_output_before_dispatch: bool = True

    def __post_init__(self):
        match self.keys:
            case "accessor":
                assert len(self.arg_accessors) == 1
                if self.direction == "scatter":
                    assert len(self.wpitch) == len(self.arg_accessors[0].shape)
            case "indices":
                assert len(self.arg_accessors) == 2
                assert len(self.arg_etypes) == 2
                source_accessor, indices_accessor = self.arg_accessors
                out_rank = (
                    len(self.shape)
                    if self.direction == "scatter"
                    else len(self.wpitch)
                )
                assert indices_accessor.shape[-1] == out_rank
                assert indices_accessor.shape[:-1] == source_accessor.shape
                if self.direction == "scatter":
                    assert self.wpitch == c_contiguous_pitch_for_shape(self.shape)
                else:
                    assert self.source_shape is not None
                    assert self.wpitch == c_contiguous_pitch_for_shape(self.source_shape)


@dataclass(frozen=True, kw_only=True)
class IrReductionKernel(IrKernel):
    operator: BinaryAssocElementOperator
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
    param_names: dict[int, str]

    def __init__(self):
        super().__init__()

        self.buffer_memo = {}
        self.buffer_view_memo = {}
        self.queue = []
        self.sinks = {}
        self.param_names = {}

    def register_param(self, name: str, view: dsl.View) -> None:
        node = view.node
        if not isinstance(node, dsl.ParamNode):
            raise TypeError(f"register_param expected a param view, got {type(node)}")
        if name in self.param_names.values():
            raise ValueError(f"duplicate param name: {name!r}")
        self.param_names[id(node)] = name

    def finish(self) -> IrProgram:
        param_buffer_items: list[tuple[str, int]] = []
        seen_names: set[str] = set()
        for index, node in enumerate(self.buffer_memo.keys()):
            if not isinstance(node, dsl.ParamNode):
                continue
            name = self.param_names.get(id(node), node.name)
            if name in seen_names:
                raise ValueError(f"duplicate param name: {name!r}")
            seen_names.add(name)
            param_buffer_items.append((name, index))
        param_buffers: frozendict[str, int] = frozendict(param_buffer_items)
        return IrProgram(
            param_buffers=param_buffers,
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
                    etype=node.etype,
                    init=marshall_pytensor(node.value, etype=node.etype),
                    readonly=True,
                )
            case _:
                return IrBuffer(
                    shape=node.shape,
                    etype=node.etype,
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
            case dsl.RemapNode():
                return self._build_kernel_for_remap_node(node)
            case _:
                raise NotImplementedError(f"Unsupported node type: {type(node)}")

    def _build_kernel_for_elementwise_node(self, node: dsl.ElementwiseNode) -> IrKernel:
        arg_etypes = tuple(view.etype for view in node.args)
        return IrElementwiseRpnKernel(
            arg_accessors=tuple(view.accessor for view in node.args),
            etype=node.etype,
            shape=node.shape,
            rpn_expr=ElementRpnExpr(string=_rpn_string_for_elementwise(node)),
            arg_etypes=arg_etypes,
        )

    def _build_kernel_for_matmul_node(self, node: dsl.MatmulNode) -> IrKernel:
        return IrMatmulKernel(
            arg_accessors=tuple(view.accessor for view in node.args),
            etype=node.etype,
            shape=node.shape,
        )

    def _build_kernel_for_reduction_node(self, node: dsl.ReductionNode) -> IrKernel:
        return IrReductionKernel(
            arg_accessors=(node.args[0].accessor,),
            etype=node.etype,
            shape=node.shape,
            operator=node.operator,
            axes=node.axes,
        )

    def _build_kernel_for_remap_node(self, node: dsl.RemapNode) -> IrKernel:
        arg_etypes = tuple(view.etype for view in node.args)
        clear_output = node.direction == "gather" or node.operator is None
        return IrRemapKernel(
            arg_accessors=tuple(view.accessor for view in node.args),
            etype=node.etype,
            shape=node.shape,
            direction=node.direction,
            keys=node.keys,
            operator=node.operator,
            woffset=node.woffset,
            wpitch=node.wpitch,
            arg_etypes=arg_etypes,
            source_shape=node.source_shape,
            clear_output_before_dispatch=clear_output,
        )


_UNARY_OPS: tuple[UnaryElementOperator, ...] = (
    "neg",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "not",
    "floor",
    "ceil",
    "bitcast_f2u",
    "bitcast_u2f",
)
_BINARY_OPS: tuple[BinaryElementOperator | BinaryCompareOperator, ...] = (
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
_BITWISE_OPS: tuple[BinaryBitwiseOperator, ...] = (
    "band",
    "bor",
    "bxor",
    "shl",
    "shr",
)


def _rpn_string_for_elementwise(
    node: dsl.ElementwiseNode,
) -> tuple[int | ElementOperator, ...]:
    op = node.operator
    if op in _UNARY_OPS:
        return (0, op)
    if op in _BINARY_OPS:
        return (0, 1, op)
    if op in _BITWISE_OPS:
        return (0, 1, op)
    raise NotImplementedError(f"{op=}")
