__all__ = [
    "IrBuffer",
    "IrBufferView",
    "IrDispatch",
    "IrElementwiseRpnKernel",
    "IrKernel",
    "IrMatmulKernel",
    "IrPrefixSumKernel",
    "IrProgram",
    "IrProgramBuilder",
    "IrReductionKernel",
    "IrRemapKernel",
    "IrSortKernel",
    "reachable_ports",
]

from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass

from frozendict import frozendict

from resin.core.accessor import Accessor
from resin.core.pytree import marshall_pytensor
from resin.dsl.node import RemapGatherInfo, RemapInfo, RemapScatterInfo
from .rpn import ElementRpnExpr
from resin.core.etype import (
    U4,
    BinaryAssocElementOperator,
    BinaryBitwiseOperator,
    BinaryCompareOperator,
    BinaryElementOperator,
    ElementType,
    ElementOperator,
    UnaryElementOperator,
)
import resin.dsl as dsl
from resin.dsl.node import DEFAULT_PORT, CustomNode

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
    etype: ElementType
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
    outputs: tuple[IrBuffer, ...]


@dataclass(frozen=True, kw_only=True)
class IrKernel(ABC):
    """Base kernel. ``etype`` is the primary output buffer element type."""

    arg_accessors: tuple[Accessor, ...]
    etype: ElementType
    shape: tuple[int, ...]
    clear_output_before_dispatch: bool = False
    # Number of storage bindings used as outputs (bindings 0..num_outputs-1).
    num_outputs: int = 1


@dataclass(frozen=True, kw_only=True)
class IrElementwiseRpnKernel(IrKernel):
    rpn_expr: ElementRpnExpr
    arg_etypes: tuple[ElementType, ...] = ()

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


@dataclass(frozen=True, kw_only=True)
class IrRemapKernel(IrKernel):
    info: RemapInfo
    arg_etypes: tuple[ElementType, ...]
    clear_output_before_dispatch: bool = True

    def __post_init__(self):
        match self.info:
            case RemapScatterInfo(accessor=accessor) if accessor is not None:
                assert len(self.arg_accessors) == 1
                assert accessor.shape == self.arg_accessors[0].shape
            case RemapScatterInfo(accessor=None):
                assert len(self.arg_accessors) == 2
                assert len(self.arg_etypes) == 2
                source_accessor, indices_accessor = self.arg_accessors
                assert indices_accessor.shape[-1] == len(self.shape)
                assert indices_accessor.shape[:-1] == source_accessor.shape
            case RemapGatherInfo(accessor=None, source_shape=None):
                assert len(self.arg_accessors) == 1
            case RemapGatherInfo(accessor=accessor, source_shape=source_shape) if (
                accessor is not None and source_shape is not None
            ):
                assert len(self.arg_accessors) == 2
                assert len(self.arg_etypes) == 2
                source_accessor, indices_accessor = self.arg_accessors
                assert indices_accessor.shape[-1] == accessor.rank
                assert indices_accessor.shape[:-1] == source_accessor.shape
                assert accessor.shape == source_shape
            case _:
                raise AssertionError(f"invalid RemapInfo: {self.info!r}")


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


@dataclass(frozen=True, kw_only=True)
class IrPrefixSumKernel(IrKernel):
    inclusive: bool
    arg_etypes: tuple[ElementType, ...]
    clear_output_before_dispatch: bool = True

    def __post_init__(self):
        assert len(self.arg_accessors) == 1
        assert self.arg_accessors[0].shape == self.shape


@dataclass(frozen=True, kw_only=True)
class IrSortKernel(IrKernel):
    arg_etypes: tuple[ElementType, ...]
    write_values: bool
    write_perm: bool
    perm_etype: ElementType = U4
    clear_output_before_dispatch: bool = True
    num_outputs: int = 2

    def __post_init__(self):
        assert len(self.arg_accessors) == 1
        assert self.write_values or self.write_perm


@dataclass(frozen=True, kw_only=True)
class IrWgslMultiOutputKernel(IrKernel):
    """Opaque multi-output WGSL kernel (custom nodes supply full source)."""

    wgsl: str
    entry_point: str
    dispatch_size: tuple[int, int, int]
    arg_etypes: tuple[ElementType, ...]
    output_etypes: tuple[ElementType, ...]
    clear_output_before_dispatch: bool = True

    def __post_init__(self):
        assert len(self.arg_etypes) == len(self.arg_accessors)
        assert len(self.output_etypes) == self.num_outputs


#
# Reachability / DCE helpers
#


def reachable_ports(roots: Iterable[dsl.View]) -> dict[dsl.Node, set[str]]:
    """Ports reachable walking backward from ``roots`` (forward sinks / grad roots)."""
    result: dict[dsl.Node, set[str]] = {}

    def visit(view: dsl.View) -> None:
        ports = result.setdefault(view.node, set())
        if view.port in ports:
            return
        ports.add(view.port)
        for arg in view.node.args:
            visit(arg)

    for root in roots:
        visit(root)
    return result


#
# IrProgramBuilder
#


class IrProgramBuilder:
    buffer_memo: dict[tuple[dsl.Node, str], IrBuffer]
    buffer_view_memo: dict[dsl.View, IrBufferView]
    queue: list[IrDispatch]
    sinks: dict[str, IrBufferView]
    param_names: dict[int, str]
    used_ports: dict[dsl.Node, set[str]]
    dispatched_nodes: set[dsl.Node]

    def __init__(self, *, used_ports: dict[dsl.Node, set[str]] | None = None):
        super().__init__()

        self.buffer_memo = {}
        self.buffer_view_memo = {}
        self.queue = []
        self.sinks = {}
        self.param_names = {}
        self.used_ports = used_ports or {}
        self.dispatched_nodes = set()

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
        seen_nodes: set[dsl.Node] = set()
        for (node, _port), _buffer in self.buffer_memo.items():
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            if not isinstance(node, dsl.ParamNode):
                continue
            name = self.param_names.get(id(node), node.name)
            if name in seen_names:
                raise ValueError(f"duplicate param name: {name!r}")
            seen_names.add(name)
            # Param buffer index is the first buffer allocated for that node.
            for index, (key_node, key_port) in enumerate(self.buffer_memo.keys()):
                if key_node is node and key_port == DEFAULT_PORT:
                    param_buffer_items.append((name, index))
                    break
            else:
                # Fall back to first port of the node.
                for index, (key_node, _key_port) in enumerate(self.buffer_memo.keys()):
                    if key_node is node:
                        param_buffer_items.append((name, index))
                        break
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

    def _ports_for_node(self, node: dsl.Node) -> tuple[str, ...]:
        if node in self.used_ports:
            used = self.used_ports[node]
            return tuple(p for p in node.output_ports() if p in used)
        return node.output_ports()

    def _build_view(self, view: dsl.View) -> IrBufferView:
        if bv := self.buffer_view_memo.get(view):
            return bv

        buffer = self._build_node_port(view.node, view.port)

        bv = IrBufferView(buffer=buffer, accessor=view.accessor)
        self.buffer_view_memo[view] = bv

        return bv

    def _build_node_port(self, node: dsl.Node, port: str) -> IrBuffer:
        key = (node, port)
        if b := self.buffer_memo.get(key):
            # Ensure the node is dispatched even if another port was built first.
            self._ensure_dispatched(node)
            return b

        # Allocate all ports for this node so dispatch can bind them together.
        for p in node.output_ports():
            pkey = (node, p)
            if pkey not in self.buffer_memo:
                self.buffer_memo[pkey] = self._allocate_buffer_for_port(node, p)

        self._ensure_dispatched(node)
        return self.buffer_memo[key]

    def _ensure_dispatched(self, node: dsl.Node) -> None:
        if node in self.dispatched_nodes:
            return
        self.dispatched_nodes.add(node)

        input_buffer_views = tuple(self._build_view(view) for view in node.args)
        used = frozenset(self._ports_for_node(node))
        kernel = self._build_kernel_for_node(node)
        if kernel is None:
            return

        # Bind only ports the kernel actually writes (DCE may drop unused ports
        # so WGSL bind-group layout matches runtime bind group entries).
        if isinstance(node, dsl.SortNode):
            ports = node.output_port_order_for_kernel(used_ports=used)
        else:
            ports = node.output_ports()
        outputs = tuple(self.buffer_memo[(node, p)] for p in ports)
        if kernel.num_outputs != len(outputs):
            raise ValueError(
                f"kernel expects {kernel.num_outputs} outputs, binding {len(outputs)}"
            )

        self.queue.append(
            IrDispatch(
                kernel=kernel,
                args=input_buffer_views,
                outputs=outputs,
            )
        )

    def _allocate_buffer_for_port(self, node: dsl.Node, port: str) -> IrBuffer:
        match node:
            case dsl.ConstNode() if port == DEFAULT_PORT:
                return IrBuffer(
                    shape=node.port_shape(port),
                    etype=node.port_etype(port),
                    init=marshall_pytensor(node.value, etype=node.port_etype(port)),
                    readonly=True,
                )
            case _:
                return IrBuffer(
                    shape=node.port_shape(port),
                    etype=node.port_etype(port),
                    init=None,
                    readonly=False,
                )

    def _build_kernel_for_node(self, node: dsl.Node) -> IrKernel | None:
        used = frozenset(self._ports_for_node(node))
        match node:
            case dsl.ConstNode() | dsl.ParamNode():
                return None
            case CustomNode():
                return node.build_kernel(used_ports=used)
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
        match node.info:
            case RemapScatterInfo(operator=operator):
                clear_output = operator is None
            case RemapGatherInfo():
                clear_output = True
        return IrRemapKernel(
            arg_accessors=tuple(view.accessor for view in node.args),
            etype=node.etype,
            shape=node.shape,
            info=node.info,
            arg_etypes=arg_etypes,
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
    "bitcast",
)
_BINARY_OPS: tuple[
    BinaryElementOperator | BinaryCompareOperator | BinaryBitwiseOperator, ...
] = (
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
    raise NotImplementedError(f"{op=}")
