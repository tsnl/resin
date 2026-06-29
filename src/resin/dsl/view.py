"""View types, annotation metadata, and tensor operation frontend."""

__all__ = [
    "TensorMeta",
    "TensorOperand",
    "View",
    "const",
    "debug_print",
    "full",
    "ones",
    "param",
    "refcount",
    "toposort",
    "zeros",
]

import math
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Annotated, overload

from resin.core.accessor import Accessor, c_contiguous_pitch_for_shape, shape_join
from resin.core.common import SupportsWrite, pascal_to_snake_case
from resin.core.etype import (
    BinaryAssocElementOperator,
    BinaryBitwiseOperator,
    ElementType,
    ElementOperator,
    F4,
    Scalar,
    UnaryElementOperator,
    etype_join,
    etype_kind,
    etype_nbytes,
    output_etype_for_unary,
)
from resin.core.pytree import PyTensor, infer_pytensor_shape
from resin.dsl.node import (
    DEFAULT_PORT,
    ConstNode,
    ElementwiseNode,
    MatmulNode,
    Node,
    ParamNode,
    PrefixSumNode,
    ReductionNode,
    RemapGatherInfo,
    RemapInfo,
    RemapNode,
    RemapScatterInfo,
)


@dataclass(frozen=True)
class TensorMeta:
    etype: ElementType
    shape: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class View:
    node: Node
    accessor: Accessor
    port: str = DEFAULT_PORT

    @overload
    def __class_getitem__(
        cls, params: tuple[ElementType, tuple[()]]
    ) -> Annotated[View | Scalar, TensorMeta]: ...

    @overload
    def __class_getitem__(
        cls, params: tuple[ElementType, tuple[int, ...]]
    ) -> TensorMeta: ...

    def __class_getitem__(cls, params: tuple[ElementType, tuple[int, ...]]) -> object:
        etype, shape = params
        if shape == ():
            return Annotated[View | Scalar, TensorMeta(etype, shape)]
        return TensorMeta(etype, shape)

    @staticmethod
    def identity(node: Node, port: str = DEFAULT_PORT) -> "View":
        return View(
            node=node,
            port=port,
            accessor=Accessor.dense(node.port_shape(port)),
        )

    @staticmethod
    def port(node: Node, port: str) -> "View":
        return View.identity(node, port=port)

    @property
    def offset(self) -> int:
        return self.accessor.offset

    @property
    def shape(self) -> tuple[int, ...]:
        return self.accessor.shape

    @property
    def pitch(self) -> tuple[int, ...]:
        return self.accessor.pitch

    @property
    def etype(self) -> ElementType | str:
        return self.node.port_etype(self.port)

    @property
    def rank(self) -> int:
        return self.accessor.rank

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * etype_nbytes(self.etype)

    def broadcast(self, ns: tuple[int, ...]) -> "View":
        return View(
            node=self.node,
            port=self.port,
            accessor=self.accessor.broadcast(ns),
        )

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> "View":
        return View(
            node=self.node,
            port=self.port,
            accessor=self.accessor.narrow(key),
        )

    def permute(self, permutation: tuple[int, ...]) -> "View":
        return View(
            node=self.node,
            port=self.port,
            accessor=self.accessor.permute(permutation),
        )

    def transpose(self) -> "View":
        return View(
            node=self.node,
            port=self.port,
            accessor=self.accessor.transpose(),
        )

    def squeeze(self, axes: tuple[int, ...]) -> "View":
        return View(
            node=self.node,
            port=self.port,
            accessor=self.accessor.squeeze(axes),
        )

    def reshape(self, shape: tuple[int, ...]) -> "View":
        if math.prod(shape) != math.prod(self.shape):
            raise ValueError(f"cannot reshape {self.shape} to {shape}")
        if not self.is_identity() and not self.accessor.is_dense_c_contiguous(
            self.node.port_shape(self.port)
        ):
            return self.copy().reshape(shape)
        return View(
            node=self.node,
            port=self.port,
            accessor=Accessor(
                offset=self.offset,
                shape=shape,
                pitch=c_contiguous_pitch_for_shape(shape),
            ),
        )

    def copy(self, *, etype: ElementType | None = None) -> "View":
        return View.remap(
            source=self,
            info=RemapGatherInfo(),
            etype=etype or self.etype,
        )

    def __pow__(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="pow")

    def __rpow__(self, other: "TensorOperand") -> "View":
        return View._from_view_or_scalar(other, etype=self.etype) ** self

    def __mul__(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="mul")

    def __rmul__(self, other: "TensorOperand") -> "View":
        return View._from_view_or_scalar(other, etype=self.etype) * self

    def __truediv__(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="div")

    def __rtruediv__(self, other: "TensorOperand") -> "View":
        return View._from_view_or_scalar(other, etype=self.etype) / self

    def __add__(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="add")

    def __radd__(self, other: "TensorOperand") -> "View":
        return View._from_view_or_scalar(other, etype=self.etype) + self

    def __sub__(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="sub")

    def __rsub__(self, other: "TensorOperand") -> "View":
        return View._from_view_or_scalar(other, etype=self.etype) - self

    def max(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="max")

    def min(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="min")

    def eq(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="eq")

    def ne(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="ne")

    def lt(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="lt")

    def gt(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="gt")

    def le(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="le")

    def ge(self, other: "TensorOperand") -> "View":
        return View.elementwise_binary(self, other, operator="ge")

    def __neg__(self) -> "View":
        return View.elementwise_unary(self, operator="neg")

    def __pos__(self) -> "View":
        return self

    def exp(self) -> "View":
        return View.elementwise_unary(self, operator="exp")

    def log(self) -> "View":
        return View.elementwise_unary(self, operator="log")

    def sqrt(self) -> "View":
        return View.elementwise_unary(self, operator="sqrt")

    def sin(self) -> "View":
        return View.elementwise_unary(self, operator="sin")

    def cos(self) -> "View":
        return View.elementwise_unary(self, operator="cos")

    def floor(self) -> "View":
        return View.elementwise_unary(self, operator="floor")

    def ceil(self) -> "View":
        return View.elementwise_unary(self, operator="ceil")

    def bitcast_f2u(self) -> "View":
        return View.elementwise_unary(self, operator="bitcast_f2u")

    def bitcast_u2f(self) -> "View":
        return View.elementwise_unary(self, operator="bitcast_u2f")

    def band(self, other: "View") -> "View":
        return View.elementwise_bitwise(self, other, operator="band")

    def bor(self, other: "View") -> "View":
        return View.elementwise_bitwise(self, other, operator="bor")

    def bxor(self, other: "View") -> "View":
        return View.elementwise_bitwise(self, other, operator="bxor")

    def shl(self, other: "View") -> "View":
        return View.elementwise_bitwise(self, other, operator="shl")

    def shr(self, other: "View") -> "View":
        return View.elementwise_bitwise(self, other, operator="shr")

    def __invert__(self) -> "View":
        return View.elementwise_unary(self, operator="not")

    def __matmul__(self, other: "View") -> "View":
        return View.matmul(self, other)

    def __rmatmul__(self, other: "View") -> "View":
        return View._from_view_or_scalar(other, etype=self.etype) @ self

    def reduce(
        self,
        *,
        operator: BinaryAssocElementOperator,
        axes: tuple[int, ...] | None,
    ) -> "View":
        axes = tuple(range(self.rank)) if axes is None else axes
        return View.reduction(self, axes=axes, operator=operator)

    def sum(self, axes: tuple[int, ...] | None = None) -> "View":
        return self.reduce(axes=axes, operator="add")

    def prod(self, axes: tuple[int, ...] | None = None) -> "View":
        return self.reduce(axes=axes, operator="mul")

    def debug_print(self, out: SupportsWrite[str]) -> None:
        debug_print(self, out)

    def is_identity(self) -> bool:
        return self.accessor.is_dense_c_contiguous(self.node.port_shape(self.port))

    def prefix_sum(self, *, inclusive: bool = False) -> "View":
        return View.identity(
            PrefixSumNode(
                shape=self.shape,
                etype=self.etype,
                args=(self,),
                inclusive=inclusive,
            )
        )


    @staticmethod
    def reverse_prefix_sum(x: "View", *, inclusive: bool = True) -> "View":
        """Prefix-sum adjoint helper: reverse, exclusive/inclusive scan, reverse."""
        n = x.shape[0] if x.rank >= 1 else 1
        if x.rank != 1:
            raise ValueError("reverse_prefix_sum expects a 1D view")
        # Build reversal indices [n-1, n-2, ..., 0] as a const u4 tensor.
        rev_idx_data = list(range(n - 1, -1, -1))
        rev_idx = const(rev_idx_data, etype="u4")
        indices = rev_idx.reshape((n, 1))
        reversed_x = View.remap(
            source=x,
            direction="gather",
            keys="indices",
            indices=indices,
            source_shape=x.shape,
        )
        scanned = reversed_x.prefix_sum(inclusive=inclusive)
        return View.remap(
            source=scanned,
            direction="gather",
            keys="indices",
            indices=indices,
            source_shape=scanned.shape,
        )

    @staticmethod
    def zeros_like(x: "View") -> "View":
        return zeros(x.shape, etype=x.etype)

    @staticmethod
    def _from_view_or_scalar(
        value: PyTensor | "View", etype: ElementType | str
    ) -> "View":
        return value if isinstance(value, View) else const(value, etype=etype)

    def _join_etypes_for_bop(self, other: "View") -> tuple["View", "View"]:
        res_etype = etype_join(self.etype, other.etype)
        s = self.copy(etype=res_etype) if self.etype != res_etype else self
        o = other.copy(etype=res_etype) if other.etype != res_etype else other
        return s, o

    def _join_shapes_for_elementwise_bop(self, other: "View") -> tuple["View", "View"]:
        join = shape_join(self.shape, self.pitch, other.shape, other.pitch)
        return (
            View(
                node=self.node,
                port=self.port,
                accessor=Accessor(
                    offset=self.offset,
                    shape=join.shape,
                    pitch=join.pitch1,
                ),
            ),
            View(
                node=other.node,
                port=other.port,
                accessor=Accessor(
                    offset=other.offset,
                    shape=join.shape,
                    pitch=join.pitch2,
                ),
            ),
        )

    def _join_shapes_for_matmul_bop(self, other: "View") -> tuple["View", "View"]:
        s = self
        o = other

        if s.rank < 2 or o.rank < 2:
            raise ValueError(
                f"Shapes {s.shape} and {o.shape} are not compatible for matmul: "
                + "both tensors must be 2D or higher"
            )

        s_shape_gat = tuple(x for i, x in enumerate(s.shape) if i != s.rank - 2)
        s_pitch_gat = tuple(x for i, x in enumerate(s.pitch) if i != s.rank - 2)
        o_shape_gat = tuple(x for i, x in enumerate(o.shape) if i != o.rank - 1)
        o_pitch_gat = tuple(x for i, x in enumerate(o.pitch) if i != o.rank - 1)
        join = shape_join(
            shape1=s_shape_gat,
            pitch1=s_pitch_gat,
            shape2=o_shape_gat,
            pitch2=o_pitch_gat,
            report_shape1=s.shape,
            report_shape2=o.shape,
        )

        bs = join.shape[:-1]
        m = s.shape[-2]
        k = join.shape[-1]
        n = o.shape[-1]

        new_s_shape = bs + (m, k)
        new_s_pitch = join.pitch1[:-1] + (s.pitch[-2], s.pitch[-1])
        new_o_shape = bs + (k, n)
        new_o_pitch = join.pitch2[:-1] + (o.pitch[-2], o.pitch[-1])

        new_self = View(
            node=s.node,
            port=s.port,
            accessor=Accessor(offset=s.offset, shape=new_s_shape, pitch=new_s_pitch),
        )
        new_other = View(
            node=o.node,
            port=o.port,
            accessor=Accessor(offset=o.offset, shape=new_o_shape, pitch=new_o_pitch),
        )
        return new_self, new_other

    @staticmethod
    def elementwise_unary(operand: "View", operator: UnaryElementOperator) -> "View":
        out_etype = output_etype_for_unary(operator, operand.etype)
        return View.identity(
            ElementwiseNode(
                shape=operand.shape,
                etype=out_etype,
                args=(operand,),
                operator=operator,
            )
        )

    @staticmethod
    def elementwise_bitwise(
        a: "View",
        b: "View",
        operator: BinaryBitwiseOperator,
    ) -> "View":
        if etype_kind(a.etype) != "uint" or etype_kind(b.etype) != "uint":
            raise TypeError(f"bitwise {operator} requires unsigned integer operands")
        a, b = a._join_shapes_for_elementwise_bop(b)
        return View.identity(
            ElementwiseNode(
                shape=a.shape,
                etype=a.etype,
                args=(a, b),
                operator=operator,
            )
        )

    @staticmethod
    def elementwise_binary(
        a: "View",
        b: "View | Scalar",
        operator: ElementOperator,
    ) -> "View":
        b = View._from_view_or_scalar(b, etype=a.etype)
        a, b = a._join_etypes_for_bop(b)
        a, b = a._join_shapes_for_elementwise_bop(b)
        return View.identity(
            ElementwiseNode(
                shape=a.shape,
                etype=a.etype,
                args=(a, b),
                operator=operator,
            )
        )

    @staticmethod
    def reduction(
        input: "View",
        axes: tuple[int, ...],
        operator: BinaryAssocElementOperator,
    ) -> "View":
        if not axes:
            return input

        out_shape = list(input.shape)
        for axis in axes:
            if axis < 0 or axis >= input.rank:
                raise IndexError(f"Axis {axis} out of bounds for shape {input.shape}")
            out_shape[axis] = 1

        return View.identity(
            ReductionNode(
                shape=tuple(out_shape),
                etype=input.etype,
                args=(input,),
                operator=operator,
                axes=axes,
            )
        )

    @staticmethod
    def matmul(a: "View", b: "View") -> "View":
        a, b = a._join_etypes_for_bop(b)
        a, b = a._join_shapes_for_matmul_bop(b)
        out_shape = a.shape[:-1] + (b.shape[-1],)
        return View.identity(MatmulNode(shape=out_shape, etype=a.etype, args=(a, b)))

    @staticmethod
    def remap(
        *,
        source: "View",
        info: RemapInfo,
        indices: "View | None" = None,
        out_shape: tuple[int, ...] | None = None,
        etype: ElementType | str | None = None,
    ) -> "View":
        match info:
            case RemapScatterInfo(accessor=accessor) if accessor is not None:
                if out_shape is None:
                    raise ValueError("scatter with accessor requires out_shape")
                if accessor.shape != source.shape:
                    raise ValueError(
                        "scatter accessor.shape must match source.shape"
                    )
                node_shape = out_shape
                args: tuple[View, ...] = (source,)
                node_info: RemapInfo = info
            case RemapScatterInfo(accessor=None):
                if out_shape is None or indices is None:
                    raise ValueError(
                        "scatter without accessor requires out_shape and indices"
                    )
                source, indices = _join_source_with_indices(source, indices, out_shape)
                node_shape = out_shape
                args = (source, indices)
                node_info = info
            case RemapGatherInfo(accessor=None, source_shape=None):
                if indices is not None:
                    raise ValueError(
                        "gather densify (no accessor) does not take indices"
                    )
                node_shape = source.shape
                args = (source,)
                node_info = info
            case RemapGatherInfo(accessor=accessor, source_shape=source_shape) if (
                accessor is not None and source_shape is not None
            ):
                if indices is None:
                    raise ValueError("gather with accessor requires indices")
                if accessor.shape != source_shape:
                    raise ValueError(
                        "gather accessor.shape must match source_shape"
                    )
                indices = _validate_indices_view(indices, source_shape)
                out_prefix = indices.shape[:-1]
                # preserve port if present in View constructor calls nearby
                source = View(
                    node=source.node,
                    port=source.port,
                    accessor=Accessor(
                        offset=source.offset,
                        shape=out_prefix,
                        pitch=c_contiguous_pitch_for_shape(out_prefix),
                    ),
                )
                node_shape = out_prefix
                args = (source, indices)
                node_info = info
            case RemapGatherInfo():
                raise ValueError(
                    "gather with accessor requires source_shape; "
                    "gather without accessor must omit source_shape"
                )

        return View.identity(
            RemapNode(
                shape=node_shape,
                etype=etype or source.etype,
                args=args,
                info=node_info,
            )
        )

    @staticmethod
    def scatter(
        *,
        source: "View",
        out_shape: tuple[int, ...],
        woffset: int,
        wpitch: tuple[int, ...],
        operator: BinaryAssocElementOperator | None = None,
        etype: ElementType | str | None = None,
    ) -> "View":
        return View.remap(
            source=source,
            info=RemapScatterInfo(
                accessor=Accessor(
                    offset=woffset, shape=source.shape, pitch=wpitch
                ),
                operator=operator,
            ),
            out_shape=out_shape,
            etype=etype,
        )


type TensorOperand = View | Scalar


def _validate_indices_view(indices: View, out_shape: tuple[int, ...]) -> View:
    out_rank = len(out_shape)
    if indices.rank == 0:
        raise ValueError("indices must have rank >= 1")
    if indices.shape[-1] != out_rank:
        raise ValueError(
            f"indices.shape[-1] must equal len(out_shape) ({out_rank}), got shape {indices.shape}"
        )
    if etype_kind(indices.etype) != "uint":
        raise ValueError(
            f"indices must have an unsigned integer etype, got {indices.etype}"
        )
    return indices


def _join_source_with_indices(
    source: View,
    indices: View,
    out_shape: tuple[int, ...],
) -> tuple[View, View]:
    indices = _validate_indices_view(indices, out_shape)
    out_rank = len(out_shape)

    prefix_join = shape_join(
        shape1=source.shape,
        pitch1=source.pitch,
        shape2=indices.shape[:-1],
        pitch2=indices.pitch[:-1],
    )
    joined_source = View(
        node=source.node,
        port=source.port,
        accessor=Accessor(
            offset=source.offset,
            shape=prefix_join.shape,
            pitch=prefix_join.pitch1,
        ),
    )
    joined_indices = View(
        node=indices.node,
        port=indices.port,
        accessor=Accessor(
            offset=indices.offset,
            shape=prefix_join.shape + (out_rank,),
            pitch=prefix_join.pitch2 + (indices.pitch[-1],),
        ),
    )
    return joined_source, joined_indices


def const(value: PyTensor, *, etype: ElementType | str = F4) -> View:
    shape = infer_pytensor_shape(value)
    return View.identity(ConstNode(shape=shape, etype=etype, args=(), value=value))


def full(shape: tuple[int, ...], v: Scalar, *, etype: ElementType | str = F4) -> View:
    return const(v, etype=etype).broadcast(shape)


def ones(shape: tuple[int, ...], *, etype: ElementType | str = F4) -> View:
    return full(shape, 1, etype=etype)


def zeros(shape: tuple[int, ...], *, etype: ElementType | str = F4) -> View:
    return full(shape, 0, etype=etype)


_param_counter = 0


def _next_param_name(explicit: str | None) -> str:
    global _param_counter
    if explicit is not None:
        return explicit
    name = f"$p{_param_counter}"
    _param_counter += 1
    return name


def param(
    *,
    shape: tuple[int, ...],
    etype: ElementType | str,
    name: str | None = None,
) -> View:
    return View.identity(
        ParamNode(
            shape=shape,
            etype=etype,
            args=(),
            name=_next_param_name(name),
        )
    )


def debug_print(root: View, out: SupportsWrite[str]) -> None:
    def build_tid_map() -> dict[Node, int]:
        reference_count_map = refcount([root])
        root_ref_count = reference_count_map[root.node]
        if root_ref_count != 1:
            raise ValueError(
                f"Root tensor must have reference count 1, got {root_ref_count}"
            )

        tid_map: dict[Node, int] = {}
        for node, ref_count in reference_count_map.items():
            if ref_count < 1:
                raise ValueError(
                    f"Node {node!r} has invalid reference count {ref_count}"
                )
            if ref_count == 1:
                continue
            tid_map[node] = len(tid_map)

        return tid_map

    def headline(node: Node) -> str:
        base_fields = {field.name for field in fields(Node)}
        extra_fields = [f.name for f in fields(node) if f.name not in base_fields]
        args = ", ".join(f"{f}={getattr(node, f)!r}" for f in extra_fields)
        name = pascal_to_snake_case(node.__class__.__name__[: -len("Node")])
        ports = node.output_ports()
        if ports == (DEFAULT_PORT,):
            return f"{name}({args}) :: {node.etype}{node.shape!r}"
        port_specs = ", ".join(
            f"{p}:{node.port_etype(p)}{node.port_shape(p)!r}" for p in ports
        )
        return f"{name}({args}) :: {{{port_specs}}}"

    def visit_view(
        view: View,
        prefix: str,
        connector: str,
        prefix_ext: str,
        *,
        is_root: bool,
    ) -> None:
        if view.is_identity():
            visit_node(view.node, prefix, connector, prefix_ext, is_root=is_root)
            return

        a = view.accessor
        port_s = f", port={view.port!r}" if view.port != DEFAULT_PORT else ""
        print(
            f"{prefix}{connector}view("
            + f"offset={a.offset}, shape={a.shape!r}, pitch={a.pitch!r}{port_s})",
            file=out,
        )
        visit_node(view.node, prefix + prefix_ext, "└ ", "  ", is_root=is_root)

    def visit_node(
        node: Node,
        prefix: str,
        connector: str,
        prefix_ext: str,
        *,
        is_root: bool,
    ) -> None:
        tid = tid_map.get(node)

        if tid is not None and not is_root:
            print(f"{prefix}{connector}%{tid}", file=out)
            return

        if tid is not None:
            print(f"{prefix}{connector}%{tid} := {headline(node)}", file=out)
        else:
            print(f"{prefix}{connector}{headline(node)}", file=out)

        child_prefix = prefix + prefix_ext
        for operand_index, operand in enumerate(node.args):
            is_last = operand_index == len(node.args) - 1
            c_connector = "└ " if is_last else "├ "
            c_ext = "  " if is_last else "│ "
            visit_view(operand, child_prefix, c_connector, c_ext, is_root=False)

    tid_map = build_tid_map()

    visit_view(root, "", "", "", is_root=True)

    for node in tid_map:
        visit_node(node, "", "", "", is_root=True)


def toposort(roots: Iterable[View]) -> list[Node]:
    visited: set[Node] = set()
    topo_order: list[Node] = []

    def visit(node: Node) -> None:
        if node in visited:
            return
        visited.add(node)

        for arg in node.args:
            visit(arg.node)

        topo_order.append(node)

    for root in roots:
        visit(root.node)

    return topo_order


def refcount(roots: Iterable[View]) -> dict[Node, int]:
    ref_counts: dict[Node, int] = {}

    def visit(node: Node) -> None:
        if node in ref_counts:
            ref_counts[node] += 1
        else:
            ref_counts[node] = 1
            for operand in node.args:
                visit(operand.node)

    for root in roots:
        visit(root.node)

    return ref_counts
