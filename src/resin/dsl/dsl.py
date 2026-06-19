"""
resin.dsl models the computational graph of tensor operations.

It is the "frontend" for the `resin` programming language, i.e. the part that users
interact with directly when writing `resin` code.
"""

__all__ = [
    "ConstNode",
    "ElementwiseNode",
    "MatmulNode",
    "Node",
    "ParamNode",
    "PyTree",
    "ReductionNode",
    "ScatterNode",
    "View",
    "const",
    "debug_print",
    "flatten_pytree",
    "full",
    "map_pytree",
    "ones",
    "param",
    "refcount",
    "toposort",
    "zeros",
]

import math
from abc import ABC
from dataclasses import dataclass, fields
from typing import Callable, Generator, Iterable

from resin.core.accessor import Accessor, c_contiguous_pitch_for_shape, shape_join
from resin.core.common import SupportsWrite, pascal_to_snake_case
from resin.core.pytree import PyTensor, PyTree, infer_pytensor_shape
from resin.core.dtype import (
    BinaryAssocScalarOperator,
    DType,
    Scalar,
    ScalarOperator,
    UnaryScalarOperator,
    dtype_join,
    dtype_nbytes,
)


@dataclass(kw_only=True, frozen=True, eq=False)
class Node(ABC):
    shape: tuple[int, ...]
    dtype: DType
    args: tuple["View", ...]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * dtype_nbytes(self.dtype)

    def view(
        self,
        *,
        offset: int = 0,
        shape: tuple[int, ...],
        pitch: tuple[int, ...],
    ) -> "View":
        new = View(
            node=self,
            accessor=Accessor(offset=offset, shape=shape, pitch=pitch),
        )
        new._raise_if_out_of_backing_node_bounds()
        return new


@dataclass(frozen=True, eq=False)
class View:
    node: "Node"
    accessor: Accessor

    @staticmethod
    def identity(node: "Node") -> "View":
        return View(node=node, accessor=Accessor.dense(node.shape))

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
    def dtype(self) -> DType:
        return self.node.dtype

    @property
    def rank(self) -> int:
        return self.accessor.rank

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * dtype_nbytes(self.dtype)

    def broadcast(self, ns: tuple[int, ...]) -> "View":
        return View(node=self.node, accessor=self.accessor.broadcast(ns))

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> "View":
        return View(node=self.node, accessor=self.accessor.narrow(key))

    def permute(self, permutation: tuple[int, ...]) -> "View":
        return View(node=self.node, accessor=self.accessor.permute(permutation))

    def transpose(self) -> "View":
        return View(node=self.node, accessor=self.accessor.transpose())

    def squeeze(self, axes: tuple[int, ...]) -> "View":
        return View(node=self.node, accessor=self.accessor.squeeze(axes))

    def copy(self, *, dtype: DType | None = None) -> "View":
        return _copy(self, dtype=dtype)

    def __pow__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="pow")

    def __rpow__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, dtype=self.dtype) ** self

    def __mul__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="mul")

    def __rmul__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, dtype=self.dtype) * self

    def __truediv__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="div")

    def __rtruediv__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, dtype=self.dtype) / self

    def __add__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="add")

    def __radd__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, dtype=self.dtype) + self

    def __sub__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="sub")

    def __rsub__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, dtype=self.dtype) - self

    def max(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="max")

    def min(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="min")

    def eq(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="eq")

    def ne(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="ne")

    def lt(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="lt")

    def gt(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="gt")

    def le(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="le")

    def ge(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="ge")

    def __neg__(self) -> "View":
        return _elementwise_unary(self, operator="neg")

    def __pos__(self) -> "View":
        return self

    def exp(self) -> "View":
        return _elementwise_unary(self, operator="exp")

    def log(self) -> "View":
        return _elementwise_unary(self, operator="log")

    def sqrt(self) -> "View":
        return _elementwise_unary(self, operator="sqrt")

    def sin(self) -> "View":
        return _elementwise_unary(self, operator="sin")

    def cos(self) -> "View":
        return _elementwise_unary(self, operator="cos")

    def __invert__(self) -> "View":
        return _elementwise_unary(self, operator="not")

    def __matmul__(self, other: "View") -> "View":
        return _matmul(self, other)

    def __rmatmul__(self, other: "View") -> "View":
        return View._from_view_or_scalar(other, dtype=self.dtype) @ self

    def reduce(
        self,
        *,
        operator: BinaryAssocScalarOperator,
        axes: tuple[int, ...] | None,
    ) -> "View":
        axes = tuple(range(self.rank)) if axes is None else axes
        return _reduction(self, axes=axes, operator=operator)

    def sum(self, axes: tuple[int, ...] | None = None) -> "View":
        return self.reduce(axes=axes, operator="add")

    def prod(self, axes: tuple[int, ...] | None = None) -> "View":
        return self.reduce(axes=axes, operator="mul")

    def debug_print(self, out: SupportsWrite[str]):
        debug_print(self, out)

    def _is_identity(self) -> bool:
        return self.accessor.is_dense_c_contiguous(self.node.shape)

    def _raise_if_out_of_backing_node_bounds(self) -> None:
        self.accessor.raise_if_addresses_out_of_bounds(math.prod(self.node.shape))

    @staticmethod
    def _from_view_or_scalar(value: "PyTensor | View", dtype: DType) -> "View":
        return value if isinstance(value, View) else const(value, dtype=dtype)

    def _join_dtypes_for_bop(self, other: "View") -> tuple["View", "View"]:
        res_dtype = dtype_join(self.dtype, other.dtype)
        s = self.copy(dtype=res_dtype) if self.dtype != res_dtype else self
        o = other.copy(dtype=res_dtype) if other.dtype != res_dtype else other
        return s, o

    def _join_shapes_for_elementwise_bop(self, other: "View") -> tuple["View", "View"]:
        join = shape_join(self.shape, self.pitch, other.shape, other.pitch)
        return (
            View(
                node=self.node,
                accessor=Accessor(
                    offset=self.offset,
                    shape=join.shape,
                    pitch=join.pitch1,
                ),
            ),
            View(
                node=other.node,
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
                "both tensors must be 2D or higher"
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
            accessor=Accessor(offset=s.offset, shape=new_s_shape, pitch=new_s_pitch),
        )
        new_other = View(
            node=o.node,
            accessor=Accessor(offset=o.offset, shape=new_o_shape, pitch=new_o_pitch),
        )
        return new_self, new_other


@dataclass(kw_only=True, frozen=True, eq=False)
class ConstNode(Node):
    value: PyTensor


def const(value: "PyTensor", *, dtype: DType = "f4") -> View:
    shape = infer_pytensor_shape(value)
    return View.identity(ConstNode(shape=shape, dtype=dtype, args=(), value=value))


def full(shape: tuple[int, ...], v: Scalar, *, dtype: DType = "f4") -> View:
    return const(v, dtype=dtype).broadcast(shape)


def ones(shape: tuple[int, ...], *, dtype: DType = "f4") -> View:
    return full(shape, 1, dtype=dtype)


def zeros(shape: tuple[int, ...], *, dtype: DType = "f4") -> View:
    return full(shape, 0, dtype=dtype)


@dataclass(kw_only=True, frozen=True, eq=False)
class ParamNode(Node):
    label: str | None


def param(
    *,
    shape: tuple[int, ...],
    dtype: DType,
    label: str | None = None,
) -> View:
    return View.identity(ParamNode(shape=shape, dtype=dtype, args=(), label=label))


@dataclass(kw_only=True, frozen=True, eq=False)
class ElementwiseNode(Node):
    operator: ScalarOperator


def _elementwise_unary(operand: View, operator: UnaryScalarOperator) -> View:
    return View.identity(
        ElementwiseNode(
            shape=operand.shape,
            dtype=operand.dtype,
            args=(operand,),
            operator=operator,
        )
    )


def _elementwise_binary(
    a: View,
    b: View | Scalar,
    operator: ScalarOperator,
) -> View:
    b = View._from_view_or_scalar(b, dtype=a.dtype)
    a, b = a._join_dtypes_for_bop(b)
    a, b = a._join_shapes_for_elementwise_bop(b)
    return View.identity(
        ElementwiseNode(
            shape=a.shape,
            dtype=a.dtype,
            args=(a, b),
            operator=operator,
        )
    )


@dataclass(kw_only=True, frozen=True, eq=False)
class ReductionNode(Node):
    operator: BinaryAssocScalarOperator
    axes: tuple[int, ...]


def _reduction(
    input: View,
    axes: tuple[int, ...],
    operator: BinaryAssocScalarOperator,
) -> View:
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
            dtype=input.dtype,
            args=(input,),
            operator=operator,
            axes=axes,
        )
    )


@dataclass(kw_only=True, frozen=True, eq=False)
class MatmulNode(Node):
    pass


def _matmul(a: View, b: View) -> View:
    a, b = a._join_dtypes_for_bop(b)
    a, b = a._join_shapes_for_matmul_bop(b)
    out_shape = a.shape[:-1] + (b.shape[-1],)
    return View.identity(MatmulNode(shape=out_shape, dtype=a.dtype, args=(a, b)))


@dataclass(kw_only=True, frozen=True, eq=False)
class ScatterNode(Node):
    operator: BinaryAssocScalarOperator | None
    woffset: int
    wpitch: tuple[int, ...]


def _scatter(
    *,
    source: View,
    out_shape: tuple[int, ...],
    woffset: int,
    wpitch: tuple[int, ...],
    operator: BinaryAssocScalarOperator | None = None,
    dtype: DType | None = None,
) -> View:
    return View.identity(
        ScatterNode(
            shape=out_shape,
            dtype=dtype or source.dtype,
            args=(source,),
            operator=operator,
            woffset=woffset,
            wpitch=wpitch,
        )
    )


def _copy(source: View, dtype: DType | None = None) -> View:
    return _scatter(
        source=source,
        out_shape=source.shape,
        woffset=0,
        wpitch=c_contiguous_pitch_for_shape(source.shape),
        dtype=dtype or source.dtype,
    )


def debug_print(root: "View", out: SupportsWrite[str]) -> None:
    def build_tid_map() -> dict["Node", int]:
        reference_count_map = refcount([root])
        assert reference_count_map[root.node] == 1, (
            "Root tensor must have reference count 1"
        )

        tid_map: dict["Node", int] = {}
        for node, ref_count in reference_count_map.items():
            assert ref_count >= 1
            if ref_count == 1:
                continue
            tid_map[node] = len(tid_map)

        return tid_map

    def headline(node: "Node") -> str:
        base_fields = {field.name for field in fields(Node)}
        extra_fields = [f.name for f in fields(node) if f.name not in base_fields]
        args = ", ".join(f"{f}={getattr(node, f)!r}" for f in extra_fields)
        name = pascal_to_snake_case(node.__class__.__name__[: -len("Node")])
        return f"{name}({args}) :: {node.dtype}{node.shape!r}"

    def visit_view(
        view: "View",
        prefix: str,
        connector: str,
        prefix_ext: str,
        *,
        is_root: bool,
    ) -> None:
        if view._is_identity():
            visit_node(view.node, prefix, connector, prefix_ext, is_root=is_root)
            return

        a = view.accessor
        print(
            f"{prefix}{connector}view("
            f"offset={a.offset}, shape={a.shape!r}, pitch={a.pitch!r})",
            file=out,
        )
        visit_node(view.node, prefix + prefix_ext, "└ ", "  ", is_root=is_root)

    def visit_node(
        node: "Node",
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


def toposort(roots: Iterable["View"]) -> list["Node"]:
    visited: set["Node"] = set()
    topo_order: list["Node"] = []

    def visit(node: "Node") -> None:
        if node in visited:
            return
        visited.add(node)

        for arg in node.args:
            visit(arg.node)

        topo_order.append(node)

    for root in roots:
        visit(root.node)

    return topo_order


def refcount(roots: Iterable["View"]) -> dict["Node", int]:
    ref_counts: dict["Node", int] = {}

    def visit(node: "Node") -> None:
        if node in ref_counts:
            ref_counts[node] += 1
        else:
            ref_counts[node] = 1
            for operand in node.args:
                visit(operand.node)

    for root in roots:
        visit(root.node)

    return ref_counts


def flatten_pytree[T](pytree: PyTree[T]) -> Generator[T, None, None]:
    if isinstance(pytree, dict):
        for v in pytree.values():
            yield from flatten_pytree(v)
    elif isinstance(pytree, list):
        for v in pytree:
            yield from flatten_pytree(v)
    else:
        yield pytree


def map_pytree[T, U](pytree: PyTree[T], f: Callable[[T], U]) -> PyTree[U]:
    if isinstance(pytree, dict):
        return {k: map_pytree(v, f) for k, v in pytree.items()}
    elif isinstance(pytree, list):
        return [map_pytree(v, f) for v in pytree]
    else:
        return f(pytree)
