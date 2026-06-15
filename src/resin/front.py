"""
resin.front models the computational graph of tensor operations.
-   Node is an operation in the graph. Its output is always a dense, C-contiguous array
    described solely by `shape`.
-   View pairs an access pattern (offset, shape, pitch) with a backing Node. A Node reads
    each of its operands through a (potentially sparse) View and writes a dense output.
    View is the user-facing tensor type: all arithmetic and view operations are defined
    on it.
-   Each Node subclass is accompanied by free functions that construct it and return a
    View onto its output (public for leaves: `const`, `ones`, `param`, ...; private for
    operations: `_matmul`, `_reduction`, ...).
-   The graph is static and acyclic, but not necessarily a tree (e.g. shared subgraphs).
-   Parameter nodes map to buffers that must be written when executing the graph.

Because every Node output is dense, view composition never produces chains of nodes:
each view operation transforms the current View's accessor in place, onto the same
backing Node. Materializing a C-contiguous intermediate is explicit, via `View.copy()`
(a ScatterNode).
"""

__all__ = [
    "ConstNode",
    "ElementwiseNode",
    "MatmulNode",
    "Node",
    "NotDifferentiableException",
    "ParamNode",
    "ReductionNode",
    "ScatterNode",
    "View",
    "c_contiguous_pitch_for_shape",
    "const",
    "full",
    "is_c_contiguous",
    "is_contiguous",
    "ones",
    "param",
    "zeros",
]

import math
from abc import ABC
from dataclasses import dataclass, fields
from typing import Callable, Generator, Iterable

import numpy.typing as npt

from .common import SupportsWrite, pascal_to_snake_case
from .scalar import (
    BinaryAssocScalarOperator,
    Scalar,
    ScalarOperator,
    ScalarType,
    UnaryScalarOperator,
    is_scalar,
    stype_join,
    stype_nbytes,
)


class NotDifferentiableException(Exception):
    pass


#
# Node
#


@dataclass(kw_only=True, frozen=True, eq=False)
class Node(ABC):
    """
    An operation in the graph. Its output is always a dense, C-contiguous array of the
    given `shape`. Operands are read through (potentially sparse) Views.
    """

    shape: tuple[int, ...]
    stype: ScalarType
    input: tuple["View", ...]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * stype_nbytes(self.stype)

    def view(
        self,
        *,
        offset: int = 0,
        shape: tuple[int, ...],
        pitch: tuple[int, ...],
    ) -> "View":
        """
        Returns a View onto this node's dense output buffer with the given access
        pattern, in backing-buffer coordinates. Panics if the view addresses memory
        outside this node.

        This is the deliberate escape hatch for reinterpreting a buffer. To reinterpret
        an existing view, reach for its backing node explicitly (`v.node.view(...)`).
        """
        new = View(node=self, offset=offset, shape=shape, pitch=pitch)
        new._raise_if_out_of_backing_node_bounds()
        return new

    def df_do(self, df_dout: "View") -> "tuple[View, ...]":
        """
        Given ∂f/∂n (df_dout, the gradient w.r.t. this node's dense output), returns
        (∂f/∂o₁, ∂f/∂o₂, ...) for each operand View oᵢ.

        Override this method in each Node subclass to implement differentiation for that
        node type. Raises NotDifferentiableException if the node is not differentiable.
        """
        _ = df_dout
        raise NotDifferentiableException(self)


#
# View
#


@dataclass(frozen=True, eq=False)
class View:
    """
    An access pattern (offset, shape, pitch) onto a backing Node's dense output buffer.

    View is the user-facing tensor type. All arithmetic operators and view operations
    are defined here. View operations transform the accessor in place, onto the same
    backing node; they never build chains of nodes.
    """

    node: "Node"
    offset: int
    shape: tuple[int, ...]
    pitch: tuple[int, ...]

    def __post_init__(self):
        assert len(self.shape) == len(self.pitch)

    @staticmethod
    def identity(node: "Node") -> "View":
        """A dense, C-contiguous view that addresses the whole of `node`'s output."""
        return View(
            node=node,
            offset=0,
            shape=node.shape,
            pitch=c_contiguous_pitch_for_shape(node.shape),
        )

    #
    # Properties:
    #

    @property
    def stype(self) -> ScalarType:
        return self.node.stype

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * stype_nbytes(self.stype)

    #
    # View operations (transform the accessor onto the same backing node):
    #

    def broadcast(self, ns: tuple[int, ...]) -> "View":
        zs = (0,) * len(ns)
        return View(
            node=self.node,
            offset=self.offset,
            shape=ns + self.shape,
            pitch=zs + self.pitch,
        )

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> "View":
        key = (key,) if isinstance(key, (int, slice)) else key
        offset, shape, pitch = from_key(self.offset, self.shape, self.pitch, key)
        return View(node=self.node, offset=offset, shape=shape, pitch=pitch)

    def permute(self, permutation: tuple[int, ...]) -> "View":
        if sorted(permutation) != list(range(self.rank)):
            raise ValueError(f"Bad permutation {permutation} for shape {self.shape}")
        return View(
            node=self.node,
            offset=self.offset,
            shape=tuple(self.shape[i] for i in permutation),
            pitch=tuple(self.pitch[i] for i in permutation),
        )

    def transpose(self) -> "View":
        """Permutes dimensions so the last two are swapped, the rest left unaltered."""
        identity = tuple(range(self.rank))
        permutation = identity[:-2] + (identity[-1], identity[-2])
        return self.permute(permutation)

    def squeeze(self, axes: tuple[int, ...]) -> "View":
        for axis in axes:
            if axis < 0 or axis >= self.rank:
                raise IndexError(f"Axis {axis} out of bounds for shape {self.shape}")
            if self.shape[axis] != 1:
                raise ValueError(f"Cannot squeeze axis {axis} with {self.shape[axis]=}")
        return View(
            node=self.node,
            offset=self.offset,
            shape=tuple(s for i, s in enumerate(self.shape) if i not in axes),
            pitch=tuple(p for i, p in enumerate(self.pitch) if i not in axes),
        )

    def copy(self, *, stype: ScalarType | None = None) -> "View":
        return _copy(self, stype=stype)

    #
    # Elementwise / reduction / matmul operations (produce a new dense Node):
    #

    def __pow__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="pow")

    def __rpow__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, stype=self.stype) ** self

    def __mul__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="mul")

    def __rmul__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, stype=self.stype) * self

    def __truediv__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="div")

    def __rtruediv__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, stype=self.stype) / self

    def __add__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="add")

    def __radd__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, stype=self.stype) + self

    def __sub__(self, other: "View | Scalar") -> "View":
        return _elementwise_binary(self, other, operator="sub")

    def __rsub__(self, other: "View | Scalar") -> "View":
        return View._from_view_or_scalar(other, stype=self.stype) - self

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

    def __invert__(self) -> "View":
        return _elementwise_unary(self, operator="not")

    def __matmul__(self, other: "View") -> "View":
        return _matmul(self, other)

    def __rmatmul__(self, other: "View") -> "View":
        return View._from_view_or_scalar(other, stype=self.stype) @ self

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

    #
    # Debug printing:
    #

    def debug_print(self, out: SupportsWrite[str]):
        debug_print(self, out)

    #
    # Differentiation:
    #

    def grad(self) -> dict["Node", "View"]:
        """
        Returns a mapping from each backing Node in the subgraph rooted at this view to
        its gradient with respect to this view.
        """
        return grad(self)

    def df_do(self, df_dview: "View") -> "tuple[View, ...]":
        """
        Given ∂f/∂(this view), returns (∂f/∂o₁, ∂f/∂o₂, ...) for each operand View of the
        backing node.

        Backprop factors into two linear steps: the accessor adjoint (push the gradient
        from this view's logical space into the backing node's dense space), then the
        node's op-adjoint (push it on to the node's operands).
        """
        return self.node.df_do(self._accessor_adjoint(df_dview))

    def _accessor_adjoint(self, g: "View") -> "View":
        """
        Pushes a gradient `g` (w.r.t. this view's logical values) back into the backing
        node's dense output space.

        - Broadcast dims (pitch 0) map one input element to many outputs, so they are
          sum-reduced.
        - The reduced gradient is then scattered back into the backing node's dense
          shape using this view's write location (offset, pitch). Unwritten elements are
          zero.
        """
        broadcast_axes = tuple(i for i, p in enumerate(self.pitch) if p == 0)
        x = g.reduce(axes=broadcast_axes, operator="add") if broadcast_axes else g

        # A dense identity view is already in the node's output space.
        if self._is_identity():
            return x

        return _scatter(
            source=x,
            out_shape=self.node.shape,
            woffset=self.offset,
            wpitch=self.pitch,
        )

    #
    # Private:
    #

    def _is_identity(self) -> bool:
        """True if this view addresses the whole backing node densely (C-contiguous)."""
        return (
            self.offset == 0
            and self.shape == self.node.shape
            and self.pitch == c_contiguous_pitch_for_shape(self.shape)
        )

    def _raise_if_out_of_backing_node_bounds(self) -> None:
        """Panics if this view addresses memory outside the (dense) backing node."""
        if self.offset < 0:
            raise ValueError(f"View offset {self.offset} is negative")
        max_address = self.offset + sum(
            (s - 1) * p for s, p in zip(self.shape, self.pitch) if p > 0
        )
        capacity = math.prod(self.node.shape)
        if self.shape and math.prod(self.shape) > 0 and max_address >= capacity:
            raise ValueError(
                f"View (offset={self.offset}, shape={self.shape}, pitch={self.pitch}) "
                f"addresses element {max_address} outside backing node of "
                f"{capacity} elements"
            )

    @staticmethod
    def _from_view_or_scalar(
        value: "npt.ArrayLike | View", stype: ScalarType
    ) -> "View":
        return value if isinstance(value, View) else const(value, stype=stype)

    def _join_dtypes_for_bop(self, other: "View") -> tuple["View", "View"]:
        res_dtype = stype_join(self.stype, other.stype)
        s = self.copy(stype=res_dtype) if self.stype != res_dtype else self
        o = other.copy(stype=res_dtype) if other.stype != res_dtype else other
        return s, o

    def _join_shapes_for_elementwise_bop(self, other: "View") -> tuple["View", "View"]:
        join = shape_join(self.shape, self.pitch, other.shape, other.pitch)
        return (
            View(
                node=self.node, offset=self.offset, shape=join.shape, pitch=join.pitch1
            ),
            View(
                node=other.node,
                offset=other.offset,
                shape=join.shape,
                pitch=join.pitch2,
            ),
        )

    def _join_shapes_for_matmul_bop(self, other: "View") -> tuple["View", "View"]:
        s = self
        o = other

        # Ensure both operands are rank 2 or higher:
        if s.rank < 2 or o.rank < 2:
            raise ValueError(
                f"Shapes {s.shape} and {o.shape} are not compatible for matmul: "
                "both tensors must be 2D or higher"
            )

        # Compute broadcasting behavior by gathering corresponding dimensions into two
        # parallel arrays, joining those arrays, and then scattering the result back for
        # the original operands.
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

        # Finalize:
        new_self = View(
            node=s.node, offset=s.offset, shape=new_s_shape, pitch=new_s_pitch
        )
        new_other = View(
            node=o.node, offset=o.offset, shape=new_o_shape, pitch=new_o_pitch
        )
        return new_self, new_other


#
# ConstNode
#


@dataclass(kw_only=True, frozen=True, eq=False)
class ConstNode(Node):
    value: npt.ArrayLike


def const(value: "npt.ArrayLike", *, stype: ScalarType = "fp32") -> View:
    shape = _infer_value_shape(value)
    return View.identity(ConstNode(shape=shape, stype=stype, input=(), value=value))


def full(shape: tuple[int, ...], v: Scalar, *, stype: ScalarType = "fp32") -> View:
    return const(v, stype=stype).broadcast(shape)


def ones(shape: tuple[int, ...], *, stype: ScalarType = "fp32") -> View:
    return full(shape, 1, stype=stype)


def zeros(shape: tuple[int, ...], *, stype: ScalarType = "fp32") -> View:
    return full(shape, 0, stype=stype)


def _infer_value_shape(value: "npt.ArrayLike") -> tuple[int, ...]:
    if is_scalar(value):
        return ()
    assert isinstance(value, list)
    if not value:
        return (0,)
    e0_shape = _infer_value_shape(value[0])
    for e1 in value[1:]:
        if _infer_value_shape(e1) != e0_shape:
            raise ValueError("Inconsistent shapes in nested list")
    return (len(value),) + e0_shape


#
# ParamNode
#


@dataclass(kw_only=True, frozen=True, eq=False)
class ParamNode(Node):
    label: str | None  # non-unique, for debug only


def param(
    *,
    shape: tuple[int, ...],
    stype: ScalarType,
    label: str | None = None,
) -> View:
    return View.identity(ParamNode(shape=shape, stype=stype, input=(), label=label))


#
# ElementwiseNode
#


@dataclass(kw_only=True, frozen=True, eq=False)
class ElementwiseNode(Node):
    operator: ScalarOperator

    def df_do(self, df_dout: "View") -> tuple[View, ...]:
        n = View.identity(self)
        match self.operator:
            case "neg":
                return (-df_dout,)
            case "exp":
                # ∂n/∂o₁ = exp(o₁) = n
                return (df_dout * n,)
            case "log":
                # ∂n/∂o₁ = 1 / o₁
                return (df_dout / self.input[0],)
            case "pow":
                # ∂n/∂o₁ = o₂ * o₁^(o₂ - 1) = o₂ * n / o₁
                # ∂n/∂o₂ = log(o₁) * o₁^o₂ = log(o₁) * n
                return (
                    df_dout * self.input[1] * n / self.input[0],
                    df_dout * self.input[0].log() * n,
                )
            case "mul":
                # ∂n/∂o₁ = o₂, ∂n/∂o₂ = o₁
                return (
                    df_dout * self.input[1],
                    df_dout * self.input[0],
                )
            case "div":
                # ∂n/∂o₁ = 1 / o₂, ∂n/∂o₂ = -o₁ / o₂² = -n / o₂
                return (
                    df_dout / self.input[1],
                    df_dout * -n / self.input[1],
                )
            case "add":
                # ∂n/∂o₁ = ∂n/∂o₂ = 1
                return (df_dout, df_dout)
            case "sub":
                # ∂n/∂o₁ = 1, ∂n/∂o₂ = -1
                return (df_dout, -df_dout)
            case "max":
                # ∂n/∂o₁ = 1 if o₁ > o₂ else 0
                # ∂n/∂o₂ = 1 if o₂ > o₁ else 0
                return (
                    df_dout * self.input[0].gt(self.input[1]),
                    df_dout * self.input[1].gt(self.input[0]),
                )
            case "min":
                # ∂n/∂o₁ = 1 if o₁ < o₂ else 0
                # ∂n/∂o₂ = 1 if o₂ < o₁ else 0
                return (
                    df_dout * self.input[0].lt(self.input[1]),
                    df_dout * self.input[1].lt(self.input[0]),
                )
            case "eq" | "ne" | "lt" | "gt" | "le" | "ge":
                raise NotDifferentiableException(self)
            case _:
                raise NotImplementedError(f"{self.operator=}")


def _elementwise_unary(operand: View, operator: UnaryScalarOperator) -> View:
    return View.identity(
        ElementwiseNode(
            shape=operand.shape,
            stype=operand.stype,
            input=(operand,),
            operator=operator,
        )
    )


def _elementwise_binary(
    a: View,
    b: "View | Scalar",
    operator: ScalarOperator,
) -> View:
    b = View._from_view_or_scalar(b, stype=a.stype)
    a, b = a._join_dtypes_for_bop(b)
    a, b = a._join_shapes_for_elementwise_bop(b)
    return View.identity(
        ElementwiseNode(shape=a.shape, stype=a.stype, input=(a, b), operator=operator)
    )


#
# ReductionNode
#


@dataclass(kw_only=True, frozen=True, eq=False)
class ReductionNode(Node):
    """
    Performs a reduction (e.g. sum, max) along one or more axes of the input tensor,
    using a specified associative binary operator (e.g. mul, add, max, min).

    The output tensor has the same shape as the input tensor, except that the reduced
    axes have been replaced with size 1.
    """

    operator: BinaryAssocScalarOperator
    axes: tuple[int, ...]

    def df_do(self, df_dout: "View") -> tuple[View, ...]:
        operand = self.input[0]

        # Broadcast df_dout (with reduced axes of size 1) back up to the operand shape.
        g = View(
            node=df_dout.node,
            offset=df_dout.offset,
            shape=operand.shape,
            pitch=tuple(
                (0 if i in self.axes else df_dout.pitch[i]) for i in range(operand.rank)
            ),
        )

        match self.operator:
            case "mul":
                # ∂n/∂o₁ = n / o₁
                return (g * (View.identity(self) / operand),)
            case "add":
                # ∂n/∂o₁ = 1
                return (g,)
            case "max" | "min":
                # ∂n/∂o₁ = 1 if o₁ is the max/min along the reduction axes, else 0.
                # The condition is identical for both "max" and "min".
                cond = operand.eq(View.identity(self))
                return (g * cond,)
            case _:
                raise NotImplementedError(f"{self.operator=}")


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
            stype=input.stype,
            input=(input,),
            operator=operator,
            axes=axes,
        )
    )


#
# MatmulNode
#


@dataclass(kw_only=True, frozen=True, eq=False)
class MatmulNode(Node):
    def df_do(self, df_dout: "View") -> tuple[View, ...]:
        # ∂n/∂o₁ = df/dn @ o₂.T
        # ∂n/∂o₂ = o₁.T @ df/dn
        return (
            df_dout @ self.input[1].transpose(),
            self.input[0].transpose() @ df_dout,
        )


def _matmul(a: View, b: View) -> View:
    a, b = a._join_dtypes_for_bop(b)
    a, b = a._join_shapes_for_matmul_bop(b)
    out_shape = a.shape[:-1] + (b.shape[-1],)
    return View.identity(MatmulNode(shape=out_shape, stype=a.stype, input=(a, b)))


#
# ScatterNode
#


@dataclass(kw_only=True, frozen=True, eq=False)
class ScatterNode(Node):
    """
    Scatter maps a dense "source" to sparse destinations in a freshly allocated,
    C-contiguous output. Unlike a read view, (woffset, wpitch) specify *write*
    locations, not read locations: source element at logical index `i` is written to
    output flat index `woffset + Σ i·wpitch`. The write shape equals the source's shape.
    All unwritten output elements are zero.

    Example NumPy code:

    ```python
    def scatter(source, out_shape, woffset, wpitch):
        res = np.zeros(shape=out_shape, dtype=source.dtype)
        # write source[i] to flat index woffset + dot(i, wpitch) of res
        ...
        return res
    ```

    `copy` (a dense round trip) is the special case where the write pattern is the dense
    C-contiguous layout of the source shape.
    """

    woffset: int
    wpitch: tuple[int, ...]

    def df_do(self, df_dout: "View") -> tuple[View, ...]:
        # The gradient w.r.t. the source is df_dout read back at the write locations.
        # This requires reading in the output's dense coordinate frame, so materialize
        # df_dout to a dense identity view first if it is not already one.
        source = self.input[0]
        dense = df_dout if df_dout._is_identity() else df_dout.copy()
        return (
            View(
                node=dense.node,
                offset=self.woffset,
                shape=source.shape,
                pitch=self.wpitch,
            ),
        )


def _scatter(
    *,
    source: View,
    out_shape: tuple[int, ...],
    woffset: int,
    wpitch: tuple[int, ...],
    stype: ScalarType | None = None,
) -> View:
    return View.identity(
        ScatterNode(
            shape=out_shape,
            stype=stype or source.stype,
            input=(source,),
            woffset=woffset,
            wpitch=wpitch,
        )
    )


def _copy(source: View, stype: ScalarType | None = None) -> View:
    return _scatter(
        source=source,
        out_shape=source.shape,
        woffset=0,
        wpitch=c_contiguous_pitch_for_shape(source.shape),
        stype=stype or source.stype,
    )


#
# Shape, Pitch:
#


def from_key(
    old_offset: int,
    old_shape: tuple[int, ...],
    old_pitch: tuple[int, ...],
    key: tuple[int | slice, ...],
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """
    Computes the (offset, shape, pitch) of the subview selected by `key` from a view
    with the given (old_offset, old_shape, old_pitch), in backing-buffer coordinates.
    """

    def bounded_index(k: int) -> int:
        d = old_shape[dim]
        if not (-d <= k < d):
            raise IndexError(f"Index {k} out of bounds for dimension {dim} of size {d}")
        return k % d

    def bounded_end(k: int) -> int:
        d = old_shape[dim]
        if not (0 <= k <= d):
            raise IndexError(
                f"Slice end {k} out of bounds for dimension {dim} of size {d}"
            )
        return k

    assert len(key) <= len(old_shape)

    new_offset = old_offset
    new_pitch = []
    new_shape = []

    for dim, k in enumerate(key):
        match k:
            case int():
                new_offset += bounded_index(k) * old_pitch[dim]
            case slice():
                b = bounded_index(k.start) if k.start is not None else 0
                e = bounded_end(k.stop) if k.stop is not None else old_shape[dim]
                s = k.step if k.step is not None else 1
                a = abs(s)

                new_offset += b * old_pitch[dim]
                new_pitch.append(old_pitch[dim] * s)
                new_shape.append(max(0, (e - b + (a - 1)) // a))
            case _:
                raise TypeError(f"Invalid index {k} for dimension {dim}")

    for dim in range(len(key), len(old_shape)):
        new_pitch.append(old_pitch[dim])
        new_shape.append(old_shape[dim])

    return new_offset, tuple(new_shape), tuple(new_pitch)


@dataclass
class ShapeJoin:
    shape: tuple[int, ...]
    pitch1: tuple[int, ...]
    pitch2: tuple[int, ...]


def shape_join(
    shape1: tuple[int, ...],
    pitch1: tuple[int, ...],
    shape2: tuple[int, ...],
    pitch2: tuple[int, ...],
    report_shape1: tuple[int, ...] | None = None,
    report_shape2: tuple[int, ...] | None = None,
) -> ShapeJoin:
    # Swap args and re-enter if needed to ensure len(shape1) <= len(shape2)
    if len(shape1) > len(shape2):
        join = shape_join(shape2, pitch2, shape1, pitch1)
        return ShapeJoin(shape=join.shape, pitch1=join.pitch2, pitch2=join.pitch1)

    # Broadcast self up to other's dim if needed and re-enter.
    if len(shape1) < len(shape2):
        p_ndim = len(shape2) - len(shape1)
        shape1 = (1,) * p_ndim + shape1
        pitch1 = (0,) * p_ndim + pitch1
        assert len(pitch1) == len(pitch2)
        return shape_join(shape1=shape1, pitch1=pitch1, shape2=shape2, pitch2=pitch2)

    # From here on, both have same ndim.
    assert len(shape1) == len(shape2)

    # If both have same ndim, check if shapes are compatible for broadcasting.
    # Each dimension must either be the same or one of them must be unity.
    # We also build the new pitch for each operand tensor at the same time.
    new_shape_list = []
    new_pitch1_list = []
    new_pitch2_list = []
    for i_dim, (s, o) in enumerate(zip(shape1, shape2)):
        if s != o and s != 1 and o != 1:
            report_shape1 = report_shape1 or shape1
            report_shape2 = report_shape2 or shape2
            raise ValueError(
                f"Shapes {report_shape1} and {report_shape2} are not compatible for "
                "broadcasting."
            )

        new_shape_list.append(max(s, o))
        new_pitch1_list.append(0 if s == 1 else pitch1[i_dim])
        new_pitch2_list.append(0 if o == 1 else pitch2[i_dim])

    out_shape = tuple(new_shape_list)
    new_pitch1 = tuple(new_pitch1_list)
    new_pitch2 = tuple(new_pitch2_list)

    # Finalize:
    return ShapeJoin(shape=out_shape, pitch1=new_pitch1, pitch2=new_pitch2)


# Contiguity and permutation:
# - Contiguous: no gaps between elements in memory (i.e. no "holes" in the tensor).
# - Permutation: a reordering of the dimensions of a tensor. E.g. transpose of matrix.
# - Permutation only involves reordering shape and pitch, no data copy needed.
# - C-contiguous: contiguous and has monotonically decreasing strides (like C arrays).
# - Lemma: tensor is contiguous if and only if it can be made C-contiguous by permuting
#   dimensions.


def c_contiguous_pitch_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Returns the C-contiguous strides for a given shape.

    An array is C-contiguous if it both...
    -   Contains no gaps between elements
    -   Has monotonically decreasing strides (like C arrays).

    Note that it is possible for a tensor to be contiguous without being C-contiguous.
    E.g. a permutation of a C-contiguous tensor.
    """
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


def is_c_contiguous(shape: tuple[int, ...], pitch: tuple[int, ...]) -> bool:
    """
    Check if a tensor with the given shape and pitch is C-contiguous.
    """
    return pitch == c_contiguous_pitch_for_shape(shape)


def is_contiguous(shape: tuple[int, ...], pitch: tuple[int, ...]) -> bool:
    new_shape, new_pitch = c_permuted(shape, pitch)
    return is_c_contiguous(new_shape, new_pitch)


def c_permuted(
    shape: tuple[int, ...], pitch: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Permute the dimensions by decreasing pitch, breaking ties by decreasing shape, and
    return the new shape and pitch.

    Note that this makes contiguous shapes C-contiguous.

    This normalization is useful even for comparing non-contiguous shapes.
    """

    permutation = c_permutation(shape, pitch)
    new_shape = permute(shape, permutation)
    new_pitch = permute(pitch, permutation)
    return new_shape, new_pitch


def c_permutation(shape: tuple[int, ...], pitch: tuple[int, ...]) -> tuple[int, ...]:
    """
    Permute the dimensions by decreasing pitch, breaking ties by decreasing
    shape, and return the new shape and pitch.

    Note that this makes contiguous shapes C-contiguous.

    This normalization is useful even for comparing non-contiguous shapes.
    """

    # argsort the dimensions by decreasing pitch, breaking ties by decreasing
    # shape.
    return tuple(
        sorted(
            range(len(pitch)),
            key=lambda i: (pitch[i], shape[i]),
            reverse=True,
        )
    )


def invert_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    """
    Computes the inverse of a permutation, i.e. a permutation `inv` such that
        forall i: inv[permutation[i]] == i
        forall i: permutation[inv[i]] == i
    """

    n = len(permutation)
    assert set(permutation) == set(range(n)), "Invalid permutation"
    return tuple(permutation.index(i) for i in range(n))


def permute[T](seq: tuple[T, ...], perm: tuple[int, ...]) -> tuple[T, ...]:
    """
    Applies a permutation to a tuple, returning the permuted tuple.
    """

    assert len(seq) == len(perm), "Permutation length must match sequence length"
    return tuple(seq[i] for i in perm)


#
# Debug print:
#


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
        return f"{name}({args}) :: {node.stype}{node.shape!r}"

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

        # Render the access pattern, then the backing node as its only child.
        print(
            f"{prefix}{connector}view("
            f"offset={view.offset}, shape={view.shape!r}, pitch={view.pitch!r})",
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
        for operand_index, operand in enumerate(node.input):
            is_last = operand_index == len(node.input) - 1
            c_connector = "└ " if is_last else "├ "
            c_ext = "  " if is_last else "│ "
            visit_view(operand, child_prefix, c_connector, c_ext, is_root=False)

    tid_map = build_tid_map()

    visit_view(root, "", "", "", is_root=True)

    for node in tid_map:
        visit_node(node, "", "", "", is_root=True)


#
# Graph operations:
#


def toposort(roots: Iterable["View"]) -> list["Node"]:
    """
    Returns a list of all nodes in the subgraph rooted at the given views, sorted in
    topological order (i.e. each node appears after all its inputs).
    """
    visited: set["Node"] = set()
    topo_order: list["Node"] = []

    def visit(node: "Node") -> None:
        if node in visited:
            return
        visited.add(node)

        for operand in node.input:
            visit(operand.node)

        topo_order.append(node)

    for root in roots:
        visit(root.node)

    return topo_order


def refcount(roots: Iterable["View"]) -> dict["Node", int]:
    """
    Compute reference counts for all nodes in the subgraph rooted at the given views.
    """
    ref_counts: dict["Node", int] = {}

    def visit(node: "Node") -> None:
        if node in ref_counts:
            ref_counts[node] += 1
        else:
            ref_counts[node] = 1
            for operand in node.input:
                visit(operand.node)

    for root in roots:
        visit(root.node)

    return ref_counts


def grad(f: "View") -> dict["Node", "View"]:
    """
    Given a forward graph `f` that computes a scalar output, returns a mapping from each
    backing Node in `f` to its gradient with respect to the output.

    The returned gradient graph holds references to views in the forward graph for
    efficient reuse of forward pass expressions in the backward pass.
    """

    if f.shape != ():
        raise ValueError("Output graph must be a scalar (i.e. have shape=())")

    # grad_node[n] = ∂f / ∂(dense output of n)
    grad_node: dict["Node", "View"] = {}

    def accumulate(view: "View", g: "View") -> None:
        contrib = view._accessor_adjoint(g)
        existing = grad_node.get(view.node)
        grad_node[view.node] = (existing + contrib) if existing is not None else contrib

    # Initialize: ∂f / ∂f = 1
    accumulate(f, ones(f.shape, stype=f.stype))

    # Traverse the graph in reverse topological order, accumulating gradients for each
    # node.
    for node in reversed(toposort([f])):
        df_dn = grad_node.get(node)
        if df_dn is None:
            continue

        try:
            df_do = node.df_do(df_dn)
        except NotDifferentiableException:
            continue

        for operand, df_do_i in zip(node.input, df_do):
            accumulate(operand, df_do_i)

    return grad_node


#
# PyTree
#


type PyTree[T] = "dict[str, PyTree[T]] | list[PyTree[T]] | T"
"""Similar to PyTree in JAX."""


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
