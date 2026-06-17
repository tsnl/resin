"""
resin.gpu.front models the computational graph of tensor operations.
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
    "const",
    "full",
    "ones",
    "param",
    "zeros",
]

import math
from abc import ABC
from dataclasses import dataclass, fields
from typing import Callable, Generator, Iterable

from ..common import SupportsWrite, pascal_to_snake_case
from .accessor import Accessor
from .pytree import PyTensor, infer_pytensor_shape
from .scalar import (
    BinaryAssocScalarOperator,
    Scalar,
    ScalarOperator,
    ScalarType,
    UnaryScalarOperator,
    stype_join,
    stype_nbytes,
)
from .shape import (
    c_contiguous_pitch_for_shape,
    shape_join,
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
    args: tuple["View", ...]

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
        new = View(
            node=self,
            accessor=Accessor(offset=offset, shape=shape, pitch=pitch),
        )
        new._raise_if_out_of_backing_node_bounds()
        return new

    def df_do(self, df_dout: "View") -> "tuple[View, ...]":
        """
        Given ∂f/∂n (df_dout, the gradient w.r.t. this node's dense output), returns
        (∂f/∂o₁, ∂f/∂o₂, ...) for each operand View oᵢ.

        Override this method in each Node subclass to implement differentiation for that
        node type. Raises NotDifferentiableException if the node is not differentiable.
        """

        # By default, nodes with no operands are trivially differentiable because they
        # have no gradients to propagate backward.
        if not self.args:
            return ()

        # If the node has operands but no df_do implementation, it's not differentiable.
        _ = df_dout
        raise NotDifferentiableException(self)


#
# View
#


@dataclass(frozen=True, eq=False)
class View:
    """
    An access pattern onto a backing Node's dense output buffer.

    View is the user-facing tensor type. All arithmetic operators and view operations
    are defined here. View operations transform the accessor in place, onto the same
    backing node; they never build chains of nodes.
    """

    node: "Node"
    accessor: Accessor

    @staticmethod
    def identity(node: "Node") -> "View":
        """A dense, C-contiguous view that addresses the whole of `node`'s output."""
        return View(node=node, accessor=Accessor.dense(node.shape))

    #
    # Properties:
    #

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
    def stype(self) -> ScalarType:
        return self.node.stype

    @property
    def rank(self) -> int:
        return self.accessor.rank

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * stype_nbytes(self.stype)

    #
    # View operations (transform the accessor onto the same backing node):
    #

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
        return self.accessor.is_dense_c_contiguous(self.node.shape)

    def _raise_if_out_of_backing_node_bounds(self) -> None:
        """Panics if this view addresses memory outside the (dense) backing node."""
        self.accessor.raise_if_addresses_out_of_bounds(math.prod(self.node.shape))

    @staticmethod
    def _from_view_or_scalar(value: "PyTensor | View", stype: ScalarType) -> "View":
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
            node=s.node,
            accessor=Accessor(offset=s.offset, shape=new_s_shape, pitch=new_s_pitch),
        )
        new_other = View(
            node=o.node,
            accessor=Accessor(offset=o.offset, shape=new_o_shape, pitch=new_o_pitch),
        )
        return new_self, new_other


#
# ConstNode
#


@dataclass(kw_only=True, frozen=True, eq=False)
class ConstNode(Node):
    value: PyTensor


def const(value: "PyTensor", *, stype: ScalarType = "f4") -> View:
    shape = infer_pytensor_shape(value)
    return View.identity(ConstNode(shape=shape, stype=stype, args=(), value=value))


def full(shape: tuple[int, ...], v: Scalar, *, stype: ScalarType = "f4") -> View:
    return const(v, stype=stype).broadcast(shape)


def ones(shape: tuple[int, ...], *, stype: ScalarType = "f4") -> View:
    return full(shape, 1, stype=stype)


def zeros(shape: tuple[int, ...], *, stype: ScalarType = "f4") -> View:
    return full(shape, 0, stype=stype)


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
    return View.identity(ParamNode(shape=shape, stype=stype, args=(), label=label))


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
                return (df_dout / self.args[0],)
            case "sqrt":
                # ∂n/∂o₁ = 1 / (2 * sqrt(o₁)) = 1 / (2 * n)
                return (df_dout / (2 * n),)
            case "sin":
                # ∂n/∂o₁ = cos(o₁)
                return (df_dout * self.args[0].cos(),)
            case "cos":
                # ∂n/∂o₁ = -sin(o₁)
                return (df_dout * -self.args[0].sin(),)
            case "pow":
                # ∂n/∂o₁ = o₂ * o₁^(o₂ - 1) = o₂ * n / o₁
                # ∂n/∂o₂ = log(o₁) * o₁^o₂ = log(o₁) * n
                return (
                    df_dout * self.args[1] * n / self.args[0],
                    df_dout * self.args[0].log() * n,
                )
            case "mul":
                # ∂n/∂o₁ = o₂, ∂n/∂o₂ = o₁
                return (
                    df_dout * self.args[1],
                    df_dout * self.args[0],
                )
            case "div":
                # ∂n/∂o₁ = 1 / o₂, ∂n/∂o₂ = -o₁ / o₂² = -n / o₂
                return (
                    df_dout / self.args[1],
                    df_dout * -n / self.args[1],
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
                    df_dout * self.args[0].gt(self.args[1]),
                    df_dout * self.args[1].gt(self.args[0]),
                )
            case "min":
                # ∂n/∂o₁ = 1 if o₁ < o₂ else 0
                # ∂n/∂o₂ = 1 if o₂ < o₁ else 0
                return (
                    df_dout * self.args[0].lt(self.args[1]),
                    df_dout * self.args[1].lt(self.args[0]),
                )
            case "not" | "eq" | "ne" | "lt" | "gt" | "le" | "ge":
                raise NotDifferentiableException(self)
            case _:
                raise NotImplementedError(f"{self.operator=}")


def _elementwise_unary(operand: View, operator: UnaryScalarOperator) -> View:
    return View.identity(
        ElementwiseNode(
            shape=operand.shape,
            stype=operand.stype,
            args=(operand,),
            operator=operator,
        )
    )


def _elementwise_binary(
    a: View,
    b: View | Scalar,
    operator: ScalarOperator,
) -> View:
    b = View._from_view_or_scalar(b, stype=a.stype)
    a, b = a._join_dtypes_for_bop(b)
    a, b = a._join_shapes_for_elementwise_bop(b)
    return View.identity(
        ElementwiseNode(
            shape=a.shape,
            stype=a.stype,
            args=(a, b),
            operator=operator,
        )
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
        operand = self.args[0]

        # Broadcast df_dout (with reduced axes of size 1) back up to the operand shape.
        g = View(
            node=df_dout.node,
            accessor=Accessor(
                offset=df_dout.offset,
                shape=operand.shape,
                pitch=tuple(
                    (0 if i in self.axes else df_dout.pitch[i])
                    for i in range(operand.rank)
                ),
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
            args=(input,),
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
            df_dout @ self.args[1].transpose(),
            self.args[0].transpose() @ df_dout,
        )


def _matmul(a: View, b: View) -> View:
    a, b = a._join_dtypes_for_bop(b)
    a, b = a._join_shapes_for_matmul_bop(b)
    out_shape = a.shape[:-1] + (b.shape[-1],)
    return View.identity(MatmulNode(shape=out_shape, stype=a.stype, args=(a, b)))


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
        source = self.args[0]
        dense = df_dout if df_dout._is_identity() else df_dout.copy()
        return (
            View(
                node=dense.node,
                accessor=Accessor(
                    offset=self.woffset,
                    shape=source.shape,
                    pitch=self.wpitch,
                ),
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
            args=(source,),
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

        for arg in node.args:
            visit(arg.node)

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
            for operand in node.args:
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

        df_do = node.df_do(df_dn)

        for operand, df_do_i in zip(node.args, df_do):
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
