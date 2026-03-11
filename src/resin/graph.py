"""
resin.graph models the computational graph of tensor operations.
-   Node is both a tensor memory and an instruction in the graph.
-   The graph is static and acyclic, but not necessarily a tree (e.g. shared subgraphs).
-   Parameter nodes map to buffers that must be written when executing the graph.

Nodes will frequently reuse their first operand's shape, pitch, etc. This makes it
easier for the backend to reuse memory and fuse kernels.

TODO: Refactor: make "views" a property of each node referencing another node instead of
a node of its own.
"""

__all__ = [
    "Accessor",
    "ConstNode",
    "ElementwiseNode",
    "MatmulNode",
    "Node",
    "NotDifferentiableException",
    "ParamNode",
    "ReductionNode",
    "ScatterNode",
    "ViewNode",
    "is_c_contiguous",
    "is_contiguous",
]

import math
from abc import ABC
from dataclasses import dataclass, fields, replace
from typing import Callable, Generator, Iterable, Literal

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
    offset: int
    shape: tuple[int, ...]
    pitch: tuple[int, ...]
    stype: ScalarType
    input: tuple["Node", ...]

    def __post_init__(self):
        assert len(self.shape) == len(self.pitch)

    #
    # Properties:
    #

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * stype_nbytes(self.stype)

    #
    # Operations:
    #

    def broadcast(self, ns: tuple[int, ...]) -> "Node":
        return ViewNode.new_broadcast(input=self, ns=ns)

    def view(
        self,
        *,
        offset: int = 0,
        shape: tuple[int, ...],
        pitch: tuple[int, ...],
    ) -> "Node":
        accessor = Accessor(offset=offset, shape=shape, pitch=pitch)
        return ViewNode.new(input=self, accessor=accessor)

    def copy(self, *, stype: ScalarType | None = None) -> "Node":
        return ScatterNode.new_copy(source=self, stype=stype)

    def __pow__(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="pow")

    def __rpow__(self, other: "Node | Scalar") -> "Node":
        return Node._from_node_or_scalar(other, stype=self.stype) ** self

    def __mul__(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="mul")

    def __rmul__(self, other: "Node | Scalar") -> "Node":
        return Node._from_node_or_scalar(other, stype=self.stype) * self

    def __truediv__(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="div")

    def __rtruediv__(self, other: "Node | Scalar") -> "Node":
        return Node._from_node_or_scalar(other, stype=self.stype) / self

    def __add__(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="add")

    def __radd__(self, other: "Node | Scalar") -> "Node":
        return Node._from_node_or_scalar(other, stype=self.stype) + self

    def __sub__(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="sub")

    def __rsub__(self, other: "Node | Scalar") -> "Node":
        return Node._from_node_or_scalar(other, stype=self.stype) - self

    def max(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="max")

    def min(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="min")

    def eq(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="eq")

    def ne(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="ne")

    def lt(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="lt")

    def gt(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="gt")

    def le(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="le")

    def ge(self, other: "Node | Scalar") -> "Node":
        return self._elementwise_bop(other, operator="ge")

    def __neg__(self) -> "Node":
        return self._elementwise_uop(operator="neg")

    def __pos__(self) -> "Node":
        return self

    def exp(self) -> "Node":
        return self._elementwise_uop(operator="exp")

    def log(self) -> "Node":
        return self._elementwise_uop(operator="log")

    def __invert__(self) -> "Node":
        return self._elementwise_uop(operator="not")

    def __matmul__(self, other: "Node") -> "Node":
        return MatmulNode.new(a=self, b=other)

    def __rmatmul__(self, other: "Node") -> "Node":
        return Node._from_node_or_scalar(other, stype=self.stype) @ self

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> "Node":
        key = (key,) if isinstance(key, (int, slice)) else key
        return ViewNode.new_index(input=self, key=key)

    def reduce(
        self,
        *,
        operator: BinaryAssocScalarOperator,
        axes: tuple[int, ...] | None,
    ) -> "Node":
        axes = tuple(range(len(self.shape))) if axes is None else axes
        return ReductionNode.new(input=self, axes=axes, operator=operator)

    def sum(self, axes: tuple[int, ...] | None = None) -> "Node":
        return self.reduce(axes=axes, operator="add")

    def prod(self, axes: tuple[int, ...] | None = None) -> "Node":
        return self.reduce(axes=axes, operator="mul")

    def permute(self, permutation: tuple[int, ...]) -> "Node":
        return ViewNode.new_permutation(input=self, permutation=permutation)

    def transpose(self) -> "Node":
        identity = tuple(range(len(self.shape)))
        permutation = identity[:-2] + (identity[-1], identity[-2])
        return self.permute(tuple(permutation))

    def squeeze(self, axes: tuple[int, ...]) -> "Node":
        for axis in axes:
            if axis < 0 or axis >= len(self.shape):
                raise IndexError(f"Axis {axis} out of bounds for shape {self.shape}")
            if self.shape[axis] != 1:
                raise ValueError(f"Cannot squeeze axis {axis} with {self.shape[axis]=}")
        new_shape = tuple(s for i, s in enumerate(self.shape) if i not in axes)
        new_pitch = tuple(p for i, p in enumerate(self.pitch) if i not in axes)
        return self.view(shape=new_shape, pitch=new_pitch)

    #
    # Debug printing:
    #

    def debug_print(self, out: SupportsWrite[str]):
        debug_print(self, out)

    #
    # Differentiation
    #

    def grad(self) -> dict["Node", "Node"]:
        """
        Returns a mapping from each input tensor in the subgraph rooted at this tensor to
        its gradient with respect to this tensor.
        """
        return grad(self)

    def df_do(self, df_dn: "Node") -> "tuple[Node, ...]":
        """
        Given ∂f/∂n (df_dn), returns (∂f/∂o₁, ∂f/∂o₂, ...) for each operand oᵢ.

        Override this method in each Node subclass to implement differentiation for that
        node type. Raises GradOfUndifferentiableNodeException if the node is not
        differentiable.
        """
        _ = df_dn
        raise NotDifferentiableException(self)

    #
    # Private:
    #

    def _elementwise_uop(self, operator: UnaryScalarOperator) -> "Node":
        return ElementwiseNode(
            offset=self.offset,
            shape=self.shape,
            pitch=self.pitch,
            stype=self.stype,
            input=(self,),
            operator=operator,
        )

    def _elementwise_bop(
        self,
        other: "Node | Scalar",
        operator: ScalarOperator,
    ) -> "Node":
        other = Node._from_node_or_scalar(other, stype=self.stype)
        self, other = self._join_dtypes_for_bop(other)
        self, other = self._join_shapes_for_elementwise_bop(other)
        return ElementwiseNode(
            offset=self.offset,
            shape=self.shape,
            pitch=self.pitch,
            stype=self.stype,
            input=(self, other),
            operator=operator,
        )

    @staticmethod
    def _from_node_or_scalar(
        value: "npt.ArrayLike | Node", stype: ScalarType
    ) -> "Node":
        return value if isinstance(value, Node) else ConstNode.new(value, stype=stype)

    def _join_dtypes_for_bop(self, other: "Node") -> tuple["Node", "Node"]:
        res_dtype = stype_join(self.stype, other.stype)
        s = self.copy(stype=res_dtype) if self.stype != res_dtype else self
        o = other.copy(stype=res_dtype) if other.stype != res_dtype else other
        return s, o

    def _join_shapes_for_elementwise_bop(self, other: "Node") -> tuple[Node, Node]:
        join = shape_join(self.shape, self.pitch, other.shape, other.pitch)
        return (
            self.view(shape=join.shape, pitch=join.pitch1),
            other.view(shape=join.shape, pitch=join.pitch2),
        )

    def _join_shapes_for_matmul_bop(self, other: "Node") -> tuple[Node, Node]:
        s = self
        o = other

        # Ensure both operands are rank 2 or higher:
        if len(s.shape) < 2 or len(o.shape) < 2:
            raise ValueError(
                f"Shapes {s.shape} and {o.shape} are not compatible for matmul: "
                "both tensors must be 2D or higher"
            )

        # Compute broadcasting behavior by gathering corresponding dimensions into two
        # parallel arrays, joining those arrays, and then scattering the result back for
        # the original operands.
        s_shape_gat = tuple(x for i, x in enumerate(s.shape) if i != len(s.shape) - 2)
        s_pitch_gat = tuple(x for i, x in enumerate(s.pitch) if i != len(s.pitch) - 2)
        o_shape_gat = tuple(x for i, x in enumerate(o.shape) if i != len(o.shape) - 1)
        o_pitch_gat = tuple(x for i, x in enumerate(o.pitch) if i != len(o.pitch) - 1)
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
        new_self = s.view(shape=new_s_shape, pitch=new_s_pitch)
        new_other = o.view(shape=new_o_shape, pitch=new_o_pitch)
        return new_self, new_other


@dataclass(kw_only=True, frozen=True, eq=False)
class ConstNode(Node):
    value: npt.ArrayLike

    @staticmethod
    def new(value: "npt.ArrayLike", *, stype: ScalarType = "fp32") -> "ConstNode":
        shape = ConstNode._infer_value_shape(value)
        pitch = compute_c_contiguous_pitch_for_shape(shape)
        return ConstNode(
            offset=0,
            shape=shape,
            pitch=pitch,
            stype=stype,
            value=value,
            input=(),
        )

    @staticmethod
    def ones(shape: tuple[int, ...], *, stype: ScalarType = "fp32") -> "Node":
        return ConstNode.full(shape, v=1, stype=stype)

    @staticmethod
    def zeros(shape: tuple[int, ...], *, stype: ScalarType = "fp32") -> "Node":
        return ConstNode.full(shape, v=0, stype=stype)

    @staticmethod
    def full(
        shape: tuple[int, ...], v: Scalar, *, stype: ScalarType = "fp32"
    ) -> "Node":
        return ConstNode.new(v, stype=stype).broadcast(shape)

    @staticmethod
    def _infer_value_shape(value: "npt.ArrayLike") -> tuple[int, ...]:
        if is_scalar(value):
            return ()
        assert isinstance(value, list)
        if not value:
            return (0,)
        e0_shape = ConstNode._infer_value_shape(value[0])
        for e1 in value[1:]:
            if ConstNode._infer_value_shape(e1) != e0_shape:
                raise ValueError("Inconsistent shapes in nested list")
        return (len(value),) + e0_shape


@dataclass(kw_only=True, frozen=True, eq=False)
class ParamNode(Node):
    label: str | None  # non-unique, for debug only

    @staticmethod
    def new(
        *,
        shape: tuple[int, ...],
        stype: ScalarType,
        label: str | None = None,
    ) -> "ParamNode":
        pitch = compute_c_contiguous_pitch_for_shape(shape)
        return ParamNode(
            offset=0,
            shape=shape,
            pitch=pitch,
            stype=stype,
            input=(),
            label=label,
        )


@dataclass(kw_only=True, frozen=True, eq=False)
class ElementwiseNode(Node):
    operator: ScalarOperator

    def df_do(self, df_dn: "Node") -> tuple[Node, ...]:
        match self.operator:
            case "neg":
                return (-df_dn,)
            case "exp":
                # ∂n/∂o₁ = exp(o₁) = n
                return (df_dn * self,)
            case "log":
                # ∂n/∂o₁ = 1 / o₁
                return (df_dn / self.input[0],)
            case "pow":
                # ∂n/∂o₁ = o₂ * o₁^(o₂ - 1) = o₂ * n / o₁
                # ∂n/∂o₂ = log(o₁) * o₁^o₂ = log(o₁) * n
                return (
                    df_dn * self.input[1] * self / self.input[0],
                    df_dn * self.input[0].log() * self,
                )
            case "mul":
                # ∂n/∂o₁ = o₂, ∂n/∂o₂ = o₁
                return (
                    df_dn * self.input[1],
                    df_dn * self.input[0],
                )
            case "div":
                # ∂n/∂o₁ = 1 / o₂, ∂n/∂o₂ = -o₁ / o₂² = -n / o₂
                return (
                    df_dn / self.input[1],
                    df_dn * -self / self.input[1],
                )
            case "add":
                # ∂n/∂o₁ = ∂n/∂o₂ = 1
                return (df_dn, df_dn)
            case "sub":
                # ∂n/∂o₁ = 1, ∂n/∂o₂ = -1
                return (df_dn, -df_dn)
            case "max":
                # ∂n/∂o₁ = 1 if o₁ > o₂ else 0
                # ∂n/∂o₂ = 1 if o₂ > o₁ else 0
                return (
                    df_dn * self.input[0].gt(self.input[1]),
                    df_dn * self.input[1].gt(self.input[0]),
                )
            case "min":
                # ∂n/∂o₁ = 1 if o₁ < o₂ else 0
                # ∂n/∂o₂ = 1 if o₂ < o₁ else 0
                return (
                    df_dn * self.input[0].lt(self.input[1]),
                    df_dn * self.input[1].lt(self.input[0]),
                )
            case "eq" | "ne" | "lt" | "gt" | "le" | "ge":
                raise NotDifferentiableException(self)
            case _:
                raise NotImplementedError(f"{self.operator=}")


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

    @staticmethod
    def new(
        input: "Node",
        axes: tuple[int, ...],
        operator: BinaryAssocScalarOperator,
    ) -> "Node":
        if not axes:
            return input

        out_shape = list(input.shape)
        for axis in axes:
            if axis < 0 or axis >= len(input.shape):
                raise IndexError(f"Axis {axis} out of bounds for shape {input.shape}")
            out_shape[axis] = 1

        out_shape = tuple(out_shape)
        out_pitch = compute_c_contiguous_pitch_for_shape(out_shape)

        return ReductionNode(
            offset=input.offset,
            shape=out_shape,
            pitch=out_pitch,
            stype=input.stype,
            input=(input,),
            operator=operator,
            axes=axes,
        )

    def df_do(self, df_dn: "Node") -> tuple[Node, ...]:
        # broadcast df_dn to the input shape
        df_dn = df_dn.view(
            shape=self.input[0].shape,
            pitch=tuple(
                (0 if i in self.axes else df_dn.pitch[i])
                for i in range(len(self.input[0].shape))
            ),
        )

        match self.operator:
            case "mul":
                # ∂n/∂o₁ = n / o₁
                return (df_dn * (self / self.input[0]),)
            case "add":
                # ∂n/∂o₁ = 1
                return (df_dn,)
            case "max" | "min":
                # ∂n/∂o₁ = 1 if o₁ is the max/min along the reduction axes, else 0
                # Mask broadcast df_dn with the condition above.
                # Note that the condition is identical for both "max" and "min".
                cond = self.input[0].eq(self)
                return (df_dn * cond,)
            case _:
                raise NotImplementedError(f"{self.operator=}")


@dataclass(kw_only=True, frozen=True, eq=False)
class MatmulNode(Node):
    @staticmethod
    def new(a: "Node", b: "Node") -> "MatmulNode":
        a, b = a._join_dtypes_for_bop(b)
        a, b = a._join_shapes_for_matmul_bop(b)
        out_offset = 0
        out_shape = a.shape[:-1] + (b.shape[-1],)
        out_pitch = compute_c_contiguous_pitch_for_shape(out_shape)
        return MatmulNode(
            offset=out_offset,
            shape=out_shape,
            pitch=out_pitch,
            stype=a.stype,
            input=(a, b),
        )

    def df_do(self, df_dn: "Node") -> tuple[Node, ...]:
        # ∂n/∂o₁ = df/dn @ o₂.T
        # ∂n/∂o₂ = o₁.T @ df/dn
        return (
            df_dn @ self.input[1].transpose(),
            self.input[0].transpose() @ df_dn,
        )


@dataclass(kw_only=True, frozen=True, eq=False)
class ViewNode(Node):
    """
    Maps each input element to zero, one, or multiple (i.e. broadcasted) output elements
    by only changing the offset, shape, pitch.
    """

    accessor: "Accessor"

    @staticmethod
    def new(input: "Node", accessor: "Accessor") -> "ViewNode":
        accessor.raise_if_not_compatible(input)
        return ViewNode(
            offset=accessor.offset,
            shape=accessor.shape,
            pitch=accessor.pitch,
            stype=input.stype,
            input=(input,),
            accessor=accessor,
        )

    @staticmethod
    def new_broadcast(input: "Node", ns: tuple[int, ...]) -> "Node":
        zs = (0,) * len(ns)
        shape = ns + input.shape
        pitch = zs + input.pitch
        return input.view(offset=input.offset, shape=shape, pitch=pitch)

    @staticmethod
    def new_permutation(input: "Node", permutation: tuple[int, ...]) -> "ViewNode":
        if sorted(permutation) != list(range(len(input.shape))):
            raise ValueError(f"Bad permutation {permutation} for shape {input.shape}")
        new_offset = input.offset
        new_shape = tuple(input.shape[i] for i in permutation)
        new_pitch = tuple(input.pitch[i] for i in permutation)
        return ViewNode.new(
            input=input,
            accessor=Accessor(offset=new_offset, shape=new_shape, pitch=new_pitch),
        )

    @staticmethod
    def new_index(input: "Node", key: tuple[int | slice, ...]) -> "ViewNode":
        accessor = Accessor.from_key(input.offset, input.shape, input.pitch, key)
        return ViewNode(
            offset=accessor.offset,
            pitch=accessor.pitch,
            shape=accessor.shape,
            stype=input.stype,
            input=(input,),
            accessor=accessor,
        )

    def df_do(self, df_dn: "Node") -> tuple[Node, ...]:
        """
        - A view simply maps each input index `x` to zero, one, or multiple output
          indices `y_o`. Each index is a "flat" index, i.e. an integer that references
          an element in some flat storage array that is viewed.
        - Consider each case:
          - x -> {}: the gradient flow is zero for that element.
          - x -> {y}: the gradient flow is identity for that element.
          - x -> {y₁, y₂, ...}: the gradient flow is a sum over elements (broadcast).
        """

        # Sum-reduce over all broadcast dimensions (i.e. pitch 0).
        x = df_dn.reduce(
            axes=tuple(i_dim for i_dim, p in enumerate(self.pitch) if p == 0),
            operator="add",
        )

        # Scatter the reduced tensor back to the input shape.
        # We can't use the original accessor because it does not account for the
        # collapsed broadcast dimensions. Instead, we replace the accessor's shape with
        # the reduced shape, keeping offset and pitch intact.
        x = ScatterNode.new(
            source=x,
            shape=self.input[0].shape,
            accessor=replace(self.accessor, shape=x.shape),
        )

        # Done:
        return (x,)


@dataclass(kw_only=True, frozen=True, eq=False)
class ScatterNode(Node):
    """
    Scatter maps a dense "source" to sparse destinations using view parameters. Unlike a
    ViewNode, the view parameters specify write locations, not read locations. The
    "shape" operand gives the shape of the output, all unwritten values are filled with
    zeros.

    Example NumPy code:

    ```python
    def scatter(source, shape, accessor) =
        res = np.zeros(shape=shape, stype=source.stype)
        res[key] = source
        return res

    # Note that `scatter` is the inverse of `view` aka `__getitem__`:
    assert scatter(source, shape, key)[key] == source
    ```

    Scatter always writes to freshly allocated C-contiguous memory.
    """

    accessor: "Accessor"

    @staticmethod
    def new(
        source: "Node",
        shape: tuple[int, ...],
        accessor: "Accessor",
    ) -> "ScatterNode":
        return ScatterNode(
            offset=0,
            shape=shape,
            pitch=compute_c_contiguous_pitch_for_shape(shape),
            stype=source.stype,
            input=(source,),
            accessor=accessor,
        )

    @staticmethod
    def new_copy(source: "Node", stype: ScalarType | None = None) -> "Node":
        stype = stype or source.stype
        pitch = compute_c_contiguous_pitch_for_shape(source.shape)
        accessor = Accessor(offset=0, pitch=pitch, shape=source.shape)
        return ScatterNode.new(source=source, shape=source.shape, accessor=accessor)

    def df_do(self, df_dn: "Node") -> tuple[Node, ...]:
        source_grad = ViewNode.new(df_dn, self.accessor)
        return (source_grad,)


#
# Shape, Pitch:
#


@dataclass(frozen=True, kw_only=True)
class Accessor:
    offset: int
    pitch: tuple[int, ...]
    shape: tuple[int, ...]

    @staticmethod
    def from_key(
        old_offset: int,
        old_shape: tuple[int, ...],
        old_pitch: tuple[int, ...],
        key: tuple[int | slice, ...],
    ) -> "Accessor":
        """
        Computes an accessor's offset, pitch, and shape for a given input tensor and
        key.

        Accessor = ViewNode (aka read accessor) or ScatterNode (aka write accessor).
        """

        def bounded_index(k: int) -> int:
            d = old_shape[dim]
            if not (-d <= k < d):
                raise IndexError(
                    f"Index {k} out of bounds for dimension {dim} of size {d}"
                )
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

        return Accessor(
            offset=new_offset,
            pitch=tuple(new_pitch),
            shape=tuple(new_shape),
        )

    def raise_if_not_compatible(self, node: "Node"):
        """
        Raises a ValueError if the view defined by this accessor is not compatible with
        the given node's memory layout (shape and pitch).

        An accessor is incompatible with a node if it cannot be implemented as a view
        (i.e. without copying).
        """

        old_shape = node.shape
        old_pitch = node.pitch
        new_shape = self.shape
        new_pitch = self.pitch

        def squeezed_c_permutation(
            shape: tuple[int, ...],
            pitch: tuple[int, ...],
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            """
            Returns the shape and pitch after deleting all dimensions of size 1 or
            dimensions of pitch 0.
            """

            new_shape = []
            new_pitch = []
            for s, p in zip(shape, pitch):
                if s != 1 and p != 0:
                    new_shape.append(s)
                    new_pitch.append(p)
            return c_permutation(tuple(new_shape), tuple(new_pitch))

        # If both old and new shapes are contiguous (not even C-contiguous), then the
        # accessor is compatible if it addresses fewer elements than in the original.
        old_cc_shape_pitch = c_permutation(old_shape, old_pitch)
        new_cc_shape_pitch = c_permutation(new_shape, new_pitch)
        if (
            old_cc_shape_pitch == new_cc_shape_pitch
            and is_c_contiguous(old_cc_shape_pitch[0], old_cc_shape_pitch[1])
            and math.prod(old_shape) >= math.prod(new_shape)
        ):
            return

        # If both old and new shapes are identical after deleting all dimensions of size
        # 1 or dimensions of pitch 0, then the view is compatible, even if not
        # contiguous.
        old_squeezed_shape_pitch = squeezed_c_permutation(old_shape, old_pitch)
        new_squeezed_shape_pitch = squeezed_c_permutation(new_shape, new_pitch)
        if old_squeezed_shape_pitch == new_squeezed_shape_pitch:
            return

        # Otherwise, the view is not compatible.
        raise ValueError(
            f"Cannot view "
            f"old tensor (shape={old_shape}, pitch={old_pitch}) as "
            f"new tensor (shape={new_shape}, pitch={new_pitch})"
        )


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


def compute_c_contiguous_pitch_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
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
    return pitch == compute_c_contiguous_pitch_for_shape(shape)


def is_contiguous(shape: tuple[int, ...], pitch: tuple[int, ...]) -> bool:
    new_shape, new_pitch = c_permutation(shape, pitch)
    return is_c_contiguous(new_shape, new_pitch)


def c_permutation(
    shape: tuple[int, ...],
    pitch: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Permute the dimensions by decreasing pitch, breaking ties by decreasing
    shape, and return the new shape and pitch.

    Note that this makes contiguous shapes C-contiguous.

    This normalization is useful even for comparing non-contiguous shapes.
    """

    # argsort the dimensions by decreasing pitch, breaking ties by decreasing
    # shape.
    perm = sorted(
        range(len(pitch)),
        key=lambda i: (pitch[i], shape[i]),
        reverse=True,
    )

    # permute the shape and pitch according to the permutation above:
    new_shape = tuple(shape[i] for i in perm)
    new_pitch = tuple(pitch[i] for i in perm)

    # Done:
    return new_shape, new_pitch


#
# Debug print:
#


def debug_print(root: "Node", out: SupportsWrite[str]) -> None:
    def build_tid_map() -> dict["Node", int]:
        reference_count_map = refcount([root])
        assert reference_count_map[root] == 1, "Root tensor must have reference count 1"

        tid_map: dict["Node", int] = {}
        for tensor, ref_count in reference_count_map.items():
            assert ref_count >= 1
            if ref_count == 1:
                continue
            tid_map[tensor] = len(tid_map)

        return tid_map

    def headline(node: "Node") -> str:
        base_fields = {field.name for field in fields(Node)}
        extra_fields = [f.name for f in fields(node) if f.name not in base_fields]
        args = ", ".join(f"{f}={getattr(node, f)!r}" for f in extra_fields)
        name = pascal_to_snake_case(node.__class__.__name__[: -len("Node")])
        return (
            f"{name}({args}) :: {node.stype}({node.offset},{node.shape},{node.pitch})"
        )

    def visit(
        node: "Node",
        out: SupportsWrite[str],
        tid_map: dict["Node", int],
        *,
        prefix: str = "",
        node_type: Literal["root", "sibling", "last-sibling"] = "root",
    ) -> None:
        connector = {"root": "", "sibling": "├ ", "last-sibling": "└ "}[node_type]
        prefix_ext = {"root": "", "sibling": "│ ", "last-sibling": "  "}[node_type]

        tid = tid_map.get(node)
        if tid is None:
            print(f"{prefix}{connector}{headline(node)}", file=out)
        else:
            if node_type == "root":
                print(
                    f"{prefix}{connector}%{tid} := {headline(node)}",
                    file=out,
                )
            else:
                print(f"{prefix}{connector}%{tid}", file=out)
                return

        child_prefix = prefix + prefix_ext
        for operand_index, operand in enumerate(node.input):
            is_last_operand = operand_index == len(node.input) - 1
            child_node_type = "last-sibling" if is_last_operand else "sibling"
            visit(
                operand,
                out,
                tid_map,
                prefix=child_prefix,
                node_type=child_node_type,
            )

    tid_map = build_tid_map()

    visit(root, out, tid_map, prefix="", node_type="root")

    for tensor in tid_map:
        visit(tensor, out, tid_map, prefix="", node_type="root")


#
# Graph operations:
#


def toposort(roots: Iterable[Node]) -> list["Node"]:
    """
    Returns a list of all tensors in the subgraph rooted at this tensor, sorted in
    topological order (i.e. each tensor appears after all its inputs).
    """
    visited: set["Node"] = set()
    topo_order: list["Node"] = []

    def visit(node: "Node") -> None:
        if node in visited:
            return
        visited.add(node)

        for operand in node.input:
            visit(operand)

        topo_order.append(node)

    for root in roots:
        visit(root)

    return topo_order


def refcount(roots: list[Node]) -> dict["Node", int]:
    """
    Compute reference counts for all tensors in the subgraph rooted at this tensor.
    """
    ref_counts: dict["Node", int] = {}

    def visit(tensor: "Node") -> None:
        if tensor in ref_counts:
            ref_counts[tensor] += 1
        else:
            ref_counts[tensor] = 1
            for operand in tensor.input:
                visit(operand)

    for root in roots:
        visit(root)

    return ref_counts


def grad(f_graph: "Node") -> dict[Node, Node]:
    """
    Given a forward graph 'f_graph' that computes a scalar output, returns a mapping
    from each input tensor in 'f_graph' to its gradient with respect to the output.

    The returned gradient graph holds references to tensor objects in the forward graph
    for efficient reuse of forward pass expressions in the backward pass.
    """

    if f_graph.shape != ():
        raise ValueError("Output graph must be a scalar (i.e. have shape=())")

    def accumulate_gradient(t: "Node", increment: "Node") -> None:
        if base := grad.get(t):
            grad[t] = base + increment
        else:
            grad[t] = increment

    # grad[t] = ∂f / ∂t
    grad: dict[Node, Node] = {}

    # Initialize: ∂f / ∂f = 1
    grad[f_graph] = ConstNode.ones(f_graph.shape, stype=f_graph.stype)

    # Traverse the graph in reverse topological order, accumulating gradients for each
    # node.
    for node in reversed(toposort([f_graph])):
        df_dn = grad.get(node)
        if not df_dn:
            continue

        try:
            df_do = node.df_do(df_dn)
        except NotDifferentiableException:
            continue

        for operand, df_do_i in zip(node.input, df_do):
            accumulate_gradient(operand, df_do_i)

    # Done:
    return grad


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
