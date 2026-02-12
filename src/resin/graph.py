"""
resin.graph models the computational graph of tensor operations.
-   Node is both a tensor memory and an instruction in the graph.
-   The graph is static and acyclic, but not necessarily a tree (e.g. shared subgraphs).
-   Parameter nodes map to buffers that must be written when executing the graph.
"""

from abc import ABC
from dataclasses import dataclass, fields
import math
from typing import Literal

import numpy.typing as npt

from .common import SupportsWrite, pascal_to_snake_case


#
# Node
#


@dataclass(kw_only=True, frozen=True, eq=False)
class Node(ABC):
    shape: tuple[int, ...]
    pitch: tuple[int, ...]
    dtype: DType
    input: tuple[Node, ...]

    def __post_init__(self):
        assert len(self.shape) == len(self.pitch)

    #
    # Properties:
    #

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * dtype_nbytes(self.dtype)

    #
    # Factory methods:
    #

    @staticmethod
    def const(value: "npt.ArrayLike", *, dtype: DType = "fp32") -> ConstNode:
        return ConstNode.new(value=value, dtype=dtype)

    @staticmethod
    def ones(shape: tuple[int, ...], *, dtype: DType = "fp32") -> Node:
        return Node.full(shape, v=1, dtype=dtype)

    @staticmethod
    def zeros(shape: tuple[int, ...], *, dtype: DType = "fp32") -> Node:
        return Node.full(shape, v=0, dtype=dtype)

    @staticmethod
    def full(shape: tuple[int, ...], v: Scalar, *, dtype: DType = "fp32") -> Node:
        return Node.const(v, dtype=dtype).broadcast(shape)

    @staticmethod
    def param(shape: tuple[int, ...], dtype: DType, label: str = "") -> ParamNode:
        return ParamNode.new(shape=shape, dtype=dtype, label=label)

    #
    # Operations:
    #

    def broadcast(self, ns: tuple[int, ...]) -> Node:
        return ViewNode.new_broadcast(input=self, ns=ns)

    def view(self, *, shape: tuple[int, ...], pitch: tuple[int, ...]) -> Node:
        return ViewNode.new(input=self, shape=shape, pitch=pitch)

    def copy(self, dtype: DType | None = None) -> Node:
        return CopyNode.new(input=self, dtype=dtype)

    def __pow__(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="pow")

    def __rpow__(self, other: Node | Scalar) -> Node:
        return Node._from_node_or_scalar(other, dtype=self.dtype) ** self

    def __mul__(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="mul")

    def __rmul__(self, other: Node | Scalar) -> Node:
        return Node._from_node_or_scalar(other, dtype=self.dtype) * self

    def __truediv__(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="div")

    def __rtruediv__(self, other: Node | Scalar) -> Node:
        return Node._from_node_or_scalar(other, dtype=self.dtype) / self

    def __add__(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="add")

    def __radd__(self, other: Node | Scalar) -> Node:
        return Node._from_node_or_scalar(other, dtype=self.dtype) + self

    def __sub__(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="sub")

    def __rsub__(self, other: Node | Scalar) -> Node:
        return Node._from_node_or_scalar(other, dtype=self.dtype) - self

    def max(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="max")

    def min(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="min")

    def eq(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="eq")

    def ne(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="ne")

    def lt(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="lt")

    def gt(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="gt")

    def le(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="le")

    def ge(self, other: Node | Scalar) -> Node:
        return self._elementwise_bop(other, operator="ge")

    def __neg__(self) -> Node:
        return self._elementwise_uop(operator="neg")

    def __pos__(self) -> Node:
        return self

    def exp(self) -> Node:
        return self._elementwise_uop(operator="exp")

    def log(self) -> Node:
        return self._elementwise_uop(operator="log")

    def __invert__(self) -> Node:
        return self._elementwise_uop(operator="not")

    def __matmul__(self, other: Node) -> Node:
        return MatmulNode.new(a=self, b=other)

    def __rmatmul__(self, other: Node) -> Node:
        return Node._from_node_or_scalar(other, dtype=self.dtype) @ self

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> Node:
        return IndexNode.new(container=self, key=key)

    def reduce(self, axes: tuple[int, ...], operator: ScalarOperator) -> Node:
        return ReductionNode.new(input=self, axes=axes, operator=operator)

    def permute(self, permutation: tuple[int, ...]) -> Node:
        return ViewNode.new_permutation(input=self, permutation=permutation)

    def transpose(self) -> Node:
        identity = tuple(range(len(self.shape)))
        permutation = identity[:-2] + (identity[-1], identity[-2])
        return self.permute(tuple(permutation))

    #
    # Graph ops:
    #

    def refcount(self) -> dict["Node", int]:
        """
        Compute reference counts for all tensors in the subgraph rooted at this tensor.
        """
        return refcount(self)

    def toposort(self) -> list["Node"]:
        """
        Returns a list of all tensors in the subgraph rooted at this tensor, sorted in
        topological order (i.e. each tensor appears after all its inputs).
        """
        return toposort(self)

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

    def df_do(self, df_dn: "Node") -> "tuple[Node, ...] | None":
        """
        Given ∂f/∂n (df_dn), returns (∂f/∂o₁, ∂f/∂o₂, ...) for each operand oᵢ,
        or None if this node is not differentiable.

        Override this method in each Node subclass to implement differentiation for that
        node type.
        """
        return None

    #
    # Private:
    #

    def _elementwise_uop(self, operator: UnaryScalarOperator) -> Node:
        return ElementwiseNode(
            shape=self.shape,
            pitch=self.pitch,
            dtype=self.dtype,
            input=(self,),
            operator=operator,
        )

    def _elementwise_bop(
        self,
        other: Node | Scalar,
        operator: ScalarOperator,
    ) -> Node:
        other = Node._from_node_or_scalar(other, dtype=self.dtype)
        self, other = self._join_dtypes_for_bop(other)
        self, other = self._join_shapes_for_elementwise_bop(other)
        return ElementwiseNode(
            shape=self.shape,
            pitch=self.pitch,
            dtype=self.dtype,
            input=(self, other),
            operator=operator,
        )

    @staticmethod
    def _from_node_or_scalar(value: "npt.ArrayLike | Node", dtype: DType) -> Node:
        return value if isinstance(value, Node) else Node.const(value, dtype=dtype)

    def _join_dtypes_for_bop(self, other: Node) -> tuple[Node, Node]:
        res_dtype = dtype_join(self.dtype, other.dtype)
        s = self.copy(dtype=res_dtype) if self.dtype != res_dtype else self
        o = other.copy(dtype=res_dtype) if other.dtype != res_dtype else other
        return s, o

    def _join_shapes_for_elementwise_bop(self, other: Node) -> tuple[Node, Node]:
        join = shape_join(self.shape, self.pitch, other.shape, other.pitch)
        return (
            self.view(shape=join.shape, pitch=join.pitch1),
            other.view(shape=join.shape, pitch=join.pitch2),
        )

    def _join_shapes_for_matmul_bop(self, other: Node) -> tuple[Node, Node]:
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
    def new(value: "npt.ArrayLike", *, dtype: DType):
        shape = ConstNode._infer_value_shape(value)
        pitch = compute_c_contiguous_pitch_for_shape(shape)
        return ConstNode(shape=shape, pitch=pitch, dtype=dtype, value=value, input=())

    @staticmethod
    def _infer_value_shape(value: "npt.ArrayLike") -> tuple[int, ...]:
        if is_scalar_instance(value):
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
        dtype: DType,
        label: str | None = None,
    ) -> "ParamNode":
        pitch = compute_c_contiguous_pitch_for_shape(shape)
        return ParamNode(shape=shape, pitch=pitch, dtype=dtype, input=(), label=label)


@dataclass(kw_only=True, frozen=True, eq=False)
class ElementwiseNode(Node):
    operator: ScalarOperator

    def df_do(self, df_dn: Node) -> tuple[Node, ...] | None:
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
                # Comparison operators are not differentiable.
                return None
            case _:
                raise NotImplementedError(f"{self.operator=}")


@dataclass(kw_only=True, frozen=True, eq=False)
class ReductionNode(Node):
    operator: ScalarOperator
    axes: tuple[int, ...]

    @staticmethod
    def new(
        input: Node,
        axes: tuple[int, ...],
        operator: ScalarOperator,
    ) -> "ReductionNode":
        out_shape = list(input.shape)
        for axis in axes:
            if axis < 0 or axis >= len(input.shape):
                raise IndexError(f"Axis {axis} out of bounds for shape {input.shape}")
            out_shape[axis] = 1

        return ReductionNode(
            shape=tuple(out_shape),
            pitch=input.pitch,
            dtype=input.dtype,
            input=(input,),
            operator=operator,
            axes=axes,
        )

    def df_do(self, df_dn: Node) -> tuple[Node, ...] | None:
        _ = df_dn
        raise NotImplementedError()


@dataclass(kw_only=True, frozen=True, eq=False)
class MatmulNode(Node):
    @staticmethod
    def new(a: Node, b: Node) -> "MatmulNode":
        a, b = a._join_dtypes_for_bop(b)
        a, b = a._join_shapes_for_matmul_bop(b)
        out_shape = a.shape[:-1] + (b.shape[-1],)
        out_pitch = compute_c_contiguous_pitch_for_shape(out_shape)
        return MatmulNode(shape=out_shape, pitch=out_pitch, dtype=a.dtype, input=(a, b))

    def df_do(self, df_dn: Node) -> tuple[Node, ...]:
        # ∂n/∂o₁ = df/dn @ o₂.T
        # ∂n/∂o₂ = o₁.T @ df/dn
        return (
            df_dn @ self.input[1].transpose(),
            self.input[0].transpose() @ df_dn,
        )


@dataclass(kw_only=True, frozen=True, eq=False)
class ViewNode(Node):
    @staticmethod
    def new(
        input: Node,
        shape: tuple[int, ...],
        pitch: tuple[int, ...],
    ) -> "Node":
        if isinstance(input, ViewNode):
            return input.input[0].view(shape=shape, pitch=pitch)
        check_view_compatibility(input.shape, input.pitch, shape, pitch)
        return ViewNode(shape=shape, pitch=pitch, dtype=input.dtype, input=(input,))

    @staticmethod
    def new_broadcast(input: Node, ns: tuple[int, ...]) -> "Node":
        zs = (0,) * len(ns)
        shape = ns + input.shape
        pitch = zs + input.pitch
        return input.view(shape=shape, pitch=pitch)

    @staticmethod
    def new_permutation(input: Node, permutation: tuple[int, ...]) -> "ViewNode":
        if sorted(permutation) != list(range(len(input.shape))):
            raise ValueError(f"Bad permutation {permutation} for shape {input.shape}")
        new_shape = tuple(input.shape[i] for i in permutation)
        new_pitch = tuple(input.pitch[i] for i in permutation)
        return ViewNode(
            shape=new_shape,
            pitch=new_pitch,
            dtype=input.dtype,
            input=(input,),
        )

    def df_do(self, df_dn: Node) -> tuple[Node, ...] | None:
        # TODO: implement this: for each view dim...
        # - if broadcast, need to reduce sum over that dim in the gradient
        # - if not broadcast, need to index with the right key
        raise NotImplementedError()


@dataclass(kw_only=True, frozen=True, eq=False)
class IndexNode(Node):
    key: tuple[int | slice, ...]

    @staticmethod
    def new(container: Node, key: int | slice | tuple[int | slice, ...]) -> "IndexNode":
        key = (key,) if isinstance(key, (int, slice)) else key

        shape_list = []
        pitch_list = []

        # Compute the new shape and pitch for indexed dimensions:
        for i, (k, s, p) in enumerate(zip(key, container.shape, container.pitch)):
            if isinstance(k, int):
                if not (-s <= k < s):
                    raise IndexError(f"Index {i} out of bounds: {key=}")
                # This dimension is removed, so we don't add to the lists.
            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else s
                step = k.step if k.step is not None else 1
                abs_step = abs(step)
                dim_size = max(0, (stop - start + (abs_step - 1)) // abs_step)
                shape_list.append(dim_size)
                pitch_list.append(p * step)
            else:
                raise TypeError(f"Index {i} must be int or slice: {key=}")

        # Add remaining dimensions that are not indexed:
        for i in range(len(key), len(container.shape)):
            shape_list.append(container.shape[i])
            pitch_list.append(container.pitch[i])

        shape = tuple(shape_list)
        pitch = tuple(pitch_list)
        dtype = container.dtype
        return IndexNode(
            key=key,
            shape=shape,
            pitch=pitch,
            dtype=dtype,
            input=(container,),
        )

    def df_do(self, df_dn: Node) -> tuple[Node, ...] | None:
        raise NotImplementedError()


@dataclass(kw_only=True, frozen=True, eq=False)
class CopyNode(Node):
    @staticmethod
    def new(input: Node, dtype: DType | None = None) -> "Node":
        dtype = dtype or input.dtype
        if is_c_contiguous(input.shape, input.pitch) and input.dtype == dtype:
            return input
        shape = input.shape
        pitch = compute_c_contiguous_pitch_for_shape(shape)
        return CopyNode(shape=shape, pitch=pitch, dtype=dtype, input=(input,))

    def df_do(self, df_dn: Node) -> tuple[Node, ...] | None:
        return (df_dn.copy(dtype=self.input[0].dtype),)


#
# DType
#


type DType = Literal["fp32", "fp16"]
type DTypeKind = Literal["float"]


def dtype_join(dtype1: DType, dtype2: DType) -> DType:
    kind = dtype_join_kind(dtype_kind(dtype1), dtype_kind(dtype2))
    nbytes = max(dtype_nbytes(dtype1), dtype_nbytes(dtype2))
    return dtype(kind, nbytes)


def dtype(kind: DTypeKind, nbytes: int) -> DType:
    match (kind, nbytes):
        case ("float", 4):
            return "fp32"
        case ("float", 2):
            return "fp16"
        case _:
            raise ValueError(f"Unsupported dtype with kind={kind} and nbytes={nbytes}")


def dtype_nbytes(dtype: DType) -> int:
    return {"fp32": 4, "fp16": 2}[dtype]


def dtype_kind(dtype: DType) -> DTypeKind:
    match dtype:
        case "fp32" | "fp16":
            return "float"
        case _:
            raise ValueError(f"Unsupported dtype {dtype}")


def dtype_join_kind(dtype1: DTypeKind, dtype2: DTypeKind) -> DTypeKind:
    if dtype1 != dtype2:
        raise ValueError(f"Cannot join different kinds' dtypes: {dtype1} and {dtype2}")
    return dtype1


#
# Shape, Pitch:
#


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


def check_view_compatibility(
    old_shape: tuple[int, ...],
    old_pitch: tuple[int, ...],
    new_shape: tuple[int, ...],
    new_pitch: tuple[int, ...],
):
    """
    Checks whether a tensor with (old_shape, old_pitch) can be viewed as a tensor with
    (new_shape, new_pitch). Returns a string describing the reason for compatibility or
    incompatibility.
    """

    def c_contiguous_permutation(
        shape: tuple[int, ...],
        pitch: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """
        Given a contiguous (but not C-contiguous) tensor, attempt to find a permutation
        of its dimensions that makes it C-contiguous. If such a permutation exists,
        return the new shape and pitch. Otherwise, return None.
        """

        # argsort the dimensions by decreasing pitch, breaking ties by decreasing shape.
        perm = sorted(
            range(len(pitch)), key=lambda i: (pitch[i], shape[i]), reverse=True
        )

        # permute the shape and pitch according to the permutation above:
        new_shape = tuple(shape[i] for i in perm)
        new_pitch = tuple(pitch[i] for i in perm)

        # Check if the new pitch is C-contiguous for the new shape:
        if is_c_contiguous(new_shape, new_pitch):
            return new_shape, new_pitch

        # If not, then the tensor cannot be made C-contiguous by permuting dimensions.
        return None

    def del_size_one_pitch_zero_dims(
        shape: tuple[int, ...],
        pitch: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Returns the shape and pitch after deleting all dimensions of size 1 or dimensions of
        pitch 0.
        """

        new_shape = []
        new_pitch = []
        for s, p in zip(shape, pitch):
            if s != 1 and p != 0:
                new_shape.append(s)
                new_pitch.append(p)
        return tuple(new_shape), tuple(new_pitch)

    # If both old and new shapes are contiguous (not even C-contiguous), then the view
    # is compatible if and only if the element count is the same.
    old_cc_shape_pitch = c_contiguous_permutation(old_shape, old_pitch)
    new_cc_shape_pitch = c_contiguous_permutation(new_shape, new_pitch)
    if (
        old_cc_shape_pitch == new_cc_shape_pitch
        and old_cc_shape_pitch is not None
        and math.prod(old_shape) == math.prod(new_shape)
    ):
        return

    # If both old and new shapes are identical after deleting all dimensions of size 1
    # or dimensions of pitch 0, then the view is compatible, even if not contiguous.
    # This is commonly called "squeezing".
    old_squeezed_shape_pitch = del_size_one_pitch_zero_dims(old_shape, old_pitch)
    new_squeezed_shape_pitch = del_size_one_pitch_zero_dims(new_shape, new_pitch)
    if old_squeezed_shape_pitch == new_squeezed_shape_pitch:
        return

    # Otherwise, the view is not compatible.
    return ValueError(
        f"Cannot view "
        f"old tensor (shape={old_shape}, pitch={old_pitch}) as "
        f"new tensor (shape={new_shape}, pitch={new_pitch})"
    )


#
# Scalar:
#


type Scalar = float | int
_SCALAR_TYPES: tuple[type, ...] = (float, int)

type ScalarOperator = UnaryScalarOperator | BinaryScalarOperator | BinaryCompareOperator
type UnaryScalarOperator = Literal["neg", "exp", "log", "not"]
type BinaryScalarOperator = Literal["pow", "mul", "div", "add", "sub", "max", "min"]
type BinaryCompareOperator = Literal["eq", "ne", "gt", "lt", "ge", "le"]


def is_scalar_instance(value: object) -> bool:
    """Check if a value is a scalar (float or int)."""
    return isinstance(value, _SCALAR_TYPES)


#
# Debug print:
#


def debug_print(root: "Node", out: SupportsWrite[str]) -> None:
    def build_tid_map() -> dict["Node", int]:
        reference_count_map = root.refcount()
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
        return f"{name}({args}) :: ({node.dtype} {node.shape} {node.pitch})"

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


def toposort(root: Node) -> list["Node"]:
    """
    Returns a list of all tensors in the subgraph rooted at this tensor, sorted in
    topological order (i.e. each tensor appears after all its inputs).
    """
    visited: set["Node"] = set()
    topo_order: list["Node"] = []

    def visit(node: "Node") -> None:
        if node not in visited:
            visited.add(node)
            for operand in node.input:
                visit(operand)
            topo_order.append(node)

    visit(root)
    return topo_order


def refcount(root: Node) -> dict["Node", int]:
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

    visit(root)
    return ref_counts


#
# Differentiation
#


def grad(f_graph: Node) -> dict[Node, Node]:
    """
    Given a forward graph 'f_graph' that computes a scalar output, returns a mapping
    from each input tensor in 'f_graph' to its gradient with respect to the output.

    The returned gradient graph holds references to tensor objects in the forward graph
    for efficient reuse of forward pass expressions in the backward pass.
    """

    def accumulate_gradient(t: Node, increment: Node) -> None:
        if base := grad.get(t):
            grad[t] = base + increment
        else:
            grad[t] = increment

    # grad[t] = ∂f / ∂t
    grad: dict[Node, Node] = {}

    # Initialize: ∂f / ∂f = 1
    grad[f_graph] = Node.ones(f_graph.shape, dtype=f_graph.dtype)

    # Traverse the graph in reverse topological order, accumulating gradients for each
    # node.
    for node in reversed(f_graph.toposort()):
        df_dn = grad.get(node)
        if not df_dn:
            continue

        df_do = node.df_do(df_dn)
        if df_do is None:
            continue

        for operand, df_do_i in zip(node.input, df_do):
            accumulate_gradient(operand, df_do_i)

    # Done:
    return grad
