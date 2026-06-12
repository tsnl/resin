import math
from typing import Sequence, Literal
from abc import ABC
from dataclasses import dataclass


#
# Expr
#


@dataclass(frozen=True)
class Expr:
    node: Node
    view: View

    def __post_init__(self):
        if not self.view.is_subset_of_dense_view(self.node.shape):
            raise ValueError(
                f"View {self.view} is not compatible with the shape {self.node.shape} "
                f"of the node {self.node}"
            )

    #
    # Property
    #

    @property
    def dtype(self) -> ScalarDType:
        return self.node.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.view.shape

    #
    # Tensor operations
    #

    def __pos__(self) -> Expr:
        return self

    def __neg__(self) -> Expr:
        return Expr._elementwise_unary("neg", self)

    def __mul__(self, other: Expr) -> Expr:
        return Expr._elementwise_binary("mul", self, other)

    def __truediv__(self, other: Expr) -> Expr:
        return Expr._elementwise_binary("div", self, other)

    def __add__(self, other: Expr) -> Expr:
        return Expr._elementwise_binary("add", self, other)

    @staticmethod
    def _elementwise_unary(op: ScalarUnaryOp, a: Expr) -> Expr:
        node = ElementwiseNode(op=op, shape=a.node.shape, args=(a,), dtype=a.node.dtype)
        view = View.c_contiguous(node.shape)
        return Expr(node=node, view=view)

    @staticmethod
    def _elementwise_binary(op: ScalarBinaryOp, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError("Binary operations are not yet implemented.")

    @staticmethod
    def _broadcast_arrays(args: Sequence[Expr]) -> tuple[Expr, ...]:
        """
        Broadcasts the input arrays together to a common shape, returning the
        broadcasted versions of the input arrays.

        Raises a ValueError if the input shapes are not broadcastable.
        """

        # Identify the broadcasted shape:
        broadcast_shape = Expr._broadcast_shape(args)

        # For each input array, if its shape is not already the broadcasted shape, we need
        # to insert a view that broadcasts it to the result shape.
        res_args = []
        for arg in args:
            if arg.shape == broadcast_shape:
                res_args.append(arg)
            else:
                # Insert a view that broadcasts this argument to the result shape.
                # The offset is 0, and the pitch is 0 for any dimension where the size is 1
                # (since we will be repeating the same value along that dimension).
                pitch = tuple(
                    0 if arg_dim == 1 else arg_pitch
                    for arg_dim, arg_pitch in zip(
                        (1,) * (len(broadcast_shape) - len(arg.shape)) + arg.shape,
                        (1,) * (len(broadcast_shape) - len(arg.shape))
                        + View._c_contiguous_pitch_for_shape(arg.shape),
                    )
                )
                res_args.append(
                    Expr(
                        node=arg.node,
                        view=View(offset=0, shape=broadcast_shape, pitch=pitch),
                    )
                )

        return tuple(res_args)

    @staticmethod
    def _broadcast_shape(args: Sequence[Expr]) -> tuple[int, ...]:
        """
        Identifies the shape resulting from broadcasting the input shapes together, or
        raises a ValueError if the shapes are not broadcastable.
        """

        if not args:
            return ()

        # The rank of the broadcasted shape is the maximum rank of the input shapes.
        rank = max(len(arg.shape) for arg in args)

        # Left-pad the input shapes with ones so that they all have the same rank.
        arg_shapes = [(1,) * (rank - len(arg.shape)) + arg.shape for arg in args]

        # Now, for each dimension, verify that all the sizes are either the same or 1.
        res_shape = []
        for dim in range(rank):
            # Gather the sizes of this dimension across all input shapes.
            dim_sizes = {shape[dim] for shape in arg_shapes}
            if len(dim_sizes - {1}) != 1:
                raise ValueError(
                    f"Shapes {tuple(arg.shape for arg in args)} are not broadcastable. "
                    f"Dimension {dim} has sizes {dim_sizes} across the input shapes. "
                    f"All sizes must be same or 1 on this dimension."
                )

            # Append the resolved size for this dimension to the result shape.
            res_shape.append(max(dim_sizes))

        # Done:
        return tuple(res_shape)


@dataclass(frozen=True)
class View:
    """
    Defines a regular pattern of elements within a multi-dimensional array.

    Each view defines a set of addresses within a multi-dimensional array, enumerated by
    multi-dimensional indices within the view's shape. E.g. if the shape is (4, 4), then
    valid indices are (0, 0), (0, 1), ..., (3, 3).

    The view's offset and pitch define how to map these multi-dimensional indices to
    addresses within the flat memory that backs the multidimensional array.

    ---

    ## View Semantics

    Mathematically, we can enumerate the addresses accessed by a view as follows, using
    an arithmetic progression per-dimension:
        address(index) := Σ_d { offset[d] + index[d] * pitch[d] }
    Where
    -   offset[d] gives the address of the first element along each dimension d.
    -   pitch[d] gives the stride along dimension d, i.e. how much to increment the
        address when we increment the index along dimension d by 1.
    -   shape[d] bounds the valid indices along dimension d, i.e. index[d] must be in
        the range [0, shape[d]) for the address to be valid.

    Note that we store Σ_d offset[d] as a single offset. This is mathematically
    equivalent to storing a separate offset for each dimension.
        address(index) := Σ_d { offset[d]   + index[d] * pitch[d]       }
                        = Σ_d { offset[d] } + Σ_d { index[d] * pitch[d] }
                        = offset            + Σ_d { index[d] * pitch[d] }

    Note that...
    -   pitch[d] may be zero or negative.
    -   offset is not necessarily the lowest address accessed by the view. This occurs
        when pitches are negative.
    """

    offset: int
    shape: tuple[int, ...]
    pitch: tuple[int, ...]

    def __post_init__(self):
        assert self.rank == len(self.shape) == len(self.pitch)

    @property
    def rank(self) -> int:
        return len(self.shape)

    #
    # C-contiguity, contiguity, and permutations
    #

    @staticmethod
    def c_contiguous(shape: tuple[int, ...]) -> View:
        return View(
            offset=0,
            shape=shape,
            pitch=View._c_contiguous_pitch_for_shape(shape),
        )

    @staticmethod
    def _c_contiguous_pitch_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        reversed_pitch = [1]
        for dim in reversed(shape):
            reversed_pitch.append(reversed_pitch[-1] * dim)
        return tuple(reversed(reversed_pitch[:-1]))

    def is_c_contiguous(self) -> bool:
        return self.pitch == View._c_contiguous_pitch_for_shape(self.shape)

    def is_contiguous(self) -> bool:
        return self.c_contiguous_permutation() is not None

    def permute(self, permutation: Sequence[int]) -> View:
        assert set(permutation) == set(range(self.rank)), "Invalid permutation"
        return View(
            offset=self.offset,
            shape=tuple(self.shape[d] for d in permutation),
            pitch=tuple(self.pitch[d] for d in permutation),
        )

    def c_contiguous_permutation(self) -> Sequence[int] | None:
        """
        If this view is a permutation of a C-contiguous view, returns the permutation.
        Otherwise, returns None.

        A view is a permutation of a C-contiguous view if there exists a permutation of
        the dimensions such that when we permute the shape and pitch according to this
        permutation, we get the shape and pitch of a C-contiguous view.

        E.g. if the original shape is (4, 4) and the original pitch is (4, 1), then
        this is a permutation of a C-contiguous view with shape (4, 4) and pitch (1,
        4), and the corresponding permutation is (1, 0).
        """

        # We can check if this view is a permutation of a C-contiguous view by sorting
        # the dimensions by their pitch and checking if the resulting shape and pitch
        # match those of a C-contiguous view.

        permutation = sorted(range(len(self.shape)), key=lambda d: self.pitch[d])
        permuted_view = self.permute(permutation)
        return permutation if permuted_view.is_c_contiguous() else None

    #
    # Subset relations
    #

    def is_subset_of_view(self, other: View) -> bool:
        pass

    def is_subset_of_dense_view(self, shape: tuple[int, ...]) -> bool:
        # The C-contiguous shape defines an address range [0, numel) where numel is the
        # total number of elements in the C-contiguous array. No gaps or exceptions.

        # Lemma:
        # If an index is in [0, numel), then it is a valid address in the C-contiguous
        # array with shape `shape`.

        # The view is compatible with the shape if every element in the view addresses
        # an element within this range.

        # We can check this by verifying that the minimum and maximum indices accessed
        # by this view are within the range [0, numel).

        # If the min and max are within the interval [0, numel), then all addresses
        # accessed by this view are within the interval [0, numel).

        numel = math.prod(shape)
        min_address, max_address = self.compute_min_max_addresses()
        return 0 <= min_address <= max_address < numel

    def is_subset_of_sparse_view(
        self,
        sparse_view: View,
    ) -> bool:

        numel = math.prod(shape)
        min_address, max_address = self.compute_min_max_addresses()
        return 0 <= min_address <= max_address < numel

    def compute_min_max_addresses(self) -> tuple[int, int]:
        """
        Computes the minimum and maximum addresses accessed by this view, assuming the
        data is laid out in C-contiguous order.
        """

        min_address = self.offset
        max_address = self.offset

        for dim_size, dim_pitch in zip(self.shape, self.pitch):
            min_address = min(
                min_address,  # at index 0
                min_address + (dim_size - 1) * dim_pitch,  # at index dim_size - 1
            )
            max_address = max(
                max_address,  # at index 0
                max_address + (dim_size - 1) * dim_pitch,  # at index dim_size - 1
            )

        assert min_address <= max_address

        return min_address, max_address

    @staticmethod
    def compose(second: View, first: View) -> View:
        """
        Returns a view that is the composition of the two given views, i.e. a view that
        applies the first view followed by the second view.

        Views define a regular pattern of elements within a multi-dimensional array.
        When a second view is composed with a first view, we apply the second view's
        pattern to the elements selected by the first view.

        E.g.
        - original array has shape (8, 8) for an 8x8 matrix
        - first view has shape (4, 4) and pitch (2, 2), selecting a 4x4 submatrix with
          stride 2.
        - second view has shape (2, 2) and pitch (1, 1), selecting a 2x2 submatrix with
          stride 1 from the top-left corner of a 4x4 matrix.
        """


@dataclass(frozen=True, eq=False)
class Node(ABC):
    dtype: ScalarDType
    shape: tuple[int, ...]
    args: tuple[Expr, ...]

    def df_da(self, value: Expr) -> tuple[Expr, ...]:
        """
        Returns the gradients of the output with respect to the input expressions.
        """
        _ = value
        raise NotDifferentiableError(f"{self} is not differentiable")


class NotDifferentiableError(Exception):
    pass


#
# ConstNode
#


@dataclass(frozen=True, eq=False)
class ConstNode(Node):
    value: ConstValue

    @staticmethod
    def new(value: ConstValue, dtype: ScalarDType) -> Expr:
        shape = const_value_shape(value)

        return Expr(
            node=ConstNode(shape=shape, args=(), value=value, dtype=dtype),
            view=View.c_contiguous(shape),
        )

    def df_da(self, value: Expr) -> tuple[Expr, ...]:
        assert self.args == ()
        _ = value
        return ()


#
# ParamNode
#


@dataclass(frozen=True, eq=False)
class ParamNode(Node):
    path: str

    @staticmethod
    def new(path: str, shape: tuple[int, ...], dtype: ScalarDType) -> Expr:
        return Expr(
            node=ParamNode(shape=shape, args=(), path=path, dtype=dtype),
            view=View.c_contiguous(shape),
        )

    def df_da(self, value: Expr) -> tuple[Expr, ...]:
        assert self.args == ()
        _ = value
        return ()


#
# ElementwiseNode
#


@dataclass(frozen=True, eq=False)
class ElementwiseNode(Node):
    op: ScalarUnaryOp | ScalarBinaryOp


#
# ReduceNode
#


@dataclass(frozen=True, eq=False)
class ReduceNode(Node):
    op: ScalarBinaryOp
    axes: tuple[int, ...]


#
# MatmulNode
#


@dataclass(frozen=True, eq=False)
class MatmulNode(Node):
    pass


#
# Scalar
#

type ScalarDType = Literal["i32", "f32"]


type ScalarUnaryOp = Literal[
    "neg",
    "exp",
    "log",
]
type ScalarBinaryOp = Literal[
    "pow",
    "mul",
    "div",
    "add",
    "sub",
    "max",
    "min",
]


#
# ConstValue
#


type ConstValue = int | float | Sequence[ConstValue]


def const_value_shape(value: ConstValue) -> tuple[int, ...]:
    match value:
        case int() | float():
            return ()

        case Sequence():
            if not value:
                return (0,)

            suffix_shape = const_value_shape(value[0])
            for element in value:
                if const_value_shape(element) != suffix_shape:
                    raise ValueError("Jagged arrays detected in ConstNode value")

            return (len(value),) + suffix_shape

        case _:
            raise ValueError(f"Unsupported type for ConstNode value: {type(value)}")
