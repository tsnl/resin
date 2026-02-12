"""
resin.tensor: tensor/computational graph module.
"""

__all__ = [
    "BatchedMatrixMultiplicationTensor",
    "BinaryOperator",
    "BroadcastTensor",
    "CompactTensor",
    "ConstantTensor",
    "DType",
    "ElementwiseOperationTensor",
    "IndexTensor",
    "Operator",
    "Pitch",
    "ReductionTensor",
    "ReshapeTensor",
    "Shape",
    "SqueezeTensor",
    "Substitution",
    "Tensor",
    "TensorFunction",
    "UnaryOperator",
    "VarTensor",
    "dtype_bytes",
]

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field
from typing import Callable, Generator, Literal

from ..common import SupportsWrite


#
# TensorDict: JSON-style objects, but with Tensor leaves.
# NOTE: a Tensor is a trivial TensorDict.
#


type TensorDict = "Tensor | list[TensorDict] | dict[str, TensorDict]"


type TensorDictType = (
    "TensorType | tuple[TensorDictType, ...] | frozenset[tuple[str, TensorDictType]]"
)


@dataclass(frozen=True)
class TensorType:
    dtype: DType
    shape: Shape


def tensordict_flatten(td: TensorDict) -> Generator[Tensor, None, None]:
    match td:
        case Tensor() as t:
            yield t
        case list() as l:
            for v in l:
                yield from tensordict_flatten(v)
        case dict() as d:
            for v in d.values():
                yield from tensordict_flatten(v)


def tensordict_type(td: TensorDict) -> TensorDictType:
    match td:
        case Tensor() as t:
            return TensorType(dtype=t.dtype, shape=t.shape)
        case list() as l:
            return tuple(tensordict_type(v) for v in l)
        case dict() as d:
            return frozenset((k, tensordict_type(v)) for k, v in d.items())
        case _:
            raise TypeError("TensorDict must be a Tensor, dict, or list")


def tensordict_type_instantiate(dt: TensorDictType, name: str) -> TensorDict:
    match dt:
        case frozenset() as dt:
            return {
                k: tensordict_type_instantiate(v, name=f"{name}.{k}")  #
                for k, v in dt
            }
        case tuple() as lt:
            return [
                tensordict_type_instantiate(v, name=f"{name}[{i}]")
                for i, v in enumerate(lt)
            ]
        case TensorType() as tt:
            return VarTensor.new(name=name, dtype=tt.dtype, shape=tt.shape)
        case _:
            raise TypeError("Invalid TensorDictType")


#
# TensorFunction: functions on TensorDict
#
#

type TensorFunction[T] = Callable[[TensorDict], T]

#
# Tensor:
#


@dataclass(eq=False)
class Tensor(ABC):
    dtype: DType
    offset: int
    shape: Shape
    pitch: Pitch
    operands: tuple["Tensor", ...]
    is_c_contiguous: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_c_contiguous = self.pitch == _c_contiguous_pitch(self.shape)

    #
    # Constructors:
    #

    @staticmethod
    def const(*, value: "list | Scalar", dtype: DType = "fp32") -> "ConstantTensor":
        return ConstantTensor.new(value=value, dtype=dtype)

    @staticmethod
    def ones(*, shape: Shape, dtype: DType = "fp32") -> "Tensor":
        return Tensor._fill(value=1.0, shape=shape, dtype=dtype)

    @staticmethod
    def zeros(*, shape: Shape, dtype: DType = "fp32") -> "Tensor":
        return Tensor._fill(value=0.0, shape=shape, dtype=dtype)

    def ones_like(self) -> "Tensor":
        return Tensor._fill(value=1.0, shape=self.shape, dtype=self.dtype)

    def zeros_like(self) -> "Tensor":
        return Tensor._fill(value=0.0, shape=self.shape, dtype=self.dtype)

    @staticmethod
    def _fill(*, value: float, shape: Shape, dtype: DType) -> "Tensor":
        result: "Tensor" = ConstantTensor.new(value=value, dtype=dtype)
        for dim_size in reversed(shape):
            result = result.broadcast(dim_size)
        return result

    @staticmethod
    def _init_broadcast(
        *,
        offset: int,
        shape: Shape,
        pitch: Pitch,
        n: int,
    ) -> tuple[int, Shape, Pitch]:
        """Returns (offset, shape, pitch) for a broadcast view."""
        new_shape = (n,) + shape
        new_pitch = (0,) + pitch
        return offset, new_shape, new_pitch

    @staticmethod
    def _init_expand(
        *,
        offset: int,
        shape: Shape,
        pitch: Pitch,
        n: int,
    ) -> tuple[int, Shape, Pitch]:
        """Returns (offset, shape, pitch) for an expand view."""
        assert shape[0] == 1, "Can only expand along dimensions of size 1"
        new_shape = (n,) + shape[1:]
        new_pitch = (0,) + pitch[1:]
        return offset, new_shape, new_pitch

    @staticmethod
    def _init_squeeze(
        *,
        offset: int,
        shape: Shape,
        pitch: Pitch,
        dim: int,
    ) -> tuple[int, Shape, Pitch]:
        """Returns (offset, shape, pitch) for a squeezed view."""
        if dim < 0 or dim >= len(shape):
            raise IndexError(f"{dim=} out of range for tensor with ndim={len(shape)}")
        if shape[dim] != 1:
            raise ValueError(f"Cannot squeeze {dim=} with size {shape[dim]}")
        new_shape = shape[:dim] + shape[dim + 1 :]
        new_pitch = pitch[:dim] + pitch[dim + 1 :]
        return offset, new_shape, new_pitch

    #
    # Tensor properties:
    #

    @property
    def nbytes(self) -> int:
        return dtype_bytes(self.dtype) * self.numel

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    #
    # Operations:
    #

    def __pow__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="pow", other=other)

    def __mul__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="mul", other=other)

    def __truediv__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="div", other=other)

    def __add__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="add", other=other)

    def __sub__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="sub", other=other)

    def max(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="max", other=other)

    def min(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="min", other=other)

    def eq(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="eq", other=other)

    def ne(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="ne", other=other)

    def __lt__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="lt", other=other)

    def __gt__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="gt", other=other)

    def __le__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="le", other=other)

    def __ge__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.elementwise_binary_operation(op="ge", other=other)

    def elementwise_binary_operation(
        self,
        *,
        op: BinaryOperator,
        other: "Tensor | Scalar",
    ) -> "ElementwiseOperationTensor":
        other = Tensor._ensure_tensor(other)
        lt, rt = self._broadcast_binary_operands(other, check=lambda s1, s2: s1 == s2)
        return ElementwiseOperationTensor.new(operator=op, operands=(lt, rt))

    def __neg__(self) -> "ElementwiseOperationTensor":
        return ElementwiseOperationTensor.new(operator="neg", operands=(self,))

    def exp(self) -> "ElementwiseOperationTensor":
        return ElementwiseOperationTensor.new(operator="exp", operands=(self,))

    def log(self) -> "ElementwiseOperationTensor":
        return ElementwiseOperationTensor.new(operator="log", operands=(self,))

    def __invert__(self) -> "ElementwiseOperationTensor":
        return ElementwiseOperationTensor.new(operator="not", operands=(self,))

    def __matmul__(self, other: "Tensor") -> "Tensor":
        assert len(self.shape) in (2, 3) and len(other.shape) in (2, 3)

        must_squeeze_result = False
        if len(other.shape) < 3:
            other = other.broadcast(n=1)
            must_squeeze_result = True

        lt, rt = self._broadcast_binary_operands(
            other,
            check=lambda s1, s2: s1[-1] == s2[-2] and s1[:-2] == s2[:-2],
        )
        res = BatchedMatrixMultiplicationTensor.new(operands=(lt, rt))

        if must_squeeze_result:
            res = res.squeeze(dim=0)

        return res

    def broadcast(self, n: int) -> "BroadcastTensor":
        return BroadcastTensor.new(operand=self, n=n)

    def expand(self, n: int) -> "ExpandTensor":
        return ExpandTensor.new(operand=self, n=n)

    def squeeze(self, dim: int) -> "Tensor":
        return SqueezeTensor.new(operand=self, dim=dim)

    def reduce(self, *, axis: int, op: BinaryOperator) -> "Tensor":
        return ReductionTensor.new(operand=self, axis=axis, operator=op)

    def compact(self) -> "Tensor":
        """Copy tensor to new contiguous storage, preserving shape."""
        return CompactTensor.new(operand=self) if not self.is_c_contiguous else self

    def reshape(self, new_shape: Shape) -> "ReshapeTensor":
        """Reshape tensor to new shape. Tensor must be contiguous."""
        return ReshapeTensor.new(operand=self, new_shape=new_shape)

    def permute(self, new_order: tuple[int, ...]) -> "PermuteTensor":
        """Permute dimensions of tensor."""
        return PermuteTensor.new(operand=self, new_order=new_order)

    def transpose(self) -> "PermuteTensor":
        """Transpose the last two dimensions of the tensor."""
        if self.ndim < 2:
            raise ValueError("Cannot transpose tensor with less than 2 dimensions")
        new_order = tuple(range(self.ndim - 2)) + (self.ndim - 1, self.ndim - 2)
        return self.permute(new_order=new_order)

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> "Tensor":
        if not isinstance(key, tuple):
            key = (key,)
        return IndexTensor.new(operand=self, key=key)

    @staticmethod
    def _ensure_tensor(value: "Tensor | Scalar") -> "Tensor":
        return value if isinstance(value, Tensor) else ConstantTensor.new(value=value)

    def _broadcast_binary_operands(
        self, other: "Tensor", check: Callable[[Shape, Shape], bool]
    ) -> tuple["Tensor", "Tensor"]:
        """
        Broadcast two tensors to a common shape for binary operations, returning the
        broadcasted operands.

        The check function determines whether the current shapes are compatible.
        This function does not need to be commutative/symmetric.
        """

        if check(self.shape, other.shape):
            return self, other
        elif len(self.shape) > len(other.shape):
            rt, lt = other._broadcast_binary_operands(
                self,
                lambda s1, s2: check(s2, s1),
            )
            return lt, rt
        else:
            n = other.shape[-1 - len(self.shape)]
            return self.broadcast(n=n)._broadcast_binary_operands(other, check)

    #
    # subgraph reference counts
    #

    def compute_subgraph_reference_counts(self) -> dict["Tensor", int]:
        """
        Compute reference counts for all tensors in the subgraph rooted at this tensor.
        """
        ref_counts: dict["Tensor", int] = {}

        def visit(tensor: "Tensor") -> None:
            if tensor in ref_counts:
                ref_counts[tensor] += 1
            else:
                ref_counts[tensor] = 1
                for operand in tensor.operands:
                    visit(operand)

        visit(self)
        return ref_counts

    #
    # debug_print
    #

    def debug_print(self, out: SupportsWrite[str]) -> None:
        reference_count_map = self.compute_subgraph_reference_counts()
        assert reference_count_map[self] == 1, "Root tensor must have reference count 1"

        tid_map: dict["Tensor", int] = {}
        for tensor, ref_count in reference_count_map.items():
            assert ref_count >= 1
            if ref_count == 1:
                continue
            tid_map[tensor] = len(tid_map)

        self._debug_print(out, tid_map, prefix="")
        for tensor in tid_map:
            tensor._debug_print(out, tid_map, prefix="", node_type="root")

    def _debug_print(
        self,
        out: SupportsWrite[str],
        tid_map: dict["Tensor", int],
        *,
        prefix: str = "",
        node_type: Literal["root", "sibling", "last-sibling"] = "root",
    ) -> None:
        connector = {"root": "", "sibling": "├ ", "last-sibling": "└ "}[node_type]
        prefix_ext = {"root": "", "sibling": "│ ", "last-sibling": "  "}[node_type]

        tid = tid_map.get(self)
        if tid is None:
            print(f"{prefix}{connector}{self._debug_print_instr()}", file=out)
        else:
            if node_type == "root":
                print(
                    f"{prefix}{connector}%{tid} := {self._debug_print_instr()}",
                    file=out,
                )
            else:
                print(f"{prefix}{connector}%{tid}", file=out)
                return

        child_prefix = prefix + prefix_ext
        for operand_index, operand in enumerate(self.operands):
            is_last_operand = operand_index == len(self.operands) - 1
            child_node_type = "last-sibling" if is_last_operand else "sibling"
            operand._debug_print(
                out,
                tid_map,
                prefix=child_prefix,
                node_type=child_node_type,
            )

    def _debug_print_instr(self) -> str:
        return f"{self._debug_print_constructor()} :: {self._debug_print_dtype_and_shape()}"

    @abstractmethod
    def _debug_print_constructor(self) -> str:
        pass

    def _debug_print_dtype_and_shape(self) -> str:
        return f"({self.dtype} ({' '.join(str(dim) for dim in self.shape)}))"

    def topological_sort(self) -> list["Tensor"]:
        visited = set()
        result: list["Tensor"] = []

        def visit(tensor: "Tensor") -> None:
            if tensor in visited:
                return
            visited.add(tensor)
            for operand in tensor.operands:
                visit(operand)
            result.append(tensor)

        visit(self)
        return result[::-1]

    def __str__(self) -> str:
        return self._debug_print_constructor()


@dataclass(eq=False)
class ConstantTensor(Tensor):
    value: "list | Scalar"

    @staticmethod
    def new(*, value: "list | Scalar", dtype: DType = "fp32") -> "ConstantTensor":
        shape = ConstantTensor._infer_shape(value)
        pitch = _c_contiguous_pitch(shape)
        return ConstantTensor(
            dtype=dtype,
            offset=0,
            shape=shape,
            pitch=pitch,
            operands=(),
            value=value,
        )

    @staticmethod
    def _infer_shape(value: "list | Scalar") -> Shape:
        if is_scalar_instance(value):
            return ()
        assert isinstance(value, list)
        if not value:
            return (0,)
        e0_shape = ConstantTensor._infer_shape(value[0])
        for e1 in value[1:]:
            if ConstantTensor._infer_shape(e1) != e0_shape:
                raise ValueError("Inconsistent shapes in nested list")
        return (len(value),) + e0_shape

    def _debug_print_constructor(self) -> str:
        return f"(constant {ConstantTensor._format_value(self.value)})"

    @staticmethod
    def _format_value(value: "list | Scalar") -> str:
        if is_scalar_instance(value):
            return str(value)
        elif isinstance(value, list):
            return "[" + " ".join(ConstantTensor._format_value(v) for v in value) + "]"
        else:
            raise TypeError("Value must be a scalar or a nested list")


@dataclass(eq=False)
class VarTensor(Tensor):
    """
    WARNING: should never be constructed by the end-user, purely used to trace/record
    inputs to functions for automatic differentiation, etc.
    """

    name: str

    @staticmethod
    def new(*, name: str, dtype: DType, shape: Shape) -> "VarTensor":
        pitch = _c_contiguous_pitch(shape)
        return VarTensor(
            dtype=dtype,
            offset=0,
            shape=shape,
            pitch=pitch,
            operands=(),
            name=name,
        )

    def _debug_print_constructor(self) -> str:
        return f"(variable {self.name})"


@dataclass(eq=False)
class ElementwiseOperationTensor(Tensor):
    operator: Operator

    @staticmethod
    def new(
        *,
        operator: Operator,
        operands: tuple[Tensor, ...],
    ) -> "ElementwiseOperationTensor":
        shape = ElementwiseOperationTensor._infer_shape(operands)
        dtype = ElementwiseOperationTensor._infer_dtype(operands)
        pitch = ElementwiseOperationTensor._infer_pitch(operands, shape)
        return ElementwiseOperationTensor(
            dtype=dtype,
            offset=0,
            shape=shape,
            pitch=pitch,
            operands=operands,
            operator=operator,
        )

    @staticmethod
    def _infer_shape(operands: tuple[Tensor, ...]) -> Shape:
        assert operands
        first = operands[0]
        for op in operands[1:]:
            if op.shape != first.shape:
                raise ValueError(
                    f"Incompatible shapes for elementwise operation: {op.shape} and {first.shape}"
                )
        return first.shape

    @staticmethod
    def _infer_dtype(operands: tuple[Tensor, ...]) -> DType:
        assert operands
        dtype = operands[0].dtype
        for op in operands[1:]:
            if op.dtype != dtype:
                raise ValueError("Incompatible dtypes for elementwise operation")
        return dtype

    @staticmethod
    def _infer_pitch(operands: tuple[Tensor, ...], shape: Shape) -> Pitch:
        assert operands
        if len(operands) == 1:
            return operands[0].pitch
        elif all(op.pitch == operands[0].pitch for op in operands):
            return operands[0].pitch
        else:
            return _c_contiguous_pitch(shape)

    def _debug_print_constructor(self) -> str:
        return f'(elementwise "{self.operator}")'


@dataclass(eq=False)
class BatchedMatrixMultiplicationTensor(Tensor):
    @staticmethod
    def new(*, operands: tuple[Tensor, Tensor]) -> "BatchedMatrixMultiplicationTensor":
        left, right = operands
        shape = BatchedMatrixMultiplicationTensor._infer_shape(left, right)
        dtype = BatchedMatrixMultiplicationTensor._infer_dtype(left, right)
        pitch = _c_contiguous_pitch(shape)
        return BatchedMatrixMultiplicationTensor(
            dtype=dtype,
            offset=0,
            shape=shape,
            pitch=pitch,
            operands=operands,
        )

    @staticmethod
    def _infer_shape(left: Tensor, right: Tensor) -> Shape:
        if len(left.shape) != 3 or len(right.shape) != 3:
            raise ValueError("Both operands must be 3D matrices")
        if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[1]:
            raise ValueError(
                f"Incompatible shapes for matrix multiplication: {left.shape} and {right.shape}"
            )
        return (left.shape[0], left.shape[1], right.shape[2])

    @staticmethod
    def _infer_dtype(left: Tensor, right: Tensor) -> DType:
        if left.dtype != right.dtype:
            raise ValueError("Incompatible dtypes for matrix multiplication")
        return left.dtype

    def _debug_print_constructor(self) -> str:
        return "(batched-matmul)"


@dataclass(eq=False)
class BroadcastTensor(Tensor):
    @staticmethod
    def new(*, operand: Tensor, n: int) -> "BroadcastTensor":
        offset, shape, pitch = Tensor._init_broadcast(
            offset=operand.offset,
            shape=operand.shape,
            pitch=operand.pitch,
            n=n,
        )
        return BroadcastTensor(
            dtype=operand.dtype,
            offset=offset,
            shape=shape,
            pitch=pitch,
            operands=(operand,),
        )

    def _debug_print_constructor(self) -> str:
        return f"(broadcast {self.shape[0]})"


@dataclass(eq=False)
class ExpandTensor(Tensor):
    @staticmethod
    def new(*, operand: Tensor, n: int) -> "ExpandTensor":
        offset, shape, pitch = Tensor._init_expand(
            offset=operand.offset,
            shape=operand.shape,
            pitch=operand.pitch,
            n=n,
        )
        return ExpandTensor(
            dtype=operand.dtype,
            offset=offset,
            shape=shape,
            pitch=pitch,
            operands=(operand,),
        )

    def _debug_print_constructor(self) -> str:
        return f"(expand {self.shape[0]})"


@dataclass(eq=False)
class SqueezeTensor(Tensor):
    dim: int

    @staticmethod
    def new(*, operand: Tensor, dim: int) -> "SqueezeTensor":
        offset, shape, pitch = Tensor._init_squeeze(
            offset=operand.offset,
            shape=operand.shape,
            pitch=operand.pitch,
            dim=dim,
        )
        return SqueezeTensor(
            dtype=operand.dtype,
            offset=offset,
            shape=shape,
            pitch=pitch,
            operands=(operand,),
            dim=dim,
        )

    def _debug_print_constructor(self) -> str:
        return f"(squeeze {self.dim})"


@dataclass(eq=False)
class ReductionTensor(Tensor):
    axis: int
    operator: BinaryOperator

    @staticmethod
    def new(
        *,
        operand: Tensor,
        axis: int,
        operator: BinaryOperator,
    ) -> "ReductionTensor":
        shape = ReductionTensor._infer_shape(operand, axis)
        dtype = operand.dtype
        pitch = _c_contiguous_pitch(shape)
        return ReductionTensor(
            dtype=dtype,
            offset=0,
            shape=shape,
            pitch=pitch,
            operands=(operand,),
            axis=axis,
            operator=operator,
        )

    @staticmethod
    def _infer_shape(operand: Tensor, axis: int) -> Shape:
        if axis < 0 or axis >= len(operand.shape):
            raise ValueError("Axis out of bounds for reduction")
        return operand.shape[:axis] + operand.shape[axis + 1 :]

    def _debug_print_constructor(self) -> str:
        return f"(reduction {self.operator} {self.axis})"


@dataclass(eq=False)
class IndexTensor(Tensor):
    key: tuple[int | slice, ...]

    @staticmethod
    def new(
        *,
        operand: Tensor,
        key: tuple[int | slice, ...],
    ) -> "IndexTensor":
        offset, shape, pitch = IndexTensor._init_index(
            offset=operand.offset,
            shape=operand.shape,
            pitch=operand.pitch,
            key=key,
        )
        return IndexTensor(
            dtype=operand.dtype,
            offset=offset,
            shape=shape,
            pitch=pitch,
            operands=(operand,),
            key=key,
        )

    @staticmethod
    def _init_index(
        *,
        offset: int,
        shape: Shape,
        pitch: Pitch,
        key: tuple[int | slice, ...],
    ) -> tuple[int, Shape, Pitch]:
        """Returns (offset, shape, pitch) for an indexed view."""
        if len(key) > len(shape):
            raise IndexError(f"Too many indices for tensor with ndim={len(shape)}")
        new_shape: list[int] = []
        new_pitch: list[int] = []
        new_offset = offset
        for i, k in enumerate(key):
            if isinstance(k, int):
                if not (-shape[i] <= k < shape[i]):
                    raise IndexError("Index out of bounds")
                new_offset += (k % shape[i]) * pitch[i]
            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else shape[i]
                step = k.step if k.step is not None else 1
                abs_step = abs(step)
                dim_size = max(0, (stop - start + (abs_step - 1)) // abs_step)
                new_shape.append(dim_size)
                new_pitch.append(pitch[i] * step)
                new_offset += start * pitch[i]
            else:
                raise TypeError("Index must be int or slice")
        for j in range(len(key), len(shape)):
            new_shape.append(shape[j])
            new_pitch.append(pitch[j])
        return new_offset, tuple(new_shape), tuple(new_pitch)

    def _debug_print_constructor(self) -> str:
        return f"(index {self._format_key(self.key)})"

    @staticmethod
    def _format_key(key: tuple[int | slice, ...]) -> str:
        formatted_parts = []
        for k in key:
            if isinstance(k, int):
                formatted_parts.append(str(k))
            elif isinstance(k, slice):
                start = "()" if k.start is None else str(k.start)
                stop = "()" if k.stop is None else str(k.stop)
                step = "()" if k.step is None else str(k.step)
                formatted_parts.append(f"(slice {start} {stop} {step})")
            else:
                raise TypeError("Key must be int or slice")
        return "[" + " ".join(formatted_parts) + "]"


@dataclass(eq=False)
class CompactTensor(Tensor):
    """Copies tensor to new contiguous storage, preserving shape."""

    @staticmethod
    def new(*, operand: Tensor) -> "CompactTensor":
        pitch = _c_contiguous_pitch(operand.shape)
        return CompactTensor(
            dtype=operand.dtype,
            offset=0,
            shape=operand.shape,
            pitch=pitch,
            operands=(operand,),
        )

    def _debug_print_constructor(self) -> str:
        return "(compact)"


@dataclass(eq=False)
class ReshapeTensor(Tensor):
    """Reshapes a contiguous tensor to a new shape (view, no copy)."""

    new_shape: Shape

    @staticmethod
    def new(*, operand: Tensor, new_shape: Shape) -> "ReshapeTensor":
        if not operand.is_c_contiguous:
            raise ValueError("Cannot reshape non-contiguous tensor")
        if math.prod(new_shape) != operand.numel:
            raise ValueError(f"Bad reshape: {operand.shape} -> {new_shape}")
        pitch = _c_contiguous_pitch(new_shape)
        return ReshapeTensor(
            dtype=operand.dtype,
            offset=operand.offset,
            shape=new_shape,
            pitch=pitch,
            operands=(operand,),
            new_shape=new_shape,
        )

    def _debug_print_constructor(self) -> str:
        shape_str = " ".join(str(d) for d in self.new_shape)
        return f"(reshape ({shape_str}))"


@dataclass(eq=False)
class PermuteTensor(Tensor):
    """
    Permutes the dimensions of a tensor.

    Does not copy data, but renders C-contiguous tensors non-C-contiguous, even though
    the output is generally contiguous.
    """

    new_order: tuple[int, ...]

    @staticmethod
    def new(*, operand: Tensor, new_order: tuple[int, ...]) -> "PermuteTensor":
        if sorted(new_order) != list(range(len(operand.shape))):
            raise ValueError("Invalid permutation order")
        new_shape = tuple(operand.shape[i] for i in new_order)
        new_pitch = tuple(operand.pitch[i] for i in new_order)
        return PermuteTensor(
            dtype=operand.dtype,
            offset=operand.offset,
            shape=new_shape,
            pitch=new_pitch,
            operands=(operand,),
            new_order=new_order,
        )

    def _debug_print_constructor(self) -> str:
        order_str = " ".join(str(i) for i in self.new_order)
        return f"(permute ({order_str}))"


#
# Shape, Pitch:
#


type Shape = tuple[int, ...]
type Pitch = tuple[int, ...]


#
# DType:
#


type DType = Literal["fp32", "fp16"]


def dtype_bytes(dtype: DType) -> int:
    return {"fp32": 4, "fp16": 2}[dtype]


#
# Operator:
#


type Operator = UnaryOperator | BinaryOperator
type UnaryOperator = Literal["neg", "exp", "log", "not"]
type BinaryOperator = Literal[
    "pow", "mul", "div", "add", "sub", "max", "min", "eq", "ne", "lt", "gt", "le", "ge"
]


#
# Scalar:
#


type Scalar = float | int
_SCALAR_TYPES: tuple[type, ...] = (float, int)


def is_scalar_instance(value: object) -> bool:
    """Check if a value is a scalar (float or int)."""
    return isinstance(value, _SCALAR_TYPES)


def _c_contiguous_pitch(shape: Shape) -> Pitch:
    """
    Returns the C-contiguous strides for a given shape.

    An array is C-contiguous if it both...
    -   Contains no gaps between elements
    -   Has monotonically decreasing strides (like C arrays).

    Note that it is possible for a tensor to be contiguous without being C-contiguous.
    E.g. a permutation of a C-contiguous tensor.
    """
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


#
# Substitution:
#


class Substitution:
    """
    Memoized graph rewriter that eliminates ParameterTensors for concrete input tensors.

    The same Substitution should be used to rewrite all tensors from the same graph to
    avoid cloning shared subgraphs.
    """

    subs: dict[str, Tensor]
    memo: dict[Tensor, Tensor]

    def __init__(self, subs: dict[str, Tensor]) -> None:
        self.subs = subs
        self.memo = {}

    @staticmethod
    def zip(
        params: list[VarTensor],
        args: tuple[Tensor, ...],
    ) -> "Substitution":
        """
        Create a Substitution that replaces params with args.
        """
        assert len(params) == len(args)
        subs = {param.name: arg for param, arg in zip(params, args)}
        return Substitution(subs=subs)

    @staticmethod
    def zip_tensordict(
        params: TensorDict,
        values: TensorDict,
    ) -> "Substitution":
        assert tensordict_type(params) == tensordict_type(values)
        params_iterator = tensordict_flatten(params)
        values_iterator = tensordict_flatten(values)
        subs = {
            param.name: value
            for param, value in zip(params_iterator, values_iterator)
            if isinstance(param, VarTensor)
        }
        return Substitution(subs=subs)

    def rewrite(self, tensor: Tensor) -> Tensor:
        if cached := self.memo.get(tensor):
            return cached

        match tensor:
            case VarTensor(name=name) if replacement := self.subs.get(name):
                res = replacement
            case leaf if not tensor.operands:
                res = leaf
            case _:
                res = replace(
                    tensor,
                    operands=tuple(
                        self.rewrite(operand) for operand in tensor.operands
                    ),
                )

        self.memo[tensor] = res
        return res

    def rewrite_tensordict(self, td: TensorDict) -> TensorDict:
        match td:
            case Tensor() as t:
                return self.rewrite(t)
            case list() as tl:
                return [self.rewrite_tensordict(t) for t in tl]
            case dict() as td:
                return {k: self.rewrite_tensordict(v) for k, v in td.items()}
            case _:
                raise TypeError("TensorDict must be a Tensor, dict, or list")
