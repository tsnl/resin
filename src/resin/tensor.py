"""
resin.tensor = tensor/computational graph module.
"""

__all__ = [
    "DType",
    "dtype_bytes",
    "Operator",
    "UnaryOperator",
    "BinaryOperator",
    "Tensor",
    "ConstantTensor",
    "ElementwiseOperationTensor",
    "BatchedMatrixMultiplicationTensor",
    "BroadcastTensor",
    "SqueezeTensor",
    "ReductionTensor",
    "IndexTensor",
    "CompactTensor",
    "ReshapeTensor",
]

from abc import ABC, abstractmethod
import math
from typing import Literal

from .common import SupportsWrite


class Tensor(ABC):
    def __init__(
        self,
        *,
        dtype: "DType",
        offset: int = 0,
        shape: tuple[int, ...],
        pitch: tuple[int, ...] | None = None,
        operands: tuple["Tensor", ...],
    ) -> None:
        super().__init__()
        self.dtype: DType = dtype
        self.offset = offset
        self.shape = shape
        self.pitch = pitch or self._contiguous_pitch()
        self.operands = operands
        self.is_contiguous = self.pitch == self._contiguous_pitch()

    def _contiguous_pitch(self) -> tuple[int, ...]:
        return tuple(math.prod(self.shape[i + 1 :]) for i in range(len(self.shape)))

    @staticmethod
    def const(*, value: list, dtype: "DType" = "fp32") -> "ConstantTensor":
        return ConstantTensor(value=value, dtype=dtype)

    @staticmethod
    def _init_broadcast(
        *,
        offset: int,
        shape: tuple[int, ...],
        pitch: tuple[int, ...],
        n: int,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """Returns (offset, shape, pitch) for a broadcast view."""
        if shape and shape[0] == 1:
            new_shape = (n,) + shape[1:]
            new_pitch = (0,) + pitch[1:]
        else:
            new_shape = (n,) + shape
            new_pitch = (0,) + pitch
        return offset, new_shape, new_pitch

    @staticmethod
    def _init_squeeze(
        *,
        offset: int,
        shape: tuple[int, ...],
        pitch: tuple[int, ...],
        dim: int,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """Returns (offset, shape, pitch) for a squeezed view."""
        if dim < 0 or dim >= len(shape):
            raise IndexError(f"{dim=} out of range for tensor with ndim={len(shape)}")
        if shape[dim] != 1:
            raise ValueError(f"Cannot squeeze {dim=} with size {shape[dim]}")
        new_shape = shape[:dim] + shape[dim + 1 :]
        new_pitch = pitch[:dim] + pitch[dim + 1 :]
        return offset, new_shape, new_pitch

    @property
    def nbytes(self) -> int:
        return dtype_bytes(self.dtype) * self.numel

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def __pow__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.bop(op="pow", other=other)

    def __mul__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.bop(op="mul", other=other)

    def __truediv__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.bop(op="div", other=other)

    def __mod__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.bop(op="rem", other=other)

    def __add__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.bop(op="add", other=other)

    def __sub__(self, other: "Tensor | Scalar") -> "ElementwiseOperationTensor":
        return self.bop(op="sub", other=other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        assert len(self.shape) in (2, 3)
        assert len(other.shape) in (2, 3)

        must_squeeze_result = False
        if len(other.shape) < 3:
            other = other.broadcast(n=1)
            must_squeeze_result = True

        lt, rt = self._broadcast_symmetric_operands(other)
        res = BatchedMatrixMultiplicationTensor(operands=(lt, rt))

        if must_squeeze_result:
            res = res.squeeze(dim=0)

        return res

    def bop(
        self,
        *,
        op: "BinaryOperator",
        other: "Tensor | Scalar",
    ) -> "ElementwiseOperationTensor":
        other = Tensor._ensure_tensor(other)
        lt, rt = self._broadcast_symmetric_operands(other)
        return ElementwiseOperationTensor(operator=op, operands=(lt, rt))

    def exp(self) -> "ElementwiseOperationTensor":
        return ElementwiseOperationTensor(operator="exp", operands=(self,))

    def log(self) -> "ElementwiseOperationTensor":
        return ElementwiseOperationTensor(operator="log", operands=(self,))

    def broadcast(self, n: int) -> "BroadcastTensor":
        return BroadcastTensor(operand=self, n=n)

    def squeeze(self, dim: int) -> "Tensor":
        return SqueezeTensor(operand=self, dim=dim)

    def reduce(self, *, axis: int, op: "BinaryOperator") -> "Tensor":
        return ReductionTensor(operand=self, axis=axis, operator=op)

    def compact(self) -> "Tensor":
        """Copy tensor to new contiguous storage, preserving shape."""
        return CompactTensor(operand=self) if not self.is_contiguous else self

    def reshape(self, new_shape: tuple[int, ...]) -> "ReshapeTensor":
        """Reshape tensor to new shape. Tensor must be contiguous."""
        return ReshapeTensor(operand=self, new_shape=new_shape)

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> "Tensor":
        if not isinstance(key, tuple):
            key = (key,)
        return IndexTensor(operand=self, key=key)

    @staticmethod
    def _ensure_tensor(value: "Tensor | Scalar") -> "Tensor":
        return value if isinstance(value, Tensor) else ConstantTensor(value=[value])

    def _broadcast_symmetric_operands(
        self, other: "Tensor"
    ) -> tuple["Tensor", "Tensor"]:
        if len(self.shape) == len(other.shape):
            return self, other
        elif len(self.shape) > len(other.shape):
            rt, lt = other._broadcast_symmetric_operands(self)
            return lt, rt
        else:
            n = other.shape[-1 - len(self.shape)]
            return self.broadcast(n=n)._broadcast_symmetric_operands(other)

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

    def debug_print(self, out: SupportsWrite[str]) -> None:
        # Compute reference counts for all tensors in the subgraph:
        reference_count_map = self.compute_subgraph_reference_counts()
        assert reference_count_map[self] == 1, "Root tensor must have reference count 1"

        # Assign TID numbers to shared tensors (reference count > 1):
        tid_map: dict["Tensor", int] = {}
        for tensor, ref_count in reference_count_map.items():
            assert ref_count >= 1
            if ref_count == 1:
                continue
            tid_map[tensor] = len(tid_map)

        # Print the root node, then all shared tensors:
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
        """
        Returns a topological sort of the subgraph rooted at this tensor.

        TODO: Verify if we can replace this with compute_subgraph_reference_counts().
        """

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

        return result[::-1]  # Reverse to get correct topological order

    def __str__(self) -> str:
        return self._debug_print_constructor()


class ConstantTensor(Tensor):
    def __init__(self, *, value: list, dtype: "DType" = "fp32") -> None:
        shape = ConstantTensor._infer_shape(value)
        super().__init__(dtype=dtype, shape=shape, operands=())
        self.value = value

    @staticmethod
    def _infer_shape(value: "list | Scalar") -> tuple[int, ...]:
        if is_scalar_instance(value):
            return ()

        assert isinstance(value, list)

        if not value:
            return (0,)

        e0_shape = ConstantTensor._infer_shape(value[0])

        for e1 in value[1:]:
            e1_shape = ConstantTensor._infer_shape(e1)
            if e1_shape != e0_shape:
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


class ElementwiseOperationTensor(Tensor):
    def __init__(self, operator: "Operator", operands: tuple[Tensor, ...]):
        shape = ElementwiseOperationTensor._infer_shape(operands)
        dtype = ElementwiseOperationTensor._infer_dtype(operands)
        super().__init__(dtype=dtype, shape=shape, operands=operands)
        self.operator = operator

    @staticmethod
    def _infer_shape(operands: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert operands

        first_operand = operands[0]

        for operand in operands[1:]:
            if operand.shape != first_operand.shape:
                raise ValueError(
                    "Incompatible shapes for elementwise operation: "
                    f"{operand.shape} and {first_operand.shape}"
                )

        return first_operand.shape

    @staticmethod
    def _infer_dtype(operands: tuple[Tensor, ...]) -> "DType":
        assert operands

        dtype = operands[0].dtype

        for operand in operands[1:]:
            if operand.dtype != dtype:
                raise ValueError("Incompatible dtypes for elementwise operation")

        return dtype

    def _debug_print_constructor(self) -> str:
        return f'(elementwise "{self.operator}")'


class BatchedMatrixMultiplicationTensor(Tensor):
    def __init__(self, *, operands: tuple[Tensor, Tensor]) -> None:
        left, right = operands
        shape = BatchedMatrixMultiplicationTensor._infer_shape(left, right)
        dtype = BatchedMatrixMultiplicationTensor._infer_dtype(left, right)
        super().__init__(dtype=dtype, shape=shape, operands=operands)

    @staticmethod
    def _infer_shape(left: Tensor, right: Tensor) -> tuple[int, ...]:
        if len(left.shape) != 3 or len(right.shape) != 3:
            raise ValueError("Both operands must be 3D matrices")

        if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[1]:
            raise ValueError(
                "Incompatible shapes for matrix multiplication: "
                f"{left.shape} and {right.shape}"
            )

        return (left.shape[0], left.shape[1], right.shape[2])

    @staticmethod
    def _infer_dtype(left: Tensor, right: Tensor) -> "DType":
        if left.dtype != right.dtype:
            raise ValueError("Incompatible dtypes for matrix multiplication")

        return left.dtype

    def _debug_print_constructor(self) -> str:
        return "(batched-matmul)"


class BroadcastTensor(Tensor):
    def __init__(self, *, operand: Tensor, n: int) -> None:
        offset, shape, pitch = Tensor._init_broadcast(
            offset=operand.offset,
            shape=operand.shape,
            pitch=operand.pitch,
            n=n,
        )
        super().__init__(
            dtype=operand.dtype,
            offset=offset,
            shape=shape,
            pitch=pitch,
            operands=(operand,),
        )

    def _debug_print_constructor(self) -> str:
        return f"(broadcast {self.shape[0]})"


class SqueezeTensor(Tensor):
    def __init__(self, *, operand: Tensor, dim: int) -> None:
        offset, shape, pitch = Tensor._init_squeeze(
            offset=operand.offset,
            shape=operand.shape,
            pitch=operand.pitch,
            dim=dim,
        )
        super().__init__(
            dtype=operand.dtype,
            offset=offset,
            shape=shape,
            pitch=pitch,
            operands=(operand,),
        )
        self.dim = dim

    def _debug_print_constructor(self) -> str:
        return f"(squeeze {self.dim})"


class ReductionTensor(Tensor):
    def __init__(
        self, *, operand: Tensor, axis: int, operator: "BinaryOperator"
    ) -> None:
        shape = ReductionTensor._infer_shape(operand, axis)
        dtype = ReductionTensor._infer_dtype(operand)
        super().__init__(dtype=dtype, shape=shape, operands=(operand,))
        self.axis = axis
        self.operator = operator

    @staticmethod
    def _infer_shape(operand: Tensor, axis: int) -> tuple[int, ...]:
        if axis < 0 or axis >= len(operand.shape):
            raise ValueError("Axis out of bounds for reduction")
        return operand.shape[:axis] + operand.shape[axis + 1 :]

    @staticmethod
    def _infer_dtype(operand: Tensor) -> "DType":
        return operand.dtype

    def _debug_print_constructor(self) -> str:
        return f"(reduction {self.operator} {self.axis})"


class IndexTensor(Tensor):
    def __init__(self, *, operand: Tensor, key: tuple[int | slice, ...]) -> None:
        offset, shape, pitch = IndexTensor._init_index(
            offset=operand.offset,
            shape=operand.shape,
            pitch=operand.pitch,
            key=key,
        )
        super().__init__(
            dtype=operand.dtype,
            offset=offset,
            shape=shape,
            pitch=pitch,
            operands=(operand,),
        )
        self.key = key

    @staticmethod
    def _init_index(
        *,
        offset: int,
        shape: tuple[int, ...],
        pitch: tuple[int, ...],
        key: tuple[int | slice, ...],
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
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


class CompactTensor(Tensor):
    """Copies tensor to new contiguous storage, preserving shape."""

    def __init__(self, *, operand: Tensor) -> None:
        super().__init__(dtype=operand.dtype, shape=operand.shape, operands=(operand,))

    def _debug_print_constructor(self) -> str:
        return "(compact)"


class ReshapeTensor(Tensor):
    """Reshapes a contiguous tensor to a new shape (view, no copy)."""

    def __init__(self, *, operand: Tensor, new_shape: tuple[int, ...]) -> None:
        if not operand.is_contiguous:
            raise ValueError("Cannot reshape non-contiguous tensor")
        if math.prod(new_shape) != operand.numel:
            raise ValueError(f"Bad reshape: {operand.shape} -> {new_shape}")
        super().__init__(
            dtype=operand.dtype,
            offset=operand.offset,
            shape=new_shape,
            operands=(operand,),
        )
        self.new_shape = new_shape

    def _debug_print_constructor(self) -> str:
        shape_str = " ".join(str(d) for d in self.new_shape)
        return f"(reshape ({shape_str}))"


type DType = Literal["fp32", "fp16"]


def dtype_bytes(dtype: DType) -> int:
    return {"fp32": 4, "fp16": 2}[dtype]


type Operator = UnaryOperator | BinaryOperator
type UnaryOperator = Literal["exp", "log"]
type BinaryOperator = Literal["pow", "mul", "div", "rem", "add", "sub"]

type Scalar = float | int
_SCALAR_TYPES: tuple[type, ...] = (float, int)


def is_scalar_instance(value: object) -> bool:
    """Check if a value is a scalar (float or int)."""
    return isinstance(value, _SCALAR_TYPES)
