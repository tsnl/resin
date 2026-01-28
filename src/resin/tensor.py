__all__ = [
    "new",
    "Tensor",
    "ConstantTensor",
    "OperatorTensor",
]

import textwrap
from typing import Literal


def new(value: list, dtype: "DType" = "float32") -> Tensor:
    return ConstantTensor(value=value, dtype=dtype)


class Tensor:
    dtype: "DType"
    shape: list[int]

    def __init__(self, *, dtype: "DType", shape: list[int]):
        self.dtype = dtype
        self.shape = shape

    def to_sexp(self, indent: int = 0) -> str:
        _ = indent
        raise NotImplementedError()

    def __pos__(self) -> "OperatorTensor":
        return OperatorTensor(
            operator="pos",
            args=[self],
            dtype=self.dtype,
            shape=self.shape,
        )

    def __neg__(self) -> "OperatorTensor":
        return OperatorTensor(
            operator="neg",
            args=[self],
            dtype=self.dtype,
            shape=self.shape,
        )

    def __invert__(self) -> "OperatorTensor":
        return OperatorTensor(
            operator="not",
            args=[self],
            dtype=self.dtype,
            shape=self.shape,
        )

    def __mul__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="mul",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __truediv__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="div",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __floordiv__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="div",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __mod__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="mod",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __add__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="add",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __sub__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="sub",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __lsh__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="lsh",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __rsh__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="rsh",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __eq__(self, other: object) -> "OperatorTensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        assert isinstance(other, Tensor)
        return OperatorTensor(
            operator="eq",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __ne__(self, other: object) -> "OperatorTensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        assert isinstance(other, Tensor)
        return OperatorTensor(
            operator="neq",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __lt__(self, other: object) -> "OperatorTensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        assert isinstance(other, Tensor)
        return OperatorTensor(
            operator="lt",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __le__(self, other: object) -> "OperatorTensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        assert isinstance(other, Tensor)
        return OperatorTensor(
            operator="le",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __gt__(self, other: object) -> "OperatorTensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        assert isinstance(other, Tensor)
        return OperatorTensor(
            operator="gt",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __ge__(self, other: object) -> "OperatorTensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        assert isinstance(other, Tensor)
        return OperatorTensor(
            operator="ge",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __and__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="and",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __or__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="or",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_elementwise_binop_shape(self.shape, other.shape),
        )

    def __matmul__(self, other: "Tensor") -> "OperatorTensor":
        return OperatorTensor(
            operator="matmul",
            args=[self, other],
            dtype=infer_binop_dtype(self.dtype, other.dtype),
            shape=infer_matmul_shape(self.shape, other.shape),
        )


class ConstantTensor(Tensor):
    def __init__(self, *, value: list, dtype: "DType"):
        super().__init__(dtype=dtype, shape=infer_shape(value))
        self.value = value

    def __str__(self) -> str:
        return self.to_sexp()

    def __repr__(self) -> str:
        return self.__str__()

    def to_sexp(self, indent: int = 0) -> str:
        return _format_array(self.value, indent)


class OperatorTensor(Tensor):
    def __init__(
        self,
        *,
        operator: Operator,
        args: list[Tensor],
        dtype: "DType",
        shape: list[int],
    ):
        super().__init__(dtype=dtype, shape=shape)

        self.operator = operator
        self.args = args

    def __str__(self) -> str:
        return self.to_sexp()

    def __repr__(self) -> str:
        return self.__str__()

    def to_sexp(self, indent: int = 0) -> str:
        # Check if everything fits on one line
        one_line = (
            f"({self.operator} " + " ".join(arg.to_sexp(0) for arg in self.args) + ")"
        )
        if "\n" not in one_line and len(one_line) <= DEFAULT_WRAP_LEN:
            return one_line

        # Multi-line format
        child_indent = indent + DEFAULT_INDENT_SIZE
        lines = [f"({self.operator}"]
        for arg in self.args:
            arg_str = arg.to_sexp(0)
            indented = textwrap.indent(arg_str, " " * child_indent)
            lines.append(indented)
        return "\n".join(lines) + ")"


def _format_array(value: list | int | float, base_indent: int = 0) -> str:
    """
    Format a nested list in NumPy-style array notation.

    The base_indent controls alignment of continuation lines relative to the
    opening bracket. The first line has no leading spaces.
    """

    if isinstance(value, (int, float)):
        return str(value)

    if not value:
        return "[]"

    # Check if this is the innermost list (contains scalars)
    if not isinstance(value[0], list):
        return "[" + " ".join(str(v) for v in value) + "]"

    # Multi-dimensional: format each sub-array on its own line
    # Continuation lines align with the first element (1 char after '[')
    continuation_indent = base_indent + 1
    formatted = [_format_array(v, continuation_indent) for v in value]

    # First element on same line as opening bracket
    lines = ["[" + formatted[0]]

    # Subsequent elements aligned with first element
    for f in formatted[1:]:
        lines.append(" " * continuation_indent + f)
    lines[-1] = lines[-1] + "]"

    return "\n".join(lines)


def infer_shape(value: list | int | float) -> list[int]:
    """
    Given a nested list, verify and compute the shape of the resulting tensor.
    """

    def help_infer_reversed_shape(value: list | int | float) -> list[int]:
        """
        Helper for infer_shape that computes the shape in reverse order.
        """

        if isinstance(value, (int, float)):
            return []

        assert isinstance(value, list)
        if not value:
            return [0]

        e0_shape = help_infer_reversed_shape(value[0])
        for element in value[1:]:
            e1_shape = help_infer_reversed_shape(element)
            if e1_shape != e0_shape:
                raise ValueError("Malformed data for ConstantExpr")

        ret_shape = e0_shape
        ret_shape.append(len(value))
        return ret_shape

    return list(reversed(help_infer_reversed_shape(value)))


def infer_binop_dtype(dtype1: "DType", dtype2: "DType") -> "DType":
    match (dtype1, dtype2):
        case ("float32", _) | (_, "float32"):
            return "float32"
        case ("float16", _) | (_, "float16"):
            return "float16"
        case ("int", "int"):
            return "int"
        case _:
            raise ValueError(f"Cannot infer binop dtype for {dtype1} and {dtype2}")


def infer_elementwise_binop_shape(shape1: list[int], shape2: list[int]) -> list[int]:
    if shape1 == shape2:
        return shape1

    if len(shape1) > len(shape2):
        return infer_elementwise_binop_shape(shape2, shape1)

    assert len(shape1) <= len(shape2)
    if not list_endswith(shape2[-len(shape1) :], shape1):
        raise ValueError(f"Incompatible shapes for binop: {shape1} and {shape2}")

    return shape2


def infer_matmul_shape(shape1: list[int], shape2: list[int]) -> list[int]:
    # Check input ranks:
    r1 = len(shape1)
    r2 = len(shape2)
    if r1 not in {2, 3} or r2 not in {2, 3}:
        raise ValueError("Matmul shapes must be rank 2 or 3")

    # Normalize to rank 3, extract 'b', 'r', 'c' for each shape:
    (b1, r1, c1) = shape1 if len(shape1) == 3 else [1] + shape1
    (b2, r2, c2) = shape2 if len(shape2) == 3 else [1] + shape2

    # Check batch dimensions are broadcastable:
    if b1 != 1 and b2 != 1 and b1 != b2:
        raise ValueError(f"Incompatible shapes for matmul: {shape1=} @ {shape2=}")

    # Check matrix multiplication dimensions:
    if c1 != r2:
        raise ValueError(f"Incompatible shapes for matmul: {shape1=} @ {shape2=}")

    # Compute output shape:
    b = max(b1, b2)
    r = r1
    c = c2
    return [b, r, c] if b > 1 else [r, c]


def list_endswith(full: list[int], suffix: list[int]) -> bool:
    if len(suffix) > len(full):
        return False
    return full[-len(suffix) :] == suffix


DType = Literal["int", "float16", "float32"]

Operator = Literal[
    "pos",
    "neg",
    "not",
    "exp",
    "log",
    "sqrt",
    "pow",
    "mul",
    "div",
    "mod",
    "add",
    "sub",
    "lsh",
    "rsh",
    "eq",
    "neq",
    "lt",
    "le",
    "gt",
    "ge",
    "and",
    "or",
    "matmul",
]

DEFAULT_WRAP_LEN = 80
DEFAULT_INDENT_SIZE = 2
