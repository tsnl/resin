import struct
from typing import cast

from .dtype import DType, Scalar, is_scalar, spell_dtype_in_pystruct

type PyTensor = Scalar | list[PyTensor] | tuple[PyTensor, ...]
type PyTree[T] = "dict[str, PyTree[T]] | list[PyTree[T]] | T"


def infer_pytensor_shape(value: "PyTensor") -> tuple[int, ...]:
    if is_scalar(value):
        return ()
    assert isinstance(value, list)
    if not value:
        return (0,)
    e0_shape = infer_pytensor_shape(value[0])
    for e1 in value[1:]:
        if infer_pytensor_shape(e1) != e0_shape:
            raise ValueError("Inconsistent shapes in nested list")
    return (len(value),) + e0_shape


def marshall_pytensor(value: "PyTensor", dtype: DType) -> bytes:
    values = flatten_pytensor(value)
    return struct.pack(f"<{len(values)}{spell_dtype_in_pystruct(dtype)}", *values)


def flatten_pytensor(value: "PyTensor") -> list[Scalar]:
    if is_scalar(value):
        return [cast(Scalar, value)]
    assert isinstance(value, list)
    result = []
    for e in value:
        result.extend(flatten_pytensor(e))
    return result