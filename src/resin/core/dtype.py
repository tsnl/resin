from typing import Literal

type Scalar = float | int
_SCALAR_TYPES: tuple[type, ...] = (float, int)


def is_scalar(value: object) -> bool:
    return isinstance(value, _SCALAR_TYPES)


type ScalarOperator = UnaryScalarOperator | BinaryScalarOperator | BinaryCompareOperator
type UnaryScalarOperator = Literal[
    "neg",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "not",
]
type BinaryAssocScalarOperator = Literal["mul", "add", "max", "min"]
type BinaryScalarOperator = Literal["pow", "div", "sub"] | BinaryAssocScalarOperator
type BinaryCompareOperator = Literal["eq", "ne", "gt", "lt", "ge", "le"]

# Tile dtypes (e.g. "mat4x4_f4") extend this union as they are added.
type DType = Literal["f4", "f2", "u4"]
type DKind = Literal["float", "uint"]


def dtype_join(dtype1: DType, dtype2: DType) -> DType:
    kind = dtype_join_kind(dtype_kind(dtype1), dtype_kind(dtype2))
    nbytes = max(dtype_nbytes(dtype1), dtype_nbytes(dtype2))
    return dtype(kind, nbytes)


def dtype(kind: DKind, nbytes: int) -> DType:
    match (kind, nbytes):
        case ("float", 4):
            return "f4"
        case ("float", 2):
            return "f2"
        case ("uint", 4):
            return "u4"
        case _:
            raise ValueError(f"Unsupported dtype with kind={kind} and nbytes={nbytes}")


def dtype_nbytes(dtype: DType) -> int:
    return {"f4": 4, "f2": 2, "u4": 4}[dtype]


def dtype_kind(dtype: DType) -> DKind:
    match dtype:
        case "f4" | "f2":
            return "float"
        case "u4":
            return "uint"
        case _:
            raise ValueError(f"Unsupported dtype {dtype}")


def dtype_join_kind(dtype1: DKind, dtype2: DKind) -> DKind:
    if dtype1 != dtype2:
        raise ValueError(f"Cannot join different kinds' dtypes: {dtype1} and {dtype2}")
    return dtype1


def spell_dtype_in_pystruct(dtype: DType) -> str:
    return {"f4": "f", "f2": "e", "u4": "I"}[dtype]


def spell_dtype_in_wgsl(dtype: DType) -> str:
    return {"f4": "f32", "f2": "f16", "u4": "u32"}[dtype]


def dtype_needs_enable_f16(dtype: DType) -> bool:
    return dtype == "f2"