from typing import Literal

#
# Scalar
#


type Scalar = float | int
_SCALAR_TYPES: tuple[type, ...] = (float, int)


def is_scalar(value: object) -> bool:
    """Check if a value is a scalar (float or int)."""
    return isinstance(value, _SCALAR_TYPES)


#
# ScalarOperator
#

type ScalarOperator = UnaryScalarOperator | BinaryScalarOperator | BinaryCompareOperator
type UnaryScalarOperator = Literal["neg", "exp", "log", "not"]
type BinaryAssocScalarOperator = Literal["mul", "add", "max", "min"]
type BinaryScalarOperator = Literal["pow", "div", "sub"] | BinaryAssocScalarOperator
type BinaryCompareOperator = Literal["eq", "ne", "gt", "lt", "ge", "le"]


#
# ScalarType
#


type ScalarType = Literal["fp32", "fp16"]  # ~ DType in NumPy
type SKind = Literal["float"]


def stype_join(dtype1: ScalarType, dtype2: ScalarType) -> ScalarType:
    kind = stype_join_kind(stype_kind(dtype1), stype_kind(dtype2))
    nbytes = max(stype_nbytes(dtype1), stype_nbytes(dtype2))
    return stype(kind, nbytes)


def stype(kind: SKind, nbytes: int) -> ScalarType:
    match (kind, nbytes):
        case ("float", 4):
            return "fp32"
        case ("float", 2):
            return "fp16"
        case _:
            raise ValueError(f"Unsupported stype with kind={kind} and nbytes={nbytes}")


def stype_nbytes(stype: ScalarType) -> int:
    return {"fp32": 4, "fp16": 2}[stype]


def stype_kind(stype: ScalarType) -> SKind:
    match stype:
        case "fp32" | "fp16":
            return "float"
        case _:
            raise ValueError(f"Unsupported stype {stype}")


def stype_join_kind(dtype1: SKind, dtype2: SKind) -> SKind:
    if dtype1 != dtype2:
        raise ValueError(f"Cannot join different kinds' dtypes: {dtype1} and {dtype2}")
    return dtype1
