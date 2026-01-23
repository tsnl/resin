__all__ = [
    "CHAR_TYPE",
    "FN_SCHEME",
    "FP32_TYPE",
    "FP64_TYPE",
    "INT_TYPE",
    "LONG_TYPE",
    "PTR_SCHEME",
    "SHORT_TYPE",
    "UCHAR_TYPE",
    "UINT_TYPE",
    "ULONG_TYPE",
    "USHORT_TYPE",
    "VOID_TYPE",
    "FnType",
    "PrimitiveType",
    "PtrType",
    "StructType",
    "Type",
]

from abc import ABC
from typing import Literal


class Type(ABC):
    """
    Type of C/GLSL types. Instances of this class are used as type annotations in
    CFunction definitions.
    """

    def __call__(self, *args, **kwargs):
        _ = args
        _ = kwargs
        raise RuntimeError(
            "`cc.Type` constructors cannot be called outside other CFunction definitions."
        )


class PrimitiveType(Type, ABC):
    """Type of a primitive C/GLSL type."""

    Identity = Literal[
        "Void",
        "Char",
        "Short",
        "Int",
        "Long",
        "UChar",
        "UShort",
        "UInt",
        "ULong",
        "Fp32",
        "Fp64",
    ]

    name: str

    def __init__(self, name: PrimitiveType.Identity):
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return self.name


class StructType(Type):
    """Type of a C/GLSL struct type."""

    raw: type

    def __init__(self, raw: type):
        super().__init__()
        self.raw = raw

    def __repr__(self) -> str:
        return f"struct {self.raw.__name__}"


class PtrType(Type):
    """Type of a pointer type."""

    pointee: Type

    def __init__(self, pointee: Type):
        super().__init__()
        self.pointee = pointee

    def __repr__(self) -> str:
        return f"Ptr[{self.pointee}]"


class FnType(Type):
    """Type of a function type."""

    param_types: list[Type]
    return_type: Type

    def __init__(self, param_types: list[Type], return_type: Type):
        super().__init__()
        self.param_types = param_types
        self.return_type = return_type

    def __repr__(self) -> str:
        params = ", ".join(str(t) for t in self.param_types)
        return f"({params}) -> {self.return_type}"


class PtrScheme:
    def __getitem__(self, key: Type) -> PtrType:
        return PtrType(pointee=key)


class FnScheme:
    def __getitem__(self, key: tuple[list[Type], Type]) -> FnType:
        param_types, return_type = key
        return FnType(param_types, return_type)


VOID_TYPE = PrimitiveType("Void")
CHAR_TYPE = PrimitiveType("Char")
SHORT_TYPE = PrimitiveType("Short")
INT_TYPE = PrimitiveType("Int")
LONG_TYPE = PrimitiveType("Long")
UCHAR_TYPE = PrimitiveType("UChar")
USHORT_TYPE = PrimitiveType("UShort")
UINT_TYPE = PrimitiveType("UInt")
ULONG_TYPE = PrimitiveType("ULong")
FP32_TYPE = PrimitiveType("Fp32")
FP64_TYPE = PrimitiveType("Fp64")
PTR_SCHEME = PtrScheme()
FN_SCHEME = FnScheme()
