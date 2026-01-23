from typing import Callable, TypeAlias, dataclass_transform
from . import ir
from . import back

#
# DSL API:
#


def function(func: Callable, /):
    """Decorator to mark a function for C/GLSL code generation."""
    return ir.Module.get(func.__module__).define_function(func)


@dataclass_transform()
def struct(cls: type, /):
    """Decorator to mark a class as a C/GLSL struct."""
    return ir.Module.get(cls.__module__).define_struct(cls)


# fmt: off
Void: TypeAlias = ir.VOID_TYPE      # pyright: ignore[reportInvalidTypeForm]
Char: TypeAlias = ir.CHAR_TYPE      # pyright: ignore[reportInvalidTypeForm]
Short: TypeAlias = ir.SHORT_TYPE    # pyright: ignore[reportInvalidTypeForm]
Int: TypeAlias = ir.INT_TYPE        # pyright: ignore[reportInvalidTypeForm]
Long: TypeAlias = ir.LONG_TYPE      # pyright: ignore[reportInvalidTypeForm]
UChar: TypeAlias = ir.UCHAR_TYPE    # pyright: ignore[reportInvalidTypeForm]
UShort: TypeAlias = ir.USHORT_TYPE  # pyright: ignore[reportInvalidTypeForm]
UInt: TypeAlias = ir.UINT_TYPE      # pyright: ignore[reportInvalidTypeForm]
ULong: TypeAlias = ir.ULONG_TYPE    # pyright: ignore[reportInvalidTypeForm]
Fp32: TypeAlias = ir.FP32_TYPE      # pyright: ignore[reportInvalidTypeForm]
Fp64: TypeAlias = ir.FP64_TYPE      # pyright: ignore[reportInvalidTypeForm]
Ptr: TypeAlias = ir.PTR_SCHEME        # pyright: ignore[reportInvalidTypeForm]
Fn: TypeAlias = ir.FN_SCHEME          # pyright: ignore[reportInvalidTypeForm]
# fmt: on


#
# Compiler API:
#


def build():
    """Compile all defined modules into executables with SPV embedded."""
    back.build(ir.Module.all())
