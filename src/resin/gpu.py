__all__ = [
    "processor",
    "BaseProcessor",
    "struct",
    "BaseTypeSpec",
    "context",
    "I32",
    "U32",
    "F16",
    "F32",
    "F64",
    "Vector",
]

from abc import ABC
from dataclasses import dataclass
from typing import Literal, dataclass_transform, Annotated, TypeAlias

from . import typed_vulkan as tv


#
# `processor` decorator:
#


@dataclass_transform(kw_only_default=True)
def processor[T](cls: type[T]) -> type[T]:
    return cls


class BaseProcessor(ABC):
    pass


class BufferView[T](ABC):
    pass


class Buffer[T](BufferView[T], ABC):
    pass


class Texture(ABC):
    pass


class Sampler(ABC):
    pass


#
# `struct` decorator:
#


@dataclass_transform()
def struct[T](cls: type[T]) -> type[T]:
    return cls


class BaseTypeSpec(ABC):
    pass


class BaseScalarTypeSpec(BaseTypeSpec, ABC):
    pass


class BaseIntegralScalarTypeSpec(BaseScalarTypeSpec, ABC):
    pass


class I32(BaseIntegralScalarTypeSpec):
    pass


class U32(BaseIntegralScalarTypeSpec):
    pass


class BaseFloatingPointScalarTypeSpec(BaseScalarTypeSpec, ABC):
    pass


class F16(BaseFloatingPointScalarTypeSpec):
    pass


class F32(BaseFloatingPointScalarTypeSpec):
    pass


class F64(BaseFloatingPointScalarTypeSpec):
    pass


class Vector[T: BaseScalarTypeSpec, N: NDim](BaseTypeSpec):
    pass


class Matrix[T: BaseScalarTypeSpec, R: NDim, C: NDim](BaseTypeSpec):
    pass


type NDim = Literal[1, 2, 3, 4]

Vec2f: TypeAlias = Vector[F32, Literal[2]]
Vec3f: TypeAlias = Vector[F32, Literal[3]]
Vec4f: TypeAlias = Vector[F32, Literal[4]]

Vec2h: TypeAlias = Vector[F16, Literal[2]]
Vec3h: TypeAlias = Vector[F16, Literal[3]]
Vec4h: TypeAlias = Vector[F16, Literal[4]]

Vec2d: TypeAlias = Vector[F64, Literal[2]]
Vec3d: TypeAlias = Vector[F64, Literal[3]]
Vec4d: TypeAlias = Vector[F64, Literal[4]]

Vec2u: TypeAlias = Vector[U32, Literal[2]]
Vec3u: TypeAlias = Vector[U32, Literal[3]]
Vec4u: TypeAlias = Vector[U32, Literal[4]]

Vec2i: TypeAlias = Vector[I32, Literal[2]]
Vec3i: TypeAlias = Vector[I32, Literal[3]]
Vec4i: TypeAlias = Vector[I32, Literal[4]]

Mat1x1f: TypeAlias = Matrix[F32, Literal[1], Literal[1]]
Mat1x2f: TypeAlias = Matrix[F32, Literal[1], Literal[2]]
Mat1x3f: TypeAlias = Matrix[F32, Literal[1], Literal[3]]
Mat1x4f: TypeAlias = Matrix[F32, Literal[1], Literal[4]]
Mat2x1f: TypeAlias = Matrix[F32, Literal[2], Literal[1]]
Mat2x2f: TypeAlias = Matrix[F32, Literal[2], Literal[2]]
Mat2x3f: TypeAlias = Matrix[F32, Literal[2], Literal[3]]
Mat2x4f: TypeAlias = Matrix[F32, Literal[2], Literal[4]]
Mat3x1f: TypeAlias = Matrix[F32, Literal[3], Literal[1]]
Mat3x2f: TypeAlias = Matrix[F32, Literal[3], Literal[2]]
Mat3x3f: TypeAlias = Matrix[F32, Literal[3], Literal[3]]
Mat3x4f: TypeAlias = Matrix[F32, Literal[3], Literal[4]]
Mat4x1f: TypeAlias = Matrix[F32, Literal[4], Literal[1]]
Mat4x2f: TypeAlias = Matrix[F32, Literal[4], Literal[2]]
Mat4x3f: TypeAlias = Matrix[F32, Literal[4], Literal[3]]
Mat4x4f: TypeAlias = Matrix[F32, Literal[4], Literal[4]]

Mat1x1h: TypeAlias = Matrix[F16, Literal[1], Literal[1]]
Mat1x2h: TypeAlias = Matrix[F16, Literal[1], Literal[2]]
Mat1x3h: TypeAlias = Matrix[F16, Literal[1], Literal[3]]
Mat1x4h: TypeAlias = Matrix[F16, Literal[1], Literal[4]]
Mat2x1h: TypeAlias = Matrix[F16, Literal[2], Literal[1]]
Mat2x2h: TypeAlias = Matrix[F16, Literal[2], Literal[2]]
Mat2x3h: TypeAlias = Matrix[F16, Literal[2], Literal[3]]
Mat2x4h: TypeAlias = Matrix[F16, Literal[2], Literal[4]]
Mat3x1h: TypeAlias = Matrix[F16, Literal[3], Literal[1]]
Mat3x2h: TypeAlias = Matrix[F16, Literal[3], Literal[2]]
Mat3x3h: TypeAlias = Matrix[F16, Literal[3], Literal[3]]
Mat3x4h: TypeAlias = Matrix[F16, Literal[3], Literal[4]]
Mat4x1h: TypeAlias = Matrix[F16, Literal[4], Literal[1]]
Mat4x2h: TypeAlias = Matrix[F16, Literal[4], Literal[2]]
Mat4x3h: TypeAlias = Matrix[F16, Literal[4], Literal[3]]
Mat4x4h: TypeAlias = Matrix[F16, Literal[4], Literal[4]]

Mat1x1d: TypeAlias = Matrix[F64, Literal[1], Literal[1]]
Mat1x2d: TypeAlias = Matrix[F64, Literal[1], Literal[2]]
Mat1x3d: TypeAlias = Matrix[F64, Literal[1], Literal[3]]
Mat1x4d: TypeAlias = Matrix[F64, Literal[1], Literal[4]]
Mat2x1d: TypeAlias = Matrix[F64, Literal[2], Literal[1]]
Mat2x2d: TypeAlias = Matrix[F64, Literal[2], Literal[2]]
Mat2x3d: TypeAlias = Matrix[F64, Literal[2], Literal[3]]
Mat2x4d: TypeAlias = Matrix[F64, Literal[2], Literal[4]]
Mat3x1d: TypeAlias = Matrix[F64, Literal[3], Literal[1]]
Mat3x2d: TypeAlias = Matrix[F64, Literal[3], Literal[2]]
Mat3x3d: TypeAlias = Matrix[F64, Literal[3], Literal[3]]
Mat3x4d: TypeAlias = Matrix[F64, Literal[3], Literal[4]]
Mat4x1d: TypeAlias = Matrix[F64, Literal[4], Literal[1]]
Mat4x2d: TypeAlias = Matrix[F64, Literal[4], Literal[2]]
Mat4x3d: TypeAlias = Matrix[F64, Literal[4], Literal[3]]
Mat4x4d: TypeAlias = Matrix[F64, Literal[4], Literal[4]]

Mat1x1u: TypeAlias = Matrix[U32, Literal[1], Literal[1]]
Mat1x2u: TypeAlias = Matrix[U32, Literal[1], Literal[2]]
Mat1x3u: TypeAlias = Matrix[U32, Literal[1], Literal[3]]
Mat1x4u: TypeAlias = Matrix[U32, Literal[1], Literal[4]]
Mat2x1u: TypeAlias = Matrix[U32, Literal[2], Literal[1]]
Mat2x2u: TypeAlias = Matrix[U32, Literal[2], Literal[2]]
Mat2x3u: TypeAlias = Matrix[U32, Literal[2], Literal[3]]
Mat2x4u: TypeAlias = Matrix[U32, Literal[2], Literal[4]]
Mat3x1u: TypeAlias = Matrix[U32, Literal[3], Literal[1]]
Mat3x2u: TypeAlias = Matrix[U32, Literal[3], Literal[2]]
Mat3x3u: TypeAlias = Matrix[U32, Literal[3], Literal[3]]
Mat3x4u: TypeAlias = Matrix[U32, Literal[3], Literal[4]]
Mat4x1u: TypeAlias = Matrix[U32, Literal[4], Literal[1]]
Mat4x2u: TypeAlias = Matrix[U32, Literal[4], Literal[2]]
Mat4x3u: TypeAlias = Matrix[U32, Literal[4], Literal[3]]
Mat4x4u: TypeAlias = Matrix[U32, Literal[4], Literal[4]]

Mat1x1i: TypeAlias = Matrix[I32, Literal[1], Literal[1]]
Mat1x2i: TypeAlias = Matrix[I32, Literal[1], Literal[2]]
Mat1x3i: TypeAlias = Matrix[I32, Literal[1], Literal[3]]
Mat1x4i: TypeAlias = Matrix[I32, Literal[1], Literal[4]]
Mat2x1i: TypeAlias = Matrix[I32, Literal[2], Literal[1]]
Mat2x2i: TypeAlias = Matrix[I32, Literal[2], Literal[2]]
Mat2x3i: TypeAlias = Matrix[I32, Literal[2], Literal[3]]
Mat2x4i: TypeAlias = Matrix[I32, Literal[2], Literal[4]]
Mat3x1i: TypeAlias = Matrix[I32, Literal[3], Literal[1]]
Mat3x2i: TypeAlias = Matrix[I32, Literal[3], Literal[2]]
Mat3x3i: TypeAlias = Matrix[I32, Literal[3], Literal[3]]
Mat3x4i: TypeAlias = Matrix[I32, Literal[3], Literal[4]]
Mat4x1i: TypeAlias = Matrix[I32, Literal[4], Literal[1]]
Mat4x2i: TypeAlias = Matrix[I32, Literal[4], Literal[2]]
Mat4x3i: TypeAlias = Matrix[I32, Literal[4], Literal[3]]
Mat4x4i: TypeAlias = Matrix[I32, Literal[4], Literal[4]]

#
# Context: global singleton instance
#


class Context:
    def __init__(self):
        super().__init__()

        self.instance = tv.vkCreateInstance(
            pCreateInfo=tv.VkInstanceCreateInfo(pApplicationInfo=None),
            pAllocator=None,
        )


def context() -> Context:
    global context_singleton
    if context_singleton is None:
        context_singleton = Context()
    return context_singleton


context_singleton: Context | None = None
