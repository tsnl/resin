import wgpu
import ctypes
import struct
from typing import Type, TypeVar, Generic, Optional, Union

T = TypeVar("T")


class BufferWrapper(Generic[T]):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        count: int,
        label: str,
        usage: int,
        element_type: Type[T],
    ):
        self.device = device
        self.count = count
        self.element_type = element_type
        self.size_in_bytes = count * ctypes.sizeof(element_type)
        self.buffer = device.create_buffer(
            label=label, size=self.size_in_bytes, usage=usage, mapped_at_creation=False
        )

    def wgpu_buffer(self) -> wgpu.GPUBuffer:
        return self.buffer

    def len(self) -> int:
        return self.count

    def copy_to_buffer(
        self, dst: "BufferWrapper", command_encoder: wgpu.GPUCommandEncoder
    ):
        command_encoder.copy_buffer_to_buffer(
            self.wgpu_buffer(), 0, dst.wgpu_buffer(), 0, dst.size_in_bytes
        )

    def copy_to_texture(
        self, dst: "TextureWrapper", command_encoder: wgpu.GPUCommandEncoder
    ):
        command_encoder.copy_buffer_to_texture(
            {"buffer": self.wgpu_buffer(), **dst.texel_copy_buffer_layout()},
            dst.texel_copy_texture_info(),
            dst.size(),
        )

    def copy_from_texture(
        self, src: "TextureWrapper", command_encoder: wgpu.GPUCommandEncoder
    ):
        command_encoder.copy_texture_to_buffer(
            src.texel_copy_texture_info(),
            {"buffer": self.wgpu_buffer(), **src.texel_copy_buffer_layout()},
            src.size(),
        )

    def clear(self, command_encoder: wgpu.GPUCommandEncoder):
        command_encoder.clear_buffer(self.wgpu_buffer(), 0, None)

    def map_sync(self, map_mode: int):
        self.buffer.map_sync(map_mode, 0, self.size_in_bytes)


class StorageBuffer(BufferWrapper[T]):
    def __init__(
        self, device: wgpu.GPUDevice, count: int, label: str, element_type: Type[T]
    ):
        super().__init__(
            device,
            count,
            label,
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
            element_type,
        )


class UniformBuffer(BufferWrapper[T]):
    def __init__(
        self, device: wgpu.GPUDevice, count: int, label: str, element_type: Type[T]
    ):
        super().__init__(
            device,
            count,
            label,
            wgpu.BufferUsage.UNIFORM
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
            element_type,
        )


class VertexBuffer(BufferWrapper[T]):
    def __init__(
        self, device: wgpu.GPUDevice, count: int, label: str, element_type: Type[T]
    ):
        super().__init__(
            device,
            count,
            label,
            wgpu.BufferUsage.VERTEX
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
            element_type,
        )


class IndexBuffer(BufferWrapper[T]):
    def __init__(
        self, device: wgpu.GPUDevice, count: int, label: str, element_type: Type[T]
    ):
        super().__init__(
            device,
            count,
            label,
            wgpu.BufferUsage.INDEX
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
            element_type,
        )


class StagingBuffer(BufferWrapper[T]):
    def __init__(
        self, device: wgpu.GPUDevice, count: int, label: str, element_type: Type[T]
    ):
        super().__init__(
            device,
            count,
            label,
            wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
            element_type,
        )

    def write(self, data: list[T] | bytes | ctypes.Array):
        # data should be a list of ctypes objects or bytes
        self.map_sync(wgpu.MapMode.WRITE)

        if isinstance(data, bytes):
            bytes_data = data
        elif isinstance(data, (list, tuple)):
            # Assuming data is list of ctypes structures
            bytes_data = b"".join(bytes(item) for item in data)
        else:
            # Assume ctypes array or similar
            bytes_data = bytes(data)

        self.buffer.write_mapped(bytes_data)
        self.buffer.unmap()


class ReadbackBuffer(BufferWrapper[T]):
    def __init__(
        self, device: wgpu.GPUDevice, count: int, label: str, element_type: Type[T]
    ):
        super().__init__(
            device,
            count,
            label,
            wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
            element_type,
        )

    def read(self) -> bytes:
        self.map_sync(wgpu.MapMode.READ)
        result = self.buffer.read_mapped(copy=True)
        self.buffer.unmap()
        return bytes(result)


class TextureWrapper:
    def __init__(self, device: wgpu.GPUDevice, texture: wgpu.GPUTexture):
        self.device = device
        self.texture = texture

    def wgpu_texture(self) -> wgpu.GPUTexture:
        return self.texture

    def size(self) -> tuple[int, int, int]:
        return (
            self.texture.width,
            self.texture.height,
            self.texture.depth_or_array_layers,
        )

    def width(self) -> int:
        return self.texture.width

    def height(self) -> int:
        return self.texture.height

    def format(self) -> str:
        return self.texture.format

    def texel_copy_buffer_layout(self) -> dict:
        bytes_per_pixel = 0
        if self.format() == wgpu.TextureFormat.rgba8unorm:
            bytes_per_pixel = 4
        elif self.format() == wgpu.TextureFormat.rgba32float:
            bytes_per_pixel = 16
        else:
            raise NotImplementedError(
                f"Format {self.format()} not supported in texel_copy_buffer_layout"
            )

        return {
            "offset": 0,
            "bytes_per_row": self.width() * bytes_per_pixel,
            "rows_per_image": self.height(),
        }

    def texel_copy_texture_info(self) -> dict:
        return {
            "texture": self.texture,
            "mip_level": 0,
            "origin": (0, 0, 0),
            "aspect": wgpu.TextureAspect.all,
        }


class Rgba8UnormTexture(TextureWrapper):
    def __init__(self, device: wgpu.GPUDevice, size_wh: tuple[int, int], label: str):
        texture = device.create_texture(
            label=label,
            size=(size_wh[0], size_wh[1], 1),
            mip_level_count=1,
            sample_count=1,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.COPY_DST
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        super().__init__(device, texture)


class Rgba32FloatTexture(TextureWrapper):
    def __init__(self, device: wgpu.GPUDevice, size_wh: tuple[int, int], label: str):
        texture = device.create_texture(
            label=label,
            size=(size_wh[0], size_wh[1], 1),
            mip_level_count=1,
            sample_count=1,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.COPY_DST
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        super().__init__(device, texture)


def fp32_to_fx_u16(value: float) -> int:
    normalized = max(0.0, min(1.0, value))
    return int(round(normalized * 65535.0))


def fp32_to_fx_i16(value: float) -> int:
    val = (max(-1.0, min(1.0, value)) + 1.0) / 2.0
    return fp32_to_fx_u16(val)
