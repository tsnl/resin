__all__ = [
    "BufferWrapper",
    "IndexBuffer",
    "ReadbackBuffer",
    "Rgba8UnormTexture",
    "Rgba32FloatTexture",
    "StagingBuffer",
    "StorageBuffer",
    "TextureWrapper",
    "UniformBuffer",
    "VertexBuffer",
]

import numpy as np
import numpy.typing as npt

import wgpu

from .basic import logger


#
# BufferWrapper
#


class BufferWrapper:
    def __init__(
        self,
        *,
        device: wgpu.GPUDevice,
        count: int,
        dtype: npt.DTypeLike,
        label: str,
        usage: int,
    ):
        super().__init__()

        self.device = device
        self.dtype = np.dtype(dtype)
        self.size_in_bytes = count * self.dtype.itemsize
        self.buffer = device.create_buffer(
            label=label,
            size=self.size_in_bytes,
            usage=usage,
            mapped_at_creation=False,
        )

        LOG.debug(f"{self.dtype=}, {self.dtype.type=}")

    def wgpu_buffer(self) -> wgpu.GPUBuffer:
        return self.buffer

    def copy_to_buffer(
        self,
        *,
        dst: "BufferWrapper",
        command_encoder: wgpu.GPUCommandEncoder,
    ) -> None:
        command_encoder.copy_buffer_to_buffer(
            self.wgpu_buffer(), 0, dst.wgpu_buffer(), 0, dst.size_in_bytes
        )

    def copy_to_texture(
        self,
        *,
        dst: "TextureWrapper",
        command_encoder: wgpu.GPUCommandEncoder,
    ) -> None:
        command_encoder.copy_buffer_to_texture(
            {"buffer": self.wgpu_buffer(), **dst.texel_copy_buffer_layout()},
            dst.texel_copy_texture_info(),
            dst.size(),
        )

    def copy_from_texture(
        self, src: "TextureWrapper", command_encoder: wgpu.GPUCommandEncoder
    ) -> None:
        command_encoder.copy_texture_to_buffer(
            src.texel_copy_texture_info(),
            {"buffer": self.wgpu_buffer(), **src.texel_copy_buffer_layout()},
            src.size(),
        )

    def clear(self, *, command_encoder: wgpu.GPUCommandEncoder) -> None:
        command_encoder.clear_buffer(self.wgpu_buffer(), 0, None)

    def map_sync(self, *, map_mode: int) -> None:
        self.buffer.map_sync(map_mode, 0, self.size_in_bytes)


class StorageBuffer(BufferWrapper):
    def __init__(
        self,
        *,
        device: wgpu.GPUDevice,
        count: int,
        dtype: npt.DTypeLike,
        label: str,
    ):
        super().__init__(
            device=device,
            count=count,
            dtype=dtype,
            label=label,
            usage=(
                wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            ),
        )


class UniformBuffer(BufferWrapper):
    def __init__(
        self,
        *,
        device: wgpu.GPUDevice,
        count: int,
        dtype: npt.DTypeLike,
        label: str,
    ):
        super().__init__(
            device=device,
            count=count,
            dtype=dtype,
            label=label,
            usage=(
                wgpu.BufferUsage.UNIFORM
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            ),
        )


class VertexBuffer(BufferWrapper):
    def __init__(
        self, device: wgpu.GPUDevice, count: int, dtype: npt.DTypeLike, label: str
    ):
        super().__init__(
            device=device,
            count=count,
            dtype=dtype,
            label=label,
            usage=(
                wgpu.BufferUsage.VERTEX
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            ),
        )


class IndexBuffer(BufferWrapper):
    def __init__(
        self, device: wgpu.GPUDevice, count: int, dtype: npt.DTypeLike, label: str
    ):
        super().__init__(
            device=device,
            count=count,
            dtype=dtype,
            label=label,
            usage=(
                wgpu.BufferUsage.INDEX
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            ),
        )


class StagingBuffer(BufferWrapper):
    def __init__(
        self, device: wgpu.GPUDevice, count: int, dtype: npt.DTypeLike, label: str
    ):
        super().__init__(
            device=device,
            count=count,
            dtype=dtype,
            label=label,
            usage=(wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC),
        )

    def write(self, *, data: npt.ArrayLike) -> None:
        self.map_sync(map_mode=wgpu.MapMode.WRITE)
        self.buffer.write_mapped(data)
        self.buffer.unmap()


class ReadbackBuffer(BufferWrapper):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        count: int,
        dtype: npt.DTypeLike,
        label: str,
    ):
        super().__init__(
            device=device,
            count=count,
            dtype=dtype,
            label=label,
            usage=(wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST),
        )

    def read(self) -> npt.NDArray:
        self.map_sync(map_mode=wgpu.MapMode.READ)
        result = np.asarray(self.buffer.read_mapped(copy=True)).astype(self.dtype.type)
        self.buffer.unmap()
        return result


#
# TextureWrapper
#


class TextureWrapper:
    def __init__(self, *, device: wgpu.GPUDevice, texture: wgpu.GPUTexture):
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
    def __init__(self, *, device: wgpu.GPUDevice, size_wh: tuple[int, int], label: str):
        texture = device.create_texture(
            label=label,
            size=(size_wh[0], size_wh[1], 1),
            mip_level_count=1,
            sample_count=1,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(
                wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.RENDER_ATTACHMENT
            ),
        )
        super().__init__(device=device, texture=texture)


class Rgba32FloatTexture(TextureWrapper):
    def __init__(self, *, device: wgpu.GPUDevice, size_wh: tuple[int, int], label: str):
        texture = device.create_texture(
            label=label,
            size=(size_wh[0], size_wh[1], 1),
            mip_level_count=1,
            sample_count=1,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba32float,
            usage=(
                wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.RENDER_ATTACHMENT
            ),
        )
        super().__init__(device=device, texture=texture)


#
# Logger
#

LOG = logger(__name__)
