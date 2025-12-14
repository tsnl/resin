__all__ = [
    "Renderer",
    "RendererAtlas",
    "RendererContext",
    "RendererFont",
    "RendererImage",
    "RendererQuadArray",
]

from collections import OrderedDict
from typing import Literal, TypeAlias

import numpy as np
import freetype as ft

from .basic import BaseResource, round_up_to_po2
from .bundled_data import BUNDLED_DATA_PATH
from .excepts import LogicError
from .gpu import (
    GpuBuffer,
    GpuBufferMeta,
    GpuCommandEncoder,
    GpuContext,
    GpuDescriptorSet,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuFence,
    GpuImage,
    GpuImageMeta,
    GpuPipeline,
    GpuPipelineLayout,
    GpuSampler,
    GpuSemaphore,
    GpuShader,
)
from . import typed_uharfbuzz as hb


#
# Renderer API:
#


class RendererContext(BaseResource):
    def __init__(
        self,
        *,
        gpu_context: GpuContext,
        parent: BaseResource | None = None,
    ):
        super().__init__(parent=parent)
        self.gpu_context = gpu_context


class Renderer(BaseResource):
    _context: RendererContext
    _gpu_device: GpuDevice
    _atlases: dict["RendererAtlasChannels", "RendererAtlas"]
    _atlas_descriptor_set_layout: GpuDescriptorSetLayout
    _atlas_descriptor_set: GpuDescriptorSet
    _default_white_image: "RendererImage"
    _quad_renderer: "QuadRenderer"
    _font_engine: "FontEngine"
    _scale: float

    def __init__(
        self,
        *,
        context: RendererContext,
        gpu_device: GpuDevice,
        scale: float = 1.0,
    ):
        super().__init__(parent=context)

        self._context = context
        self._gpu_device = gpu_device
        self._scale = scale

        self._atlases = {
            1: RendererAtlas(renderer=self, channels=1),
            4: RendererAtlas(renderer=self, channels=4),
        }

        self._atlas_descriptor_set_layout = GpuDescriptorSetLayout(
            device=self._gpu_device,
            bindings=OrderedDict(
                {
                    "atlasMono": GpuDescriptorSetLayoutBinding(
                        type="sampled-image", count=self._atlases[1]._max_pages
                    ),
                    "atlasRgba": GpuDescriptorSetLayoutBinding(
                        type="sampled-image", count=self._atlases[4]._max_pages
                    ),
                    "rectsMono": GpuDescriptorSetLayoutBinding(type="storage-buffer"),
                    "rectsRgba": GpuDescriptorSetLayoutBinding(type="storage-buffer"),
                    "sampler": GpuDescriptorSetLayoutBinding(type="sampler"),
                }.items()
            ),
        )
        self._sampler = GpuSampler(
            device=self._gpu_device,
            mag_filter="nearest",
            min_filter="nearest",
        )
        self._atlas_descriptor_set = GpuDescriptorSet(
            device=self._gpu_device,
            layout=self._atlas_descriptor_set_layout,
            bindings={
                "atlasMono": self._atlases[1]._page_gpu_image_list,
                "atlasRgba": self._atlases[4]._page_gpu_image_list,
                "rectsMono": self._atlases[1]._uv_rect_array_device_buf,
                "rectsRgba": self._atlases[4]._uv_rect_array_device_buf,
                "sampler": self._sampler,
            },
        )

        self._default_white_image = RendererImage(
            renderer=self,
            data=np.ones((1, 1, 4), dtype=np.float32),
        )

        self._quad_renderer = QuadRenderer(renderer=self, gpu_device=gpu_device)

        self._font_engine = FontEngine(renderer=self)

    @property
    def scale(self) -> float:
        return self._scale

    def _on_dispose(self) -> None:
        self._quad_renderer.dispose()
        self._atlas_descriptor_set.dispose()
        self._atlas_descriptor_set_layout.dispose()

        # Dispose atlases:
        for _, atlas in self._atlases.items():
            atlas.dispose()

    def draw(
        self,
        *,
        canvas: "RendererCanvas",
        target: GpuImage,
        wait_semaphores: list[GpuSemaphore],
        done_semaphores: list[GpuSemaphore],
        fence: GpuFence,
    ):
        """
        Render the given array of quads to the given target `GpuImage`.

        The quads array should have dtype RENDERER_QUAD_DTYPE.
        """

        for atlas in self._atlases.values():
            atlas.flush()

        self._quad_renderer.draw(
            quads=canvas.as_quad_array(),
            target=target,
            wait_semaphores=wait_semaphores,
            done_semaphores=done_semaphores,
            fence=fence,
        )

    def atlas(self, channels: "RendererAtlasChannels") -> "RendererAtlas":
        return self._atlases[channels]


#
# RendererAtlas, RendererImage
#


class RendererImage:
    _atlas: "RendererAtlas"
    _index: int
    _allocation_uv_xywh: tuple[float, float, float, float] | None
    _allocation_px_xywh: tuple[int, int, int, int] | None

    def __init__(self, *, renderer: "Renderer", data: np.ndarray):
        assert data.ndim in (2, 3)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        self._atlas = renderer._atlases[data.shape[2]]
        self._data = data.copy()
        self._data.flags.writeable = False
        self._index = -1
        self._allocation_uv_xywh = None
        self._allocation_px_xywh = None

        self._atlas.insert(self)

    @property
    def image_id(self) -> int:
        if self._index < 0:
            raise LogicError("Image not yet allocated in atlas")
        return self._index

    @property
    def px_width(self) -> int:
        return self._data.shape[1]

    @property
    def px_height(self) -> int:
        return self._data.shape[0]

    @property
    def allocation_uv_xywh(self) -> tuple[float, float, float, float]:
        assert self._allocation_uv_xywh is not None
        return self._allocation_uv_xywh

    @property
    def allocation_px_xywh(self) -> tuple[int, int, int, int]:
        assert self._allocation_px_xywh is not None
        return self._allocation_px_xywh

    @property
    def page_index(self) -> int:
        assert self._allocation_uv_xywh is not None
        return int(self._allocation_uv_xywh[1])


RendererAtlasChannels: TypeAlias = Literal[1, 4]


class RendererAtlas(BaseResource):
    _renderer: Renderer
    _gpu_device: GpuDevice
    _channels: RendererAtlasChannels
    _page_size: int
    _max_pages: int
    _max_rects: int
    _images: list[RendererImage]
    _unallocated_images: list[RendererImage]
    _page_cursor_array: np.ndarray
    _page_count: int
    _page_pixel_data: np.ndarray
    _page_gpu_image_list: list[GpuImage]
    _uv_rect_array_device_buf: GpuBuffer

    def __init__(
        self,
        *,
        renderer: Renderer,
        channels: RendererAtlasChannels,
        max_pages: int = 4,
        page_size: int = 4096,
        max_rects: int = 1 << 20,
    ):
        super().__init__(parent=renderer)
        self._renderer = renderer
        self._gpu_device = renderer._gpu_device

        self._channels = channels
        self._page_size = page_size
        self._max_pages = max_pages
        self._max_rects = max_rects

        self._images: list[RendererImage] = []
        self._unallocated_images: list[RendererImage] = []

        self._page_cursor_array = np.zeros(
            (max_pages,),
            dtype=np.dtype(
                [
                    ("insert_x", np.float32),
                    ("insert_y", np.float32),
                    ("row_height", np.float32),
                ]
            ),
        )
        self._page_count = 0
        self._page_pixel_data = np.empty(
            (max_pages, page_size, page_size, channels),
            dtype=np.float32,
        )
        self._page_gpu_image_list = self._create_page_gpu_images()

        self._uv_rect_array_device_buf = GpuBuffer(
            device=self._gpu_device,
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=max_rects,
                element_dtype=UV_RECT_DTYPE,
            ),
        )

    def _create_page_gpu_images(self) -> list[GpuImage]:
        return [
            GpuImage(
                device=self._renderer._gpu_device,
                usages=["texture-binding", "transfer-dst"],
                meta=GpuImageMeta(
                    shape=(self._page_size, self._page_size, self._channels),
                    dtype=np.float32,
                    color_space="linear",
                ),
            )
            for _ in range(self._max_pages)
        ]

    def insert(self, image: RendererImage):
        assert image._data.shape[2] == self._channels
        image._index = len(self._images)
        self._images.append(image)
        self._unallocated_images.append(image)

    def flush(self):
        if not self._unallocated_images:
            return

        self._unallocated_images.sort(
            key=lambda img: img._data.shape[0] * img._data.shape[1], reverse=True
        )

        failed_to_alloc = False
        dirty_images = []

        for img in self._unallocated_images:
            if not self._alloc_image(img):
                failed_to_alloc = True
                break
            dirty_images.append(img)

        if failed_to_alloc:
            self._compact()
            dirty_images = self._images  # All images dirty

        self._upload_pages(dirty_images)
        self._upload_rects()
        self._unallocated_images.clear()

    def _alloc_image(self, img: RendererImage) -> bool:
        h_px, w_px = img._data.shape[0], img._data.shape[1]
        w = w_px / self._page_size
        h = h_px / self._page_size

        # Iterate over pages in reverse order
        for page_index in reversed(range(self._page_count)):
            page = self._page_cursor_array[page_index]

            # Try insert on current row
            x, y = page["insert_x"], page["insert_y"]
            if x + w <= 1.0 and y + h <= 1.0:
                page["insert_x"] += w
                page["row_height"] = max(page["row_height"], h)
                self._set_image_rect(img, page_index, x, y, w, h)
                return True

            # Try insert on new row
            x = 0
            y += page["row_height"]
            if x + w <= 1.0 and y + h <= 1.0:
                page["insert_x"] = w
                page["insert_y"] = y
                page["row_height"] = h
                self._set_image_rect(img, page_index, x, y, w, h)
                return True

        # Try add page
        if self._page_count < self._max_pages:
            page_index = self._page_count
            self._page_count += 1
            page = self._page_cursor_array[page_index]
            # New page starts at 0,0
            page["insert_x"] = w
            page["insert_y"] = 0.0
            page["row_height"] = h
            self._set_image_rect(img, page_index, 0.0, 0.0, w, h)
            return True

        return False

    def _set_image_rect(
        self,
        img: RendererImage,
        page_index: int,
        x: float,
        y: float,
        w: float,
        h: float,
    ):
        img._allocation_uv_xywh = (x, float(page_index) + y, w, h)

        x_px = int(x * self._page_size)
        y_px = int(y * self._page_size)
        w_px = int(w * self._page_size)
        h_px = int(h * self._page_size)
        img._allocation_px_xywh = (x_px, y_px, w_px, h_px)

        # Update CPU pixel data
        self._page_pixel_data[page_index, y_px : y_px + h_px, x_px : x_px + w_px] = (
            img._data
        )

    def _compact(self):
        # Clear pages
        self._page_cursor_array.fill(0)
        self._page_count = 0
        self._page_pixel_data.fill(0)

        # Sort ALL images by size
        all_images = sorted(
            self._images,
            key=lambda img: img._data.shape[0] * img._data.shape[1],
            reverse=True,
        )

        for img in all_images:
            if not self._alloc_image(img):
                raise MemoryError("Out of atlas memory during compaction")

    def _upload_pages(self, dirty_images: list[RendererImage]):
        if not dirty_images:
            return

        staging_buffer = GpuBuffer(
            device=self._gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=self._page_size * self._page_size * self._channels,
                element_dtype=np.float32,
            ),
        )

        command_encoder = GpuCommandEncoder(
            device=self._gpu_device,
            queue_type="transfer",
        )

        # Group by page
        pages: dict[int, list[RendererImage]] = {}
        for img in dirty_images:
            pages.setdefault(img.page_index, []).append(img)

        for page_index, images in pages.items():
            offset = 0
            for img in images:
                staging_buffer.memory.write(data=img._data, offset=offset)

                command_encoder.transition_image_layout(
                    image=self._page_gpu_image_list[page_index],
                    layout="transfer-dst-optimal",
                )
                command_encoder.copy_buffer_to_image(
                    src=staging_buffer,
                    dst=self._page_gpu_image_list[page_index],
                    buffer_offset=offset,
                    image_offset=(
                        img.allocation_px_xywh[0],
                        img.allocation_px_xywh[1],
                        0,
                    ),
                    image_extent=(
                        img.allocation_px_xywh[2],
                        img.allocation_px_xywh[3],
                        1,
                    ),
                )
                offset += img._data.nbytes

            # Must submit per page because we reuse the staging buffer
            command_encoder.submit().wait()
            command_encoder = GpuCommandEncoder(
                device=self._gpu_device,
                queue_type="transfer",
            )

        staging_buffer.dispose()

    def _upload_rects(self):
        rects = np.zeros((len(self._images),), dtype=UV_RECT_DTYPE)
        for i, img in enumerate(self._images):
            if img._allocation_uv_xywh is not None:
                rects[i] = img._allocation_uv_xywh

        staging_buffer = GpuBuffer(
            device=self._gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta.from_array(rects),
        )
        staging_buffer.memory.write(data=rects)

        command_encoder = GpuCommandEncoder(
            device=self._gpu_device,
            queue_type="transfer",
        )
        command_encoder.copy_buffer_to_buffer(
            src=staging_buffer,
            dst=self._uv_rect_array_device_buf,
            size=rects.nbytes,
        )
        command_encoder.submit().wait()
        staging_buffer.dispose()


UV_RECT_DTYPE = np.dtype(
    [
        ("x", np.float32),  # 0 <= x < 1
        ("y", np.float32),  # int(y) is the page_index, fmod(y, 1) is the UV y
        ("w", np.float32),  # 0 < x_uv + w <= 1
        ("h", np.float32),  # 0 < y_uv + h <= 1
    ]
)


#
# QuadRenderer
#


class QuadRenderer(BaseResource):
    """Encapsulates the quads rendering pipeline with all GPU resources and logic."""

    COMMON_UNIFORM_DTYPE = np.dtype(
        [
            ("framebuffer_size_px", np.uint32, (2,)),
            ("total_quad_count", np.float32),
            ("atlas_size_px", np.uint32),
        ]
    )
    BATCH_UNIFORM_DTYPE = np.dtype(
        [
            ("instance_count", np.uint32),
            ("is_opaque", np.uint32),
            ("is_mono", np.uint32),
            ("_rsv1", np.uint32),
        ]
    )
    QUAD_DTYPE = np.dtype(
        [
            ("dst_px", np.int32, (4, 2)),
            ("src_uv", np.float32, (4, 2)),
            ("color", np.float32, (4,)),
            ("border_color", np.float32, (4,)),
            ("border_thickness_px", np.uint32, (4,)),
            ("height", np.float32),
            ("image_id", np.uint32),
            ("_rsv0", np.uint32),
            ("_rsv1", np.uint32),
        ]
    )

    renderer: "Renderer"
    gpu_device: GpuDevice

    _vertex_shader: GpuShader
    _fragment_shader: GpuShader
    _pipeline_layout: GpuPipelineLayout
    _common_uniform_staging_buf: GpuBuffer
    _common_uniform_device_buf: GpuBuffer
    _common_uniform_descriptor_set: GpuDescriptorSet
    _cached_gpu_pipeline: GpuPipeline | None
    _cached_depth_image: GpuImage | None

    _quad_capacity: int
    _quad_array_staging_buf: GpuBuffer | None
    _quad_array_device_buf: GpuBuffer | None
    _batch_uniform_staging_buf: GpuBuffer | None
    _batch_uniform_device_buf: GpuBuffer | None
    _batch_descriptor_set: GpuDescriptorSet | None

    def __init__(
        self,
        *,
        renderer: "Renderer",
        gpu_device: GpuDevice,
    ):
        super().__init__(parent=renderer)
        self.renderer = renderer
        self.gpu_device = gpu_device

        self._vertex_shader = self._new_quads_vertex_shader()
        self._fragment_shader = self._new_quads_fragment_shader()
        self._pipeline_layout = self._new_quads_pipeline_layout()
        self._common_uniform_staging_buf = self._new_common_uniform_buffer(staging=True)
        self._common_uniform_device_buf = self._new_common_uniform_buffer(staging=False)
        self._common_uniform_descriptor_set = self._new_common_uniform_descriptor_set()
        self._cached_gpu_pipeline = None
        self._cached_depth_image = None

        self._quad_capacity = 0
        self._quad_array_staging_buf = None
        self._quad_array_device_buf = None
        self._batch_uniform_staging_buf = None
        self._batch_uniform_device_buf = None
        self._batch_descriptor_set = None

    def _on_dispose(self) -> None:
        self._vertex_shader.dispose()
        self._fragment_shader.dispose()

        self._pipeline_layout.dispose()

        self._common_uniform_descriptor_set.dispose()

        self._common_uniform_staging_buf.dispose()
        self._common_uniform_device_buf.dispose()

        if self._batch_descriptor_set is not None:
            self._batch_descriptor_set.dispose()
        if self._batch_uniform_staging_buf is not None:
            self._batch_uniform_staging_buf.dispose()
        if self._batch_uniform_device_buf is not None:
            self._batch_uniform_device_buf.dispose()
        if self._quad_array_staging_buf is not None:
            self._quad_array_staging_buf.dispose()
        if self._quad_array_device_buf is not None:
            self._quad_array_device_buf.dispose()

        if self._cached_depth_image is not None:
            self._cached_depth_image.dispose()
        if self._cached_gpu_pipeline is not None:
            self._cached_gpu_pipeline.dispose()

    def _ensure_batch_capacity(self, capacity: int):
        if capacity <= self._quad_capacity:
            return

        # Dispose old resources
        if self._batch_descriptor_set is not None:
            self._batch_descriptor_set.dispose()
        if self._batch_uniform_staging_buf is not None:
            self._batch_uniform_staging_buf.dispose()
        if self._batch_uniform_device_buf is not None:
            self._batch_uniform_device_buf.dispose()
        if self._quad_array_staging_buf is not None:
            self._quad_array_staging_buf.dispose()
        if self._quad_array_device_buf is not None:
            self._quad_array_device_buf.dispose()

        # Create new resources
        new_capacity = max(8, round_up_to_po2(capacity))
        self._quad_capacity = new_capacity

        self._batch_uniform_device_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["uniform", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=1,
                element_dtype=QuadRenderer.BATCH_UNIFORM_DTYPE,
            ),
        )
        self._batch_uniform_staging_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=1,
                element_dtype=QuadRenderer.BATCH_UNIFORM_DTYPE,
            ),
        )
        self._quad_array_device_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=new_capacity,
                element_dtype=QuadRenderer.QUAD_DTYPE,
            ),
        )
        self._quad_array_staging_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=new_capacity,
                element_dtype=QuadRenderer.QUAD_DTYPE,
            ),
        )
        self._batch_descriptor_set = GpuDescriptorSet(
            device=self.gpu_device,
            layout=self._pipeline_layout.descriptor_set_layouts[2],
            bindings={
                "quads": self._quad_array_device_buf,
                "batchUniform": self._batch_uniform_device_buf,
            },
        )

    def draw(
        self,
        *,
        quads: "RendererQuadArray",
        target: GpuImage,
        wait_semaphores: list[GpuSemaphore],
        done_semaphores: list[GpuSemaphore],
        fence: GpuFence,
    ):
        """
        Render the given array of quads to the given target `GpuImage`.

        The quads array should have dtype RENDERER_QUAD_DTYPE.
        """

        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="graphics",
        )

        command_encoder.transition_image_layout(
            image=target,
            layout="color-attachment-optimal",
        )

        assert "color-attachment" in target.usages

        gpu_pipeline = self._get_gpu_pipeline(target=target)
        depth_image = self._get_depth_image(width=target.width, height=target.height)

        self._ensure_batch_capacity(len(quads))

        if len(quads) > 0:
            # Quads:
            assert self._quad_array_staging_buf is not None
            assert self._quad_array_device_buf is not None
            self._quad_array_staging_buf.memory.write(data=quads)
            command_encoder.copy_buffer_to_buffer(
                src=self._quad_array_staging_buf,
                dst=self._quad_array_device_buf,
                size=len(quads) * QuadRenderer.QUAD_DTYPE.itemsize,
            )

            # Batch uniform:
            assert self._batch_uniform_staging_buf is not None
            assert self._batch_uniform_device_buf is not None
            uniform_array = np.empty((1,), dtype=QuadRenderer.BATCH_UNIFORM_DTYPE)
            uniform_array[0]["instance_count"] = int(len(quads))
            uniform_array[0]["is_opaque"] = 1  # TODO: support transparency
            uniform_array[0]["is_mono"] = 0
            self._batch_uniform_staging_buf.memory.write(data=uniform_array)
            command_encoder.copy_buffer_to_buffer(
                src=self._batch_uniform_staging_buf,
                dst=self._batch_uniform_device_buf,
                size=QuadRenderer.BATCH_UNIFORM_DTYPE.itemsize,
            )

        self._write_common_uniform(
            encoder=command_encoder,
            target=target,
            total_quad_count=len(quads),
        )

        self._draw(
            encoder=command_encoder,
            pipeline=gpu_pipeline,
            depth_image=depth_image,
            target=target,
            instance_count=len(quads),
        )

        command_encoder.transition_image_layout(
            image=target,
            layout=(
                "present-src"
                if self.gpu_device.present_support_enabled
                else "transfer-src-optimal"
            ),
        )

        command_encoder.submit(
            fence=fence,
            wait_semaphores=wait_semaphores,
            signal_semaphores=done_semaphores,
        )

    def _new_quads_vertex_shader(self) -> GpuShader:
        return GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "r2d.vert.spv",
            stage="vertex",
        )

    def _new_quads_fragment_shader(self) -> GpuShader:
        return GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "r2d.frag.spv",
            stage="fragment",
        )

    def _new_quads_pipeline_layout(self) -> GpuPipelineLayout:
        return GpuPipelineLayout(
            device=self.gpu_device,
            descriptor_set_layouts=[
                GpuDescriptorSetLayout(
                    device=self.gpu_device,
                    bindings=OrderedDict(
                        {
                            "commonUniform": GpuDescriptorSetLayoutBinding(
                                type="uniform-buffer",
                                stages=["vertex", "fragment"],
                            ),
                        }.items()
                    ),
                ),
                self.renderer._atlas_descriptor_set_layout,
                GpuDescriptorSetLayout(
                    device=self.gpu_device,
                    bindings=OrderedDict(
                        {
                            "quads": GpuDescriptorSetLayoutBinding(
                                type="storage-buffer",
                                stages=["vertex", "fragment"],
                            ),
                            "batchUniform": GpuDescriptorSetLayoutBinding(
                                type="uniform-buffer",
                                stages=["vertex", "fragment"],
                            ),
                        }.items()
                    ),
                ),
            ],
        )

    def _new_common_uniform_buffer(self, *, staging: bool) -> GpuBuffer:
        return GpuBuffer(
            device=self.gpu_device,
            usages=["uniform", "copy-dst"] if not staging else ["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=1,
                element_dtype=QuadRenderer.COMMON_UNIFORM_DTYPE,
            ),
        )

    def _new_common_uniform_descriptor_set(self) -> GpuDescriptorSet:
        return GpuDescriptorSet(
            device=self.gpu_device,
            layout=self._pipeline_layout.descriptor_set_layouts[0],
            bindings={"commonUniform": self._common_uniform_device_buf},
        )

    def _get_gpu_pipeline(self, target: GpuImage) -> GpuPipeline:
        if cached_pipeline := self._get_cached_gpu_pipeline(target=target):
            return cached_pipeline
        gpu_pipeline = self._new_gpu_pipeline(target=target)
        self._cached_gpu_pipeline = gpu_pipeline
        return gpu_pipeline

    def _get_cached_gpu_pipeline(self, target: GpuImage) -> GpuPipeline | None:
        if self._cached_gpu_pipeline is None:
            return None
        if self._cached_gpu_pipeline.vk_color_format != target.vk_format:
            return None
        if self._cached_gpu_pipeline.viewport_width != target.width:
            return None
        if self._cached_gpu_pipeline.viewport_height != target.height:
            return None
        return self._cached_gpu_pipeline

    def _new_gpu_pipeline(self, target: GpuImage) -> GpuPipeline:
        return GpuPipeline(
            device=self.gpu_device,
            vertex_shader=self._vertex_shader,
            fragment_shader=self._fragment_shader,
            vk_color_format=target.vk_format,
            enable_depth_test=False,
            enable_alpha_blending=True,
            viewport_width=target.width,
            viewport_height=target.height,
            layout=self._pipeline_layout,
        )

    def _get_depth_image(self, *, width: int, height: int) -> GpuImage:
        if (
            self._cached_depth_image is not None
            and self._cached_depth_image.width == width
            and self._cached_depth_image.height == height
        ):
            return self._cached_depth_image

        depth_image = self._new_depth_image(width=width, height=height)
        self._cached_depth_image = depth_image

        return depth_image

    def _new_depth_image(self, *, width: int, height: int) -> GpuImage:
        return GpuImage(
            device=self.gpu_device,
            usages=["depth-attachment"],
            meta=GpuImageMeta(shape=(height, width, 1), dtype=np.float32),
        )

    def _write_common_uniform(
        self,
        *,
        encoder: GpuCommandEncoder,
        target: GpuImage,
        total_quad_count: int,
    ):
        uniform_data = np.zeros((1,), dtype=QuadRenderer.COMMON_UNIFORM_DTYPE)
        uniform_data[0]["framebuffer_size_px"] = [target.width, target.height]
        uniform_data[0]["total_quad_count"] = float(total_quad_count)
        uniform_data[0]["atlas_size_px"] = self.renderer._atlases[4]._page_size
        self._common_uniform_staging_buf.memory.write(data=uniform_data)
        encoder.copy_buffer_to_buffer(
            src=self._common_uniform_staging_buf,
            dst=self._common_uniform_device_buf,
            size=QuadRenderer.COMMON_UNIFORM_DTYPE.itemsize,
        )

    def _draw(
        self,
        *,
        encoder: GpuCommandEncoder,
        pipeline: GpuPipeline,
        depth_image: GpuImage,
        target: GpuImage,
        instance_count: int,
    ):
        with encoder.render(
            color_attachment=target,
            depth_attachment=depth_image,
            clear_color="black",
        ) as render_pass:
            render_pass.bind_pipeline(pipeline=pipeline)
            render_pass.bind_descriptor_set(
                set_index=0,
                descriptor_set=self._common_uniform_descriptor_set,
            )
            render_pass.bind_descriptor_set(
                set_index=1,
                descriptor_set=self.renderer._atlas_descriptor_set,
            )

            if instance_count > 0:
                assert self._batch_descriptor_set is not None
                render_pass.bind_descriptor_set(
                    set_index=2,
                    descriptor_set=self._batch_descriptor_set,
                )
                render_pass.draw(
                    vertex_count=6,
                    instance_count=instance_count,
                    first_vertex=0,
                    first_instance=0,
                )


assert QuadRenderer.COMMON_UNIFORM_DTYPE.itemsize == 4 * 4
assert QuadRenderer.BATCH_UNIFORM_DTYPE.itemsize == 4 * 4
assert QuadRenderer.QUAD_DTYPE.itemsize == 32 * 4


class RendererQuadArray(np.ndarray):
    def __new__(cls, shape: tuple[int, ...] | int) -> "RendererQuadArray":
        return np.zeros(shape, dtype=QuadRenderer.QUAD_DTYPE).view(cls)


#
# RendererFont, RendererFontAtlas:
#


RendererFont: TypeAlias = Literal["sans-serif", "serif"]


class FontEngine(BaseResource):
    _renderer: Renderer
    _all_fonts: list[RendererFont]
    _hb_font_map: dict[RendererFont, hb.Font]
    _ft_image_cache: dict[
        tuple[RendererFont, int, int, int],
        tuple["RendererImage | None", int, int],
    ]
    _ft_weight_axis_index: dict[RendererFont, int]

    def __init__(
        self,
        renderer: Renderer,
    ) -> None:
        super().__init__(parent=renderer)

        self._renderer = renderer

        self._all_fonts = ["sans-serif", "serif"]
        self._hb_font_map = {
            font: FontEngine._load_harfbuzz_font(font)  #
            for font in self._all_fonts
        }
        self._ft_face_map = {
            font: FontEngine._load_freetype_font(font)  #
            for font in self._all_fonts
        }
        self._ft_image_cache = {}
        self._ft_weight_axis_index = {}

        for font in self._all_fonts:
            face = self._ft_face_map[font]
            try:
                info = face.get_variation_info()
                for i, axis in enumerate(info.axes):
                    if axis.tag == "wght":
                        self._ft_weight_axis_index[font] = i
                        break
            except Exception:
                pass

    @staticmethod
    def _load_harfbuzz_font(font: RendererFont) -> hb.Font:
        file_path = {
            "sans-serif": BUNDLED_DATA_PATH / "data/font-Inter_4_1/InterVariable.ttf",
            "serif": BUNDLED_DATA_PATH / "data/font-Lora/Lora-VariableFont_wght.ttf",
        }[font]

        with open(file_path, "rb") as f:
            hb_blob = f.read()

        hb_face = hb.Face(hb_blob)
        hb_font = hb.Font(hb_face)
        return hb_font

    @staticmethod
    def _load_freetype_font(font: RendererFont) -> ft.Face:
        file_path = {
            "sans-serif": BUNDLED_DATA_PATH / "data/font-Inter_4_1/InterVariable.ttf",
            "serif": BUNDLED_DATA_PATH / "data/font-Lora/Lora-VariableFont_wght.ttf",
        }[font]

        ft_face = ft.Face(str(file_path))
        return ft_face

    def _set_freetype_weight(self, font: RendererFont, weight: int) -> None:
        if font not in self._ft_weight_axis_index:
            return

        face = self._ft_face_map[font]
        axis_idx = self._ft_weight_axis_index[font]
        coords = list(face.get_var_design_coords())
        coords[axis_idx] = float(weight)
        face.set_var_design_coords(coords)

    def _shape_text(
        self,
        font: RendererFont,
        text: str,
        font_size_px: int,
        font_weight: int,
    ) -> tuple[list[hb.GlyphInfo], list[hb.GlyphPosition]]:
        hb_font = self._hb_font_map[font]
        # Use renderer's scale factor for HiDPI support
        renderer_scale = self._renderer.scale
        effective_size_px = int(font_size_px * renderer_scale)
        scale = effective_size_px * 64  # HarfBuzz uses 26.6 fixed point
        hb_font.scale = (scale, scale)
        hb_font.set_variations({"wght": font_weight})

        hb_buffer = hb.Buffer()
        hb_buffer.add_str(text)
        hb_buffer.guess_segment_properties()

        hb.shape(hb_font, hb_buffer)

        return hb_buffer.glyph_infos, hb_buffer.glyph_positions

    def _get_glyph_image(
        self,
        font: RendererFont,
        glyph_index: int,
        font_size_px: int,
        font_weight: int,
    ) -> tuple["RendererImage | None", int, int]:
        # Use renderer's scale factor for HiDPI support
        renderer_scale = self._renderer.scale
        effective_size_px = int(font_size_px * renderer_scale)
        image_cache_key = (font, glyph_index, effective_size_px, font_weight)

        if image_cache_key in self._ft_image_cache:
            return self._ft_image_cache[image_cache_key]

        ft_face = self._ft_face_map[font]
        ft_face.set_pixel_sizes(0, effective_size_px)
        self._set_freetype_weight(font, font_weight)
        ft_face.load_glyph(glyph_index, ft.FT_LOAD_RENDER | ft.FT_LOAD_TARGET_NORMAL)

        bitmap_left = ft_face.glyph.bitmap_left
        bitmap_top = ft_face.glyph.bitmap_top
        bitmap = ft_face.glyph.bitmap

        if not bitmap.buffer or bitmap.width == 0 or bitmap.rows == 0:
            result = (None, 0, 0)
            self._ft_image_cache[image_cache_key] = result
            return result

        h, w = bitmap.rows, bitmap.width
        pitch = bitmap.pitch

        # Load buffer as (h, pitch)
        buffer_array = np.array(bitmap.buffer, dtype=np.uint8).reshape(h, pitch)

        # Prepare RGBA data array
        data = np.empty((h, w, 4), dtype=np.float32)

        # Slice to remove padding if any
        if pitch != w:
            buffer_array = buffer_array[:, :w]

        # Grayscale glyph: use gray value for all RGB channels and alpha
        gray_norm = buffer_array / 255.0
        data[..., 0] = 1.0
        data[..., 1] = 1.0
        data[..., 2] = 1.0
        data[..., 3] = gray_norm

        # Create RendererImage
        image = RendererImage(renderer=self._renderer, data=data)
        result = (image, bitmap_left, bitmap_top)
        self._ft_image_cache[image_cache_key] = result

        # Done:
        return result

    def _add_quads_to_canvas(
        self,
        *,
        canvas: "RendererCanvas",
        text: str,
        font: RendererFont,
        font_size_px: int,
        color: tuple[float, float, float, float],
        dst_xy: tuple[int, int],
        dst_wh: tuple[int, int],
        wrap: bool,
        font_weight: int,
    ):
        if not text:
            return

        # Strategy: Accumulate pen position in 26.6 fixed-point to preserve
        # subpixel precision. Only round to integer pixels when placing glyphs.
        # This prevents accumulated rounding error.

        renderer_scale = self._renderer.scale
        effective_size_px = int(font_size_px * renderer_scale)

        ft_face = self._ft_face_map[font]
        ft_face.set_pixel_sizes(0, effective_size_px)
        self._set_freetype_weight(font, font_weight)
        metrics = ft_face.size

        # FreeType metrics are in 26.6 fixed point - keep in 26.6
        ascender_26_6 = metrics.ascender
        height_26_6 = metrics.height

        infos, positions = self._shape_text(font, text, font_size_px, font_weight)

        dst_x, dst_y = dst_xy
        dst_w, dst_h = dst_wh

        # Convert destination rect to physical 26.6 fixed-point
        dst_x_26_6 = int(dst_x * renderer_scale * 64)
        dst_y_26_6 = int(dst_y * renderer_scale * 64)
        dst_w_26_6 = int(dst_w * renderer_scale * 64)
        dst_h_26_6 = int(dst_h * renderer_scale * 64)

        # Physical pixel versions for clipping (integers)
        dst_x_phys = int(dst_x * renderer_scale)
        dst_y_phys = int(dst_y * renderer_scale)
        dst_w_phys = int(dst_w * renderer_scale)
        dst_h_phys = int(dst_h * renderer_scale)

        # Pen position in 26.6 fixed-point (physical)
        pen_x_26_6 = dst_x_26_6
        pen_y_26_6 = dst_y_26_6 + ascender_26_6
        start_x_26_6 = dst_x_26_6

        for i in range(len(infos)):
            info = infos[i]
            pos = positions[i]

            codepoint = info.codepoint
            cluster = info.cluster
            char = text[cluster] if cluster < len(text) else " "

            # Handle explicit newlines
            if char == "\n":
                pen_x_26_6 = start_x_26_6
                pen_y_26_6 += height_26_6
                continue

            # HarfBuzz positions are in 26.6 fixed point - use directly
            x_advance_26_6 = pos.x_advance
            y_advance_26_6 = pos.y_advance
            x_offset_26_6 = pos.x_offset
            y_offset_26_6 = pos.y_offset

            # Word wrapping (use 26.6 for accurate measurement)
            if wrap and not char.isspace():
                is_word_start = False
                if i == 0:
                    is_word_start = True
                else:
                    prev_cluster = infos[i - 1].cluster
                    prev_char = text[prev_cluster] if prev_cluster < len(text) else " "
                    if prev_char.isspace():
                        is_word_start = True

                if is_word_start:
                    word_width_26_6 = 0
                    for j in range(i, len(infos)):
                        c = infos[j].cluster
                        c_char = text[c] if c < len(text) else " "
                        if c_char.isspace():
                            break
                        word_width_26_6 += positions[j].x_advance

                    # Wrap if word doesn't fit and we're not at start of line
                    if (pen_x_26_6 + word_width_26_6 > dst_x_26_6 + dst_w_26_6) and (
                        pen_x_26_6 > start_x_26_6
                    ):
                        pen_x_26_6 = start_x_26_6
                        pen_y_26_6 += height_26_6

            image, bitmap_left, bitmap_top = self._get_glyph_image(
                font, codepoint, font_size_px, font_weight
            )

            if image is not None:
                # Convert pen position to physical pixels for this glyph
                # Round 26.6 to nearest integer pixel
                pen_x_phys = (pen_x_26_6 + 32) >> 6
                pen_y_phys = (pen_y_26_6 + 32) >> 6
                x_offset_phys = (x_offset_26_6 + 32) >> 6
                y_offset_phys = (y_offset_26_6 + 32) >> 6

                # bitmap_left and bitmap_top are already in physical pixels
                # Glyph quad position in physical pixels
                qx_phys = pen_x_phys + x_offset_phys + bitmap_left
                qy_phys = pen_y_phys - bitmap_top - y_offset_phys

                # Glyph dimensions in physical pixels
                qw_phys = image.px_width
                qh_phys = image.px_height

                # Intersection with dst rect (in physical pixels)
                ix_phys = max(qx_phys, dst_x_phys)
                iy_phys = max(qy_phys, dst_y_phys)
                ir_phys = min(qx_phys + qw_phys, dst_x_phys + dst_w_phys)
                ib_phys = min(qy_phys + qh_phys, dst_y_phys + dst_h_phys)

                if ir_phys > ix_phys and ib_phys > iy_phys:
                    # Convert to logical pixel coordinates for output
                    dst_ix = int(ix_phys / renderer_scale)
                    dst_iy = int(iy_phys / renderer_scale)
                    dst_ir = int((ir_phys + renderer_scale - 1) / renderer_scale)
                    dst_ib = int((ib_phys + renderer_scale - 1) / renderer_scale)
                    dst_iw = dst_ir - dst_ix
                    dst_ih = dst_ib - dst_iy

                    # Clipping offset and size in physical pixels (for src rect)
                    src_off_x = ix_phys - qx_phys
                    src_off_y = iy_phys - qy_phys
                    src_w = ir_phys - ix_phys
                    src_h = ib_phys - iy_phys

                    # Ensure we don't exceed the image bounds
                    src_w = min(src_w, image.px_width - src_off_x)
                    src_h = min(src_h, image.px_height - src_off_y)

                    if src_w > 0 and src_h > 0 and dst_iw > 0 and dst_ih > 0:
                        canvas.add_quad(
                            dst_xy=(dst_ix, dst_iy),
                            dst_wh=(dst_iw, dst_ih),
                            src_xy=(src_off_x, src_off_y),
                            src_wh=(src_w, src_h),
                            color=color,
                            image=image,
                        )

            # Accumulate in 26.6 to preserve precision
            pen_x_26_6 += x_advance_26_6
            pen_y_26_6 += y_advance_26_6


#
# RendererCanvas:
#


class RendererCanvas:
    def __init__(self, renderer: Renderer, capacity: int = 64):
        self.renderer = renderer
        self._quad_array = RendererQuadArray(capacity)
        self._quad_count = 0

    #
    # Getters and properties:
    #

    def __len__(self) -> int:
        return self._quad_count

    @property
    def capacity(self) -> int:
        return len(self._quad_array)

    def as_quad_array(self) -> RendererQuadArray:
        return self._quad_array[: self._quad_count].view(RendererQuadArray)

    #
    # clear, reserve
    #

    def clear(self):
        self._quad_count = 0

    def reserve(self, new_capacity: int):
        """Ensure that the quad list has at least the given capacity."""
        if new_capacity <= self.capacity:
            return
        self.reserve_exact(new_capacity=round_up_to_po2(new_capacity))

    def reserve_exact(self, new_capacity: int):
        """Like 'reserve', but will never allocate more than requested."""
        if new_capacity <= self.capacity:
            return
        new_array = RendererQuadArray(new_capacity)
        new_array[: len(self._quad_array)] = self._quad_array
        self._quad_array = new_array

    #
    # add_quad
    #

    def add_quad(
        self,
        *,
        dst_xy: tuple[int, int],
        dst_wh: tuple[int, int] | None = None,
        src_xy: tuple[int, int] = (0, 0),
        src_wh: tuple[int, int] | None = None,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        border_thickness_px: tuple[int, int, int, int] = (0, 0, 0, 0),  # TRBL
        image: RendererImage | None = None,
    ):
        """Adds a single quad to the list."""

        # Ensure capacity
        if len(self) >= self.capacity:
            self.reserve(1 + len(self))
        assert len(self) < self.capacity

        # Reserve index
        index = self._quad_count
        self._quad_count += 1

        # Resolve:
        src_wh = RendererCanvas._resolve_src_wh(dst_wh, src_wh, image)
        assert src_wh is not None

        # write: dst_px
        dst_w, dst_h = RendererCanvas._eval_dst_px_wh(dst_wh, src_wh)

        scale = self.renderer.scale
        x = int(dst_xy[0] * scale)
        y = int(dst_xy[1] * scale)
        w = int(dst_w * scale)
        h = int(dst_h * scale)

        self._quad_array[index]["dst_px"][0] = [x, y]
        self._quad_array[index]["dst_px"][1] = [x + w, y]
        self._quad_array[index]["dst_px"][2] = [x + w, y + h]
        self._quad_array[index]["dst_px"][3] = [x, y + h]

        # write: src_uv
        uv_xywh = RendererCanvas._eval_src_uv_xywh(src_xy, src_wh, image)
        uv_x, uv_y, uv_w, uv_h = uv_xywh
        self._quad_array[index]["src_uv"][0] = [uv_x, uv_y]
        self._quad_array[index]["src_uv"][1] = [uv_x + uv_w, uv_y]
        self._quad_array[index]["src_uv"][2] = [uv_x + uv_w, uv_y + uv_h]
        self._quad_array[index]["src_uv"][3] = [uv_x, uv_y + uv_h]

        # write: image_id
        quad_image = image or self.renderer._default_white_image
        self._quad_array[index]["image_id"] = quad_image.image_id

        # write: color, border_color, border_thickness_px
        self._quad_array[index]["color"] = color
        self._quad_array[index]["border_color"] = border_color
        self._quad_array[index]["border_thickness_px"] = [
            int(border_thickness_px[0] * scale),
            int(border_thickness_px[1] * scale),
            int(border_thickness_px[2] * scale),
            int(border_thickness_px[3] * scale),
        ]

        # write: height
        self._quad_array[index]["height"] = float(index)

    @staticmethod
    def _resolve_src_wh(
        dst_wh: tuple[int, int] | None,
        src_wh: tuple[int, int] | None,
        image: RendererImage | None,
    ) -> tuple[int, int]:
        if image is not None:
            # Image present: check if src_wh is given to crop, else use full image size
            if src_wh is not None:
                return src_wh
            else:
                return image.px_width, image.px_height

        # No image: src_wh is irrelevant even if given. Use dst_wh.
        if dst_wh is not None:
            return dst_wh

        # Neither image nor dst_wh given: error
        raise LogicError("Cannot determine quad size: supply dst_wh or image.")

    @staticmethod
    def _eval_dst_px_wh(
        dst_wh: tuple[int, int] | None,
        src_wh: tuple[int, int],
    ) -> tuple[int, int]:
        if dst_wh is not None:
            return dst_wh
        else:
            return src_wh

    @staticmethod
    def _eval_src_uv_xywh(
        src_xy: tuple[int, int],
        src_wh: tuple[int, int],
        image: RendererImage | None,
    ) -> tuple[float, float, float, float]:
        if image is None:
            return (0.0, 0.0, 1.0, 1.0)

        src_uv_xy = (
            src_xy[0] / image.px_width,
            src_xy[1] / image.px_height,
        )
        src_uv_wh = (
            src_wh[0] / image.px_width,
            src_wh[1] / image.px_height,
        )
        return (src_uv_xy[0], src_uv_xy[1], src_uv_wh[0], src_uv_wh[1])

    #
    # add_text
    #

    def add_text(
        self,
        *,
        text: str,
        font: "RendererFont",
        dst_xy: tuple[int, int],
        dst_wh: tuple[int, int],
        font_size_px: int = 16,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        wrap: bool = True,
        font_weight: int = 400,
    ):
        """
        Adds quads for rendering the given text string with the given font.
        """

        font_cache = self.renderer._font_engine

        font_cache._add_quads_to_canvas(
            canvas=self,
            text=text,
            font=font,
            font_size_px=font_size_px,
            color=color,
            dst_xy=dst_xy,
            dst_wh=dst_wh,
            wrap=wrap,
            font_weight=font_weight,
        )
