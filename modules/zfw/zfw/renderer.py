__all__ = [
    "Renderer",
    "RendererContext",
    "RendererImage",
    "RendererQuadArray",
]

from collections import OrderedDict
from typing import Literal, TypeAlias

import numpy as np

from .basic import BaseResource, StructuredNDArray, expect, next_po2
from .bundled_data import BUNDLED_DATA_PATH
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
    context: RendererContext
    gpu_device: GpuDevice
    default_white_image_atlas: "RendererAtlas"
    default_white_image: "RendererImage"
    quad_pipeline: "RendererQuadPipeline"

    _atlases: dict["RendererAtlasChannels", "RendererAtlas"]

    def __init__(
        self,
        *,
        context: RendererContext,
        gpu_device: GpuDevice,
    ):
        super().__init__(parent=context)

        self.context = context
        self.gpu_device = gpu_device

        self._atlases = {
            1: RendererAtlas(renderer=self, channels=1),
            4: RendererAtlas(renderer=self, channels=4),
        }

        self.renderer_descriptor_set_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "atlasMono": GpuDescriptorSetLayoutBinding(
                        type="sampled-image", count=self._atlases[1].max_pages
                    ),
                    "atlasRgba": GpuDescriptorSetLayoutBinding(
                        type="sampled-image", count=self._atlases[4].max_pages
                    ),
                    "rectsMono": GpuDescriptorSetLayoutBinding(type="storage-buffer"),
                    "rectsRgba": GpuDescriptorSetLayoutBinding(type="storage-buffer"),
                    "samplerLinear": GpuDescriptorSetLayoutBinding(type="sampler"),
                    "samplerNearest": GpuDescriptorSetLayoutBinding(type="sampler"),
                }.items()
            ),
        )

        self.renderer_descriptor_set = GpuDescriptorSet(
            device=self.gpu_device,
            layout=self.renderer_descriptor_set_layout,
            bindings={
                "atlasMono": self._atlases[1]._page_gpu_image_list,
                "atlasRgba": self._atlases[4]._page_gpu_image_list,
                "rectsMono": self._atlases[1].rects_buffer,
                "rectsRgba": self._atlases[4].rects_buffer,
                "samplerLinear": self._atlases[4].linear_sampler,
                "samplerNearest": self._atlases[4].nearest_sampler,
            },
        )

        self.default_white_image = RendererImage(
            renderer=self,
            data=np.ones((1, 1, 4), dtype=np.float32),
        )
        self.default_white_image_atlas = self.default_white_image.atlas

        self.quad_pipeline = RendererQuadPipeline(renderer=self, gpu_device=gpu_device)

    def _on_dispose(self) -> None:
        self.quad_pipeline.dispose()
        self.renderer_descriptor_set.dispose()
        self.renderer_descriptor_set_layout.dispose()

        # Dispose atlases:
        for _, atlas in self._atlases.items():
            atlas.dispose()

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

        self.quad_pipeline.draw(
            quads=quads,
            target=target,
            wait_semaphores=wait_semaphores,
            done_semaphores=done_semaphores,
            fence=fence,
        )


#
# RendererAtlas, RendererImage
#


class RendererImage:
    renderer: Renderer
    atlas: "RendererAtlas"
    index: int

    def __init__(self, *, renderer: Renderer, data: np.ndarray):
        super().__init__()

        assert data.ndim in (2, 3)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        self.renderer = renderer
        self.atlas = renderer._atlases[data.shape[2]]
        self.index = self.atlas.insert(data=data)

    @property
    def uv_xywh(self) -> tuple[float, float, float, float]:
        return tuple(self.atlas._page_rect_allocator.rects[self.index])

    @property
    def px_xywh(self) -> tuple[int, int, int, int]:
        x_uv, y_uv, w_uv, h_uv = self.uv_xywh
        page_size = self.atlas.page_size

        x_px = int(x_uv * page_size)
        y_px = int((y_uv % 1.0) * page_size)
        w_px = int(w_uv * page_size)
        h_px = int(h_uv * page_size)

        return (x_px, y_px, w_px, h_px)

    @property
    def page_index(self) -> int:
        _, y_uv, _, _ = self.uv_xywh
        return int(y_uv)


RendererAtlasChannels: TypeAlias = Literal[1, 4]


class RendererAtlas(BaseResource):
    renderer: Renderer
    gpu_device: GpuDevice

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
        self.renderer = renderer
        self.gpu_device = renderer.gpu_device

        self.channels = channels
        self.atlas_id = channels
        self.page_size = page_size
        self.max_pages = max_pages
        self.max_rects = max_rects

        self._page_rect_allocator = PageRectAllocator(
            max_pages=max_pages,
            max_rects=max_rects,
        )
        self._page_pixel_data = np.empty(
            (max_pages, page_size, page_size, channels),
            dtype=np.float32,
        )
        self._page_gpu_image_list = self._create_page_gpu_images()

        self.rects_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=max_rects,
                element_dtype=UV_RECT_DTYPE,
            ),
        )
        self.linear_sampler = GpuSampler(
            device=self.gpu_device,
            mag_filter="linear",
            min_filter="linear",
        )
        self.nearest_sampler = GpuSampler(
            device=self.gpu_device,
            mag_filter="nearest",
            min_filter="nearest",
        )

    def _create_page_gpu_images(self) -> list[GpuImage]:
        return [
            GpuImage(
                device=self.renderer.gpu_device,
                usages=["texture-binding", "transfer-dst"],
                meta=GpuImageMeta(
                    shape=(self.page_size, self.page_size, self.channels),
                    dtype=np.float32,
                    color_space="linear",
                ),
            )
            for _ in range(self.max_pages)
        ]

    def insert(self, *, data: np.ndarray) -> int:
        assert data.ndim == 3 and data.shape[2] == self.channels

        # First, try to allocate from an existing page:
        image = self._try_simple_insert(data=data)
        if image is not None:
            return image

        # Otherwise, try running compaction and try again:
        self.compact()
        image = self._try_simple_insert(data=data)
        if image is not None:
            return image

        # If that fails, add a new page and try again:
        if (
            self._try_add_page()
            and (image := self._try_simple_insert(data=data)) is not None
        ):
            return image

        # If that fails, we're out of memory:
        raise MemoryError("out of atlas memory")

    def _try_simple_insert(self, *, data: np.ndarray) -> int | None:
        assert data.ndim == 3 and data.shape[2] == self.channels

        w_px, h_px = data.shape[1], data.shape[0]
        w = w_px / self.page_size
        h = h_px / self.page_size

        alloc_index = self._page_rect_allocator.alloc(w=w, h=h)
        if alloc_index is not None:
            self._upload_image(alloc_index, data)
            return alloc_index

        return None

    def compact(self):
        # First, compact just the allocation table.
        # We obtain a copy of the old allocation table.
        old_rect_array = self._page_rect_allocator.compact()

        # Next, compact the pages on the CPU:
        self._compact_pages_on_cpu(
            structured_old_uv_rect_array=old_rect_array,
            structured_new_uv_rect_array=self._page_rect_allocator.rects,
        )

        # Finally, upload all the pages from the CPU to the GPU:
        self._upload_all_pages()

    def _compact_pages_on_cpu(
        self,
        *,
        structured_old_uv_rect_array: "UvRectArray",
        structured_new_uv_rect_array: "UvRectArray",
    ):
        assert structured_old_uv_rect_array.shape == structured_new_uv_rect_array.shape

        old_uv_rect_array = structured_old_uv_rect_array.view(np.float32).reshape(-1, 4)
        new_uv_rect_array = structured_new_uv_rect_array.view(np.float32).reshape(-1, 4)

        rect_count = old_uv_rect_array.shape[0]
        page_count = len(self._page_gpu_image_list)

        old_px_rect_array = (old_uv_rect_array * self.page_size).astype(int)
        new_px_rect_array = (new_uv_rect_array * self.page_size).astype(int)

        src_page_data = self._page_pixel_data[:page_count].copy()
        for rect_index in range(rect_count):
            src_x, src_y, src_w, src_h = old_px_rect_array[rect_index]
            dst_x, dst_y, dst_w, dst_h = new_px_rect_array[rect_index]

            src_page = src_y // self.page_size
            src_y = src_y % self.page_size

            dst_page = dst_y // self.page_size
            dst_y = dst_y % self.page_size

            s = src_page_data[src_page, src_y : src_y + src_h, src_x : src_x + src_w]
            self._page_pixel_data[
                dst_page, dst_y : dst_y + dst_h, dst_x : dst_x + dst_w
            ] = s

    def _upload_all_pages(self):
        page_count = len(self._page_gpu_image_list)

        # If no pages, then nothing to do:
        if not page_count:
            return

        # Create staging buffer for rects:
        rects = self._page_rect_allocator.rects
        rects_staging_buffer: GpuBuffer | None = None
        if rects.nbytes > 0:
            rects_staging_buffer = GpuBuffer(
                device=self.gpu_device,
                usages=["staging", "copy-src"],
                meta=GpuBufferMeta.from_array(rects),
            )
            rects_staging_buffer.memory.write(data=rects)

        # Create and write to a staging buffer for pixels:
        pixel_staging_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta.from_array(self._page_pixel_data[:page_count]),
        )
        pixel_staging_buffer.memory.write(data=self._page_pixel_data[:page_count])

        # Using a one-time command buffer, copy from the staging buffer to the images,
        # blocking until done:
        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="transfer",
        )

        # Copy rects:
        if rects_staging_buffer is not None:
            command_encoder.copy_buffer_to_buffer(
                src=rects_staging_buffer,
                dst=self.rects_buffer,
                size=rects_staging_buffer.meta.size,
            )

        for page_index, page in enumerate(self._page_gpu_image_list):
            command_encoder.transition_image_layout(
                image=page,
                layout="transfer-dst-optimal",
            )
            command_encoder.copy_buffer_to_image(
                src=pixel_staging_buffer,
                dst=page,
                buffer_offset=self._page_pixel_data[page_index].nbytes * page_index,
            )
        command_encoder.submit().wait()

        if rects_staging_buffer is not None:
            rects_staging_buffer.dispose()
        pixel_staging_buffer.dispose()

    def _try_add_page(self) -> bool:
        return self._page_rect_allocator.try_add_page()

    def _upload_image(self, alloc_index: int, data: np.ndarray):
        assert data.shape[2] == self.channels
        assert data.dtype == np.float32

        rects = self._page_rect_allocator.rects.view(np.float32).reshape(-1, 4)

        x_px, y_px, w_px, h_px = (rects[alloc_index] * self.page_size).astype(int)
        page_index = y_px // self.page_size
        y_px = y_px % self.page_size

        # Update pixel data in CPU mirror:
        self._page_pixel_data[page_index, y_px : y_px + h_px, x_px : x_px + w_px] = data

        # Create staging buffer for rects:
        rect_data = self._page_rect_allocator.rects[alloc_index : alloc_index + 1]
        rects_staging_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta.from_array(rect_data),
        )
        rects_staging_buffer.memory.write(data=rect_data)

        # Create and write to a staging buffer for pixels:
        pixel_staging_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta.from_array(data),
        )
        pixel_staging_buffer.memory.write(data=data)

        # Using a one-time command buffer, copy from the staging buffer to the image,
        # blocking until done:
        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="transfer",
        )

        # Copy rects:
        command_encoder.copy_buffer_to_buffer(
            src=rects_staging_buffer,
            dst=self.rects_buffer,
            dst_offset=alloc_index * UV_RECT_DTYPE.itemsize,
            size=rects_staging_buffer.meta.size,
        )

        command_encoder.transition_image_layout(
            image=self._page_gpu_image_list[page_index],
            layout="transfer-dst-optimal",
        )
        command_encoder.copy_buffer_to_image(
            src=pixel_staging_buffer,
            dst=self._page_gpu_image_list[page_index],
            image_offset=(x_px, y_px, 0),
            image_extent=(w_px, h_px, 1),
        )
        command_encoder.submit().wait()

        rects_staging_buffer.dispose()
        pixel_staging_buffer.dispose()


class PageRectAllocator:
    """
    Inserts rectangles into pages, each a unit square of size 1.0 x 1.0 in UV space.
    """

    def __init__(self, *, max_pages: int, max_rects: int):
        super().__init__()

        self.max_pages = max_pages
        self.max_rects = max_rects

        self.page_cursor_array = np.zeros(
            (max_pages,),
            dtype=np.dtype(
                [
                    ("insert_x", np.float32),
                    ("insert_y", np.float32),
                    ("row_height", np.float32),
                ]
            ),
        )
        self.page_count = 0

        self.rect_array = UvRectArray((self.max_rects,))
        self.rect_count = 0

    @property
    def pages(self) -> np.ndarray:
        return self.page_cursor_array[: self.page_count]

    @property
    def rects(self) -> "UvRectArray":
        return self.rect_array[: self.rect_count].view(UvRectArray)

    @rects.setter
    def rects(self, value: "UvRectArray | np.ndarray"):
        self.rect_array[: self.rect_count] = value

    def alloc(self, *, w: float, h: float) -> int | None:
        # Iterate over pages in reverse order, trying to insert:
        for page_index in reversed(range(self.page_count)):
            page = self.page_cursor_array[page_index]

            # Try to insert on the current row:
            x, y = page["insert_x"], page["insert_y"]
            if x + w <= 1.0 and y + h <= 1.0:
                page["insert_x"] += w
                page["row_height"] = max(page["row_height"], h)
                return self._push_allocation(x, y, w, h, page_index)

            # Try to insert on a new row:
            x = 0
            y += page["row_height"]
            if x + w <= 1.0 and y + h <= 1.0:
                page["insert_x"] = w
                page["insert_y"] = y
                page["row_height"] = h
                return self._push_allocation(x, y, w, h, page_index)
        else:
            return None

    def compact(self) -> "UvRectArray":
        # Create a new, empty allocator that can hold all the current rects:
        new_allocator = PageRectAllocator(
            max_pages=self.max_pages,
            max_rects=self.rect_count,
        )

        # Sort rects by area in descending order, ordering alloc indices for insertion:
        insert_idx_array = np.argsort(-1 * self.rects["w"] * self.rects["h"])

        # Insert rects in the new allocator, building an array mapping old to new
        # indices:
        new_idx_array = np.empty_like(insert_idx_array)
        for old_idx in insert_idx_array:
            new_idx_array[old_idx] = expect(
                new_allocator.alloc(
                    w=self.rects[old_idx]["w"].item(),
                    h=self.rects[old_idx]["h"].item(),
                )
            )

        # Swap the current 'rect_array' for the new one.
        # Return the old rect array.
        # The user can map the old array entries to the new ones currently in self.
        old_rect_array = self.rects
        self.rects = new_allocator.rects[new_idx_array]

        # Done:
        assert old_rect_array.shape == self.rects.shape
        return old_rect_array.view(UvRectArray)

    def try_add_page(self) -> bool:
        if self.page_count >= self.max_pages:
            return False

        self.page_count += 1
        assert self.page_count <= self.max_pages
        return True

    def _push_allocation(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        page_index: int,
    ) -> int:
        if self.rect_count >= self.max_rects:
            raise MemoryError("Out of rect allocations")

        assert 0.0 <= x < 1.0 and 0.0 <= y < 1.0
        assert 0.0 < x + w <= 1.0 and 0.0 < y + h <= 1.0
        assert isinstance(page_index, int)

        allocation_index = self.rect_count
        self.rect_count += 1
        assert self.rect_count <= self.max_rects

        self.rect_array[allocation_index] = (x, float(page_index) + y, w, h)

        return allocation_index


UV_RECT_DTYPE = np.dtype(
    [
        ("x", np.float32),  # 0 <= x < 1
        ("y", np.float32),  # int(y) is the page_index, fmod(y, 1) is the UV y
        ("w", np.float32),  # 0 < x_uv + w <= 1
        ("h", np.float32),  # 0 < y_uv + h <= 1
    ]
)


class UvRectArray(StructuredNDArray):
    DTYPE = UV_RECT_DTYPE


#
# RendererQuadPipeline
#


class RendererQuadPipeline(BaseResource):
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
            ("flags", np.uint32),
            ("_pad", np.uint32),
        ]
    )

    renderer: "Renderer"
    gpu_device: GpuDevice

    _quads_vertex_shader: GpuShader
    _quads_fragment_shader: GpuShader
    _quads_pipeline_layout: GpuPipelineLayout
    _quads_common_uniform_staging_buf: GpuBuffer
    _quads_common_uniform_device_buf: GpuBuffer
    _quads_common_uniform_descriptor_set: GpuDescriptorSet
    _quads_cached_pipeline: GpuPipeline | None
    _quads_cached_depth_image: GpuImage | None

    _batch_capacity: int
    _batch_quads_staging_buf: GpuBuffer | None
    _batch_quads_device_buf: GpuBuffer | None
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

        self._quads_vertex_shader = self._new_quads_vertex_shader()
        self._quads_fragment_shader = self._new_quads_fragment_shader()
        self._quads_pipeline_layout = self._new_quads_pipeline_layout()
        self._quads_common_uniform_staging_buf = self._new_common_uniform_buffer(
            staging=True
        )
        self._quads_common_uniform_device_buf = self._new_common_uniform_buffer(
            staging=False
        )
        self._quads_common_uniform_descriptor_set = (
            self._new_common_uniform_descriptor_set()
        )
        self._quads_cached_pipeline = None
        self._quads_cached_depth_image = None

        self._batch_capacity = 0
        self._batch_quads_staging_buf = None
        self._batch_quads_device_buf = None
        self._batch_uniform_staging_buf = None
        self._batch_uniform_device_buf = None
        self._batch_descriptor_set = None

    def _on_dispose(self) -> None:
        self._quads_vertex_shader.dispose()
        self._quads_fragment_shader.dispose()

        self._quads_pipeline_layout.dispose()

        self._quads_common_uniform_descriptor_set.dispose()

        self._quads_common_uniform_staging_buf.dispose()
        self._quads_common_uniform_device_buf.dispose()

        if self._batch_descriptor_set is not None:
            self._batch_descriptor_set.dispose()
        if self._batch_uniform_staging_buf is not None:
            self._batch_uniform_staging_buf.dispose()
        if self._batch_uniform_device_buf is not None:
            self._batch_uniform_device_buf.dispose()
        if self._batch_quads_staging_buf is not None:
            self._batch_quads_staging_buf.dispose()
        if self._batch_quads_device_buf is not None:
            self._batch_quads_device_buf.dispose()

        if self._quads_cached_depth_image is not None:
            self._quads_cached_depth_image.dispose()
        if self._quads_cached_pipeline is not None:
            self._quads_cached_pipeline.dispose()

    def _ensure_batch_capacity(self, capacity: int):
        if capacity <= self._batch_capacity:
            return

        # Dispose old resources
        if self._batch_descriptor_set is not None:
            self._batch_descriptor_set.dispose()
        if self._batch_uniform_staging_buf is not None:
            self._batch_uniform_staging_buf.dispose()
        if self._batch_uniform_device_buf is not None:
            self._batch_uniform_device_buf.dispose()
        if self._batch_quads_staging_buf is not None:
            self._batch_quads_staging_buf.dispose()
        if self._batch_quads_device_buf is not None:
            self._batch_quads_device_buf.dispose()

        # Create new resources
        new_capacity = max(8, next_po2(capacity))
        self._batch_capacity = new_capacity

        self._batch_uniform_device_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["uniform", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=1,
                element_dtype=RendererQuadPipeline.BATCH_UNIFORM_DTYPE,
            ),
        )
        self._batch_uniform_staging_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=1,
                element_dtype=RendererQuadPipeline.BATCH_UNIFORM_DTYPE,
            ),
        )
        self._batch_quads_device_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=new_capacity,
                element_dtype=RendererQuadPipeline.QUAD_DTYPE,
            ),
        )
        self._batch_quads_staging_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=new_capacity,
                element_dtype=RendererQuadPipeline.QUAD_DTYPE,
            ),
        )
        self._batch_descriptor_set = GpuDescriptorSet(
            device=self.gpu_device,
            layout=self._quads_pipeline_layout.descriptor_set_layouts[2],
            bindings={
                "quads": self._batch_quads_device_buf,
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
            assert self._batch_quads_staging_buf is not None
            assert self._batch_quads_device_buf is not None
            self._batch_quads_staging_buf.memory.write(data=quads)
            command_encoder.copy_buffer_to_buffer(
                src=self._batch_quads_staging_buf,
                dst=self._batch_quads_device_buf,
                size=len(quads) * RendererQuadPipeline.QUAD_DTYPE.itemsize,
            )

            # Batch uniform:
            assert self._batch_uniform_staging_buf is not None
            assert self._batch_uniform_device_buf is not None
            uniform_array = np.empty(
                (1,), dtype=RendererQuadPipeline.BATCH_UNIFORM_DTYPE
            )
            uniform_array[0]["instance_count"] = int(len(quads))
            uniform_array[0]["is_opaque"] = 1  # TODO: support transparency
            uniform_array[0]["is_mono"] = 0
            self._batch_uniform_staging_buf.memory.write(data=uniform_array)
            command_encoder.copy_buffer_to_buffer(
                src=self._batch_uniform_staging_buf,
                dst=self._batch_uniform_device_buf,
                size=RendererQuadPipeline.BATCH_UNIFORM_DTYPE.itemsize,
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
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "zfw" / "r2d.vert.spv",
            stage="vertex",
        )

    def _new_quads_fragment_shader(self) -> GpuShader:
        return GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "zfw" / "r2d.frag.spv",
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
                self.renderer.renderer_descriptor_set_layout,
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
                element_dtype=RendererQuadPipeline.COMMON_UNIFORM_DTYPE,
            ),
        )

    def _new_common_uniform_descriptor_set(self) -> GpuDescriptorSet:
        return GpuDescriptorSet(
            device=self.gpu_device,
            layout=self._quads_pipeline_layout.descriptor_set_layouts[0],
            bindings={"commonUniform": self._quads_common_uniform_device_buf},
        )

    def _get_gpu_pipeline(self, target: GpuImage) -> GpuPipeline:
        if cached_pipeline := self._get_cached_gpu_pipeline(target=target):
            return cached_pipeline
        gpu_pipeline = self._new_gpu_pipeline(target=target)
        self._quads_cached_pipeline = gpu_pipeline
        return gpu_pipeline

    def _get_cached_gpu_pipeline(self, target: GpuImage) -> GpuPipeline | None:
        if self._quads_cached_pipeline is None:
            return None
        if self._quads_cached_pipeline.vk_color_format != target.vk_format:
            return None
        if self._quads_cached_pipeline.viewport_width != target.width:
            return None
        if self._quads_cached_pipeline.viewport_height != target.height:
            return None
        return self._quads_cached_pipeline

    def _new_gpu_pipeline(self, target: GpuImage) -> GpuPipeline:
        return GpuPipeline(
            device=self.gpu_device,
            vertex_shader=self._quads_vertex_shader,
            fragment_shader=self._quads_fragment_shader,
            vk_color_format=target.vk_format,
            enable_depth_test=False,
            enable_alpha_blending=True,
            viewport_width=target.width,
            viewport_height=target.height,
            layout=self._quads_pipeline_layout,
        )

    def _get_depth_image(self, *, width: int, height: int) -> GpuImage:
        if (
            self._quads_cached_depth_image is not None
            and self._quads_cached_depth_image.width == width
            and self._quads_cached_depth_image.height == height
        ):
            return self._quads_cached_depth_image

        depth_image = self._new_depth_image(width=width, height=height)
        self._quads_cached_depth_image = depth_image

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
        uniform_data = np.zeros((1,), dtype=RendererQuadPipeline.COMMON_UNIFORM_DTYPE)
        uniform_data[0]["framebuffer_size_px"] = [target.width, target.height]
        uniform_data[0]["total_quad_count"] = float(total_quad_count)
        uniform_data[0]["atlas_size_px"] = self.renderer._atlases[4].page_size
        self._quads_common_uniform_staging_buf.memory.write(data=uniform_data)
        encoder.copy_buffer_to_buffer(
            src=self._quads_common_uniform_staging_buf,
            dst=self._quads_common_uniform_device_buf,
            size=RendererQuadPipeline.COMMON_UNIFORM_DTYPE.itemsize,
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
                descriptor_set=self._quads_common_uniform_descriptor_set,
            )
            render_pass.bind_descriptor_set(
                set_index=1,
                descriptor_set=self.renderer.renderer_descriptor_set,
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


assert RendererQuadPipeline.COMMON_UNIFORM_DTYPE.itemsize == 4 * 4
assert RendererQuadPipeline.BATCH_UNIFORM_DTYPE.itemsize == 4 * 4
assert RendererQuadPipeline.QUAD_DTYPE.itemsize == 32 * 4


class RendererQuadArray(np.ndarray):
    def __new__(cls, shape: tuple[int, ...] | int) -> "RendererQuadArray":
        return np.zeros(shape, dtype=RendererQuadPipeline.QUAD_DTYPE).view(cls)
