__all__ = [
    "Image",
    "ImageHeap",
    "QuadArray",
    "Renderer",
    "RendererContext",
]

from collections import OrderedDict
from pathlib import Path
from typing import Literal, Callable

import numpy as np

from .basic import (
    BaseResource,
    logger,
    round_up_to_po2,
    Font,
    HorizontalAlignment,
    VerticalAlignment,
)
from .bundled_data import BUNDLED_DATA_PATH
from .excepts import LogicError
from .gpu import (
    GpuBuffer,
    GpuBufferImageCopyRegion,
    GpuBufferMeta,
    GpuBufferUsage,
    GpuCommandEncoder,
    GpuContext,
    GpuDescriptorSet,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuImage,
    GpuImageMeta,
    GpuPipeline,
    GpuPipelineLayout,
    GpuSampler,
    GpuShader,
)
from . import typed_freetype as ft  # Must import before uharfbuzz
from . import typed_uharfbuzz as hb


_logger = logger(__name__)


#
# Renderer API:
#


class RendererContext(BaseResource):
    def __init__(
        self,
        *,
        gpu_context: GpuContext,
        parent_resource: BaseResource | None = None,
    ):
        super().__init__(parent_resource=parent_resource)
        self.gpu_context = gpu_context


class Renderer(BaseResource):
    _context: RendererContext
    _gpu_device: GpuDevice
    _max_frames_in_flight: int
    _scale: float
    _basic_uniform: "BasicUniform"
    _atlas: "Atlas"
    _default_white_image: "Image"
    _r2d: "Renderer2d"
    _r3d: "Renderer3d"

    def __init__(
        self,
        *,
        context: RendererContext,
        gpu_device: GpuDevice,
        max_frames_in_flight: int,
        scale: float = 1.0,
    ):
        super().__init__(parent_resource=context)

        self._context = context
        self._gpu_device = gpu_device
        self._max_frames_in_flight = max_frames_in_flight
        self._scale = scale

        self._basic_uniform = BasicUniform(renderer=self)
        self._atlas = Atlas(renderer=self)

        self._default_white_image = Image(
            renderer=self,
            data=np.ones((1, 1, 4), dtype=np.float32),
            sampler="nearest",
        )

        self._r2d = Renderer2d(renderer=self)
        self._r3d = Renderer3d(renderer=self)

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def max_frames_in_flight(self) -> int:
        return self._max_frames_in_flight

    def _on_dispose_resource(self) -> None:
        self._r3d.dispose_resource()
        self._r2d.dispose_resource()

        self._atlas.dispose_resource()
        self._basic_uniform.dispose_resource()

    def draw(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        canvas: "Canvas",
        target: GpuImage,
        frame_index: int,
    ):
        """
        Render the given array of quads to the given target `GpuImage`.

        The quads array should have dtype RENDERER_QUAD_DTYPE.
        """
        assert frame_index < self._max_frames_in_flight

        # Upload:
        #

        # Update common uniforms in `b0_basic`:
        self._basic_uniform.flush(
            command_encoder=command_encoder,
            framebuffer_size_px=(target.width, target.height),
            frame_index=frame_index,
        )

        # Update texture atlas in `b1_atlas.slang`:
        self._atlas.flush(command_encoder=command_encoder, frame_index=frame_index)

        # Draw:
        #

        # Ensure target is in correct layout
        command_encoder.transition_image_layout(
            image=target,
            layout="color-attachment-optimal",
        )

        # Run r2d pipelines:
        self._r2d.draw(
            command_encoder=command_encoder,
            quads=canvas.as_quad_array(),
            target=target,
            frame_index=frame_index,
        )

    @property
    def atlas(self) -> "Atlas":
        return self._atlas


#
# BasicUniform
#


class BasicUniform(BaseResource):
    DTYPE = np.dtype(
        [
            ("framebuffer_size_px", np.uint32, (2,)),
            ("_rsv0", np.uint32),
            ("_rsv1", np.uint32),
        ]
    )

    _renderer: "Renderer"
    _gpu_device: GpuDevice
    _buffer_ring: "GpuBufferRing"
    _descriptor_set_layout: GpuDescriptorSetLayout
    _descriptor_set_ring: "GpuDescriptorSetRing"

    def __init__(self, *, renderer: "Renderer"):
        super().__init__(parent_resource=renderer)

        self._renderer = renderer
        self._gpu_device = renderer._gpu_device

        buffer_meta = GpuBufferMeta(
            element_count=1,
            element_dtype=BasicUniform.DTYPE,
        )
        self._buffer_ring = GpuBufferRing(
            device=self._gpu_device,
            meta=buffer_meta,
            ring_capacity=renderer._max_frames_in_flight,
            device_usages=["uniform", "copy-dst"],
        )
        self._descriptor_set_layout = GpuDescriptorSetLayout(
            device=self._gpu_device,
            bindings=OrderedDict(
                {
                    "u": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["vertex", "fragment"],
                    ),
                }.items()
            ),
        )
        self._descriptor_set_ring = GpuDescriptorSetRing(
            device=self._gpu_device,
            layout=self._descriptor_set_layout,
            ring_capacity=renderer._max_frames_in_flight,
            bindings_fn=lambda i: {"u": self._buffer_ring.get_device_buffer(i)},
        )

    def _on_dispose_resource(self) -> None:
        self._descriptor_set_ring.dispose_resource()
        self._descriptor_set_layout.dispose_resource()
        self._buffer_ring.dispose_resource()
        super()._on_dispose_resource()

    def flush(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        framebuffer_size_px: tuple[int, int],
        frame_index: int,
    ) -> GpuDescriptorSet:
        data = np.empty((1,), dtype=BasicUniform.DTYPE)
        data[0]["framebuffer_size_px"] = framebuffer_size_px

        self._buffer_ring.write(frame_index, data)
        self._buffer_ring.flush(frame_index, command_encoder)

        return self._descriptor_set_ring.get(frame_index)


#
# Ring buffer helpers for multi-frame-in-flight synchronization:
#


class GpuBufferRing(BaseResource):
    """
    A ring of staging + device buffer pairs for safe multi-frame-in-flight rendering.

    Each frame uses a different slot to avoid race conditions where the GPU is still
    reading from a buffer while the CPU is writing to it for the next frame.
    """

    _device: GpuDevice
    _meta: GpuBufferMeta
    _ring_capacity: int
    _staging_usages: list[GpuBufferUsage]
    _device_usages: list[GpuBufferUsage]
    _staging_buffers: list[GpuBuffer]
    _device_buffers: list[GpuBuffer]

    def __init__(
        self,
        *,
        device: GpuDevice,
        meta: GpuBufferMeta,
        ring_capacity: int,
        staging_usages: list[GpuBufferUsage] | None = None,
        device_usages: list[GpuBufferUsage],
    ):
        super().__init__(parent_resource=None)
        self._device = device
        self._meta = meta
        self._ring_capacity = ring_capacity
        self._staging_usages = staging_usages or ["staging", "copy-src"]
        self._device_usages = device_usages

        self._staging_buffers = [
            GpuBuffer(
                device=device,
                usages=self._staging_usages,
                meta=meta,
            )
            for _ in range(ring_capacity)
        ]
        self._device_buffers = [
            GpuBuffer(
                device=device,
                usages=self._device_usages,
                meta=meta,
            )
            for _ in range(ring_capacity)
        ]

    def _on_dispose_resource(self) -> None:
        for buf in self._staging_buffers:
            buf.dispose_resource()
        for buf in self._device_buffers:
            buf.dispose_resource()
        super()._on_dispose_resource()

    @property
    def meta(self) -> GpuBufferMeta:
        return self._meta

    @property
    def ring_capacity(self) -> int:
        return self._ring_capacity

    def write(self, index: int, data: np.ndarray) -> None:
        """Write data to the staging buffer at the given ring index."""
        assert 0 <= index < self._ring_capacity
        self._staging_buffers[index].memory.write(data=data)

    def flush(
        self,
        index: int,
        command_encoder: GpuCommandEncoder,
        size: int | None = None,
    ) -> GpuBuffer:
        """
        Copy from staging to device buffer at the given ring index.
        Returns the device buffer for binding.
        """
        assert 0 <= index < self._ring_capacity
        copy_size = size if size is not None else self._meta.size
        command_encoder.copy_buffer_to_buffer(
            src=self._staging_buffers[index],
            dst=self._device_buffers[index],
            size=copy_size,
        )
        return self._device_buffers[index]

    def get_device_buffer(self, index: int) -> GpuBuffer:
        """Get the device buffer at the given ring index without flushing."""
        assert 0 <= index < self._ring_capacity
        return self._device_buffers[index]


class GpuDescriptorSetRing(BaseResource):
    """
    A ring of descriptor sets for safe multi-frame-in-flight rendering.

    Each frame uses a different descriptor set slot to allow different buffer
    bindings per frame.
    """

    _device: GpuDevice
    _layout: GpuDescriptorSetLayout
    _ring_capacity: int
    _descriptor_sets: list[GpuDescriptorSet]

    def __init__(
        self,
        *,
        device: GpuDevice,
        layout: GpuDescriptorSetLayout,
        ring_capacity: int,
        bindings_fn: Callable[[int], dict],
    ):
        super().__init__(parent_resource=None)
        self._device = device
        self._layout = layout
        self._ring_capacity = ring_capacity

        self._descriptor_sets = [
            GpuDescriptorSet(
                device=device,
                layout=layout,
                bindings=bindings_fn(i),
            )
            for i in range(ring_capacity)
        ]

    def _on_dispose_resource(self) -> None:
        for ds in self._descriptor_sets:
            ds.dispose_resource()
        super()._on_dispose_resource()

    @property
    def ring_capacity(self) -> int:
        return self._ring_capacity

    def get(self, index: int) -> GpuDescriptorSet:
        """Get the descriptor set at the given ring index."""
        assert 0 <= index < self._ring_capacity
        return self._descriptor_sets[index]


#
# Image, Atlas
#


class Image:
    _atlas: "ImageHeap"
    _index: int
    _allocation_uv_xywh: tuple[float, float, float, float] | None
    _allocation_px_xywh: tuple[int, int, int, int] | None
    _sampler: "SamplerType"

    def __init__(
        self,
        *,
        renderer: "Renderer",
        data: np.ndarray,
        sampler: "SamplerType",
    ):
        assert data.ndim in (2, 3)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        self._atlas = renderer._atlas.heap(channels=data.shape[2])
        self._data = data.copy()
        self._data.flags.writeable = False
        self._index = -1
        self._allocation_uv_xywh = None
        self._allocation_px_xywh = None
        self._sampler = sampler

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

    @property
    def sampler(self) -> "SamplerType":
        return self._sampler


type ImageChannels = Literal[1, 4]
type SamplerType = Literal["linear", "nearest"]


class Atlas(BaseResource):
    _gpu_device: GpuDevice
    _mono_heap: "ImageHeap"
    _rgba_heap: "ImageHeap"
    _heap_index: dict[ImageChannels, "ImageHeap"]
    _descriptor_set_layout: GpuDescriptorSetLayout
    _descriptor_set: GpuDescriptorSet

    def __init__(self, *, renderer: "Renderer"):
        super().__init__(parent_resource=renderer)
        self._gpu_device = renderer._gpu_device
        self._mono_heap = ImageHeap(renderer=renderer, channels=1)
        self._rgba_heap = ImageHeap(renderer=renderer, channels=4)
        self._heap_index = {1: self._mono_heap, 4: self._rgba_heap}

        self._descriptor_set_layout = GpuDescriptorSetLayout(
            device=self._gpu_device,
            bindings=OrderedDict(
                {
                    "atlasMono": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        count=self._mono_heap._max_pages,
                    ),
                    "atlasRgba": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        count=self._rgba_heap._max_pages,
                    ),
                    "rectsMono": GpuDescriptorSetLayoutBinding(type="storage-buffer"),
                    "rectsRgba": GpuDescriptorSetLayoutBinding(type="storage-buffer"),
                    "nearestSampler": GpuDescriptorSetLayoutBinding(type="sampler"),
                    "linearSampler": GpuDescriptorSetLayoutBinding(type="sampler"),
                }.items()
            ),
        )
        self._nearest_sampler = GpuSampler(
            device=self._gpu_device,
            mag_filter="nearest",
            min_filter="nearest",
        )
        self._linear_sampler = GpuSampler(
            device=self._gpu_device,
            mag_filter="linear",
            min_filter="linear",
        )
        self._descriptor_set = GpuDescriptorSet(
            device=self._gpu_device,
            layout=self._descriptor_set_layout,
            bindings={
                "atlasMono": self._mono_heap._page_gpu_image_list,
                "atlasRgba": self._rgba_heap._page_gpu_image_list,
                "rectsMono": self._mono_heap._uv_rect_array_device_buf,
                "rectsRgba": self._rgba_heap._uv_rect_array_device_buf,
                "nearestSampler": self._nearest_sampler,
                "linearSampler": self._linear_sampler,
            },
        )

    def _on_dispose_resource(self) -> None:
        self._descriptor_set.dispose_resource()
        self._descriptor_set_layout.dispose_resource()

        self._nearest_sampler.dispose_resource()
        self._linear_sampler.dispose_resource()

        self._mono_heap.dispose_resource()
        self._rgba_heap.dispose_resource()

        super()._on_dispose_resource()

    def heap(self, *, channels: ImageChannels) -> "ImageHeap":
        return self._heap_index[channels]

    def flush(self, *, command_encoder: GpuCommandEncoder, frame_index: int) -> None:
        self._mono_heap.flush(command_encoder=command_encoder, frame_index=frame_index)
        self._rgba_heap.flush(command_encoder=command_encoder, frame_index=frame_index)


class ImageHeap(BaseResource):
    """
    A GPU texture array that stores multiple images of the same number of channels and
    same format.
    """

    _renderer: Renderer
    _gpu_device: GpuDevice
    _channels: ImageChannels
    _page_size: int
    _max_pages: int
    _max_rects: int
    _images: list[Image]
    _unallocated_images: list[Image]
    _page_cursor_array: np.ndarray
    _page_count: int
    _page_pixel_data: np.ndarray
    _page_gpu_image_list: list[GpuImage]
    # Ring-buffered staging buffers (one list per frame slot):
    _page_staging_buffer_ring: list[list[GpuBuffer]]
    _uv_rect_array_device_buf: GpuBuffer
    _uv_rect_array_staging_ring: list[GpuBuffer]

    def __init__(
        self,
        *,
        renderer: Renderer,
        channels: ImageChannels,
        max_pages: int = 4,
        page_size: int = 4096,
        max_rects: int = 1 << 20,
    ):
        super().__init__(parent_resource=renderer)
        self._renderer = renderer
        self._gpu_device = renderer._gpu_device

        self._channels = channels
        self._page_size = page_size
        self._max_pages = max_pages
        self._max_rects = max_rects

        self._images: list[Image] = []
        self._unallocated_images: list[Image] = []

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
        # Ring-buffered staging buffers:
        self._page_staging_buffer_ring = [
            self._create_page_staging_buffers()
            for _ in range(renderer._max_frames_in_flight)
        ]

        self._uv_rect_array_device_buf = GpuBuffer(
            device=self._gpu_device,
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=max_rects,
                element_dtype=UV_RECT_DTYPE,
            ),
        )
        # Ring-buffered rect staging buffers:
        self._uv_rect_array_staging_ring = [
            GpuBuffer(
                device=self._gpu_device,
                usages=["staging", "copy-src"],
                meta=GpuBufferMeta(
                    element_count=max_rects,
                    element_dtype=UV_RECT_DTYPE,
                ),
            )
            for _ in range(renderer._max_frames_in_flight)
        ]

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

    def _create_page_staging_buffers(self) -> list[GpuBuffer]:
        page_element_count = self._page_size * self._page_size * self._channels
        return [
            GpuBuffer(
                device=self._gpu_device,
                usages=["staging", "copy-src"],
                meta=GpuBufferMeta(
                    element_count=page_element_count,
                    element_dtype=np.float32,
                ),
            )
            for _ in range(self._max_pages)
        ]

    def _on_dispose_resource(self) -> None:
        for gpu_image in self._page_gpu_image_list:
            gpu_image.dispose_resource()
        for staging_list in self._page_staging_buffer_ring:
            for staging_buf in staging_list:
                staging_buf.dispose_resource()
        self._uv_rect_array_device_buf.dispose_resource()
        for staging_buf in self._uv_rect_array_staging_ring:
            staging_buf.dispose_resource()
        super()._on_dispose_resource()

    def insert(self, image: Image):
        assert image._data.shape[2] == self._channels
        image._index = len(self._images)
        self._images.append(image)
        self._unallocated_images.append(image)

    def flush(self, *, command_encoder: GpuCommandEncoder, frame_index: int) -> None:
        if not self._unallocated_images:
            return

        # Sort unallocated images by area (largest first)
        self._unallocated_images.sort(
            key=lambda img: img._data.shape[0] * img._data.shape[1],
            reverse=True,
        )

        # First, we'll consider all unallocated images as "dirty" (i.e. needing upload)
        # and will try to allocate them.
        dirty_images = self._unallocated_images
        self._unallocated_images = []

        # Try to allocate unallocated images
        failed_to_alloc = False
        for img in dirty_images:
            if not self._alloc_image(img):
                failed_to_alloc = True
                break

        # If any failed to allocate, compact and re-allocate all images.
        # In this case, all images (including those previously allocated) are dirty.
        if failed_to_alloc:
            # Compaction requires a blocking sync to avoid race conditions:
            # we must wait for all in-flight frames to finish before modifying
            # the shared page images.
            _logger.warning(
                "ImageHeap compaction triggered. This causes a GPU sync and may "
                "impact performance. Consider increasing max_pages or page_size.",
            )

            # Create a temporary command encoder, submit, and wait.
            blocking_encoder = GpuCommandEncoder(
                device=self._gpu_device,
                queue_type="transfer",
            )
            self._compact()
            dirty_images = self._images
            self._upload_pages(
                dirty_images=dirty_images,
                command_encoder=blocking_encoder,
                frame_index=frame_index,
            )
            self._upload_rects(
                dirty_images=dirty_images,
                command_encoder=blocking_encoder,
                frame_index=frame_index,
            )
            blocking_encoder.submit().wait()
        else:
            # No compaction needed: upload only the newly allocated images.
            self._upload_pages(
                dirty_images=dirty_images,
                command_encoder=command_encoder,
                frame_index=frame_index,
            )
            self._upload_rects(
                dirty_images=dirty_images,
                command_encoder=command_encoder,
                frame_index=frame_index,
            )

    def _alloc_image(self, img: Image) -> bool:
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
        img: Image,
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
                raise MemoryError("Out of image-heap memory during compaction")

    def _upload_pages(
        self,
        *,
        dirty_images: list[Image],
        command_encoder: GpuCommandEncoder,
        frame_index: int,
    ):
        if not dirty_images:
            return

        # Group by page
        pages: dict[int, list[Image]] = {}
        for img in dirty_images:
            pages.setdefault(img.page_index, []).append(img)

        # Use ring-buffered staging buffers for this frame
        page_staging_list = self._page_staging_buffer_ring[frame_index]

        # Upload per-page with batched copy regions to avoid WRITE_AFTER_WRITE hazards
        for page_index, images in pages.items():
            page_staging = page_staging_list[page_index]
            offset = 0

            # Build all copy regions and write staging data
            regions: list[GpuBufferImageCopyRegion] = []
            for img in images:
                page_staging.memory.write(data=img._data, offset=offset)
                regions.append(
                    GpuBufferImageCopyRegion(
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
                )
                offset += img._data.nbytes

            # Transition once per page before issuing the batched copy
            command_encoder.transition_image_layout(
                image=self._page_gpu_image_list[page_index],
                layout="transfer-dst-optimal",
            )
            # Single batched copy command for all regions
            command_encoder.copy_buffer_to_image(
                src=page_staging,
                dst=self._page_gpu_image_list[page_index],
                regions=regions,
            )

    def _upload_rects(
        self,
        *,
        dirty_images: list[Image],
        command_encoder: GpuCommandEncoder,
        frame_index: int,
    ):
        if not dirty_images:
            return

        # Find the minimum index among dirty images for partial update
        min_index = min(img._index for img in dirty_images)

        # Build rect array starting from min_index
        rect_count = len(self._images) - min_index
        rects = np.zeros((rect_count,), dtype=UV_RECT_DTYPE)
        for i, img in enumerate(self._images[min_index:]):
            if img._allocation_uv_xywh is not None:
                rects[i] = img._allocation_uv_xywh

        # Use ring-buffered staging buffer for this frame
        staging_buf = self._uv_rect_array_staging_ring[frame_index]

        # Upload rect array with offset
        staging_buf.memory.write(data=rects)
        offset_bytes = min_index * UV_RECT_DTYPE.itemsize
        command_encoder.copy_buffer_to_buffer(
            src=staging_buf,
            dst=self._uv_rect_array_device_buf,
            size=rects.nbytes,
            dst_offset=offset_bytes,
        )


UV_RECT_DTYPE = np.dtype(
    [
        ("x", np.float32),  # 0 <= x < 1
        ("y", np.float32),  # int(y) is the page_index, fmod(y, 1) is the UV y
        ("w", np.float32),  # 0 < x_uv + w <= 1
        ("h", np.float32),  # 0 < y_uv + h <= 1
    ]
)


#
# Renderer2d
#


class Renderer2d(BaseResource):
    """
    A GPU-accelerated 2D renderer that just draws textured quads orthographically
    projected into screen space. Takes a `Canvas` as input, which describes the quads
    to draw.

    All coordinates and dimensions are PHYSICAL pixels, NOT LOGICAL pixels.
    The "Canvas" class takes care of converting logical pixels to physical pixels with
    a configured HiDPI or LoDPI scale factor.
    """

    QUAD_LIST_HEADER_DTYPE = np.dtype(
        [
            ("count", np.uint32),
            ("_rsv0", np.uint32),
            ("_rsv1", np.uint32),
            ("_rsv2", np.uint32),
        ]
    )
    QUAD_DTYPE = np.dtype(
        [
            ("dst_px", np.int32, (4, 2)),
            ("src_uv", np.float32, (4, 2)),
            ("color", np.float32, (4,)),
            ("border_color", np.float32, (4,)),
            ("border_thickness", np.uint32, (4,)),
            ("height", np.float32),
            ("image_id", np.uint32),
            ("flags", np.uint32),
            ("_rsv0", np.uint32),
        ]
    )
    QUAD_FLAG_USE_LINEAR_SAMPLER = 0x1

    renderer: "Renderer"
    gpu_device: GpuDevice

    # Shaders and pipeline:
    _vertex_shader: GpuShader
    _fragment_shader: GpuShader
    _descriptor_set_layout: GpuDescriptorSetLayout
    _pipeline_layout: GpuPipelineLayout
    _cached_gpu_pipeline: GpuPipeline | None
    _cached_depth_image: GpuImage | None

    # Quad array ring buffers:
    _quad_capacity: int
    _quad_array_buffer_ring: GpuBufferRing | None
    _quad_array_header_buffer_ring: GpuBufferRing | None
    _descriptor_set_ring: GpuDescriptorSetRing | None

    def __init__(self, *, renderer: "Renderer"):
        super().__init__(parent_resource=renderer)
        self.renderer = renderer
        self.gpu_device = renderer._gpu_device

        self._vertex_shader = self._new_vertex_shader()
        self._fragment_shader = self._new_fragment_shader()
        self._descriptor_set_layout = self._new_descriptor_set_layout()
        self._pipeline_layout = self._new_pipeline_layout()
        self._cached_gpu_pipeline = None
        self._cached_depth_image = None

        self._quad_capacity = 0
        self._quad_array_buffer_ring = None
        self._quad_array_header_buffer_ring = None
        self._descriptor_set_ring = None

    def _on_dispose_resource(self) -> None:
        self._vertex_shader.dispose_resource()
        self._fragment_shader.dispose_resource()

        self._pipeline_layout.dispose_resource()
        self._descriptor_set_layout.dispose_resource()

        if self._descriptor_set_ring is not None:
            self._descriptor_set_ring.dispose_resource()
        if self._quad_array_header_buffer_ring is not None:
            self._quad_array_header_buffer_ring.dispose_resource()
        if self._quad_array_buffer_ring is not None:
            self._quad_array_buffer_ring.dispose_resource()

        if self._cached_depth_image is not None:
            self._cached_depth_image.dispose_resource()
        if self._cached_gpu_pipeline is not None:
            self._cached_gpu_pipeline.dispose_resource()

    def _maybe_realloc_quad_array_buffers(self, capacity: int):
        # Early out if current capacity is sufficient
        if capacity <= self._quad_capacity:
            return

        # Dispose old quad array resources
        #

        if self._descriptor_set_ring is not None:
            self._descriptor_set_ring.dispose_resource()
        if self._quad_array_header_buffer_ring is not None:
            self._quad_array_header_buffer_ring.dispose_resource()
        if self._quad_array_buffer_ring is not None:
            self._quad_array_buffer_ring.dispose_resource()

        # Allocate new quad array resources
        #

        # Compute new capacity:
        new_capacity = max(8, round_up_to_po2(capacity))
        self._quad_capacity = new_capacity
        ring_capacity = self.renderer._max_frames_in_flight

        # Create quad array header ring buffer:
        self._quad_array_header_buffer_ring = GpuBufferRing(
            device=self.gpu_device,
            meta=GpuBufferMeta(
                element_count=1,
                element_dtype=Renderer2d.QUAD_LIST_HEADER_DTYPE,
            ),
            ring_capacity=ring_capacity,
            device_usages=["uniform", "copy-dst"],
        )

        # Create quad array ring buffer:
        self._quad_array_buffer_ring = GpuBufferRing(
            device=self.gpu_device,
            meta=GpuBufferMeta(
                element_count=new_capacity,
                element_dtype=Renderer2d.QUAD_DTYPE,
            ),
            ring_capacity=ring_capacity,
            device_usages=["storage", "copy-dst"],
        )

        # Create descriptor set ring:
        # Capture the ring buffers explicitly to satisfy type checker
        quad_buffer_ring = self._quad_array_buffer_ring
        header_buffer_ring = self._quad_array_header_buffer_ring
        self._descriptor_set_ring = GpuDescriptorSetRing(
            device=self.gpu_device,
            layout=self._descriptor_set_layout,
            ring_capacity=ring_capacity,
            bindings_fn=lambda i: {
                "quadArray": quad_buffer_ring.get_device_buffer(i),
                "quadArrayHeader": header_buffer_ring.get_device_buffer(i),
            },
        )

    def draw(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        quads: "QuadArray",
        target: GpuImage,
        frame_index: int,
    ):
        """
        Render the given array of quads to the given target `GpuImage`.

        The quads array should have dtype RENDERER_QUAD_DTYPE.
        """

        assert "color-attachment" in target.usages

        # Get or create pipeline and depth image for this target:
        gpu_pipeline = self._get_gpu_pipeline(target=target)
        depth_image = self._get_depth_image(width=target.width, height=target.height)

        # Ensure quad array buffers have enough capacity:
        self._maybe_realloc_quad_array_buffers(len(quads))

        # Write quad array header buffer:
        assert self._quad_array_header_buffer_ring is not None
        uniform_array = np.empty((1,), dtype=Renderer2d.QUAD_LIST_HEADER_DTYPE)
        uniform_array[0]["count"] = int(len(quads))
        self._quad_array_header_buffer_ring.write(frame_index, uniform_array)
        self._quad_array_header_buffer_ring.flush(frame_index, command_encoder)

        # Write quad array buffer:
        assert self._quad_array_buffer_ring is not None
        self._quad_array_buffer_ring.write(frame_index, quads)
        self._quad_array_buffer_ring.flush(
            frame_index, command_encoder, size=quads.nbytes
        )

        self._draw(
            encoder=command_encoder,
            pipeline=gpu_pipeline,
            depth_image=depth_image,
            target=target,
            instance_count=len(quads),
            frame_index=frame_index,
        )

    def _new_vertex_shader(self) -> GpuShader:
        return GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "r2d.vert.spv",
            stage="vertex",
        )

    def _new_fragment_shader(self) -> GpuShader:
        return GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "r2d.frag.spv",
            stage="fragment",
        )

    def _new_descriptor_set_layout(self) -> GpuDescriptorSetLayout:
        return GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "quadArray": GpuDescriptorSetLayoutBinding(
                        type="storage-buffer",
                        stages=["vertex", "fragment"],
                    ),
                    "quadArrayHeader": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["vertex", "fragment"],
                    ),
                }.items()
            ),
        )

    def _new_pipeline_layout(self) -> GpuPipelineLayout:
        return GpuPipelineLayout(
            device=self.gpu_device,
            descriptor_set_layouts=[
                self.renderer._basic_uniform._descriptor_set_layout,
                self.renderer._atlas._descriptor_set_layout,
                self._descriptor_set_layout,
            ],
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

    def _draw(
        self,
        *,
        encoder: GpuCommandEncoder,
        pipeline: GpuPipeline,
        depth_image: GpuImage,
        target: GpuImage,
        instance_count: int,
        frame_index: int,
    ):
        # Get the descriptor set for BasicUniform for this frame
        basic_uniform_ds = self.renderer._basic_uniform._descriptor_set_ring.get(
            frame_index
        )

        # Transition depth image to depth-stencil-attachment-optimal before rendering
        encoder.transition_image_layout(
            image=depth_image,
            layout="depth-stencil-attachment-optimal",
        )

        with encoder.render(
            color_attachment=target,
            depth_attachment=depth_image,
            clear_color="black",
        ) as render_pass:
            render_pass.bind_pipeline(pipeline=pipeline)
            render_pass.bind_descriptor_set(
                set_index=0,
                descriptor_set=basic_uniform_ds,
            )
            render_pass.bind_descriptor_set(
                set_index=1,
                descriptor_set=self.renderer._atlas._descriptor_set,
            )

            if instance_count > 0:
                assert self._descriptor_set_ring is not None
                render_pass.bind_descriptor_set(
                    set_index=2,
                    descriptor_set=self._descriptor_set_ring.get(frame_index),
                )
                render_pass.draw(
                    vertex_count=6,
                    first_vertex=0,
                    instance_count=instance_count,
                    first_instance=0,
                )


assert Renderer2d.QUAD_LIST_HEADER_DTYPE.itemsize == 4 * 4
assert Renderer2d.QUAD_DTYPE.itemsize == 32 * 4


class QuadArray(np.ndarray):
    def __new__(cls, shape: tuple[int, ...] | int) -> "QuadArray":
        return np.zeros(shape, dtype=Renderer2d.QUAD_DTYPE).view(cls)


#
# Canvas: 2D scene
#


class Canvas:
    """
    A list of textured quads to be rendered.
    """

    def __init__(self, renderer: Renderer, capacity: int = 64):
        self.renderer = renderer
        self._quad_array = QuadArray(capacity)
        self._quad_count = 0

        # Font rendering support
        self._all_fonts: list[Font] = ["sans-serif", "serif", "monospaced"]
        self._hb_font_map: dict[Font, hb.Font] = {
            font: self._load_harfbuzz_font(font) for font in self._all_fonts
        }
        self._ft_face_map: dict[Font, ft.Face] = {
            font: self._load_freetype_font(font) for font in self._all_fonts
        }
        self._ft_image_cache: dict[
            tuple[Font, int, int, int],
            tuple[Image | None, int, int],
        ] = {}
        self._ft_weight_axis_index: dict[Font, int] = {}

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

    #
    # Getters and properties:
    #

    def __len__(self) -> int:
        return self._quad_count

    @property
    def capacity(self) -> int:
        return len(self._quad_array)

    def as_quad_array(self) -> QuadArray:
        return self._quad_array[: self._quad_count].view(QuadArray)

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
        new_array = QuadArray(new_capacity)
        new_array[: len(self._quad_array)] = self._quad_array
        self._quad_array = new_array

    #
    # font loading helpers
    #

    @staticmethod
    def _get_font_file_path(font: Font) -> Path:
        return {
            "sans-serif": (BUNDLED_DATA_PATH / "data/font-Inter_4_1/InterVariable.ttf"),
            "serif": (BUNDLED_DATA_PATH / "data/font-Lora/Lora-VariableFont_wght.ttf"),
            "monospaced": (
                BUNDLED_DATA_PATH
                / "data/font-SourceCodePro/SourceCodePro-VariableFont_wght.ttf"
            ),
        }[font]

    @staticmethod
    def _load_harfbuzz_font(font: Font) -> hb.Font:
        file_path = Canvas._get_font_file_path(font)

        with open(file_path, "rb") as f:
            hb_blob = f.read()

        hb_face = hb.Face(hb_blob)
        hb_font = hb.Font(hb_face)
        return hb_font

    @staticmethod
    def _load_freetype_font(font: Font) -> ft.Face:
        file_path = Canvas._get_font_file_path(font)
        ft_face = ft.Face(str(file_path))
        return ft_face

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
        border_thickness: tuple[int, int, int, int] = (0, 0, 0, 0),  # TRBL
        image: Image | None = None,
        _dip: bool = True,
    ) -> None:
        """
        Adds a single quad using logical (device-independent) pixel coordinates.

        Coordinates are converted to physical pixels internally based on the
        renderer's scale factor.
        """

        # If dip is True, convert to physical pixel coordinates
        if _dip:
            scale = self.renderer.scale

            # Resolve src_wh (in physical pixels, from image)
            src_wh_resolved = Canvas._resolve_src_wh(dst_wh, src_wh, image)

            # Compute dst_wh in logical pixels
            dst_wh_logical = Canvas._eval_dst_px_wh(dst_wh, src_wh_resolved)

            # Convert logical to physical
            dst_xy = (int(dst_xy[0] * scale), int(dst_xy[1] * scale))
            dst_wh = (
                int(dst_wh_logical[0] * scale),
                int(dst_wh_logical[1] * scale),
            )
            border_thickness = (
                int(border_thickness[0] * scale),
                int(border_thickness[1] * scale),
                int(border_thickness[2] * scale),
                int(border_thickness[3] * scale),
            )
        else:
            src_wh_resolved = Canvas._resolve_src_wh(dst_wh, src_wh, image)
            dst_wh = Canvas._eval_dst_px_wh(dst_wh, src_wh_resolved)

        # Ensure capacity
        if len(self) >= self.capacity:
            self.reserve(1 + len(self))
        assert len(self) < self.capacity

        # Reserve index
        index = self._quad_count
        self._quad_count += 1

        # Physical pixel coordinates - no scaling
        x, y = dst_xy
        w, h = dst_wh

        self._quad_array[index]["dst_px"][0] = [x, y]
        self._quad_array[index]["dst_px"][1] = [x + w, y]
        self._quad_array[index]["dst_px"][2] = [x + w, y + h]
        self._quad_array[index]["dst_px"][3] = [x, y + h]

        # write: src_uv
        resolved_src_wh = (
            src_wh
            if src_wh is not None
            else (image.px_width if image else w, image.px_height if image else h)
        )
        uv_xywh = Canvas._eval_src_uv_xywh(src_xy, resolved_src_wh, image)
        uv_x, uv_y, uv_w, uv_h = uv_xywh
        self._quad_array[index]["src_uv"][0] = [uv_x, uv_y]
        self._quad_array[index]["src_uv"][1] = [uv_x + uv_w, uv_y]
        self._quad_array[index]["src_uv"][2] = [uv_x + uv_w, uv_y + uv_h]
        self._quad_array[index]["src_uv"][3] = [uv_x, uv_y + uv_h]

        # write: image_id
        quad_image = image or self.renderer._default_white_image
        self._quad_array[index]["image_id"] = quad_image.image_id

        # write: flags (bit 0: 1 for linear, 0 for nearest)
        flags = 0
        if quad_image.sampler == "linear":
            flags |= Renderer2d.QUAD_FLAG_USE_LINEAR_SAMPLER
        self._quad_array[index]["flags"] = flags

        # write: color, border_color, border_thickness
        self._quad_array[index]["color"] = color
        self._quad_array[index]["border_color"] = border_color
        self._quad_array[index]["border_thickness"] = list(border_thickness)

        # write: height
        self._quad_array[index]["height"] = float(index)

    #
    # add_text
    #

    def add_text(
        self,
        *,
        text: str,
        font: "Font",
        dst_xy: tuple[int, int],
        dst_wh: tuple[int, int],
        font_size_px: int = 16,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        wrap: bool = True,
        font_weight: int = 400,
        horizontal_alignment: HorizontalAlignment = "left",
        vertical_alignment: VerticalAlignment = "top",
        optical_alignment: bool = True,
    ):
        """
        Adds quads for rendering the given text string with the given font.

        Supports horizontal and vertical alignment within the destination rectangle.

        If `optical_alignment` is True, aligns based on the visible ink bounds
        rather than the logical metric bounds.
        """
        if not text:
            return

        # Strategy: Accumulate pen position in 26.6 fixed-point to preserve
        # subpixel precision. Only round to integer pixels when placing glyphs.
        # This prevents accumulated rounding error.

        renderer_scale = self.renderer.scale
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

        lines = self._layout_text_lines(infos, positions, text, wrap, dst_w_26_6)

        total_text_height_26_6 = len(lines) * height_26_6

        # Vertical alignment
        start_y_26_6 = dst_y_26_6
        if vertical_alignment == "middle":
            start_y_26_6 += (dst_h_26_6 - total_text_height_26_6) // 2
        elif vertical_alignment == "bottom":
            start_y_26_6 += dst_h_26_6 - total_text_height_26_6

        pen_y_26_6 = start_y_26_6 + ascender_26_6

        for start_idx, end_idx, line_width_26_6 in lines:
            # Horizontal alignment
            pen_x_26_6 = dst_x_26_6

            if optical_alignment:
                min_ink, max_ink = self._get_line_optical_bounds(
                    font, infos, positions, start_idx, end_idx
                )
                optical_width = max_ink - min_ink

                if horizontal_alignment == "center":
                    pen_x_26_6 += (dst_w_26_6 - optical_width) // 2 - min_ink
                elif horizontal_alignment == "right":
                    pen_x_26_6 += dst_w_26_6 - max_ink
                elif horizontal_alignment == "left":
                    pen_x_26_6 -= min_ink
            else:
                if horizontal_alignment == "center":
                    pen_x_26_6 += (dst_w_26_6 - line_width_26_6) // 2
                elif horizontal_alignment == "right":
                    pen_x_26_6 += dst_w_26_6 - line_width_26_6

            for i in range(start_idx, end_idx):
                info = infos[i]
                pos = positions[i]

                codepoint = info.codepoint
                # cluster = info.cluster

                # HarfBuzz positions are in 26.6 fixed point - use directly
                x_advance_26_6 = pos.x_advance
                y_advance_26_6 = pos.y_advance
                x_offset_26_6 = pos.x_offset
                y_offset_26_6 = pos.y_offset

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
                        # Clipping offset and size in physical pixels (for src rect)
                        src_off_x = ix_phys - qx_phys
                        src_off_y = iy_phys - qy_phys
                        src_w = ir_phys - ix_phys
                        src_h = ib_phys - iy_phys

                        # Ensure we don't exceed the image bounds
                        src_w = min(src_w, qw_phys - src_off_x)
                        src_h = min(src_h, qh_phys - src_off_y)

                        # Add quad with physical pixel coordinates
                        self.add_quad(
                            dst_xy=(ix_phys, iy_phys),
                            dst_wh=(src_w, src_h),
                            src_xy=(src_off_x, src_off_y),
                            src_wh=(src_w, src_h),
                            color=color,
                            image=image,
                            _dip=False,
                        )

                # Accumulate in 26.6 to preserve precision
                pen_x_26_6 += x_advance_26_6
                pen_y_26_6 += y_advance_26_6

            pen_y_26_6 += height_26_6

    @staticmethod
    def _resolve_src_wh(
        dst_wh: tuple[int, int] | None,
        src_wh: tuple[int, int] | None,
        image: Image | None,
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
        image: Image | None,
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
    # Text layout
    #

    def _layout_text_lines(
        self,
        infos: list[hb.GlyphInfo],
        positions: list[hb.GlyphPosition],
        text: str,
        wrap: bool,
        max_width_26_6: int,
    ) -> list[tuple[int, int, int]]:
        lines = []
        line_start_index = 0
        current_line_width_26_6 = 0

        i = 0
        while i < len(infos):
            info = infos[i]
            pos = positions[i]
            cluster = info.cluster
            char = text[cluster] if cluster < len(text) else " "

            if char == "\n":
                lines.append((line_start_index, i, current_line_width_26_6))
                line_start_index = i + 1
                current_line_width_26_6 = 0
                i += 1
                continue

            if wrap and not char.isspace():
                is_word_start = False
                if i == line_start_index:
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
                        if c_char.isspace() or c_char == "\n":
                            break
                        word_width_26_6 += positions[j].x_advance

                    if (
                        current_line_width_26_6 + word_width_26_6 > max_width_26_6
                    ) and (current_line_width_26_6 > 0):
                        lines.append((line_start_index, i, current_line_width_26_6))
                        line_start_index = i
                        current_line_width_26_6 = 0

            current_line_width_26_6 += pos.x_advance
            i += 1

        lines.append((line_start_index, len(infos), current_line_width_26_6))
        return lines

    def _get_line_optical_bounds(
        self,
        font: Font,
        infos: list[hb.GlyphInfo],
        positions: list[hb.GlyphPosition],
        start_idx: int,
        end_idx: int,
    ) -> tuple[int, int]:
        """
        Returns (min_x, max_x) of the ink bounds for the given range of glyphs,
        relative to the start of the line (pen_x = 0).
        Values are in 26.6 fixed point.
        """
        hb_font = self._hb_font_map[font]
        min_x = 2147483647
        max_x = -2147483648

        pen_x = 0
        has_ink = False

        for i in range(start_idx, end_idx):
            info = infos[i]
            pos = positions[i]

            extents = hb_font.get_glyph_extents(info.codepoint)

            # If width/height are 0, it's likely whitespace or invisible
            if extents.width != 0 and extents.height != 0:
                # Glyph origin relative to pen
                # HarfBuzz extents are relative to glyph origin
                # x_offset is applied to glyph origin

                # Ink rect:
                # left = pen_x + x_offset + x_bearing
                # right = left + width

                left = pen_x + pos.x_offset + extents.x_bearing
                right = left + extents.width

                if left < min_x:
                    min_x = left
                if right > max_x:
                    max_x = right
                has_ink = True

            pen_x += pos.x_advance

        if not has_ink:
            return 0, 0

        return min_x, max_x

    #
    # Glyph images
    #

    def _set_freetype_weight(self, font: Font, weight: int) -> None:
        if font not in self._ft_weight_axis_index:
            return

        face = self._ft_face_map[font]
        axis_idx = self._ft_weight_axis_index[font]
        coords = list(face.get_var_design_coords())
        coords[axis_idx] = float(weight)
        face.set_var_design_coords(coords)

    def _shape_text(
        self,
        font: Font,
        text: str,
        font_size_px: int,
        font_weight: int,
    ) -> tuple[list[hb.GlyphInfo], list[hb.GlyphPosition]]:
        hb_font = self._hb_font_map[font]
        # Use renderer's scale factor for HiDPI support
        renderer_scale = self.renderer.scale
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
        font: Font,
        glyph_index: int,
        font_size_px: int,
        font_weight: int,
    ) -> tuple[Image | None, int, int]:
        # Use renderer's scale factor for HiDPI support
        renderer_scale = self.renderer.scale
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
        image = Image(renderer=self.renderer, data=data, sampler="nearest")
        result = (image, bitmap_left, bitmap_top)
        self._ft_image_cache[image_cache_key] = result

        # Done:
        return result


#
# Rendererer3d
#


class Renderer3d(BaseResource):
    """
    A GPU-accelerated 3D renderer.
    """

    def __init__(self, *, renderer: Renderer):
        super().__init__(parent_resource=renderer)
        self._renderer = renderer
        self._gpu_device = renderer._gpu_device


class Scene:
    pass
