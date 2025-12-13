__all__ = [
    "Renderer",
    "RendererContext",
    "RendererImage",
    "RendererQuadArray",
]

from collections import OrderedDict
from typing import Literal, TypeAlias

import numpy as np

from .basic import BaseResource, next_po2
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
            data=np.ones((1, 1, 4), dtype=np.float32),
        )
        self._atlases[4].insert(self.default_white_image)
        self.default_white_image_atlas = self._atlases[4]

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

        for atlas in self._atlases.values():
            atlas.flush()

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
    index: int
    _uv_xywh: tuple[float, float, float, float] | None
    _px_xywh: tuple[int, int, int, int] | None

    def __init__(self, *, data: np.ndarray):
        assert data.ndim in (2, 3)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        self.data = data.copy()
        self.data.flags.writeable = False
        self.index = -1
        self._uv_xywh = None
        self._px_xywh = None

    @property
    def uv_xywh(self) -> tuple[float, float, float, float]:
        if self._uv_xywh is None:
            raise RuntimeError("Image not allocated (call flush())")
        return self._uv_xywh

    @property
    def px_xywh(self) -> tuple[int, int, int, int]:
        if self._px_xywh is None:
            raise RuntimeError("Image not allocated (call flush())")
        return self._px_xywh

    @property
    def page_index(self) -> int:
        if self._uv_xywh is None:
            raise RuntimeError("Image not allocated (call flush())")
        return int(self._uv_xywh[1])


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

        self.images: list[RendererImage] = []
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

    def insert(self, image: RendererImage) -> int:
        assert image.data.shape[2] == self.channels
        image.index = len(self.images)
        self.images.append(image)
        self._unallocated_images.append(image)
        return image.index

    def flush(self):
        if not self._unallocated_images:
            return

        self._unallocated_images.sort(
            key=lambda img: img.data.shape[0] * img.data.shape[1], reverse=True
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
            dirty_images = self.images  # All images dirty

        self._upload_pages(dirty_images)
        self._upload_rects()
        self._unallocated_images.clear()

    def _alloc_image(self, img: RendererImage) -> bool:
        h_px, w_px = img.data.shape[0], img.data.shape[1]
        w = w_px / self.page_size
        h = h_px / self.page_size

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
        if self._page_count < self.max_pages:
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
        img._uv_xywh = (x, float(page_index) + y, w, h)

        x_px = int(x * self.page_size)
        y_px = int(y * self.page_size)
        w_px = int(w * self.page_size)
        h_px = int(h * self.page_size)
        img._px_xywh = (x_px, y_px, w_px, h_px)

        # Update CPU pixel data
        self._page_pixel_data[page_index, y_px : y_px + h_px, x_px : x_px + w_px] = (
            img.data
        )

    def _compact(self):
        # Clear pages
        self._page_cursor_array.fill(0)
        self._page_count = 0
        self._page_pixel_data.fill(0)

        # Sort ALL images by size
        all_images = sorted(
            self.images,
            key=lambda img: img.data.shape[0] * img.data.shape[1],
            reverse=True,
        )

        for img in all_images:
            if not self._alloc_image(img):
                raise MemoryError("Out of atlas memory during compaction")

    def _upload_pages(self, dirty_images: list[RendererImage]):
        if not dirty_images:
            return

        staging_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=self.page_size * self.page_size * self.channels,
                element_dtype=np.float32,
            ),
        )

        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="transfer",
        )

        # Group by page
        pages: dict[int, list[RendererImage]] = {}
        for img in dirty_images:
            pages.setdefault(img.page_index, []).append(img)

        for page_index, images in pages.items():
            offset = 0
            for img in images:
                staging_buffer.memory.write(data=img.data, offset=offset)

                command_encoder.transition_image_layout(
                    image=self._page_gpu_image_list[page_index],
                    layout="transfer-dst-optimal",
                )
                command_encoder.copy_buffer_to_image(
                    src=staging_buffer,
                    dst=self._page_gpu_image_list[page_index],
                    buffer_offset=offset,
                    image_offset=(img.px_xywh[0], img.px_xywh[1], 0),
                    image_extent=(img.px_xywh[2], img.px_xywh[3], 1),
                )
                offset += img.data.nbytes

            # Must submit per page because we reuse the staging buffer
            command_encoder.submit().wait()
            command_encoder = GpuCommandEncoder(
                device=self.gpu_device,
                queue_type="transfer",
            )

        staging_buffer.dispose()

    def _upload_rects(self):
        rects = np.zeros((len(self.images),), dtype=UV_RECT_DTYPE)
        for i, img in enumerate(self.images):
            if img._uv_xywh is not None:
                rects[i] = img._uv_xywh

        staging_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta.from_array(rects),
        )
        staging_buffer.memory.write(data=rects)

        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="transfer",
        )
        command_encoder.copy_buffer_to_buffer(
            src=staging_buffer,
            dst=self.rects_buffer,
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
        new_capacity = max(8, next_po2(capacity))
        self._quad_capacity = new_capacity

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
        self._quad_array_device_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=new_capacity,
                element_dtype=RendererQuadPipeline.QUAD_DTYPE,
            ),
        )
        self._quad_array_staging_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=new_capacity,
                element_dtype=RendererQuadPipeline.QUAD_DTYPE,
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
        uniform_data = np.zeros((1,), dtype=RendererQuadPipeline.COMMON_UNIFORM_DTYPE)
        uniform_data[0]["framebuffer_size_px"] = [target.width, target.height]
        uniform_data[0]["total_quad_count"] = float(total_quad_count)
        uniform_data[0]["atlas_size_px"] = self.renderer._atlases[4].page_size
        self._common_uniform_staging_buf.memory.write(data=uniform_data)
        encoder.copy_buffer_to_buffer(
            src=self._common_uniform_staging_buf,
            dst=self._common_uniform_device_buf,
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
                descriptor_set=self._common_uniform_descriptor_set,
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
