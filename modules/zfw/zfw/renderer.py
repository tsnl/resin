__all__ = [
    "RENDERER_QUADS_BATCH_UNIFORM_DTYPE",
    "RENDERER_QUADS_COMMON_UNIFORM_DTYPE",
    "RENDERER_QUAD_DTYPE",
    "Renderer",
    "RendererAtlas",
    "RendererContext",
    "RendererImage",
    "RendererQuadArray",
]

from collections import OrderedDict

import numpy as np

from .basic import BaseResource
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
    GpuSamplerAddressMode,
    GpuSamplerFilter,
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

    _quads_vertex_shader: GpuShader
    _quads_fragment_shader: GpuShader
    _quads_pipeline_layout: GpuPipelineLayout
    _quads_common_uniform_staging_buf: GpuBuffer
    _quads_common_uniform_device_buf: GpuBuffer
    _quads_common_uniform_descriptor_set: GpuDescriptorSet
    _quads_cached_pipeline: GpuPipeline | None
    _quads_cached_depth_image: GpuImage | None
    _quads_cached_gpu_desc_sets: dict[int, "RendererQuadBatch"]
    _atlases: list["RendererAtlas"]

    def __init__(
        self,
        *,
        context: RendererContext,
        gpu_device: GpuDevice,
    ):
        super().__init__(parent=context)

        self.context = context
        self.gpu_device = gpu_device

        self._atlases = []

        self.default_white_image_atlas = RendererAtlas(
            renderer=self,
            data=np.full((8, 8, 4), 0xFF, dtype=np.uint8),
            image_rect_map={"default": (0, 0, 8, 8)},
        )
        self.default_white_image = self.default_white_image_atlas["default"]

        self._quads_vertex_shader = self._new_vertex_shader()
        self._quads_fragment_shader = self._new_fragment_shader()
        self._quads_pipeline_layout = self._new_pipeline_layout()
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
        self._quads_cached_gpu_desc_sets = {}

    def _on_dispose(self) -> None:
        self._quads_vertex_shader.dispose()
        self._quads_fragment_shader.dispose()

        self._quads_pipeline_layout.dispose()

        self._quads_common_uniform_descriptor_set.dispose()

        self._quads_common_uniform_staging_buf.dispose()
        self._quads_common_uniform_device_buf.dispose()

        for gpu_batch in self._quads_cached_gpu_desc_sets.values():
            gpu_batch.dispose()

        if self._quads_cached_depth_image is not None:
            self._quads_cached_depth_image.dispose()
        if self._quads_cached_pipeline is not None:
            self._quads_cached_pipeline.dispose()

        # Dispose default white image and atlas:
        self.default_white_image_atlas.dispose()

    def register_new_atlas(self, atlas: "RendererAtlas") -> int:
        """Register a new atlas and return its atlas_id."""
        atlas_id = len(self._atlases)
        self._atlases.append(atlas)
        return atlas_id

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

        The quads array should have dtype RENDERER_QUAD_DTYPE and contain
        atlas_id fields indicating which atlas each quad belongs to.
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
        gpu_batches = self._get_gpu_batches_from_quads(
            quads=quads,
            encoder=command_encoder,
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
            gpu_batches=gpu_batches,
            target=target,
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

    def _new_vertex_shader(self) -> GpuShader:
        return GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "zfw" / "r2d.vert.spv",
            stage="vertex",
        )

    def _new_fragment_shader(self) -> GpuShader:
        return GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "zfw" / "r2d.frag.spv",
            stage="fragment",
        )

    def _new_pipeline_layout(self) -> GpuPipelineLayout:
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
                GpuDescriptorSetLayout(
                    device=self.gpu_device,
                    bindings=OrderedDict(
                        {
                            "atlasTexture": GpuDescriptorSetLayoutBinding(
                                type="combined-image-sampler",
                                stages=["vertex", "fragment"],
                            ),
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
                element_count=1, element_dtype=RENDERER_QUADS_COMMON_UNIFORM_DTYPE
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

    def _get_gpu_batch_dict(
        self,
        *,
        quads: np.ndarray,
        encoder: GpuCommandEncoder,
    ) -> dict[int, "RendererQuadBatch"]:
        """
        Group quads by atlas_id and get or create a GPU batch for each group.
        Returns a dictionary mapping atlas_id to RendererQuadBatch.
        """
        if len(quads) == 0:
            return {}

        # Create a copy and sort by atlas_id
        sorted_quads = quads.copy()
        sorted_quads.sort(order="atlas_id")

        # Find boundaries between different atlas_ids
        gpu_batches: dict[int, "RendererQuadBatch"] = {}

        if len(sorted_quads) > 0:
            # Use broadcasting to find where atlas_id changes
            atlas_ids = sorted_quads["atlas_id"]
            changes = np.concatenate(
                ([0], np.where(np.diff(atlas_ids) != 0)[0] + 1, [len(atlas_ids)])
            )

            for i in range(len(changes) - 1):
                start_idx = changes[i]
                end_idx = changes[i + 1]
                quad_batch = sorted_quads[start_idx:end_idx]
                atlas_id = int(quad_batch[0]["atlas_id"])

                # Get the atlas from the registry
                atlas = self._atlases[atlas_id]
                instance_count = end_idx - start_idx

                # Get or create GPU batch
                gpu_batch = self._get_gpu_batch(
                    atlas=atlas,
                    quads=quad_batch,
                    instance_count=instance_count,
                    command_encoder=encoder,
                )
                gpu_batches[atlas_id] = gpu_batch

        # Only keep cached GPU batches that are still in use:
        self._quads_cached_gpu_desc_sets = gpu_batches

        return gpu_batches

    def _get_gpu_batches_from_quads(
        self,
        *,
        quads: np.ndarray,
        encoder: GpuCommandEncoder,
    ) -> dict[int, "RendererQuadBatch"]:
        """Process quads and return GPU batches organized by atlas_id."""
        return self._get_gpu_batch_dict(quads=quads, encoder=encoder)

    def _get_gpu_batch(
        self,
        *,
        atlas: "RendererAtlas",
        quads: np.ndarray,
        instance_count: int,
        command_encoder: GpuCommandEncoder,
    ) -> "RendererQuadBatch":
        atlas_id = atlas.atlas_id

        if gpu_batch := self._get_cached_gpu_batch(
            atlas_id=atlas_id, instance_count=instance_count
        ):
            gpu_batch.write(quads=quads, command_encoder=command_encoder)
            return gpu_batch

        gpu_batch = self._new_gpu_batch(atlas=atlas, instance_count=instance_count)
        gpu_batch.write(quads=quads, command_encoder=command_encoder)

        self._quads_cached_gpu_desc_sets[atlas_id] = gpu_batch

        return gpu_batch

    def _get_cached_gpu_batch(
        self,
        *,
        atlas_id: int,
        instance_count: int,
    ) -> "RendererQuadBatch | None":
        gpu_batch = self._quads_cached_gpu_desc_sets.get(atlas_id)
        if gpu_batch is None:
            return None
        if gpu_batch.capacity < instance_count:
            return None
        return gpu_batch

    def _new_gpu_batch(
        self,
        *,
        atlas: "RendererAtlas",
        instance_count: int,
    ) -> "RendererQuadBatch":
        # Start with at least 8 quads, use power-of-2 for capacity
        capacity = max(8, _next_po2(instance_count))

        uniform_device_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["uniform", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=1,
                element_dtype=RENDERER_QUADS_BATCH_UNIFORM_DTYPE,
            ),
        )
        uniform_staging_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=1,
                element_dtype=RENDERER_QUADS_BATCH_UNIFORM_DTYPE,
            ),
        )
        quads_device_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=capacity,
                element_dtype=RENDERER_QUAD_DTYPE,
            ),
        )
        quads_staging_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=capacity,
                element_dtype=RENDERER_QUAD_DTYPE,
            ),
        )
        binding = GpuDescriptorSet(
            device=self.gpu_device,
            layout=self._quads_pipeline_layout.descriptor_set_layouts[1],
            bindings={
                "atlasTexture": (atlas.gpu_image, atlas.gpu_sampler),
                "quads": quads_device_buf,
                "batchUniform": uniform_device_buf,
            },
        )
        return RendererQuadBatch(
            renderer=self,
            atlas=atlas,
            instance_count=instance_count,
            capacity=capacity,
            quads_staging_buf=quads_staging_buf,
            quads_device_buf=quads_device_buf,
            uniform_staging_buf=uniform_staging_buf,
            uniform_device_buf=uniform_device_buf,
            descriptor_set=binding,
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
        uniform_data = np.zeros((1,), dtype=RENDERER_QUADS_COMMON_UNIFORM_DTYPE)
        uniform_data[0]["framebuffer_size_px"] = [target.width, target.height]
        uniform_data[0]["total_quad_count"] = float(total_quad_count)
        self._quads_common_uniform_staging_buf.memory.write(data=uniform_data)
        encoder.copy_buffer_to_buffer(
            src=self._quads_common_uniform_staging_buf,
            dst=self._quads_common_uniform_device_buf,
            size=RENDERER_QUADS_COMMON_UNIFORM_DTYPE.itemsize,
        )

    def _draw(
        self,
        *,
        encoder: GpuCommandEncoder,
        pipeline: GpuPipeline,
        depth_image: GpuImage,
        gpu_batches: dict[int, "RendererQuadBatch"],
        target: GpuImage,
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

            for gpu_batch in gpu_batches.values():
                render_pass.bind_descriptor_set(
                    set_index=1,
                    descriptor_set=gpu_batch.descriptor_set,
                )
                render_pass.draw(
                    vertex_count=6,
                    instance_count=gpu_batch.instance_count,
                    first_vertex=0,
                    first_instance=0,
                )


#
# Atlas, Image API: shared by 2D and 3D renderer workflows
#


class RendererAtlas(BaseResource):
    renderer: "Renderer"
    gpu_device: GpuDevice
    gpu_image: GpuImage
    gpu_sampler: GpuSampler
    image_map: dict[str, "RendererImage"]
    atlas_id: int

    def __init__(
        self,
        *,
        renderer: "Renderer",
        data: np.ndarray,
        image_rect_map: dict[str, tuple[int, int, int, int]],
        mag_filter: GpuSamplerFilter = "linear",
        min_filter: GpuSamplerFilter = "linear",
        address_mode: GpuSamplerAddressMode = "clamp-to-edge",
    ):
        super().__init__(parent=renderer)

        assert data.ndim == 3 and data.shape[2] == 4

        self.renderer = renderer
        self.gpu_device = renderer.gpu_device

        self.gpu_image = GpuImage(
            device=renderer.gpu_device,
            usages=["texture-binding"],
            meta=GpuImageMeta.from_array(data),
        )
        self.gpu_sampler = GpuSampler(
            device=renderer.gpu_device,
            mag_filter=mag_filter,
            min_filter=min_filter,
            address_mode=address_mode,
        )

        self.image_map = {
            name: RendererImage(atlas=self, entry_name=name, rect_xywh=rect_xywh)
            for name, rect_xywh in image_rect_map.items()
        }

        self.atlas_id = renderer.register_new_atlas(self)

        self.write(data=data)

    def _on_dispose(self) -> None:
        self.gpu_sampler.dispose()
        self.gpu_image.dispose()

    def __getitem__(self, image_name: str) -> "RendererImage":
        return self.image_map[image_name]

    def write(self, *, data: np.ndarray):
        assert self.gpu_image.memory

        # Create and write to a staging buffer:
        staging_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta.from_array(data),
        )
        staging_buffer.memory.write(data=data)

        # Using a one-time command buffer, copy from the staging buffer to the image,
        # blocking until done:
        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="transfer",
        )
        command_encoder.transition_image_layout(
            image=self.gpu_image,
            layout="transfer-dst-optimal",
        )
        command_encoder.copy_buffer_to_image(
            src=staging_buffer,
            dst=self.gpu_image,
        )
        command_encoder.transition_image_layout(
            image=self.gpu_image,
            layout="texture-binding",
        )
        command_encoder.submit().wait()


class RendererImage(BaseResource):
    atlas: RendererAtlas
    entry_name: str
    width_px: int
    height_px: int
    rect_xy_xy_px: tuple[tuple[int, int], tuple[int, int]]
    rect_xy_xy_uv: tuple[tuple[float, float], tuple[float, float]]

    def __init__(
        self,
        atlas: RendererAtlas,
        entry_name: str,
        rect_xywh: tuple[int, int, int, int],
    ):
        super().__init__(parent=atlas)
        self.atlas = atlas
        self.entry_name = entry_name
        x0_px, y0_px, w_px, h_px = rect_xywh
        x1_px, y1_px = x0_px + w_px - 1, y0_px + h_px - 1
        self.width_px = w_px
        self.height_px = h_px
        self.rect_xy_xy_px = ((x0_px, y0_px), (x1_px, y1_px))
        self.rect_xy_xy_uv = (self._uv(x0_px, y0_px), self._uv(x1_px, y1_px))

    def _uv(self, x_px: int, y_px: int) -> tuple[float, float]:
        atlas_width = self.atlas.gpu_image.width
        atlas_height = self.atlas.gpu_image.height
        u = x_px / atlas_width
        v = y_px / atlas_height
        return (u, v)


class RendererQuadBatch(BaseResource):
    renderer: Renderer
    atlas: RendererAtlas
    instance_count: int
    capacity: int
    quads_device_buf: GpuBuffer
    quads_staging_buf: GpuBuffer
    uniform_device_buf: GpuBuffer
    uniform_staging_buf: GpuBuffer
    descriptor_set: GpuDescriptorSet

    def __init__(
        self,
        *,
        renderer: Renderer,
        atlas: RendererAtlas,
        instance_count: int,
        capacity: int,
        quads_device_buf: GpuBuffer,
        quads_staging_buf: GpuBuffer,
        uniform_device_buf: GpuBuffer,
        uniform_staging_buf: GpuBuffer,
        descriptor_set: GpuDescriptorSet,
    ):
        super().__init__(parent=renderer)
        self.renderer = renderer
        self.atlas = atlas
        self.instance_count = instance_count
        self.capacity = capacity
        self.quads_device_buf = quads_device_buf
        self.quads_staging_buf = quads_staging_buf
        self.uniform_device_buf = uniform_device_buf
        self.uniform_staging_buf = uniform_staging_buf
        self.descriptor_set = descriptor_set

    def _on_dispose(self) -> None:
        self.descriptor_set.dispose()
        self.uniform_staging_buf.dispose()
        self.uniform_device_buf.dispose()
        self.quads_staging_buf.dispose()
        self.quads_device_buf.dispose()

    def write(self, *, quads: np.ndarray, command_encoder: GpuCommandEncoder):
        """Write quads array to GPU buffers."""
        # Quads:
        self.quads_staging_buf.memory.write(data=quads)
        command_encoder.copy_buffer_to_buffer(
            src=self.quads_staging_buf,
            dst=self.quads_device_buf,
            size=len(quads) * RENDERER_QUAD_DTYPE.itemsize,
        )

        # Batch uniform:
        uniform_array = np.empty((1,), dtype=RENDERER_QUADS_BATCH_UNIFORM_DTYPE)
        uniform_array[0]["instance_count"] = int(len(quads))
        uniform_array[0]["is_opaque"] = 1  # TODO: support transparency
        self.uniform_staging_buf.memory.write(data=uniform_array)
        command_encoder.copy_buffer_to_buffer(
            src=self.uniform_staging_buf,
            dst=self.uniform_device_buf,
            size=RENDERER_QUADS_BATCH_UNIFORM_DTYPE.itemsize,
        )


def _next_po2(x: int) -> int:
    """Return the next power of two greater than or equal to x."""
    if x <= 0:
        return 1
    v = 1
    while v < x:
        v *= 2
    return v


RENDERER_QUADS_COMMON_UNIFORM_DTYPE = np.dtype(
    [
        ("framebuffer_size_px", np.int32, (2,)),
        ("total_quad_count", np.float32),
        ("atlas_id", np.uint32),
    ]
)
assert RENDERER_QUADS_COMMON_UNIFORM_DTYPE.itemsize == 4 * 4


RENDERER_QUAD_DTYPE = np.dtype(
    [
        ("dst_px", np.int32, (4, 2)),
        ("src_uv", np.float32, (4, 2)),
        ("color", np.float32, (4,)),
        ("border_color", np.float32, (4,)),
        ("border_thickness_px", np.uint32, (4,)),
        ("height", np.float32),
        ("atlas_id", np.uint32),
        ("_rsv0", np.uint32),
        ("_rsv1", np.uint32),
    ]
)
assert RENDERER_QUAD_DTYPE.itemsize == 32 * 4

RENDERER_QUADS_BATCH_UNIFORM_DTYPE = np.dtype(
    [
        ("instance_count", np.uint32),
        ("is_opaque", np.uint32),
        ("_rsv0", np.uint32),
        ("_rsv1", np.uint32),
    ]
)
assert RENDERER_QUADS_BATCH_UNIFORM_DTYPE.itemsize == 4 * 4

RENDERER_QUADS_BATCH_UNIFORM_DTYPE = np.dtype(
    [
        ("instance_count", np.uint32),
        ("is_opaque", np.uint32),
        ("_rsv0", np.uint32),
        ("_rsv1", np.uint32),
    ]
)
assert RENDERER_QUADS_BATCH_UNIFORM_DTYPE.itemsize == 4 * 4


class RendererQuadArray(np.recarray):
    def __new__(cls, shape: tuple[int, ...] | int) -> "RendererQuadArray":
        return np.zeros(shape, dtype=RENDERER_QUAD_DTYPE).view(cls)
