__all__ = [
    "RendererContext",
    "Renderer",
    "RendererAtlas",
    "RendererImage",
    "RendererCanvas",
]

from collections import OrderedDict

import numpy as np

from .basic import BaseResource
from .bundled_data import BUNDLED_DATA_PATH
from .excepts import LogicError
from .gpu import (
    GpuBuffer,
    GpuBufferMeta,
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
    GpuSamplerAddressMode,
    GpuSamplerFilter,
    GpuShader,
)

#
# Renderer API:
#


class RendererContext(BaseResource):
    def __init__(self, gpu_context: GpuContext):
        super().__init__(parent=gpu_context)


class Renderer(BaseResource):
    context: RendererContext
    gpu_device: GpuDevice
    default_white_image_atlas: "RendererAtlas"
    default_white_image: "RendererImage"
    renderer_2d: "Renderer2d"

    def __init__(
        self,
        *,
        context: RendererContext,
        gpu_device: GpuDevice,
    ):
        super().__init__(parent=context)

        self.context = context
        self.gpu_device = gpu_device

        self.default_white_image_atlas = RendererAtlas(
            renderer=self,
            data=np.full((32, 32, 4), 0xFF, dtype=np.uint8),
            image_rect_map={"default": (0, 0, 32, 32)},
        )
        self.default_white_image = self.default_white_image_atlas["default"]

        self.renderer_2d = Renderer2d(
            renderer=self,
            default_white_image=self.default_white_image,
        )

    def show(self, *, canvas: "RendererCanvas", target: GpuImage):
        """
        Render the given `RendererCanvas` to the given target `GpuImage`.
        """

        self.renderer_2d.show(canvas=canvas, target=target)


#
# Atlas, Image API: shared by 2D and 3D renderer workflows
#


class RendererAtlas(BaseResource):
    renderer: "Renderer"
    gpu_image: GpuImage
    gpu_sampler: GpuSampler
    image_map: dict[str, "RendererImage"]

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
        self.gpu_image.write(data=data)

        self.image_map = {
            name: RendererImage(atlas=self, entry_name=name, rect_xywh=rect_xywh)
            for name, rect_xywh in image_rect_map.items()
        }

    def __getitem__(self, image_name: str) -> "RendererImage":
        return self.image_map[image_name]


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


#
# Canvas API:
#


class RendererCanvas(BaseResource):
    renderer: "Renderer"
    cpu_quad_collection: R2dCpuQuadCollection

    def __init__(self, *, renderer: "Renderer"):
        super().__init__(parent=renderer)
        self.renderer = renderer
        self.cpu_quad_collection = R2dCpuQuadCollection(
            default_white_image=renderer.default_white_image
        )

    def draw(
        self,
        *,
        dst_xy: tuple[int, int],
        image: RendererImage | None = None,
        dst_wh: tuple[int, int] | None = None,
        scale: float | None = None,
        tint_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        border_thickness_px: tuple[int, int, int, int] = (0, 0, 0, 0),
        corner_radius: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        """
        Draw a quad with an optional image, border, tint, corner radius, etc to the
        canvas.
        """

        self.cpu_quad_collection.add(
            dst_xy=dst_xy,
            image=image,
            dst_wh=dst_wh,
            scale=scale,
            tint_color=tint_color,
            border_color=border_color,
            border_thickness_px=border_thickness_px,
            corner_radius=corner_radius,
        )


#
# Renderer2d: private implementation of 2D renderer
#


class Renderer2d(BaseResource):
    renderer: Renderer
    default_white_image: RendererImage
    _vertex_shader: GpuShader
    _fragment_shader: GpuShader
    _pipeline_layout: GpuPipelineLayout
    _common_uniform_buf: GpuBuffer
    _common_uniform_descriptor_set: GpuDescriptorSet
    _cached_pipeline: GpuPipeline | None
    _cached_depth_image: GpuImage | None
    _cached_gpu_batches: dict["RendererAtlas", "R2dGpuQuadBatch"]

    def __init__(
        self,
        *,
        renderer: Renderer,
        default_white_image: RendererImage,
    ):
        super().__init__(parent=renderer)

        self.renderer = renderer
        self.default_white_image = default_white_image
        self._vertex_shader = self._new_vertex_shader()
        self._fragment_shader = self._new_fragment_shader()
        self._pipeline_layout = self._new_pipeline_layout()
        self._common_uniform_buf = self._new_common_uniform_buffer()
        self._common_uniform_descriptor_set = self._new_common_uniform_descriptor_set()
        self._cached_pipeline = None
        self._cached_depth_image = None
        self._cached_gpu_batches = {}

    @property
    def gpu_device(self) -> GpuDevice:
        return self.renderer.gpu_device

    def show(self, *, canvas: "RendererCanvas", target: GpuImage):
        assert "color-attachment" in target.usages

        # Get GPU pipeline while updating cache:
        gpu_pipeline = self._get_gpu_pipeline(target=target)

        # Get depth image while updating cache:
        depth_image = self._get_depth_image(width=target.width, height=target.height)

        # Get GPU batches for this canvas while updating cache:
        gpu_batches = self._get_gpu_batch_dict(canvas=canvas)

        # TODO: implement rendering to target
        raise NotImplementedError("Renderer2d.show() WIP")

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

    def _new_pipeline_layout(self) -> GpuPipelineLayout:
        return GpuPipelineLayout(
            device=self.gpu_device,
            descriptor_set_layouts=[
                GpuDescriptorSetLayout(
                    device=self.gpu_device,
                    bindings=OrderedDict(
                        {
                            "uniform": GpuDescriptorSetLayoutBinding(
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
                                stages=["fragment"],
                            ),
                            "quads": GpuDescriptorSetLayoutBinding(
                                type="storage-buffer",
                                stages=["fragment"],
                            ),
                        }.items()
                    ),
                ),
            ],
        )

    def _new_common_uniform_buffer(self) -> GpuBuffer:
        return GpuBuffer(
            device=self.gpu_device,
            usages=["uniform", "copy-dst"],
            meta=GpuBufferMeta(element_count=1, element_dtype=R2D_UNIFORM_DTYPE),
        )

    def _new_common_uniform_descriptor_set(self) -> GpuDescriptorSet:
        return GpuDescriptorSet(
            device=self.gpu_device,
            layout=self._pipeline_layout.descriptor_set_layouts[0],
            bindings={"uniform": self._common_uniform_buf},
        )

    def _get_gpu_pipeline(self, target: GpuImage) -> GpuPipeline:
        if cached_pipeline := self._get_cached_gpu_pipeline(target=target):
            return cached_pipeline
        gpu_pipeline = self._new_gpu_pipeline(target=target)
        self._cached_pipeline = gpu_pipeline
        return gpu_pipeline

    def _get_cached_gpu_pipeline(self, target: GpuImage) -> GpuPipeline | None:
        if self._cached_pipeline is None:
            return None
        if self._cached_pipeline.vk_color_format != target.vk_format:
            return None
        if self._cached_pipeline.viewport_width != target.width:
            return None
        if self._cached_pipeline.viewport_height != target.height:
            return None
        return self._cached_pipeline

    def _new_gpu_pipeline(self, target: GpuImage) -> GpuPipeline:
        return GpuPipeline(
            device=self.gpu_device,
            vertex_shader=self._vertex_shader,
            fragment_shader=self._fragment_shader,
            vk_color_format=target.vk_format,
            viewport_width=target.width,
            viewport_height=target.height,
            layout=self._pipeline_layout,
        )

    def _get_gpu_batch_dict(
        self,
        *,
        canvas: "RendererCanvas",
    ) -> dict["RendererAtlas", "R2dGpuQuadBatch"]:
        # Compute a fresh set of GPU batches:
        gpu_batches = {
            atlas: self._get_gpu_batch(atlas=atlas, cpu_batch=cpu_batch)
            for atlas, cpu_batch in canvas.cpu_quad_collection.batches.items()
        }

        # Only keep cached GPU batches that are still in use:
        self._cached_gpu_batches = gpu_batches

        # Done:
        return gpu_batches

    def _get_gpu_batch(
        self,
        *,
        atlas: RendererAtlas,
        cpu_batch: R2dCpuQuadBatch,
    ) -> "R2dGpuQuadBatch":
        if gpu_batch := self._get_cached_gpu_batch(atlas=atlas, cpu_batch=cpu_batch):
            gpu_batch.upload(cpu_batch=cpu_batch)
            return gpu_batch

        gpu_batch = self._new_gpu_batch(atlas=atlas, cpu_batch=cpu_batch)
        gpu_batch.upload(cpu_batch=cpu_batch)

        self._cached_gpu_batches[atlas] = gpu_batch

        return gpu_batch

    def _get_cached_gpu_batch(
        self,
        *,
        atlas: RendererAtlas,
        cpu_batch: R2dCpuQuadBatch,
    ) -> "R2dGpuQuadBatch | None":
        gpu_batch = self._cached_gpu_batches.get(atlas)
        if gpu_batch is None:
            return None
        if gpu_batch.capacity < cpu_batch.instance_count:
            return None
        return gpu_batch

    def _new_gpu_batch(
        self,
        *,
        atlas: RendererAtlas,
        cpu_batch: "R2dCpuQuadBatch",
    ) -> "R2dGpuQuadBatch":
        gpu_quads_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=cpu_batch.capacity,
                element_dtype=R2D_QUAD_NP_DTYPE,
            ),
        )
        binding = GpuDescriptorSet(
            device=self.gpu_device,
            layout=self._pipeline_layout.descriptor_set_layouts[1],
            bindings={
                "atlasTexture": (atlas.gpu_image, atlas.gpu_sampler),
                "quads": gpu_quads_buf,
            },
        )
        return R2dGpuQuadBatch(
            renderer_2d=self,
            atlas=atlas,
            instance_count=cpu_batch.instance_count,
            capacity=cpu_batch.capacity,
            buffer=gpu_quads_buf,
            binding=binding,
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


class R2dGpuQuadBatch(BaseResource):
    atlas: RendererAtlas
    instance_count: int
    capacity: int
    buffer: GpuBuffer
    binding: GpuDescriptorSet

    def __init__(
        self,
        *,
        renderer_2d: Renderer2d,
        atlas: RendererAtlas,
        instance_count: int,
        capacity: int,
        buffer: GpuBuffer,
        binding: GpuDescriptorSet,
    ):
        super().__init__(parent=renderer_2d)

        self.atlas = atlas
        self.instance_count = instance_count
        self.capacity = capacity
        self.buffer = buffer
        self.binding = binding

    def upload(self, cpu_batch: R2dCpuQuadBatch):
        self.buffer.memory.write(data=cpu_batch.data[: cpu_batch.instance_count])


#
# R2dCpuQuadCollection and R2dCpuQuadBatch:
#


class R2dCpuQuadCollection:
    batches: dict[RendererAtlas, "R2dCpuQuadBatch"]
    total_added_image_count: int

    def __init__(self, *, default_white_image: RendererImage):
        super().__init__()
        self.batches = {}
        self.default_white_image = default_white_image

    def add(
        self,
        *,
        dst_xy: tuple[int, int],
        image: RendererImage | None = None,
        dst_wh: tuple[int, int] | None = None,
        scale: float | None = None,
        tint_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        border_thickness_px: tuple[int, int, int, int] = (0, 0, 0, 0),
        corner_radius: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        """
        Draws a quad with an optional image, border, tint, corner radius, etc to the
        canvas.

        Under the hood, we add the quad to a batch corresponding to the image's atlas.
        The actual rendering is performed later when the canvas is submitted to the
        renderer.
        """

        # Check arguments:
        if not image:
            if dst_wh is None:
                raise LogicError("`dst_wh` must be provided if `image` is None")
        if dst_wh is not None and scale is not None:
            raise LogicError("Cannot provide both `dst_wh` and `scale`")

        # Fill-in defaults:
        # - If no image is provided, we use the default white image, but the caller must
        #   provide `dst_wh` in that case, so the default image size is never leaked,
        #   it remains an implementation detail.
        # - Scale is applied on top of `dst_wh`, but the caller can never provide both,
        #   so this order is an implementation detail.
        image = image or self.default_white_image
        dst_wh = dst_wh if dst_wh is not None else (image.width_px, image.height_px)
        scale = scale if scale is not None else 1.0

        # Get the batch for this image's atlas:
        batch = self._get_batch_for_atlas(image.atlas)

        # Compute source rectangle in UVs:
        (src_x0_uv, src_y0_uv), (src_x1_uv, src_y1_uv) = image.rect_xy_xy_uv

        # Compute destination rectangle in pixels:
        dst_x0_px, dst_y0_px = dst_xy
        dst_w_px, dst_h_px = dst_wh
        dst_x1_px, dst_y1_px = (
            dst_x0_px + dst_w_px * scale,
            dst_y0_px + dst_h_px * scale,
        )

        # Add instance to batch:
        batch.add_instance(
            dst_xy=(
                (int(dst_x0_px), int(dst_y0_px)),  # TL
                (int(dst_x1_px), int(dst_y0_px)),  # TR
                (int(dst_x1_px), int(dst_y1_px)),  # BR
                (int(dst_x0_px), int(dst_y1_px)),  # BL
            ),
            src_uv=(
                (src_x0_uv, src_y0_uv),  # TL
                (src_x1_uv, src_y0_uv),  # TR
                (src_x1_uv, src_y1_uv),  # BR
                (src_x0_uv, src_y1_uv),  # BL
            ),
            tint_color=tint_color,
            border_color=border_color,
            border_thickness_px=border_thickness_px,
            corner_radius=corner_radius,
            height=self.total_added_image_count,
        )
        self.total_added_image_count += 1

    def _get_batch_for_atlas(self, atlas: RendererAtlas) -> R2dCpuQuadBatch:
        batch = self.batches.get(atlas)
        if batch is None:
            batch = R2dCpuQuadBatch()
            self.batches[atlas] = batch
        return batch


class R2dCpuQuadBatch:
    instance_count: int
    data: np.ndarray

    def __init__(self, capacity: int = 8):
        super().__init__()
        self.instance_count = 0
        self.data = np.empty(capacity, dtype=R2D_QUAD_NP_DTYPE)

    @property
    def capacity(self) -> int:
        return self.data.shape[0]

    def add_instance(
        self,
        dst_xy: tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
        ],
        src_uv: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
        tint_color: tuple[float, float, float, float],
        border_color: tuple[float, float, float, float],
        border_thickness_px: tuple[int, int, int, int],
        corner_radius: tuple[int, int, int, int],
        height: int,
    ):
        self._ensure_capacity(self.instance_count)

        idx = self.instance_count
        assert idx < self.capacity, "self._ensure_capacity() failed"

        self.data["dst_px"][idx] = dst_xy
        self.data["src_uv"][idx] = src_uv
        self.data["tint_color"][idx] = tint_color
        self.data["border_color"][idx] = border_color
        self.data["border_thickness_px"][idx] = border_thickness_px
        self.data["corner_radius"][idx] = corner_radius
        self.data["height"][idx] = height

        self.instance_count += 1

    def _ensure_capacity(self, n: int):
        if n <= self.capacity:
            return

        new_capacity = _next_po2(n)
        assert new_capacity > self.dst_px.shape[0]

        growth = new_capacity - self.dst_px.shape[0]

        self.dst_px = np.concatenate(
            [
                self.dst_px,
                np.zeros((growth, 4, 2), dtype=np.int32),
            ]
        )
        self.src_uv = np.concatenate(
            [
                self.src_uv,
                np.zeros((growth, 4, 2), dtype=np.float32),
            ]
        )
        self.tint_color = np.concatenate(
            [
                self.tint_color,
                np.zeros((growth, 4), dtype=np.float32),
            ]
        )
        self.border_color = np.concatenate(
            [
                self.border_color,
                np.zeros((growth, 4), dtype=np.float32),
            ]
        )
        self.border_thickness = np.concatenate(
            [
                self.border_thickness,
                np.zeros((growth, 4), dtype=np.int32),
            ]
        )
        self.corner_radius = np.concatenate(
            [
                self.corner_radius,
                np.zeros((growth, 4), dtype=np.int32),
            ]
        )
        self.height = np.concatenate(
            [
                self.height,
                np.zeros((growth, 1), dtype=np.int32),
            ]
        )


def _next_po2(x: int) -> int:
    """Return the next power of two greater than or equal to x."""
    if x <= 0:
        return 1
    v = 1
    while v < x:
        v *= 2
    return v


R2D_UNIFORM_DTYPE = np.dtype(
    [
        ("framebuffer_size_px", np.int32, (2,)),
    ]
)
assert R2D_UNIFORM_DTYPE.itemsize == 2 * 4


R2D_QUAD_NP_DTYPE = np.dtype(
    [
        ("dst_px", np.int32, (4, 2)),
        ("src_uv", np.float32, (4, 2)),
        ("tint_color", np.float32, (4,)),
        ("border_color", np.float32, (4,)),
        ("border_thickness_px", np.uint32, (4,)),
        ("corner_radius", np.uint32),
        ("height", np.uint32),
        ("_rsv0", np.uint32),
        ("_rsv1", np.uint32),
    ]
)
assert R2D_QUAD_NP_DTYPE.itemsize == 32 * 4
