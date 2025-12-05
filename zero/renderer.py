__all__ = [
    "RendererContext",
    "Renderer",
    "RendererAtlas",
    "RendererImage",
    "RendererCanvas",
]

from collections import OrderedDict

import torch
import numpy as np

from .bundled_data import BUNDLED_DATA_PATH
from .core import BaseResource
from .excepts import LogicError
from .gpu import (
    GpuBuffer,
    GpuBufferMeta,
    GpuContext,
    GpuDescriptorSet,
    GpuDescriptorSetBinding,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuImage,
    GpuImageMeta,
    GpuPipeline,
    GpuPipelineLayout,
    GpuShader,
)

#
# Renderer API:
#


class RendererContext(BaseResource):
    def __init__(self, gpu_context: GpuContext):
        super().__init__(parent=gpu_context)

    def create_renderer(self, *, gpu_device: GpuDevice) -> "Renderer":
        """
        Create a new `Renderer` instance using the given GPU device.
        """

        return Renderer(context=self, gpu_device=gpu_device)


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
        """
        DO NOT CALL THIS CONSTRUCTOR DIRECTLY.

        Use `RendererContext.create_renderer()` instead.
        """

        super().__init__(parent=context)

        self.gpu_device = gpu_device

        self.default_white_image_atlas = self.create_atlas(
            data=torch.full((32, 32, 4), 0xFF, dtype=torch.uint8),
            image_rect_map={"default": (0, 0, 32, 32)},
        )
        self.default_white_image = self.default_white_image_atlas["default"]

        self.renderer_2d = Renderer2d(
            renderer=self,
            default_white_image=self.default_white_image,
        )

    def create_atlas(
        self,
        *,
        data: torch.Tensor,
        image_rect_map: dict[str, tuple[int, int, int, int]],
    ) -> "RendererAtlas":
        """
        Create a new `RendererAtlas` from the given image data and image rectangle map.
        """

        assert data.ndim == 3 and data.shape[2] == 4

        gpu_image = self.gpu_device.create_image(
            usages=["texture-binding"],
            meta=GpuImageMeta.from_tensor(data),
        )
        gpu_image.write(data=data)

        return RendererAtlas(
            context=self.context,
            gpu_image=gpu_image,
            image_rect_map=image_rect_map,
        )

    def create_canvas(self) -> "RendererCanvas":
        """
        Create a fresh `RendererCanvas` instance.

        You should create a new canvas for each frame you want to render.

        You can render quads to the canvas using the `RendererCanvas.draw()` method, and
        then submit the canvas for rendering using the `Renderer.show()` method.
        """

        return self.renderer_2d.create_canvas()

    def show(self, *, canvas: "RendererCanvas", target: GpuImage):
        """
        Render the given `RendererCanvas` to the given target `GpuImage`.
        """

        self.renderer_2d.show(canvas=canvas, target=target)


#
# Atlas, Image API: shared by 2D and 3D renderer workflows
#


class RendererAtlas(BaseResource):
    context: RendererContext
    gpu_image: GpuImage
    image_map: dict[str, "RendererImage"]

    def __init__(
        self,
        *,
        context: RendererContext,
        gpu_image: GpuImage,
        image_rect_map: dict[str, tuple[int, int, int, int]],
    ):
        super().__init__(parent=context)
        self.gpu_image = gpu_image
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
    cpu_quad_collection: R2dCpuQuadCollection

    def __init__(self, *, default_white_image: RendererImage):
        super().__init__()
        self.cpu_quad_collection = R2dCpuQuadCollection(
            default_white_image=default_white_image
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
        border_thickness_px: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        corner_radius: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
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
        self._cached_pipeline = None
        self._cached_depth_image = None
        self._cached_gpu_batches = {}

    @property
    def gpu_device(self) -> GpuDevice:
        return self.renderer.gpu_device

    def create_canvas(self) -> "RendererCanvas":
        return RendererCanvas(default_white_image=self.default_white_image)

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
        return self.gpu_device.create_shader(
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "r2d.vert.spv",
            stage="vertex",
        )

    def _new_fragment_shader(self) -> GpuShader:
        return self.gpu_device.create_shader(
            spirv_path=BUNDLED_DATA_PATH / "shaders" / "r2d.frag.spv",
            stage="fragment",
        )

    def _new_pipeline_layout(self) -> GpuPipelineLayout:
        return self.gpu_device.create_pipeline_layout(
            descriptor_set_layouts=[
                self.gpu_device.create_descriptor_set_layout(
                    bindings=OrderedDict(
                        {
                            "framebufferSize": GpuDescriptorSetLayoutBinding(
                                type="uniform-buffer",
                                stages=["vertex", "fragment"],
                            ),
                            "atlasSize": GpuDescriptorSetLayoutBinding(
                                type="uniform-buffer",
                                stages=["vertex", "fragment"],
                            ),
                        }.items()
                    )
                ),
                self.gpu_device.create_descriptor_set_layout(
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
                    )
                ),
            ]
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
        return self.gpu_device.create_pipeline(
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
        cpu_batch: R2dCpuQuadBatch,
    ) -> "R2dGpuQuadBatch":
        quads_buf_cpu = np.empty(shape=cpu_batch.capacity, dtype=R2D_QUAD_NP_DTYPE)

        n = cpu_batch.instance_count
        quads_buf_cpu["dst_px"][:n] = cpu_batch.dst_px.numpy()
        quads_buf_cpu["src_uv"][:n] = cpu_batch.src_uv.numpy()
        quads_buf_cpu["tint_color"][:n] = cpu_batch.tint_color.numpy()
        quads_buf_cpu["border_color"][:n] = cpu_batch.border_color.numpy()
        quads_buf_cpu["border_thickness_px"][:n] = cpu_batch.border_thickness.numpy()
        quads_buf_cpu["corner_radius"][:n] = cpu_batch.corner_radius.numpy()
        quads_buf_cpu["height"][:n] = cpu_batch.height.numpy()

        quads_buf = self.gpu_device.create_buffer(
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(
                element_count=cpu_batch.capacity,
                element_dtype=R2D_QUAD_NP_DTYPE,
            ),
        )

        binding = self.gpu_device.create_descriptor_set()

        return R2dGpuQuadBatch(
            renderer_2d=self,
            atlas=atlas,
            instance_count=cpu_batch.instance_count,
            capacity=cpu_batch.capacity,
            dst_px=dst_px_buf,
            src_uv=src_uv_buf,
            tint_color=tint_color_buf,
            border_color=border_color_buf,
            border_width=border_width_buf,
            corner_radius=corner_radius_buf,
            height=height_buf,
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
        return self.gpu_device.create_image(
            usages=["depth-attachment"],
            meta=GpuImageMeta(shape=(height, width, 1), dtype=torch.float32),
        )


class R2dGpuQuadBatch(BaseResource):
    atlas: RendererAtlas
    instance_count: int
    capacity: int
    data: GpuBuffer
    binding: GpuDescriptorSet

    def __init__(
        self,
        *,
        renderer_2d: Renderer2d,
        atlas: RendererAtlas,
        instance_count: int,
        capacity: int,
        data: GpuBuffer,
        binding: GpuDescriptorSet,
    ):
        super().__init__(parent=renderer_2d)

        self.atlas = atlas
        self.instance_count = instance_count
        self.capacity = capacity
        self.data = data
        self.binding = binding

    def upload(self, cpu_batch: R2dCpuQuadBatch):
        # TODO: need to pack data: see shader
        raise NotImplementedError()


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
        border_thickness_px: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        corner_radius: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
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
    dst_px: torch.Tensor  # int32(N, 4, 2): TL, TR, BR, BL
    src_uv: torch.Tensor  # float32(N, 4, 2): TL, TR, BR, BL
    tint_color: torch.Tensor  # float32(N, 4)
    border_color: torch.Tensor  # float32(N, 4)
    border_thickness: torch.Tensor  # int32(N, 4): T, R, B, L
    corner_radius: torch.Tensor  # int32(N, 4): TL, TR, BR, BL
    height: torch.Tensor  # int32(N, 1)

    def __init__(self, capacity: int = 8):
        super().__init__()
        self.instance_count = 0
        self.dst_px = torch.zeros((capacity, 4, 2), dtype=torch.int32)
        self.src_uv = torch.zeros((capacity, 4, 2), dtype=torch.float32)
        self.tint_color = torch.zeros((capacity, 4), dtype=torch.float32)
        self.border_color = torch.zeros((capacity, 4), dtype=torch.float32)
        self.border_thickness = torch.zeros((capacity, 4), dtype=torch.int32)
        self.corner_radius = torch.zeros((capacity, 4), dtype=torch.int32)
        self.height = torch.zeros((capacity, 1), dtype=torch.int32)

    @property
    def capacity(self) -> int:
        return self.dst_px.shape[0]

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
        idx = self.instance_count
        if idx >= self.capacity:
            self._double_capacity()

        assert idx < self.capacity

        self.dst_px[idx] = torch.tensor(dst_xy, dtype=torch.int32)
        self.src_uv[idx] = torch.tensor(src_uv, dtype=torch.float32)
        self.tint_color[idx] = torch.tensor(tint_color, dtype=torch.float32)
        self.border_color[idx] = torch.tensor(border_color, dtype=torch.float32)
        self.border_thickness[idx] = torch.tensor(
            border_thickness_px, dtype=torch.int32
        )
        self.corner_radius[idx] = torch.tensor(corner_radius, dtype=torch.int32)
        self.height[idx] = torch.tensor(height, dtype=torch.int32)

        self.instance_count += 1

    def _double_capacity(self):
        new_capacity = self.dst_px.shape[0] * 2
        assert new_capacity > self.dst_px.shape[0]

        growth = new_capacity - self.dst_px.shape[0]

        self.dst_px = torch.cat(
            [
                self.dst_px,
                torch.zeros((growth, 4, 2), dtype=torch.int32),
            ]
        )
        self.src_uv = torch.cat(
            [
                self.src_uv,
                torch.zeros((growth, 4, 2), dtype=torch.float32),
            ]
        )
        self.tint_color = torch.cat(
            [
                self.tint_color,
                torch.zeros((growth, 4), dtype=torch.float32),
            ]
        )
        self.border_color = torch.cat(
            [
                self.border_color,
                torch.zeros((growth, 4), dtype=torch.float32),
            ]
        )
        self.border_thickness = torch.cat(
            [
                self.border_thickness,
                torch.zeros((growth, 4), dtype=torch.int32),
            ]
        )
        self.corner_radius = torch.cat(
            [
                self.corner_radius,
                torch.zeros((growth, 4), dtype=torch.int32),
            ]
        )
        self.height = torch.cat(
            [
                self.height,
                torch.zeros((growth, 1), dtype=torch.int32),
            ]
        )


R2D_QUAD_NP_DTYPE = np.dtype(
    [
        ("dst_px", np.uint32, (4, 2)),
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
