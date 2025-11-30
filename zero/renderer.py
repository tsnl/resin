__all__ = [
    "RendererContext",
    "Renderer",
    "RendererAtlas",
    "RendererImage",
    "RendererCanvas",
]

import torch

from .excepts import LogicError
from .gpu import (
    GpuContext,
    GpuImage,
    GpuDevice,
    GpuBuffer,
    GpuBufferMeta,
    GpuImageMeta,
    GpuPipeline,
    GpuDescriptorSet,
    GpuDescriptorSetLayout,
    GpuDescriptorBinding,
    GpuDescriptorBindingResourceType,
    GpuShader,
)
from .core import BaseResource
from .bundled_data import BUNDLED_DATA_PATH


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
    _cached_gpu_pipeline_vertex_shader: GpuShader | None
    _cached_gpu_pipeline_fragment_shader: GpuShader | None
    _cached_gpu_pipeline: GpuPipeline | None
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
        self._cached_gpu_pipeline_vertex_shader = None
        self._cached_gpu_pipeline_fragment_shader = None
        self._cached_gpu_pipeline = None
        self._cached_depth_image: GpuImage | None = None
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
        gpu_batches = self._get_gpu_batches(canvas=canvas)

        # TODO: implement rendering to target
        raise NotImplementedError("Renderer2d.show() WIP")

    def _get_gpu_pipeline(self, target: GpuImage) -> GpuPipeline:
        # If the cached pipeline matches the target, return it:
        if (
            self._cached_gpu_pipeline is not None
            and self._cached_gpu_pipeline.vk_color_format == target.vk_format
            and self._cached_gpu_pipeline.viewport_width == target.width
            and self._cached_gpu_pipeline.viewport_height == target.height
        ):
            return self._cached_gpu_pipeline

        # Get shaders:
        if self._cached_gpu_pipeline_vertex_shader is None:
            self._cached_gpu_pipeline_vertex_shader = self.gpu_device.create_shader(
                spirv_path=BUNDLED_DATA_PATH / "shaders" / "r2d.vert.spv",
                stage="vertex",
            )
        if self._cached_gpu_pipeline_fragment_shader is None:
            self._cached_gpu_pipeline_fragment_shader = self.gpu_device.create_shader(
                spirv_path=BUNDLED_DATA_PATH / "shaders" / "r2d.frag.spv",
                stage="fragment",
            )

        # Create pipeline, cache it:
        gpu_pipeline = self.gpu_device.create_pipeline(
            vertex_shader=self._cached_gpu_pipeline_vertex_shader,
            fragment_shader=self._cached_gpu_pipeline_fragment_shader,
            vk_color_format=target.vk_format,
            viewport_width=target.width,
            viewport_height=target.height,
        )
        self._cached_gpu_pipeline = gpu_pipeline

        # Return it:
        return gpu_pipeline

    def _get_gpu_batches(
        self,
        *,
        canvas: "RendererCanvas",
    ) -> dict["RendererAtlas", "R2dGpuQuadBatch"]:
        gpu_batches = {}

        for atlas, cpu_batch in canvas.cpu_quad_collection.batches.items():
            gpu_batch = self._cached_gpu_batches.get(atlas)
            if gpu_batch is None or gpu_batch.capacity < cpu_batch.instance_count:
                gpu_batch = self._new_gpu_batch(atlas=atlas, cpu_batch=cpu_batch)
                self._cached_gpu_batches[atlas] = gpu_batch

            gpu_batch.upload(cpu_batch=cpu_batch)

            gpu_batches[atlas] = gpu_batch

        self._cached_gpu_batches = gpu_batches

        return gpu_batches

    def _get_depth_image(self, *, width: int, height: int) -> GpuImage:
        if (
            self._cached_depth_image is not None
            and self._cached_depth_image.width == width
            and self._cached_depth_image.height == height
        ):
            return self._cached_depth_image

        depth_image = self.gpu_device.create_image(
            usages=["depth-attachment"],
            meta=GpuImageMeta(shape=(height, width, 1), dtype=torch.float32),
        )
        self._cached_depth_image = depth_image

        return depth_image

    def _new_gpu_batch(
        self,
        *,
        atlas: RendererAtlas,
        cpu_batch: R2dCpuQuadBatch,
    ) -> "R2dGpuQuadBatch":
        def help_create_gpu_buffer(data: torch.Tensor) -> GpuBuffer:
            return self.gpu_device.create_buffer(
                usages=["uniform", "copy-dst"],
                meta=GpuBufferMeta.from_tensor(data),
            )

        return R2dGpuQuadBatch(
            renderer_2d=self,
            atlas=atlas,
            instance_count=cpu_batch.instance_count,
            capacity=cpu_batch.capacity,
            dst_px=help_create_gpu_buffer(cpu_batch.dst_px),
            src_uv=help_create_gpu_buffer(cpu_batch.src_uv),
            tint_color=help_create_gpu_buffer(cpu_batch.tint_color),
            border_color=help_create_gpu_buffer(cpu_batch.border_color),
            border_width=help_create_gpu_buffer(cpu_batch.border_width),
            corner_radius=help_create_gpu_buffer(cpu_batch.corner_radius),
            height=help_create_gpu_buffer(cpu_batch.height),
        )


class R2dGpuQuadBatch(BaseResource):
    atlas: RendererAtlas
    instance_count: int
    capacity: int
    dst_px: GpuBuffer
    src_uv: GpuBuffer
    tint_color: GpuBuffer
    border_color: GpuBuffer
    border_width: GpuBuffer
    corner_radius: GpuBuffer
    height: GpuBuffer

    def __init__(
        self,
        *,
        renderer_2d: Renderer2d,
        atlas: RendererAtlas,
        instance_count: int,
        capacity: int,
        dst_px: GpuBuffer,
        src_uv: GpuBuffer,
        tint_color: GpuBuffer,
        border_color: GpuBuffer,
        border_width: GpuBuffer,
        corner_radius: GpuBuffer,
        height: GpuBuffer,
    ):
        super().__init__(parent=renderer_2d)

        self.atlas = atlas
        self.instance_count = instance_count
        self.capacity = capacity
        self.dst_px = dst_px
        self.src_uv = src_uv
        self.tint_color = tint_color
        self.border_color = border_color
        self.border_width = border_width
        self.corner_radius = corner_radius
        self.height = height

    def upload(self, cpu_batch: R2dCpuQuadBatch):
        self.dst_px.write(data=cpu_batch.dst_px[: cpu_batch.instance_count])
        self.src_uv.write(data=cpu_batch.src_uv[: cpu_batch.instance_count])
        self.tint_color.write(data=cpu_batch.tint_color[: cpu_batch.instance_count])
        self.border_color.write(data=cpu_batch.border_color[: cpu_batch.instance_count])
        self.border_width.write(data=cpu_batch.border_width[: cpu_batch.instance_count])
        self.corner_radius.write(
            data=cpu_batch.corner_radius[: cpu_batch.instance_count]
        )
        self.height.write(data=cpu_batch.height[: cpu_batch.instance_count])


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
                (float(dst_x0_px), float(dst_y0_px)),  # TL
                (float(dst_x1_px), float(dst_y0_px)),  # TR
                (float(dst_x1_px), float(dst_y1_px)),  # BR
                (float(dst_x0_px), float(dst_y1_px)),  # BL
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
    dst_px: torch.Tensor  # (N, 4, 2): TL, TR, BR, BL
    src_uv: torch.Tensor  # (N, 4, 2): TL, TR, BR, BL
    tint_color: torch.Tensor  # (N, 4)
    border_color: torch.Tensor  # (N, 4)
    border_width: torch.Tensor  # (N, 4): T, R, B, L
    corner_radius: torch.Tensor  # (N, 4): TL, TR, BR, BL
    height: torch.Tensor  # (N, 1)

    def __init__(self, capacity: int = 8):
        super().__init__()
        self.instance_count = 0
        self.dst_px = torch.zeros((capacity, 4, 2), dtype=torch.uint32)
        self.src_uv = torch.zeros((capacity, 4, 2), dtype=torch.float32)
        self.tint_color = torch.zeros((capacity, 4), dtype=torch.float32)
        self.border_color = torch.zeros((capacity, 4), dtype=torch.float32)
        self.border_width = torch.zeros((capacity, 4), dtype=torch.float32)
        self.corner_radius = torch.zeros((capacity, 4), dtype=torch.float32)
        self.height = torch.zeros((capacity, 1), dtype=torch.float32)

    @property
    def capacity(self) -> int:
        return self.dst_px.shape[0]

    def add_instance(
        self,
        dst_xy: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
        src_uv: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
        tint_color: tuple[float, float, float, float],
        border_color: tuple[float, float, float, float],
        border_thickness_px: tuple[float, float, float, float],
        corner_radius: tuple[float, float, float, float],
        height: float,
    ):
        idx = self.instance_count
        if idx >= self.capacity:
            self._double_capacity()

        assert idx < self.capacity

        self.dst_px[idx] = torch.tensor(dst_xy, dtype=torch.float32)
        self.src_uv[idx] = torch.tensor(src_uv, dtype=torch.float32)
        self.tint_color[idx] = torch.tensor(tint_color, dtype=torch.float32)
        self.border_color[idx] = torch.tensor(border_color, dtype=torch.float32)
        self.border_width[idx] = torch.tensor(border_thickness_px, dtype=torch.float32)
        self.corner_radius[idx] = torch.tensor(corner_radius, dtype=torch.float32)
        self.height[idx] = height

        self.instance_count += 1

    def _double_capacity(self):
        new_capacity = self.dst_px.shape[0] * 2
        assert new_capacity > self.dst_px.shape[0]

        growth = new_capacity - self.dst_px.shape[0]

        self.dst_px = torch.cat([self.dst_px, torch.zeros((growth, 4, 2))])
        self.src_uv = torch.cat([self.src_uv, torch.zeros((growth, 4, 2))])
        self.tint_color = torch.cat([self.tint_color, torch.zeros((growth, 4))])
        self.border_color = torch.cat([self.border_color, torch.zeros((growth, 4))])
        self.border_width = torch.cat([self.border_width, torch.zeros((growth, 4))])
        self.corner_radius = torch.cat([self.corner_radius, torch.zeros((growth, 4))])
        self.height = torch.cat([self.height, torch.zeros((growth, 1))])
