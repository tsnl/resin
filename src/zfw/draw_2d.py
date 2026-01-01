"""
Draw2d is a low-level 2D renderer for textured quads.

You probably want to use the higher-level `draw_2d_ex` module instead that builds on top
of this one with more features.
"""

__all__ = [
    "Draw2dQuad",
    "Draw2dRenderer",
    "Draw2dTarget",
]

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .basic import BaseResource, logger, round_up_to_po2
from .bundled_data import BUNDLED_DATA_PATH
from .gpu import (
    GpuBuffer,
    GpuBufferMeta,
    GpuCommandEncoder,
    GpuDescriptorSet,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuImage,
    GpuGraphicsPipeline,
    GpuImageMeta,
    GpuPipelineLayout,
    GpuSampler,
    GpuShader,
)


class Draw2dRenderer(BaseResource):
    """
    Draw2dRenderer draws a list of textured 2D quads to a GPU image. It can be used to
    render user interfaces, HUDs, and other 2D elements.

    To maximize throughput, a single renderer can render to multiple targets
    concurrently, e.g. for double or triple buffering. Create multiple `Draw2dTarget`
    instances for each frame in flight.
    """

    _gpu_device: GpuDevice
    _target_width_px: int
    _target_height_px: int
    _clear_color: Literal["transparent", "black"] | None

    _gpu_uniform_descriptor_set_layout: GpuDescriptorSetLayout
    _gpu_quad_batch_descriptor_set_layout: GpuDescriptorSetLayout
    _gpu_pipeline_layout: GpuPipelineLayout

    _gpu_vertex_shader_module: GpuShader
    _gpu_fragment_shader_module: GpuShader
    _gpu_pipeline: GpuGraphicsPipeline

    _gpu_default_white_image: GpuImage
    _gpu_default_sampler: GpuSampler

    _target_list: list["Draw2dTarget"]

    def __init__(
        self,
        *,
        gpu_device: GpuDevice,
        target_width_px: int,
        target_height_px: int,
        clear_color: Literal["transparent", "black"] | None = "transparent",
    ):
        super().__init__()

        self._gpu_device = gpu_device
        self._target_width_px = target_width_px
        self._target_height_px = target_height_px
        self._clear_color = clear_color

        self._gpu_uniform_descriptor_set_layout = GpuDescriptorSetLayout(
            device=gpu_device,
            bindings=OrderedDict(
                {
                    "uniform": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["vertex"],
                    ),
                }.items()
            ),
        )
        self._gpu_quad_batch_descriptor_set_layout = GpuDescriptorSetLayout(
            device=gpu_device,
            bindings=OrderedDict(
                {
                    "quads": GpuDescriptorSetLayoutBinding(
                        type="storage-buffer",
                        stages=["vertex", "fragment"],
                    ),
                    "image": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "sampler": GpuDescriptorSetLayoutBinding(
                        type="sampler",
                        stages=["fragment"],
                    ),
                }.items()
            ),
        )
        self._gpu_pipeline_layout = GpuPipelineLayout(
            device=gpu_device,
            descriptor_set_layouts=[
                self._gpu_uniform_descriptor_set_layout,
                self._gpu_quad_batch_descriptor_set_layout,
            ],
        )

        self._gpu_vertex_shader_module = GpuShader(
            device=gpu_device,
            spirv_path=(BUNDLED_DATA_PATH / "shaders/draw_2d.vert.spv"),
            stage="vertex",
        )
        self._gpu_fragment_shader_module = GpuShader(
            device=gpu_device,
            spirv_path=(BUNDLED_DATA_PATH / "shaders/draw_2d.frag.spv"),
            stage="fragment",
        )
        self._gpu_pipeline = GpuGraphicsPipeline(
            device=gpu_device,
            vertex_shader=self._gpu_vertex_shader_module,
            fragment_shader=self._gpu_fragment_shader_module,
            enable_depth_test=False,
            enable_alpha_blending=True,
            viewport_width=target_width_px,
            viewport_height=target_height_px,
            layout=self._gpu_pipeline_layout,
        )

        self._gpu_default_white_image = GpuImage(
            device=gpu_device,
            usages=["texture-binding"],
            data=np.ones((1, 1, 4), dtype=np.float32),
        )
        self._gpu_default_sampler = GpuSampler(
            device=gpu_device,
            min_filter="nearest",
            mag_filter="nearest",
        )

        self._target_list = []

    @property
    def gpu_device(self) -> GpuDevice:
        return self._gpu_device

    def resize(self, *, target_width_px: int, target_height_px: int):
        if (
            self._target_width_px == target_width_px
            and self._target_height_px == target_height_px
        ):
            return

        self._gpu_device.wait_idle()

        self._target_width_px = target_width_px
        self._target_height_px = target_height_px

        self._gpu_pipeline.dispose()
        self._gpu_pipeline = GpuGraphicsPipeline(
            device=self._gpu_device,
            vertex_shader=self._gpu_vertex_shader_module,
            fragment_shader=self._gpu_fragment_shader_module,
            enable_depth_test=False,
            enable_alpha_blending=True,
            viewport_width=target_width_px,
            viewport_height=target_height_px,
            layout=self._gpu_pipeline_layout,
        )

        for target in self._target_list:
            target._resize_impl(
                target_width_px=target_width_px,
                target_height_px=target_height_px,
            )

    def record_gpu_commands(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        target: "Draw2dTarget",
        quads: list["Draw2dQuad"],
    ):
        target._record_gpu_commands(command_encoder=command_encoder, quads=quads)

    def _on_dispose(self) -> None:
        for target in self._target_list:
            target.dispose()

        self._gpu_default_sampler.dispose()
        self._gpu_default_white_image.dispose()
        self._gpu_pipeline.dispose()
        self._gpu_fragment_shader_module.dispose()
        self._gpu_vertex_shader_module.dispose()
        self._gpu_pipeline_layout.dispose()
        self._gpu_quad_batch_descriptor_set_layout.dispose()
        self._gpu_uniform_descriptor_set_layout.dispose()


class Draw2dTarget(BaseResource):
    """
    Draw2dTarget encapsulates a single render target and associated mutable state.

    Multiple targets can be rendered to concurrently by the same renderer to maximize
    throughput.
    """

    _renderer: Draw2dRenderer

    _gpu_color_image: GpuImage

    _uniform: "Uniform"
    _quad_batch_cache: dict[GpuImage | None, "QuadGroup"]

    def __init__(self, *, renderer: Draw2dRenderer):
        super().__init__(parent_resource=renderer)

        self._renderer = renderer

        self._gpu_color_image = GpuImage(
            device=renderer._gpu_device,
            usages=["color-attachment", "texture-binding", "transfer-src"],
            meta=GpuImageMeta(
                shape=(
                    self._renderer._target_height_px,
                    self._renderer._target_width_px,
                    4,
                ),
                dtype=np.float32,
            ),
        )

        self._uniform = Uniform(renderer=renderer)
        self._quad_batch_cache = {}

        # Register with renderer:
        self._renderer._target_list.append(self)

    @property
    def color_image(self) -> GpuImage:
        return self._gpu_color_image

    def _resize_impl(self, *, target_width_px: int, target_height_px: int):
        self._gpu_color_image.dispose()
        self._gpu_color_image = GpuImage(
            device=self._renderer._gpu_device,
            usages=["color-attachment", "transfer-src"],
            meta=GpuImageMeta(
                shape=(target_height_px, target_width_px, 4),
                dtype=np.float32,
            ),
        )

    def _record_gpu_commands(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        quads: list["Draw2dQuad"],
    ):
        # Prepare batches on the CPU:
        #

        quads_with_depth = Draw2dTarget._augment_quads_with_depth(quads=quads)
        group_quads, group_spans = self._compute_batches(quads=quads_with_depth)

        # Record GPU commands:
        #

        # Transition the target image to color-attachment layout:
        command_encoder.transition_image_layout(
            image=self._gpu_color_image,
            layout="color-attachment-optimal",
        )

        # Flush uniform data for this target:
        uniform_descriptor_set = self._uniform.flush_data_and_get_descriptor_set(
            command_encoder=command_encoder,
            target_width_px=self._gpu_color_image.width,
            target_height_px=self._gpu_color_image.height,
        )

        # Flush grouped quad data to per-group buffers and get descriptor sets:
        group_descriptor_sets = {
            image: self._get_quad_group(
                image=image,
            ).flush_data_and_get_descriptor_set(
                command_encoder=command_encoder,
                quads=quads,
            )
            for image, quads in group_quads.items()
        }
        batches = [
            (group_descriptor_sets[image], span)  #
            for image, span in group_spans
        ]

        # Record draw calls:
        with command_encoder.render(
            color_attachment=self._gpu_color_image,
            clear_color=self._renderer._clear_color,
        ) as rp:
            rp.bind_pipeline(pipeline=self._renderer._gpu_pipeline)
            rp.bind_descriptor_set(set_=uniform_descriptor_set, set_index=0)
            for batch_descriptor_set, batch_span in batches:
                rp.bind_descriptor_set(set_=batch_descriptor_set, set_index=1)
                rp.draw(
                    vertex_count=6,
                    first_instance=batch_span.begin,
                    instance_count=batch_span.size,
                )

    def _compute_batches(
        self,
        *,
        quads: list["QuadWithDepth"],
    ) -> tuple[
        dict[GpuImage | None, list["QuadWithDepth"]],
        list[tuple[GpuImage | None, "Span"]],
    ]:
        """
        Computes batches of quads, each of which shares the same image and can be drawn
        in a single multi-instance draw call.

        ---

        To correctly order quads, we scan the 'quads' list for contiguous runs that use
        the same image. Each such run becomes a batch.

        Definitions:
        - Quad: a single 2D quad to be drawn. Has an associated image (or None).
        - Group: quads sharing the same image.
        - Group index: index of a quad within its group.
        - Global index: index of a quad in the full (ungrouped) quad list.
        - Span: a contiguous run of quads in either global or group space.
          - Group span: a span whose indices point to the per-group quad list.
          - Global span: a span whose indices point to the full quad list.
        - Batch: a group span.

        Goal: given a global, flat quad list, compute a list of group spans.
        """

        # Group quads into global spans that share the same image.
        global_spans = Draw2dTarget._compute_global_spans(quads=quads)

        # Convert global spans into group spans while accumulating all the quads
        # per-group:
        return Draw2dTarget._compute_group_spans_from_global_spans(
            quads=quads,
            global_spans=global_spans,
        )

    @staticmethod
    def _augment_quads_with_depth(
        *,
        quads: list["Draw2dQuad"],
    ) -> list["QuadWithDepth"]:
        total_quad_count = len(quads)
        return [
            QuadWithDepth(
                inner=quad,
                depth=float(total_quad_count - index),
            )
            for index, quad in enumerate(quads)
        ]

    @staticmethod
    def _compute_global_spans(
        *,
        quads: list["QuadWithDepth"],
    ) -> list[tuple[GpuImage | None, "Span"]]:
        global_runs: list[tuple[GpuImage | None, "Span"]] = [
            (
                quads[0].inner.fill_image,
                Span(begin=0, size=1),
            )
        ]
        for global_index in range(1, len(quads)):
            curr_run_image, curr_run_span = global_runs[-1]

            quad = quads[global_index]

            if quad.inner.fill_image is curr_run_image:
                curr_run_span.size += 1
            else:
                global_runs.append(
                    (
                        quad.inner.fill_image,
                        Span(begin=global_index, size=1),
                    )
                )
        return global_runs

    @staticmethod
    def _compute_group_spans_from_global_spans(
        *,
        quads: list["QuadWithDepth"],
        global_spans: list[tuple[GpuImage | None, "Span"]],
    ) -> tuple[
        dict[GpuImage | None, list["QuadWithDepth"]],
        list[tuple[GpuImage | None, "Span"]],
    ]:
        group_quads = defaultdict(list)
        group_spans: list[tuple[GpuImage | None, "Span"]] = []
        for image, global_run in global_spans:
            # Accumulate quads into a per-group list, noting the begin index of the
            # added span within the group:
            group_run_begin = len(group_quads[image])
            group_quads[image] += quads[
                global_run.begin : global_run.begin + global_run.size
            ]

            # Create the group span, append:
            group_run = Span(begin=group_run_begin, size=global_run.size)
            group_spans.append((image, group_run))

        # Next, we can use `group_quads`
        return group_quads, group_spans

    def _get_quad_group(self, *, image: GpuImage | None) -> "QuadGroup":
        if qb := self._quad_batch_cache.get(image):
            return qb

        qb = QuadGroup(
            renderer=self._renderer,
            gpu_image=(image or self._renderer._gpu_default_white_image),
        )
        self._quad_batch_cache[image] = qb
        return qb


@dataclass(kw_only=True, frozen=True)
class Draw2dQuad:
    dst_xywh_px: tuple[int, int, int, int] | None = None  # None = full target
    src_xywh_px: tuple[int, int, int, int] | None = None  # None = full image
    fill_image: GpuImage | None = None  # None = white 1x1 texture
    fill_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    border_thickness_px: tuple[int, int, int, int] = (0, 0, 0, 0)  # TRBL
    border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


#
# Implementation: Uniform:
#


class Uniform(BaseResource):
    gpu_device: GpuDevice
    gpu_descriptor_set_layout: GpuDescriptorSetLayout

    cached_framebuffer_size_px: tuple[int, int] | None
    cached_resources: "UniformResources | None"

    def __init__(
        self,
        *,
        renderer: Draw2dRenderer,
    ):
        super().__init__(parent_resource=renderer)

        self.gpu_device = renderer._gpu_device
        self.gpu_descriptor_set_layout = renderer._gpu_uniform_descriptor_set_layout

        self.cached_framebuffer_size_px = None
        self.cached_resources = None

    def flush_data_and_get_descriptor_set(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        target_width_px: int,
        target_height_px: int,
    ) -> "GpuDescriptorSet":
        resources = self._acquire_resources()

        if self.cached_framebuffer_size_px == (target_width_px, target_height_px):
            return resources.gpu_descriptor_set

        self.cached_framebuffer_size_px = (target_width_px, target_height_px)

        with resources.gpu_staging_buffer.memory.map() as mv:
            mem = np.frombuffer(mv, dtype=UNIFORM_DTYPE, count=1)
            mem[0]["viewport_size_px"] = [target_width_px, target_height_px]

        command_encoder.copy_buffer_to_buffer(
            src=resources.gpu_staging_buffer,
            dst=resources.gpu_device_buffer,
            size=resources.gpu_staging_buffer.meta.size,
        )

        return resources.gpu_descriptor_set

    def _acquire_resources(self) -> "UniformResources":
        if self.cached_resources:
            return self.cached_resources

        gpu_buffer_meta = GpuBufferMeta(
            element_dtype=UNIFORM_DTYPE,
            element_count=1,
        )
        gpu_staging_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=gpu_buffer_meta,
        )
        gpu_device_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["uniform", "copy-dst"],
            meta=gpu_buffer_meta,
        )
        gpu_descriptor_set = GpuDescriptorSet(
            device=self.gpu_device,
            bindings={
                "uniform": gpu_device_buffer,
            },
            layout=self.gpu_descriptor_set_layout,
        )
        ret = UniformResources(
            uniform=self,
            gpu_staging_buffer=gpu_staging_buffer,
            gpu_device_buffer=gpu_device_buffer,
            gpu_descriptor_set=gpu_descriptor_set,
        )

        self.cached_resources = ret

        return ret


class UniformResources(BaseResource):
    gpu_staging_buffer: GpuBuffer
    gpu_device_buffer: GpuBuffer
    gpu_descriptor_set: GpuDescriptorSet

    def __init__(
        self,
        *,
        uniform: "Uniform",
        gpu_staging_buffer: GpuBuffer,
        gpu_device_buffer: GpuBuffer,
        gpu_descriptor_set: GpuDescriptorSet,
    ):
        super().__init__(parent_resource=uniform)

        self.gpu_staging_buffer = gpu_staging_buffer
        self.gpu_device_buffer = gpu_device_buffer
        self.gpu_descriptor_set = gpu_descriptor_set

    def _on_dispose(self) -> None:
        self.gpu_staging_buffer.dispose()
        self.gpu_device_buffer.dispose()
        self.gpu_descriptor_set.dispose()


UNIFORM_DTYPE = np.dtype(
    [
        ("viewport_size_px", "<f4", 2),
        ("_rsv", "<f4", 2),
    ]
)


#
# Implementation: QuadGroupByImage: several quads sharing the same image:
#


class QuadGroup(BaseResource):
    renderer: Draw2dRenderer

    gpu_image: GpuImage

    cached_resource_capacity: int
    cached_resources: "QuadGroupResources | None"

    def __init__(self, *, renderer: Draw2dRenderer, gpu_image: GpuImage):
        super().__init__(parent_resource=renderer)

        self.renderer = renderer

        self.gpu_image = gpu_image

        self.cached_resource_capacity = 0
        self.cached_resources = None

    @property
    def gpu_device(self) -> GpuDevice:
        return self.renderer._gpu_device

    @property
    def gpu_descriptor_set_layout(self) -> GpuDescriptorSetLayout:
        return self.renderer._gpu_quad_batch_descriptor_set_layout

    @property
    def gpu_sampler(self) -> GpuSampler:
        return self.renderer._gpu_default_sampler

    def flush_data_and_get_descriptor_set(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        quads: list["QuadWithDepth"],
    ) -> "GpuDescriptorSet":
        # Acquire resources for this batch with sufficient capacity:
        resources = self._acquire_resources(capacity=len(quads))

        # Marshall quad data into staging buffer:
        self._marshall_quad_data(
            quads=quads,
            staging_buffer=resources.gpu_staging_buffer,
        )

        # Copy staging buffer to device buffer:
        command_encoder.copy_buffer_to_buffer(
            src=resources.gpu_staging_buffer,
            dst=resources.gpu_device_buffer,
            size=resources.gpu_staging_buffer.meta.size,
        )

        # Return the descriptor set:
        return resources.gpu_descriptor_set

    def _acquire_resources(self, *, capacity: int) -> "QuadGroupResources":
        assert capacity > 0

        if self.cached_resources and capacity <= self.cached_resource_capacity:
            return self.cached_resources

        if self.cached_resources:
            self.cached_resources.dispose()
            self.cached_resources = None

        assert self.cached_resources is None

        capacity = round_up_to_po2(capacity)

        gpu_buffer_meta = GpuBufferMeta(
            element_dtype=QUAD_DTYPE,
            element_count=capacity,
        )
        gpu_staging_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=gpu_buffer_meta,
        )
        gpu_device_buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["storage", "copy-dst"],
            meta=gpu_buffer_meta,
        )
        gpu_descriptor_set = GpuDescriptorSet(
            device=self.gpu_device,
            bindings={
                "quads": gpu_device_buffer,
                "image": self.gpu_image,
                "sampler": self.gpu_sampler,
            },
            layout=self.gpu_descriptor_set_layout,
        )
        ret = QuadGroupResources(
            parent_resource=self,
            gpu_staging_buffer=gpu_staging_buffer,
            gpu_device_buffer=gpu_device_buffer,
            gpu_descriptor_set=gpu_descriptor_set,
        )

        self.cached_resources = ret
        self.cached_resource_capacity = capacity

        return ret

    def _marshall_quad_data(
        self,
        *,
        quads: list["QuadWithDepth"],
        staging_buffer: GpuBuffer,
    ):
        with staging_buffer.memory.map() as mv:
            mem = np.frombuffer(mv, dtype=QUAD_DTYPE, count=len(quads))
            for i, quad_with_depth in enumerate(quads):
                quad = quad_with_depth.inner
                depth = quad_with_depth.depth
                mem[i]["dst_xywh_px"] = self._opt_xywh_px_to_xywh_px(quad.dst_xywh_px)
                mem[i]["src_xywh_uv"] = self._opt_xywh_px_to_xywh_uv(quad.src_xywh_px)
                mem[i]["fill_color"] = quad.fill_color
                mem[i]["border_thickness"] = quad.border_thickness_px
                mem[i]["border_color"] = quad.border_color
                mem[i]["depth"] = depth
                mem[i]["flags"] = 0

    def _opt_xywh_px_to_xywh_uv(
        self,
        xywh_px: tuple[int, int, int, int] | None,
    ) -> tuple[float, float, float, float]:
        xywh_px = self._opt_xywh_px_to_xywh_px(xywh_px)
        return (
            xywh_px[0] / self.gpu_image.width,
            xywh_px[1] / self.gpu_image.height,
            xywh_px[2] / self.gpu_image.width,
            xywh_px[3] / self.gpu_image.height,
        )

    def _opt_xywh_px_to_xywh_px(
        self,
        xywh_px: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        return xywh_px or (0, 0, self.gpu_image.width, self.gpu_image.height)


class QuadGroupResources(BaseResource):
    gpu_staging_buffer: GpuBuffer
    gpu_device_buffer: GpuBuffer
    gpu_descriptor_set: GpuDescriptorSet

    def __init__(
        self,
        *,
        parent_resource: "QuadGroup",
        gpu_staging_buffer: GpuBuffer,
        gpu_device_buffer: GpuBuffer,
        gpu_descriptor_set: GpuDescriptorSet,
    ):
        super().__init__(parent_resource=parent_resource)

        self.gpu_staging_buffer = gpu_staging_buffer
        self.gpu_device_buffer = gpu_device_buffer
        self.gpu_descriptor_set = gpu_descriptor_set

    def _on_dispose(self) -> None:
        self.gpu_staging_buffer.dispose()
        self.gpu_device_buffer.dispose()
        self.gpu_descriptor_set.dispose()


QUAD_DTYPE = np.dtype(
    [
        ("dst_xywh_px", "<i4", 4),
        ("src_xywh_uv", "<f4", 4),
        ("fill_color", "<f4", 4),
        ("border_thickness", "<i4", 4),
        ("border_color", "<f4", 4),
        ("depth", "<f4"),
        ("flags", "<u4"),
        ("_rsv", "<i4", 2),
    ]
)


#
# Helpers:
#


@dataclass(kw_only=True, frozen=True)
class QuadWithDepth:
    inner: Draw2dQuad
    depth: float


@dataclass(kw_only=True)
class Span:
    begin: int
    size: int

    @property
    def end(self) -> int:
        return self.begin + self.size


#
# Logging:
#

LOG = logger(__name__)
