from collections import OrderedDict
import numpy as np

from .gpu import (
    GpuBuffer,
    GpuBufferMeta,
    GpuDescriptorSet,
    GpuDescriptorSetBinding,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuEzBuffer,
    GpuImage,
    GpuSampler,
)
from .basic import BaseResource, logger, round_up_to_po2


#
# Context
#


class RendererContext(BaseResource):
    pass


#
# Renderer
#


class Renderer(BaseResource):
    context: RendererContext
    gpu_device: GpuDevice

    def __init__(self, *, context: RendererContext, device: GpuDevice):
        super().__init__(parent_resource=context)
        self.context = context
        self.gpu_device = device


#
# Canvas
#


class Canvas(BaseResource):
    renderer: Renderer
    gpu_device: GpuDevice
    gpu_quad_batch_descriptor_set_layout: GpuDescriptorSetLayout
    gpu_sampler: GpuSampler
    quad_batch_dict: dict[int, "QuadBatch"]
    quad_count: float

    def __init__(self, renderer: Renderer):
        super().__init__(parent_resource=renderer)
        self.renderer = renderer
        self.gpu_device = self.renderer.gpu_device
        self.gpu_quad_batch_descriptor_set_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
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
        self.gpu_sampler = GpuSampler(
            device=self.gpu_device,
            min_filter="nearest",
            mag_filter="nearest",
        )
        self.quad_batch_dict = {}
        self.quad_count: float = 0.0

    def clear(self):
        self.quad_count = 0

        for batch in self.quad_batch_dict.values():
            batch.gpu_quad_buffer.clear()

    def add_quad(
        self,
        image: GpuImage,
        dst_xywh_px: tuple[int, int, int, int],
        src_xywh_px: tuple[int, int, int, int],
        border_thickness: tuple[int, int, int, int],
        color: tuple[int, int, int, int],
        border_color: tuple[int, int, int, int],
    ):
        assert all(0 <= ct <= 255 for ct in border_thickness)
        assert all(0 <= c <= 255 for c in color)
        assert all(0 <= bc <= 255 for bc in border_color)

        src_xywh_uv = (
            src_xywh_px[0] / image.width,
            src_xywh_px[1] / image.height,
            src_xywh_px[2] / image.width,
            src_xywh_px[3] / image.height,
        )

        self._get_quad_batch(image).gpu_quad_buffer.extend(
            values=np.array(
                [
                    (
                        dst_xywh_px,
                        src_xywh_uv,
                        self.quad_count,
                        border_thickness,
                        color,
                        border_color,
                    )
                ],
                dtype=QUAD_DTYPE,
            )
        )
        self.quad_count += 1.0

    def _get_quad_batch(self, image: GpuImage) -> "QuadBatch":
        image_id = image.get_unique_image_id()

        if quad_batch := self.quad_batch_dict.get(image_id):
            return quad_batch

        quad_batch = QuadBatch(canvas=self, gpu_image=image)
        self.quad_batch_dict[image_id] = quad_batch

        return quad_batch


class QuadBatch(BaseResource):
    # Parent/owner resource references:
    canvas: Canvas
    gpu_device: GpuDevice

    # Owned resources:
    gpu_image: GpuImage
    gpu_quad_buffer: GpuEzBuffer
    gpu_descriptor_set: GpuDescriptorSet

    def __init__(self, *, canvas: Canvas, gpu_image: GpuImage):
        super().__init__(parent_resource=canvas)

        self.canvas = canvas
        self.gpu_device = canvas.gpu_device

        self.gpu_image = gpu_image
        self.gpu_quad_buffer = GpuEzBuffer(
            device=self.gpu_device,
            dtype=QUAD_DTYPE,
            capacity=1,
            usages=["storage"],
        )
        self.gpu_descriptor_set = GpuDescriptorSet(
            layout=self.canvas.gpu_quad_batch_descriptor_set_layout,
            device=self.gpu_device,
            bindings={
                "quads": self.gpu_quad_buffer.device_buffer,
                "image": self.gpu_image,
                "sampler": self.canvas.gpu_sampler,
            },
        )

    def _on_dispose(self) -> None:
        self.gpu_image.dispose()
        self.gpu_quad_buffer.dispose()
        self.gpu_descriptor_set.dispose()


QUAD_DTYPE = np.dtype(
    [
        ("dst_xywh_px", np.int32, 4),
        ("src_xywh_uv", np.float32, 4),
        ("depth", np.float32),
        ("border_thickness", np.uint8, 4),
        ("color", np.uint8, 4),
        ("border_color", np.uint8, 4),
    ]
)


#
# RendererDynamicAtlas
#


#
# Logger
#

LOG = logger(__name__)
