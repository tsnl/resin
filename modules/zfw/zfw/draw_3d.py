from collections import OrderedDict
import numpy as np

from .basic import BaseResource
from .gpu import (
    GpuCommandEncoder,
    GpuDescriptorSet,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuEzBuffer,
    GpuImage,
    GpuSampler,
)


#
# Draw3dContext
#


class Draw3dContext(BaseResource):
    def __init__(self):
        super().__init__(parent_resource=None)


#
# Draw3dRenderer
#


class Draw3dRenderer(BaseResource):
    # References to parent resources
    context: Draw3dContext

    # GPU device for this renderer:
    gpu_device: GpuDevice

    # GPU resources:
    material_gpu_descriptor_set_layout: GpuDescriptorSetLayout
    material_linear_sampler: GpuSampler

    def __init__(self, context: Draw3dContext, gpu_device: GpuDevice):
        super().__init__(parent_resource=context)

        self.context = context

        self.gpu_device = gpu_device

        self.material_gpu_descriptor_set_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "colorTint": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["fragment"],
                    ),
                    "colorImage": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "normalTint": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["fragment"],
                    ),
                    "normalImage": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "metalnessTint": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["fragment"],
                    ),
                    "metalnessImage": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "roughnessTint": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["fragment"],
                    ),
                    "roughnessImage": GpuDescriptorSetLayoutBinding(
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
        self.material_linear_sampler = GpuSampler(
            device=self.gpu_device,
            min_filter="linear",
            mag_filter="linear",
        )


#
# Draw3dGeometry
#


class Draw3dGeometry(BaseResource):
    # References to parent resources
    renderer: Draw3dRenderer
    gpu_device: GpuDevice

    # GPU resources:
    vertex_buffer: GpuEzBuffer
    index_buffer: GpuEzBuffer

    def __init__(
        self,
        renderer: Draw3dRenderer,
        vertex_data: np.ndarray,
        index_data: np.ndarray,
    ):
        """
        Initialize a Draw3dGeometry.
        """

        assert vertex_data.dtype == VERTEX_DTYPE
        assert index_data.dtype

        super().__init__(parent_resource=renderer)

        self.renderer = renderer
        self.gpu_device = renderer.gpu_device

        self.vertex_buffer = GpuEzBuffer(
            device=self.gpu_device,
            capacity=len(vertex_data),
            dtype=VERTEX_DTYPE,
            usages=["vertex"],
        )
        self.vertex_buffer.array[:] = vertex_data

        self.index_buffer = GpuEzBuffer(
            device=self.gpu_device,
            capacity=len(index_data),
            dtype=np.dtype(np.uint16),
            usages=["index"],
        )
        self.index_buffer.array[:] = index_data

        # Flush buffers to GPU
        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="transfer",
        )
        self.vertex_buffer.flush(command_encoder=command_encoder)
        self.index_buffer.flush(command_encoder=command_encoder)
        command_encoder.submit().wait()

    def _on_dispose(self) -> None:
        if vertex_buffer := getattr(self, "vertex_buffer", None):
            vertex_buffer.dispose()
        if index_buffer := getattr(self, "index_buffer", None):
            index_buffer.dispose()


VERTEX_DTYPE = np.dtype(
    [
        ("position", np.float32, 3),
        ("normal", np.float32, 3),
        ("texcoord0", np.float32, 2),
    ]
)


#
# Material
#


class Draw3dMaterial(BaseResource):
    # References to parent resources
    renderer: Draw3dRenderer
    gpu_device: GpuDevice

    # Material properties
    color_tint: tuple[float, float, float]
    color_image: GpuImage | None
    normal_tint: tuple[float, float, float]
    normal_image: GpuImage | None
    metalness_tint: float
    metalness_image: GpuImage | None
    roughness_tint: float
    roughness_image: GpuImage | None

    # Descriptor set:
    descriptor_set: GpuDescriptorSet

    def __init__(
        self,
        renderer: Draw3dRenderer,
        color_tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
        color_image: GpuImage | None = None,
        normal_tint: tuple[float, float, float] = (0.0, 0.0, 1.0),
        normal_image: GpuImage | None = None,
        metalness_tint: float = 1.0,
        metalness_image: GpuImage | None = None,
        roughness_tint: float = 1.0,
        roughness_image: GpuImage | None = None,
    ):
        super().__init__(parent_resource=renderer)

        self.renderer = renderer
        self.gpu_device = renderer.gpu_device

        self.color_tint = color_tint
        self.color_image = color_image
        self.normal_tint = normal_tint
        self.normal_image = normal_image
        self.metalness_tint = metalness_tint
        self.metalness_image = metalness_image
        self.roughness_tint = roughness_tint
        self.roughness_image = roughness_image

        raise NotImplementedError()
        # self.descriptor_set = GpuDescriptorSet(device=self.gpu_device, bindings={})
