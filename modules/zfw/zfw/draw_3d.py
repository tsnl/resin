from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

from .basic import BaseResource
from .gpu import (
    GpuCommandEncoder,
    GpuDescriptorSet,
    GpuDescriptorSetBinding,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuEzBuffer,
    GpuImage,
    GpuImageMeta,
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
    material_default_color_image: GpuImage
    material_default_normal_image: GpuImage
    material_default_metalness_image: GpuImage
    material_default_roughness_image: GpuImage

    def __init__(self, context: Draw3dContext, gpu_device: GpuDevice):
        super().__init__(parent_resource=context)

        self.context = context

        self.gpu_device = gpu_device

        self.material_gpu_descriptor_set_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "tint": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["fragment"],
                    ),
                    "colorImage": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "normalImage": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "metalnessImage": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
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
        self.material_default_color_image = GpuImage(
            device=self.gpu_device,
            usages=["texture-binding"],
            data=np.ones((32, 32, 4), dtype=np.float32),
        )
        self.material_default_normal_image = GpuImage(
            device=self.gpu_device,
            usages=["texture-binding"],
            data=np.full(
                (32, 32, 4),
                fill_value=(0.0, 0.0, 1.0, 1.0),
                dtype=np.float32,
            ),
        )
        self.material_default_metalness_image = GpuImage(
            device=self.gpu_device,
            usages=["texture-binding"],
            data=np.ones(shape=(32, 32, 1), dtype=np.float32),
        )
        self.material_default_roughness_image = GpuImage(
            device=self.gpu_device,
            usages=["texture-binding"],
            data=np.ones(shape=(32, 32, 1), dtype=np.float32),
        )

        self.geometry_id_to_instance_map = {}
        self.material_id_to_instance_map = {}
        self.mesh_instances = {}

    def _on_dispose(self) -> None:
        if it := getattr(self, "material_gpu_descriptor_set_layout", None):
            it.dispose()

        if it := getattr(self, "material_linear_sampler", None):
            it.dispose()

        if it := getattr(self, "material_default_color_image", None):
            it.dispose()

        if it := getattr(self, "material_default_normal_image", None):
            it.dispose()

        if it := getattr(self, "material_default_metalness_image", None):
            it.dispose()

        if it := getattr(self, "material_default_roughness_image", None):
            it.dispose()

    def draw(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        mesh_instances: dict[tuple["Draw3dGeometry", "Draw3dMaterial"], np.ndarray],
    ) -> None:
        """
        Draws the given mesh instances.

        :param command_encoder: The GPU command encoder to record commands to.
        :param mesh_instances: A mapping of (geometry, material) pairs to an Nx4x4 array
            of model matrices for N instances of that geometry with that material.
        """

        raise NotImplementedError()


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
        self.vertex_buffer.extend(values=vertex_data)

        self.index_buffer = GpuEzBuffer(
            device=self.gpu_device,
            capacity=len(index_data),
            dtype=np.dtype(np.uint16),
            usages=["index"],
        )
        self.index_buffer.extend(values=index_data)

        # Flush buffers to GPU
        self.vertex_buffer.flush()
        self.index_buffer.flush()

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
# Draw3dMaterial
#


class Draw3dMaterial(BaseResource):
    # References to parent resources
    renderer: Draw3dRenderer
    gpu_device: GpuDevice

    # Material properties
    tint_buffer: GpuEzBuffer
    color_image: GpuImage | None
    normal_image: GpuImage | None
    metalness_image: GpuImage | None
    roughness_image: GpuImage | None

    # Descriptor set:
    descriptor_set: GpuDescriptorSet

    def __init__(
        self,
        renderer: Draw3dRenderer,
        color_tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
        color_image: GpuImage | None = None,
        normal_image: GpuImage | None = None,
        metalness_tint: float = 1.0,
        metalness_image: GpuImage | None = None,
        roughness_tint: float = 1.0,
        roughness_image: GpuImage | None = None,
    ):
        super().__init__(parent_resource=renderer)

        self.renderer = renderer
        self.gpu_device = renderer.gpu_device

        self.tint_buffer = GpuEzBuffer(
            device=self.gpu_device,
            capacity=1,
            dtype=TINT_BUFFER_DTYPE,
            usages=["uniform"],
        )
        self.tint_buffer.extend(
            values=np.array(
                [
                    (
                        (*color_tint, 1.0),
                        metalness_tint,
                        roughness_tint,
                    ),
                ],
                dtype=TINT_BUFFER_DTYPE,
            )
        )
        self.tint_buffer.flush()

        self.color_image = color_image
        self.normal_image = normal_image
        self.metalness_image = metalness_image
        self.roughness_image = roughness_image

        self.descriptor_set = GpuDescriptorSet(
            device=self.gpu_device,
            bindings={
                "tint": self.tint_buffer.device_buffer,
                "colorImage": (
                    self.color_image or self.renderer.material_default_color_image
                ),
                "normalImage": (
                    self.normal_image or self.renderer.material_default_normal_image
                ),
                "metalnessImage": (
                    self.metalness_image
                    or self.renderer.material_default_metalness_image
                ),
                "roughnessImage": (
                    self.roughness_image
                    or self.renderer.material_default_roughness_image
                ),
                "sampler": self.renderer.material_linear_sampler,
            },
            layout=self.renderer.material_gpu_descriptor_set_layout,
        )

    def _on_dispose(self) -> None:
        if descriptor_set := getattr(self, "descriptor_set", None):
            descriptor_set.dispose()


TINT_BUFFER_DTYPE = np.dtype(
    [
        ("color", np.float32, 4),
        ("metalness", np.float32, 1),
        ("roughness", np.float32, 1),
    ]
)
