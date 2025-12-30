from collections import OrderedDict, defaultdict
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

    # GPU resources: camera uniform
    camera_pv_matrix_uniform_ds_layout: GpuDescriptorSetLayout
    camera_pv_matrix_uniform_buffer: GpuEzBuffer
    camera_pv_matrix_uniform_ds: GpuDescriptorSet

    # GPU resources: material descriptor set layout and default resources
    material_ds_layout: GpuDescriptorSetLayout
    material_linear_sampler: GpuSampler
    material_default_color_image: GpuImage
    material_default_normal_image: GpuImage
    material_default_metalness_image: GpuImage
    material_default_roughness_image: GpuImage

    # GPU resources: per-instance model matrix buffer and descriptor set.
    # We use a single large buffer and dynamic offsets.
    instance_transforms_buffer: GpuEzBuffer
    instance_transforms_gpu_ds_layout: GpuDescriptorSetLayout
    instance_transforms_gpu_ds: GpuDescriptorSet

    def __init__(self, context: Draw3dContext, gpu_device: GpuDevice):
        super().__init__(parent_resource=context)

        self.context = context

        self.gpu_device = gpu_device

        # Camera uniform
        self.camera_pv_matrix_uniform_ds_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "cameraPv": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["vertex"],
                    ),
                }.items()
            ),
        )
        self.camera_pv_matrix_uniform_buffer = GpuEzBuffer(
            device=self.gpu_device,
            capacity=1,
            dtype=MAT4X4F_DTYPE,
            usages=["uniform"],
        )
        self.camera_pv_matrix_uniform_ds = GpuDescriptorSet(
            device=self.gpu_device,
            bindings={
                "cameraPv": self.camera_pv_matrix_uniform_buffer.device_buffer,
            },
            layout=self.camera_pv_matrix_uniform_ds_layout,
        )

        # Material descriptor set layout and default resources
        self.material_ds_layout = GpuDescriptorSetLayout(
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

        # Per-instance model matrix buffer and descriptor set
        MAX_INSTANCES = 16 << 10  # 16K instances
        self.instance_transforms_buffer = GpuEzBuffer(
            device=self.gpu_device,
            capacity=MAX_INSTANCES,
            dtype=MAT4X4F_DTYPE,
            usages=["storage"],
        )
        self.instance_transforms_gpu_ds_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "instanceTransforms": GpuDescriptorSetLayoutBinding(
                        type="storage-buffer",
                        stages=["vertex"],
                    ),
                }.items()
            ),
        )
        self.instance_transforms_gpu_ds = GpuDescriptorSet(
            device=self.gpu_device,
            bindings={
                "instanceTransforms": self.instance_transforms_buffer.device_buffer,
            },
            layout=self.instance_transforms_gpu_ds_layout,
        )

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
        meshes: dict[tuple["Draw3dGeometry", "Draw3dMaterial"], np.ndarray],
        camera_transform: np.ndarray,
        camera_intrinsics: "Draw3dCameraIntrinsics",
        target: GpuImage,
    ) -> None:
        """
        Draws the given mesh instances.

        :param command_encoder: The GPU command encoder to record commands to.
        :param mesh_instances: A mapping of (geometry, material) pairs to an Nx4x4
            tensor of N 4x4 model matrices for N instances of that geometry with that
            material. The matrices are in world space, row-major.
        :param camera_transform: A 4x4 matrix representing the camera's world transform.
            The matrix is in world-space, row-major.
        :param camera_focal_length: The camera's focal length, in meters.
        :param camera_sensor_height: The camera's sensor height, in meters.
        :param target: The GPU image to render to. Used to compute aspect ratio.
        """

        # Compute camera parameters:
        camera_aspect_ratio = target.width / target.height
        camera_proj = camera_intrinsics._projection_matrix(camera_aspect_ratio)
        camera_view = np.linalg.inv(camera_transform)
        camera_pv = camera_proj @ camera_view

        # Update camera uniform buffer:
        self.camera_pv_matrix_uniform_buffer.clear()
        self.camera_pv_matrix_uniform_buffer.extend(
            values=np.array([camera_pv], dtype=MAT4X4F_DTYPE)
        )
        self.camera_pv_matrix_uniform_buffer.flush(command_encoder=command_encoder)

        # Group mesh instances: matrix_batches[material][geometry] = model_matrices
        per_material_matrix_batches: defaultdict[
            Draw3dMaterial,
            dict[Draw3dGeometry, np.ndarray],
        ] = defaultdict(dict)
        for (geometry, material), model_matrices in meshes.items():
            per_material_matrix_batches[material][geometry] = model_matrices

        # Write each model_matrices array to a subspan of the global instance transforms
        # buffer, extending the buffer as we go. Record the spans for each batch. These
        # become dynamic offsets when binding the instance transforms buffer.
        per_material_span_batches: defaultdict[
            Draw3dMaterial,
            dict[Draw3dGeometry, tuple[int, int]],
        ] = defaultdict(dict)
        self.instance_transforms_buffer.clear()
        for material, geometry_dict in per_material_matrix_batches.items():
            for geometry, model_matrices in geometry_dict.items():
                span = self.instance_transforms_buffer.extend(values=model_matrices)
                per_material_span_batches[material][geometry] = span
        self.instance_transforms_buffer.flush(command_encoder=command_encoder)

        # Draw all mesh instances:
        with command_encoder.render(color_attachment=target) as rp:
            # Bind camera descriptor set:
            rp.bind_descriptor_set(
                set_index=0,
                set_=self.camera_pv_matrix_uniform_ds,
            )

            # Bind global per-instance transforms descriptor set:
            rp.bind_descriptor_set(
                set_index=1,
                set_=self.instance_transforms_gpu_ds,
            )

            # For each material, bind material and draw all associated geometries:
            for material, geometry_dict in per_material_span_batches.items():
                # Bind material descriptor set:
                rp.bind_descriptor_set(
                    set_index=2,
                    set_=material.descriptor_set,
                )

                # Bind each geometry and draw:
                for geometry, instance_span in geometry_dict.items():
                    matrix_span_offset, matrix_span_count = instance_span
                    rp.bind_vertex_buffer(buffer=geometry.vertex_buffer.device_buffer)
                    rp.bind_index_buffer(buffer=geometry.index_buffer.device_buffer)
                    rp.draw_indexed(
                        index_count=len(geometry.index_buffer),
                        first_instance=matrix_span_offset,
                        instance_count=matrix_span_count,
                    )


@dataclass
class Draw3dCameraIntrinsics:
    vertical_fov: float
    """
    Vertical field of view, in radians. The vertical field of view is computed based on 
    the aspect ratio of the target image.
    """

    clip_near: float = 1e-2
    """Near clipping plane distance."""

    clip_far: float = 1e3
    """Far clipping plane distance."""

    def _projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """
        Compute a 4x4 perspective projection matrix for rendering.

        :param aspect_ratio: The aspect ratio (width / height) of the target image.
        :return: A 4x4 projection matrix, row-major.
        """
        f = 1.0 / np.tan(self.vertical_fov / 2.0)
        a = aspect_ratio
        n = self.clip_near
        m = self.clip_far  # m for 'max'

        return np.array(
            [
                [f / a, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (m + n) / (n - m), (2 * m * n) / (n - m)],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )


MAT4X4F_DTYPE = np.dtype(("<f4", (4, 4)))


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
            layout=self.renderer.material_ds_layout,
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
