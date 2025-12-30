from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import numpy as np

from .basic import BaseResource
from .bundled_data import BUNDLED_DATA_PATH
from .gpu import (
    GpuCommandEncoder,
    GpuDescriptorSet,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuEzBuffer,
    GpuImage,
    GpuImageMeta,
    GpuPipeline,
    GpuPipelineLayout,
    GpuSampler,
    GpuShader,
    GpuVertexAttribute,
    GpuVertexBufferLayout,
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


# Camera uniform buffer dtype matching the shader CameraUniform struct
CAMERA_UNIFORM_DTYPE = np.dtype(
    [
        ("projectionView", np.float32, (4, 4)),
        ("inverseProjectionView", np.float32, (4, 4)),
        ("cameraPosition", np.float32, 3),
        ("iblSamples", np.uint32),
    ]
)


class Draw3dRenderer(BaseResource):
    # References to parent resources
    context: Draw3dContext

    # GPU device for this renderer:
    gpu_device: GpuDevice

    # GPU resources: camera uniform (for main mesh rendering)
    camera_uniform_ds_layout: GpuDescriptorSetLayout
    camera_uniform_buffer: GpuEzBuffer
    camera_uniform_ds: GpuDescriptorSet

    # GPU resources: environment camera uniform (for environment rendering)
    env_camera_uniform_ds_layout: GpuDescriptorSetLayout
    env_camera_uniform_buffer: GpuEzBuffer
    env_camera_uniform_ds: GpuDescriptorSet

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

    # GPU resources: environment map descriptor set layout
    environment_ds_layout: GpuDescriptorSetLayout
    environment_sampler: GpuSampler
    default_environment_image: GpuImage
    default_environment_ds: GpuDescriptorSet

    # Pipeline resources for main mesh rendering:
    _main_pipeline_layout: GpuPipelineLayout
    _main_vertex_shader: GpuShader
    _main_fragment_shader: GpuShader
    _cached_main_pipeline: GpuPipeline | None
    _cached_depth_image: GpuImage | None

    # Pipeline resources for environment background rendering:
    _env_pipeline_layout: GpuPipelineLayout
    _env_vertex_shader: GpuShader
    _env_fragment_shader: GpuShader
    _cached_env_pipeline: GpuPipeline | None

    # Cache for environment descriptor sets (keyed by GpuImage id)
    _environment_ds_cache: dict[int, GpuDescriptorSet]

    def __init__(self, context: Draw3dContext, gpu_device: GpuDevice):
        super().__init__(parent_resource=context)

        self.context = context

        self.gpu_device = gpu_device

        # Camera uniform for main mesh rendering (includes inverse PV, position, IBL samples)
        self.camera_uniform_ds_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "cameraUniform": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["vertex", "fragment"],
                    ),
                }.items()
            ),
        )
        # Size: 4x4 matrix (64 bytes) + 4x4 matrix (64 bytes) + vec3 + uint (16 bytes) = 144 bytes
        # But buffer stores as float32 count: 16 + 16 + 4 = 36 floats
        self.camera_uniform_buffer = GpuEzBuffer(
            device=self.gpu_device,
            capacity=1,
            dtype=CAMERA_UNIFORM_DTYPE,
            usages=["uniform"],
        )
        self.camera_uniform_ds = GpuDescriptorSet(
            device=self.gpu_device,
            bindings={
                "cameraUniform": self.camera_uniform_buffer.device_buffer,
            },
            layout=self.camera_uniform_ds_layout,
        )

        # Environment camera uniform (for environment background rendering)
        self.env_camera_uniform_ds_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "cameraUniform": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["vertex", "fragment"],
                    ),
                }.items()
            ),
        )
        # Environment camera needs: inverse PV (64 bytes) + camera position + padding (16 bytes)
        self.env_camera_uniform_buffer = GpuEzBuffer(
            device=self.gpu_device,
            capacity=20,  # 16 floats for inverse PV + 4 floats for position+padding
            dtype=np.float32,
            usages=["uniform"],
        )
        self.env_camera_uniform_ds = GpuDescriptorSet(
            device=self.gpu_device,
            bindings={
                "cameraUniform": self.env_camera_uniform_buffer.device_buffer,
            },
            layout=self.env_camera_uniform_ds_layout,
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
            capacity=MAX_INSTANCES * 16,  # 16 floats per 4x4 matrix
            dtype=np.float32,
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

        # Environment map descriptor set layout and default resources
        self.environment_ds_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "environmentMap": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "environmentSampler": GpuDescriptorSetLayoutBinding(
                        type="sampler",
                        stages=["fragment"],
                    ),
                }.items()
            ),
        )
        self.environment_sampler = GpuSampler(
            device=self.gpu_device,
            min_filter="linear",
            mag_filter="linear",
            address_mode="repeat",
        )
        # Default environment: neutral gray
        self.default_environment_image = GpuImage(
            device=self.gpu_device,
            usages=["texture-binding"],
            data=np.full((32, 32, 4), fill_value=0.5, dtype=np.float32),
        )
        self.default_environment_ds = GpuDescriptorSet(
            device=self.gpu_device,
            bindings={
                "environmentMap": self.default_environment_image,
                "environmentSampler": self.environment_sampler,
            },
            layout=self.environment_ds_layout,
        )

        # Main mesh pipeline layout and shaders
        self._main_pipeline_layout = GpuPipelineLayout(
            device=self.gpu_device,
            descriptor_set_layouts=[
                self.camera_uniform_ds_layout,
                self.instance_transforms_gpu_ds_layout,
                self.material_ds_layout,
                self.environment_ds_layout,
            ],
        )

        self._main_vertex_shader = GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders/draw_3d_main.vert.spv",
            stage="vertex",
        )
        self._main_fragment_shader = GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders/draw_3d_main.frag.spv",
            stage="fragment",
        )

        self._cached_main_pipeline = None
        self._cached_depth_image = None

        # Environment background pipeline layout and shaders
        self._env_pipeline_layout = GpuPipelineLayout(
            device=self.gpu_device,
            descriptor_set_layouts=[
                self.env_camera_uniform_ds_layout,
                self.environment_ds_layout,
            ],
        )

        self._env_vertex_shader = GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders/draw_3d_environment.vert.spv",
            stage="vertex",
        )
        self._env_fragment_shader = GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders/draw_3d_environment.frag.spv",
            stage="fragment",
        )

        self._cached_env_pipeline = None

        # Environment descriptor set cache
        self._environment_ds_cache = {}

    def _on_dispose(self) -> None:
        # Dispose cached pipelines and depth image
        if it := getattr(self, "_cached_main_pipeline", None):
            it.dispose()
        if it := getattr(self, "_cached_env_pipeline", None):
            it.dispose()
        if it := getattr(self, "_cached_depth_image", None):
            it.dispose()

        # Dispose shaders
        if it := getattr(self, "_main_vertex_shader", None):
            it.dispose()
        if it := getattr(self, "_main_fragment_shader", None):
            it.dispose()
        if it := getattr(self, "_env_vertex_shader", None):
            it.dispose()
        if it := getattr(self, "_env_fragment_shader", None):
            it.dispose()

        # Dispose pipeline layouts
        if it := getattr(self, "_main_pipeline_layout", None):
            it.dispose()
        if it := getattr(self, "_env_pipeline_layout", None):
            it.dispose()

        # Dispose descriptor set layouts
        if it := getattr(self, "camera_uniform_ds_layout", None):
            it.dispose()
        if it := getattr(self, "env_camera_uniform_ds_layout", None):
            it.dispose()
        if it := getattr(self, "instance_transforms_gpu_ds_layout", None):
            it.dispose()
        if it := getattr(self, "material_ds_layout", None):
            it.dispose()
        if it := getattr(self, "environment_ds_layout", None):
            it.dispose()

        # Dispose buffers
        if it := getattr(self, "camera_uniform_buffer", None):
            it.dispose()
        if it := getattr(self, "env_camera_uniform_buffer", None):
            it.dispose()
        if it := getattr(self, "instance_transforms_buffer", None):
            it.dispose()

        # Dispose material resources
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

        # Dispose environment resources
        if it := getattr(self, "environment_sampler", None):
            it.dispose()
        if it := getattr(self, "default_environment_image", None):
            it.dispose()

    def get_environment_descriptor_set(
        self, environment_map: GpuImage
    ) -> GpuDescriptorSet:
        """
        Get or create a descriptor set for an environment map.

        :param environment_map: The environment map image (equirectangular projection).
        :return: A descriptor set binding the environment map.
        """
        cache_key = id(environment_map)
        if cache_key not in self._environment_ds_cache:
            self._environment_ds_cache[cache_key] = GpuDescriptorSet(
                device=self.gpu_device,
                bindings={
                    "environmentMap": environment_map,
                    "environmentSampler": self.environment_sampler,
                },
                layout=self.environment_ds_layout,
            )
        return self._environment_ds_cache[cache_key]

    def draw(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        meshes: dict[tuple["Draw3dGeometry", "Draw3dMaterial"], np.ndarray],
        camera_transform: np.ndarray,
        camera_intrinsics: "Draw3dCameraIntrinsics",
        target: GpuImage,
        environment_map: GpuImage | None = None,
    ) -> None:
        """
        Draws the given mesh instances with PBR and IBL.

        :param command_encoder: The GPU command encoder to record commands to.
        :param meshes: A mapping of (geometry, material) pairs to a flat float32
            array of N 4x4 model matrices (N*16 floats). Each matrix should be
            in column-major layout (transposed from numpy's row-major default).
        :param camera_transform: A 4x4 matrix representing the camera's world transform.
            The matrix is in world-space, row-major (standard numpy convention).
        :param camera_intrinsics: Camera intrinsic parameters (FOV, clip planes, IBL samples).
        :param target: The GPU image to render to. Used to compute aspect ratio.
        :param environment_map: Optional environment map for IBL (equirectangular projection).
            If None, a default gray environment is used.
        """

        # Compute camera parameters:
        camera_aspect_ratio = target.width / target.height
        camera_proj = camera_intrinsics._projection_matrix(camera_aspect_ratio)
        camera_view = np.linalg.inv(camera_transform)
        camera_pv = camera_proj @ camera_view
        camera_inv_pv = np.linalg.inv(camera_pv)
        camera_position = camera_transform[:3, 3]

        # Update camera uniform buffer for main mesh rendering
        # IMPORTANT: Slang/HLSL float4x4 uses column-major layout by default.
        # NumPy arrays are row-major, so we transpose before flattening.
        camera_uniform_data = np.array(
            [
                (
                    camera_pv.T,  # projectionView (column-major)
                    camera_inv_pv.T,  # inverseProjectionView (column-major)
                    camera_position,  # cameraPosition
                    camera_intrinsics.ibl_samples,  # iblSamples
                )
            ],
            dtype=CAMERA_UNIFORM_DTYPE,
        )
        self.camera_uniform_buffer.clear()
        self.camera_uniform_buffer.extend(values=camera_uniform_data)
        self.camera_uniform_buffer.flush(command_encoder=command_encoder)

        # Update environment camera uniform buffer
        env_camera_data = np.zeros(20, dtype=np.float32)
        env_camera_data[:16] = camera_inv_pv.T.ravel()  # inverseProjectionView
        env_camera_data[16:19] = camera_position  # cameraPosition
        env_camera_data[19] = 0.0  # padding
        self.env_camera_uniform_buffer.clear()
        self.env_camera_uniform_buffer.extend(values=env_camera_data)
        self.env_camera_uniform_buffer.flush(command_encoder=command_encoder)

        # Transpose model matrices to column-major layout and ravel, preparing instance
        # transforms buffer for GPU upload:
        meshes_column_major = {}
        for key, model_matrices in meshes.items():
            if model_matrices.dtype.type != np.float32:
                raise ValueError("Model matrices array must have dtype float32")
            if model_matrices.ndim != 3 or model_matrices.shape[1:] != (4, 4):
                raise ValueError("Model matrices array must have shape (N, 4, 4)")
            meshes_column_major[key] = model_matrices.transpose(0, 2, 1).reshape(-1)

        # Group mesh instances: matrix_batches[material][geometry] = model_matrices
        per_material_col_major_matrix_batches: defaultdict[
            Draw3dMaterial,
            dict[Draw3dGeometry, np.ndarray],
        ] = defaultdict(dict)
        for (geometry, material), model_matrices in meshes_column_major.items():
            per_material_col_major_matrix_batches[material][geometry] = model_matrices

        # Write each model_matrices array to a subspan of the global instance transforms
        # buffer, extending the buffer as we go. Record the spans for each batch. These
        # become dynamic offsets when binding the instance transforms buffer.
        # Note: model_matrices is a flat float32 array (N matrices = N*16 floats).
        # The spans are stored as (matrix_offset, matrix_count) for draw_indexed.
        per_material_span_batches: defaultdict[
            Draw3dMaterial,
            dict[Draw3dGeometry, tuple[int, int]],
        ] = defaultdict(dict)
        self.instance_transforms_buffer.clear()
        matrix_offset = 0
        for material, geometry_dict in per_material_col_major_matrix_batches.items():
            for geometry, model_matrices in geometry_dict.items():
                float_count = len(model_matrices)
                assert float_count % 16 == 0, "4x4 model matrices must have 16 floats"
                matrix_count = float_count // 16
                per_material_span_batches[material][geometry] = (
                    matrix_offset,
                    matrix_count,
                )
                matrix_offset += matrix_count
                self.instance_transforms_buffer.extend(values=model_matrices)
        self.instance_transforms_buffer.flush(command_encoder=command_encoder)

        # Get or create pipelines and depth image:
        env_pipeline = self._get_env_pipeline(target)
        main_pipeline = self._get_main_pipeline(target)
        depth_image = self._get_depth_image(target.width, target.height)

        # Get or create environment descriptor set
        if environment_map is not None:
            env_ds = self.get_environment_descriptor_set(environment_map)
        else:
            env_ds = self.default_environment_ds

        # Transition images to correct layouts:
        command_encoder.transition_image_layout(
            image=target,
            layout="color-attachment-optimal",
        )
        command_encoder.transition_image_layout(
            image=depth_image,
            layout="depth-stencil-attachment-optimal",
        )
        if environment_map is not None:
            command_encoder.transition_image_layout(
                image=environment_map,
                layout="texture-binding",
            )

        # Layer 1: Draw environment map background at infinity
        with command_encoder.render(
            color_attachment=target,
            depth_attachment=None,
            clear_color="black",
        ) as rp:
            rp.bind_pipeline(pipeline=env_pipeline)
            rp.bind_descriptor_set(set_index=0, set_=self.env_camera_uniform_ds)
            rp.bind_descriptor_set(set_index=1, set_=env_ds)
            rp.draw(vertex_count=3, instance_count=1)

        # Layer 2: Draw meshes with depth test
        with command_encoder.render(
            color_attachment=target,
            depth_attachment=depth_image,
            clear_color=None,  # Load previous content (environment background)
            clear_depth=True,  # Clear depth buffer (fresh start for mesh rendering)
        ) as rp:
            # Set pipeline:
            rp.bind_pipeline(pipeline=main_pipeline)

            # Bind camera descriptor set:
            rp.bind_descriptor_set(
                set_index=0,
                set_=self.camera_uniform_ds,
            )

            # Bind global per-instance transforms descriptor set:
            rp.bind_descriptor_set(
                set_index=1,
                set_=self.instance_transforms_gpu_ds,
            )

            # Bind environment map descriptor set:
            rp.bind_descriptor_set(set_index=3, set_=env_ds)

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

        # Clean up temporary descriptor set if we created one
        if environment_map is not None:
            env_ds.dispose()

    def _get_main_pipeline(self, target: GpuImage) -> GpuPipeline:
        """Get or create a pipeline for main mesh rendering."""
        if self._cached_main_pipeline is not None:
            if (
                self._cached_main_pipeline.vk_color_format == target._vk_format
                and self._cached_main_pipeline.viewport_width == target.width
                and self._cached_main_pipeline.viewport_height == target.height
            ):
                return self._cached_main_pipeline
            self._cached_main_pipeline.dispose()

        # Vertex layout matching VERTEX_DTYPE:
        # - position: float32x3 at offset 0
        # - normal: float32x3 at offset 12
        # - texcoord0: float32x2 at offset 24
        # Total stride: 32 bytes
        vertex_layout = GpuVertexBufferLayout(
            stride=VERTEX_DTYPE.itemsize,
            attributes=[
                GpuVertexAttribute(format="float32x3", offset=0),  # position
                GpuVertexAttribute(format="float32x3", offset=12),  # normal
                GpuVertexAttribute(format="float32x2", offset=24),  # texcoord0
            ],
        )

        self._cached_main_pipeline = GpuPipeline(
            device=self.gpu_device,
            vertex_shader=self._main_vertex_shader,
            fragment_shader=self._main_fragment_shader,
            vk_color_format=target._vk_format,
            enable_depth_test=True,
            enable_alpha_blending=False,
            viewport_width=target.width,
            viewport_height=target.height,
            layout=self._main_pipeline_layout,
            vertex_buffer_layouts=[vertex_layout],
        )
        return self._cached_main_pipeline

    def _get_env_pipeline(self, target: GpuImage) -> GpuPipeline:
        """Get or create a pipeline for environment background rendering."""
        if self._cached_env_pipeline is not None:
            if (
                self._cached_env_pipeline.vk_color_format == target._vk_format
                and self._cached_env_pipeline.viewport_width == target.width
                and self._cached_env_pipeline.viewport_height == target.height
            ):
                return self._cached_env_pipeline
            self._cached_env_pipeline.dispose()

        # No vertex buffers needed - fullscreen triangle generated in shader
        self._cached_env_pipeline = GpuPipeline(
            device=self.gpu_device,
            vertex_shader=self._env_vertex_shader,
            fragment_shader=self._env_fragment_shader,
            vk_color_format=target._vk_format,
            enable_depth_test=False,
            enable_alpha_blending=False,
            viewport_width=target.width,
            viewport_height=target.height,
            layout=self._env_pipeline_layout,
            vertex_buffer_layouts=[],
        )
        return self._cached_env_pipeline

    def _get_depth_image(self, width: int, height: int) -> GpuImage:
        """Get or create a depth image for the given dimensions."""
        if (
            self._cached_depth_image is not None
            and self._cached_depth_image.width == width
            and self._cached_depth_image.height == height
        ):
            return self._cached_depth_image

        if self._cached_depth_image is not None:
            self._cached_depth_image.dispose()

        self._cached_depth_image = GpuImage(
            device=self.gpu_device,
            usages=["depth-attachment"],
            meta=GpuImageMeta(shape=(height, width, 1), dtype="<f4"),
        )
        return self._cached_depth_image


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

    ibl_samples: int = 1
    """Number of samples for IBL specular importance sampling. Default is 4."""

    def _projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """
        Compute a 4x4 perspective projection matrix for rendering.

        Uses reverse-Z for Vulkan's [0, 1] depth range:
        - Near plane maps to depth 1.0
        - Far plane maps to depth 0.0

        This provides better depth precision for distant objects.

        :param aspect_ratio: The aspect ratio (width / height) of the target image.
        :return: A 4x4 projection matrix, row-major.
        """
        f = 1.0 / np.tan(self.vertical_fov / 2.0)
        a = aspect_ratio
        n = self.clip_near
        m = self.clip_far  # m for 'max'

        # Reverse-Z projection for Vulkan [0, 1] depth range:
        # z_ndc = (A * z_eye + B) / (-z_eye)
        # At z_eye = -n: z_ndc = 1  =>  A = n / (m - n)
        # At z_eye = -m: z_ndc = 0  =>  B = n * m / (m - n)
        #
        # Note: Vulkan's clip space has Y pointing down, so we negate the Y
        # component to flip the image right-side up.
        return np.array(
            [
                [f / a, 0.0, 0.0, 0.0],
                [0.0, -f, 0.0, 0.0],
                [0.0, 0.0, n / (m - n), (n * m) / (m - n)],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
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
                        (0.0, 0.0),  # padding
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
        if tint_buffer := getattr(self, "tint_buffer", None):
            tint_buffer.dispose()


TINT_BUFFER_DTYPE = np.dtype(
    [
        ("color", np.float32, 4),
        ("metalness", np.float32),
        ("roughness", np.float32),
        ("_pad0", np.float32, 2),
    ]
)
