__all__ = [
    "Draw3dFrame",
    "Draw3dRenderer",
    "Draw3dScene",
]

import math
from dataclasses import dataclass, field

import numba
import numpy as np
import jaxtyping as jt
from typeguard import typechecked
import wgpu

from .basic import BaseDisposable, StructuredNDArray

#
# Renderer
#


class Draw3dRenderer:
    device: wgpu.GPUDevice
    queue: wgpu.GPUQueue

    target_size_wh_px: tuple[int, int]

    instance_capacity: int
    geometry_capacity: int
    bvh_node_capacity: int
    triangle_capacity: int

    def __init__(
        self,
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
        target_size_wh_px: tuple[int, int],
        instance_capacity: int = 1 << 10,
        geometry_capacity: int = 1 << 8,
        bvh_node_capacity: int = 1 << 18,
        triangle_capacity: int = 1 << 20,
    ):
        self.device = device
        self.queue = queue

        self.target_size_wh_px = target_size_wh_px

        self.instance_capacity = instance_capacity
        self.geometry_capacity = geometry_capacity
        self.bvh_node_capacity = bvh_node_capacity
        self.triangle_capacity = triangle_capacity

        self.renderer_bind_group_layout = device.create_bind_group_layout(
            label="Draw3dRenderer.RendererBindGroupLayout",
            entries=[
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=2,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
            ],
        )

        self.per_frame_bind_group_layout = device.create_bind_group_layout(
            label="Draw3dRenderer.PerFrameBindGroupLayout",
            entries=[
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba32float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="uniform"),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=2,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="uniform"),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=3,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="read-only-storage"),
                ),
            ],
        )

        pipeline_layout = device.create_pipeline_layout(
            label="Draw3dRenderer.PipelineLayout",
            bind_group_layouts=[
                self.renderer_bind_group_layout,
                self.per_frame_bind_group_layout,
            ],
        )

        with open(__file__.replace(".py", ".wgsl"), "r") as f:
            shader_source = f.read()

        draw_shader = device.create_shader_module(code=shader_source)

        self.pipeline = device.create_compute_pipeline(
            label="Draw3dRenderer.DrawPipeline",
            layout=pipeline_layout,
            compute=wgpu.ProgrammableStage(
                module=draw_shader,
                entry_point="main_wrapper",
            ),
        )

        self.geometry_heap_device_buffer = device.create_buffer(
            label="Draw3dRenderer.GeometryHeapDeviceBuffer",
            size=PodGeometryArray.array_size(shape=(self.geometry_capacity,)),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.bvh_node_heap_device_buffer = device.create_buffer(
            label="Draw3dRenderer.BvhNodeHeapDeviceBuffer",
            size=PodBvhNodeArray.array_size(shape=(self.bvh_node_capacity,)),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.triangle_vertex_offset_heap_device_buffer = device.create_buffer(
            label="Draw3dRenderer.TriangleVertexOffsetHeapDeviceBuffer",
            size=PodVertexOffsetArray.array_size(shape=(self.triangle_capacity, 3)),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.triangle_vertex_detail_heap_device_buffer = device.create_buffer(
            label="Draw3dRenderer.TriangleVertexDetailHeapDeviceBuffer",
            size=0,  # Not used yet
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        self.renderer_bind_group = device.create_bind_group(
            label="Draw3dRenderer.RendererBindGroup",
            layout=self.renderer_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=wgpu.BufferBinding(
                        buffer=self.geometry_heap_device_buffer,
                        offset=0,
                        size=self.geometry_heap_device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=wgpu.BufferBinding(
                        buffer=self.bvh_node_heap_device_buffer,
                        offset=0,
                        size=self.bvh_node_heap_device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=wgpu.BufferBinding(
                        buffer=self.triangle_vertex_offset_heap_device_buffer,
                        offset=0,
                        size=self.triangle_vertex_offset_heap_device_buffer.size,
                    ),
                ),
            ],
        )

        self.allocated_geometry_count = 0
        self.allocated_bvh_node_count = 0
        self.allocated_triangle_count = 0

        # Initialization: clear device buffers to zero
        encoder = device.create_command_encoder(
            label="Draw3dRenderer.InitializationEncoder"
        )
        encoder.clear_buffer(self.geometry_heap_device_buffer, offset=0)
        encoder.clear_buffer(self.bvh_node_heap_device_buffer, offset=0)
        encoder.clear_buffer(self.triangle_vertex_offset_heap_device_buffer, offset=0)
        queue.submit([encoder.finish()])

    def _add_triangles(self, vertices: PodVertexOffsetArray) -> int:
        assert vertices.ndim == 1 and vertices.dtype == PodVertexOffsetArray.DTYPE

        vertex_count = vertices.shape[0]
        triangle_count = vertex_count // 3

        # Allocate:
        allocation_offset_in_triangles = self.allocated_triangle_count
        allocation_offset_in_bytes = (
            allocation_offset_in_triangles * PodVertexOffsetArray.DTYPE.itemsize * 3
        )
        self.allocated_triangle_count += triangle_count
        if self.allocated_triangle_count > self.triangle_capacity:
            raise RuntimeError("Draw3dRenderer triangle heap capacity exceeded.")

        # Upload via staging buffer:
        staging_buffer = self.device.create_buffer(
            label="Draw3dRenderer.GeometryUploadStagingBuffer",
            size=vertices.nbytes,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )
        staging_buffer.map_sync(mode=wgpu.MapMode.WRITE)
        staging_buffer.write_mapped(data=vertices)
        staging_buffer.unmap()

        encoder = self.device.create_command_encoder(
            label="Draw3dRenderer.GeometryUploadEncoder"
        )
        encoder.copy_buffer_to_buffer(
            source=staging_buffer,
            source_offset=0,
            destination=self.triangle_vertex_offset_heap_device_buffer,
            destination_offset=allocation_offset_in_bytes,
            size=vertices.nbytes,
        )
        self.queue.submit([encoder.finish()])

        # Return offset in triangles:
        return allocation_offset_in_triangles

    def _add_geometry(
        self,
        triangle_span_begin: int,
        triangle_count: int,
    ):
        data = PodGeometryArray.empty(shape=(1,))

        data["triangle_span"]["begin"][0] = triangle_span_begin
        data["triangle_span"]["end"][0] = triangle_span_begin + triangle_count

        # TODO: build BVH for the geometry
        data["bvh_node_span"][0]["begin"] = 0
        data["bvh_node_span"][0]["end"] = 0

        # Allocate:
        allocation_offset = self.allocated_geometry_count
        self.allocated_geometry_count += 1
        if self.allocated_geometry_count > self.geometry_capacity:
            raise RuntimeError("Draw3dRenderer geometry heap capacity exceeded.")

        # Upload via staging buffer:
        staging_buffer = self.device.create_buffer(
            label="Draw3dRenderer.GeometryUploadStagingBuffer",
            size=PodGeometryArray.array_size(shape=(1,)),
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )
        staging_buffer.map_sync(mode=wgpu.MapMode.WRITE)
        staging_buffer.write_mapped(data=data)
        staging_buffer.unmap()

        encoder = self.device.create_command_encoder(
            label="Draw3dRenderer.GeometryUploadEncoder"
        )
        encoder.copy_buffer_to_buffer(
            source=staging_buffer,
            source_offset=0,
            destination=self.geometry_heap_device_buffer,
            destination_offset=allocation_offset
            * PodGeometryArray.array_size(shape=(1,)),
            size=PodGeometryArray.array_size(shape=(1,)),
        )
        self.queue.submit([encoder.finish()])

        # Return offset in geometry:
        return allocation_offset

    def record(
        self,
        scene: Draw3dScene,
        frame: "Draw3dFrame",
        command_encoder: wgpu.GPUCommandEncoder,
    ) -> None:
        frame.record(
            self.pipeline,
            self.renderer_bind_group,
            self.target_size_wh_px,
            command_encoder,
            scene,
        )


class Draw3dFrame:
    renderer: Draw3dRenderer

    def __init__(self, renderer: Draw3dRenderer) -> None:
        self.renderer = renderer
        self._debug_flags = 0

        self.output_image = self._device.create_texture(
            label="Draw3dFrame.OutputImage",
            size=(renderer.target_size_wh_px[0], renderer.target_size_wh_px[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.frame_info_device_buffer = self._device.create_buffer(
            label="Draw3dFrame.FrameInfoDeviceBuffer",
            size=PodFrameInfoArray.array_size(shape=(1,)),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.frame_info_staging_buffer = self._device.create_buffer(
            label="Draw3dFrame.FrameInfoStagingBuffer",
            size=PodFrameInfoArray.array_size(shape=(1,)),
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )
        self.camera_device_buffer = self._device.create_buffer(
            label="Draw3dFrame.CameraDeviceBuffer",
            size=PodCameraArray.array_size(shape=(1,)),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.camera_staging_buffer = self._device.create_buffer(
            label="Draw3dFrame.CameraStagingBuffer",
            size=PodCameraArray.array_size(shape=(1,)),
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )
        self.instances_list_device_buffer = self._device.create_buffer(
            label="Draw3dFrame.InstanceHeapDeviceBuffer",
            size=PodInstanceArray.array_size(shape=(self.renderer.instance_capacity,)),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.instances_list_staging_buffer = self._device.create_buffer(
            label="Draw3dFrame.InstanceHeapStagingBuffer",
            size=PodInstanceArray.array_size(shape=(self.renderer.instance_capacity,)),
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )

        self.bind_group = self._device.create_bind_group(
            label="Draw3dFrame.BindGroup",
            layout=renderer.per_frame_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=self.output_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=wgpu.BufferBinding(
                        buffer=self.frame_info_device_buffer,
                        offset=0,
                        size=self.frame_info_device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=wgpu.BufferBinding(
                        buffer=self.camera_device_buffer,
                        offset=0,
                        size=self.camera_device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=3,
                    resource=wgpu.BufferBinding(
                        buffer=self.instances_list_device_buffer,
                        offset=0,
                        size=self.instances_list_device_buffer.size,
                    ),
                ),
            ],
        )

    @property
    def _device(self) -> wgpu.GPUDevice:
        return self.renderer.device

    def get_output_image(self) -> wgpu.GPUTexture:
        return self.output_image

    def set_debug_flags(
        self,
        *,
        emit_primary_ray_direction: bool = False,
        emit_closest_hit_depth_in_r: bool = False,
        emit_hit_world_position: bool = False,
    ) -> None:
        """Set debug visualization flags.

        Args:
            emit_primary_ray_direction: If True, output normalized ray direction as RGB.
            emit_closest_hit_depth_in_r: If True, output normalized hit depth in red channel.
            emit_hit_world_position: If True, output world-space hit position as RGB.
        """
        self._debug_flags = 0
        if emit_primary_ray_direction:
            self._debug_flags |= _FRAME_FLAG_EMIT_PRIMARY_RAY_DIRECTION
        if emit_closest_hit_depth_in_r:
            self._debug_flags |= _FRAME_FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R
        if emit_hit_world_position:
            self._debug_flags |= _FRAME_FLAG_EMIT_HIT_WORLD_POSITION

    def record(
        self,
        pipeline: wgpu.GPUComputePipeline,
        renderer_bind_group: wgpu.GPUBindGroup,
        target_size_wh: tuple[int, int],
        encoder: wgpu.GPUCommandEncoder,
        scene: Draw3dScene,
    ) -> None:
        instance_count = sum(len(transforms) for transforms in scene.meshes.values())

        self._upload_frame_info(instance_count, encoder, self._debug_flags)
        self._upload_camera_info(scene.camera, encoder)
        self._upload_instances_info(scene.meshes, encoder)

        compute_pass = encoder.begin_compute_pass(label="Draw3dFrame.ComputePass")
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, renderer_bind_group, [], 0, 0)
        compute_pass.set_bind_group(1, self.bind_group, [], 0, 0)
        compute_pass.dispatch_workgroups(
            workgroup_count_x=math.ceil(target_size_wh[0] / 8),
            workgroup_count_y=math.ceil(target_size_wh[1] / 8),
            workgroup_count_z=1,
        )
        compute_pass.end()

    def _upload_frame_info(
        self,
        instance_count: int,
        command_encoder: wgpu.GPUCommandEncoder,
        debug_flags: int = 0,
    ) -> None:
        frame_info_data = PodFrameInfoArray.empty(shape=(1,))
        frame_info_data["instance_count"] = instance_count
        frame_info_data["target_size_w_px"] = self.renderer.target_size_wh_px[0]
        frame_info_data["target_size_h_px"] = self.renderer.target_size_wh_px[1]
        frame_info_data["debug_flags"] = debug_flags

        self.frame_info_staging_buffer.map_sync(wgpu.MapMode.WRITE)
        self.frame_info_staging_buffer.write_mapped(data=frame_info_data)
        self.frame_info_staging_buffer.unmap()

        command_encoder.copy_buffer_to_buffer(
            source=self.frame_info_staging_buffer,
            source_offset=0,
            destination=self.frame_info_device_buffer,
            destination_offset=0,
            size=frame_info_data.nbytes,
        )

    def _upload_camera_info(
        self,
        camera: "Draw3dCamera",
        command_encoder: wgpu.GPUCommandEncoder,
    ) -> None:
        camera_data = PodCameraArray.empty(shape=(1,))
        camera_data["transform"] = camera.transform
        camera_data["fov_y_rad"] = camera.fov_y_rad
        camera_data["aspect_ratio"] = camera.aspect_ratio
        camera_data["max_distance"] = camera.max_distance

        self.camera_staging_buffer.map_sync(wgpu.MapMode.WRITE)
        self.camera_staging_buffer.write_mapped(data=camera_data)
        self.camera_staging_buffer.unmap()

        command_encoder.copy_buffer_to_buffer(
            source=self.camera_staging_buffer,
            source_offset=0,
            destination=self.camera_device_buffer,
            destination_offset=0,
            size=camera_data.nbytes,
        )

    def _upload_instances_info(
        self,
        instances: dict[
            tuple["Draw3dGeometry", "Draw3dMaterial"],
            jt.Float32[np.ndarray, "n 4 4"],
        ],
        command_encoder: wgpu.GPUCommandEncoder,
    ) -> None:
        total_instance_count = sum(len(t) for t in instances.values())
        if total_instance_count > self.renderer.instance_capacity:
            raise RuntimeError("Draw3dRenderer instance heap capacity exceeded.")

        if total_instance_count == 0:
            # No instances to upload
            return

        data = PodInstanceArray.empty(shape=(total_instance_count,))
        offset = 0
        for (geometry, material), transforms in instances.items():
            n = transforms.shape[0]
            data["geometry_id"][offset : offset + n] = geometry.geometry_id
            data["material_id"][offset : offset + n] = 0  # TODO: material ID
            data["transform"][offset : offset + n] = transforms

            # Compute inverse transforms
            for i in range(n):
                transform_4x4 = transforms[i]  # Shape (4, 4)
                inv_transform_4x4 = np.linalg.inv(transform_4x4)
                data["inv_transform"][offset + i] = inv_transform_4x4

            offset += n

        self.instances_list_staging_buffer.map_sync(wgpu.MapMode.WRITE)
        self.instances_list_staging_buffer.write_mapped(data=data)
        self.instances_list_staging_buffer.unmap()

        command_encoder.copy_buffer_to_buffer(
            source=self.instances_list_staging_buffer,
            source_offset=0,
            destination=self.instances_list_device_buffer,
            destination_offset=0,
            size=data.nbytes,
        )


class Draw3dGeometry(BaseDisposable):
    renderer: Draw3dRenderer

    triangle_count: int
    vertex_count: int

    geometry_id: int
    geometry_heap_offset_in_triangles: int

    def __init__(
        self,
        renderer: Draw3dRenderer,
        *,
        t_indices: jt.UInt32[jt.Array, "nt 3"],
        v_p_array: jt.Float32[jt.Array, "nv 3"],
        v_n_array: jt.Float32[jt.Array, "nv 3"] | None = None,
        v_t_array: jt.Float32[jt.Array, "nv 2"] | None = None,
    ) -> None:
        """
        Constructs a Draw3dGeometry from given vertex and triangle data.

        :param renderer: The Draw3dRenderer instance to use.
        :param t_indices: Triangle indices array of shape (nt, 3) and dtype uint32.
        :param v_p_array: Vertex positions array of shape (nv, 3) and dtype float32.
        :param v_n_array: (Optional) Vertex normals array of shape (nv, 3) and dtype float32.
        :param v_t_array: (Optional) Vertex texture coordinates array of shape (nv, 2) and dtype float32.
        """

        super().__init__()

        self.renderer = renderer
        self.triangle_count = t_indices.shape[0]

        pod_vertex_offset_array: PodVertexOffsetArray
        pod_vertex_offset_array = v_p_array.view(PodVertexOffsetArray)

        # Interleave the vertex arrays:
        v = np.concatenate([v_p_array, v_n_array, v_t_array], axis=-1)
        v = v.view(dtype=PodVertexOffsetArray.DTYPE).squeeze()
        assert (
            v.shape == (v_p_array.shape[0],) and v.dtype == PodVertexOffsetArray.DTYPE
        )

        # Compute BVH, reordering indices as needed:
        # TODO

        # Get rid of the index buffer: load the vertices for each triangle:
        v = PodVertexOffsetArray(v[t_indices])
        assert (
            v.shape == (self.triangle_count, 3)
            and v.dtype == PodVertexOffsetArray.DTYPE
        )

        # Flatten to 1D array for _add_geometry
        v = v.reshape(-1).view(PodVertexOffsetArray)
        assert (
            v.shape == (self.triangle_count * 3,)
            and v.dtype == PodVertexOffsetArray.DTYPE
        )

        # Allocate space in renderer heaps, upload data
        self.geometry_heap_offset_in_triangles = renderer._add_triangles(v)
        self.geometry_id = renderer._add_geometry(
            self.geometry_heap_offset_in_triangles,
            self.triangle_count,
        )

    @staticmethod
    def _build_bvh(
        i: jt.UInt32[jt.Array, "nt 3"],
        v: jt.Float32[jt.Array, "nv 3"],
    ) -> tuple[
        PodBvhNodeArray,
        jt.UInt32[jt.Array, "nt 3"],
    ]:
        """
        Builds a BVH for the given triangles and vertices.

        :param i: Original triangle indices array of shape (nt, 3).
        :param v: Vertex position array of shape (nv, 3).
        :return: A tuple containing the BVH node array and the reordered triangle indices.
        """

        nt = i.shape[0]
        nv = v.shape[0]

        # Compute triangle centroids
        c: jt.Float32[jt.Array, "nt"] = v[i].mean(axis=-2)
        assert c.shape == (nt, 3)

        # For each axis, compute the SAH:
        for axis in range(3):
            pass

        raise NotImplementedError()

    @staticmethod
    def _build_bvh_partition(
        i: jt.UInt32[jt.Array, "nt 3"],
        c: jt.Float32[jt.Array, "nt 3"],
        v: jt.Float32[jt.Array, "nv 3"],
        p: jt.Float32[jt.Array, "3"],
        x: int,
    ) -> tuple[
        jt.UInt32[jt.Array, "nt 3"],
        jt.UInt32[jt.Array, "nt 3"],
    ]:
        """
        Recursively builds the BVH partition.

        :param i: Triangle indices array of shape (nt, 3).
        :param c: Triangle centroids array of shape (nt, 3).
        :param v: Vertex position array of shape (nv, 3).
        :param p: Pivot value for partitioning.
        :param x: Axis index (0, 1, or 2) for partitioning.
        :return: A tuple containing the left and right partitioned triangle indices.
        """

        lt_mask = c[:, x] < p[x]
        rt_mask = ~lt_mask

        i_lt = i[lt_mask]
        i_rt = i[rt_mask]

        return i_lt, i_rt


class Draw3dMaterial(BaseDisposable):
    renderer: Draw3dRenderer

    color_map: jt.Float32[np.ndarray, "h w 3"] | None
    color_factor: tuple[float, float, float]
    normal_map: jt.Float32[np.ndarray, "h w 3"] | None
    metalness_map: jt.Float32[np.ndarray, "h w 1"] | None
    metalness_factor: float
    roughness_map: jt.Float32[np.ndarray, "h w 1"] | None
    roughness_factor: float

    def __init__(
        self,
        renderer: Draw3dRenderer,
        *,
        color_map: jt.Float32[np.ndarray, "h w 3"] | None = None,
        color_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        normal_map: jt.Float32[np.ndarray, "h w 3"] | None = None,
        metalness_map: jt.Float32[np.ndarray, "h w 1"] | None = None,
        metalness_factor: float = 1.0,
        roughness_map: jt.Float32[np.ndarray, "h w 1"] | None = None,
        roughness_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.renderer = renderer

        self.color_map = color_map
        self.color_factor = color_factor
        self.normal_map = normal_map
        self.metalness_map = metalness_map
        self.metalness_factor = metalness_factor
        self.roughness_map = roughness_map
        self.roughness_factor = roughness_factor

        # TODO: upload material data to GPU


@dataclass(kw_only=True)
class Draw3dScene:
    camera: Draw3dCamera

    meshes: dict[
        tuple[Draw3dGeometry, Draw3dMaterial],
        jt.Float32[np.ndarray, "n 4 4"],
    ] = field(default_factory=dict)

    environment_map: jt.Float32[np.ndarray, "eh ew 3"] | None = None


@dataclass
class Draw3dCamera:
    transform: jt.Float32[np.ndarray, "4 4"]
    fov_y_rad: float
    aspect_ratio: float
    max_distance: float = 1e3


#
# POD Types (NumPy structured dtypes)
#

POD_SPAN_DTYPE = np.dtype(
    [
        ("begin", np.uint32),
        ("end", np.uint32),
    ]
)

POD_AABB_DTYPE = np.dtype(
    [
        ("min", np.float32, (3,)),
        ("max", np.float32, (3,)),
    ]
)


class PodVertexOffsetArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("offset", np.float32, (3,)),
        ]
    )


class PodVertexDetailArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("tangent", np.float32, (3,)),
            ("bitangent", np.float32, (3,)),
            ("normal", np.float32, (3,)),
            ("texcoord", np.float32, (2,)),
        ]
    )


class PodBvhNodeArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("triangle_span", POD_SPAN_DTYPE),
            ("children", np.uint32, (2,)),
            ("aabb", POD_AABB_DTYPE),
        ]
    )


class PodGeometryArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("bvh_node_span", POD_SPAN_DTYPE),
            ("triangle_span", POD_SPAN_DTYPE),
        ]
    )


class PodFrameInfoArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("instance_count", np.uint32),
            ("target_size_w_px", np.uint32),
            ("target_size_h_px", np.uint32),
            ("debug_flags", np.uint32),
        ]
    )


# Frame info flags (private)
_FRAME_FLAG_EMIT_PRIMARY_RAY_DIRECTION = 1 << 0
_FRAME_FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R = 1 << 1
_FRAME_FLAG_EMIT_HIT_WORLD_POSITION = 1 << 2


class PodCameraArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("transform", np.float32, (4, 4)),
            ("fov_y_rad", np.float32),
            ("aspect_ratio", np.float32),
            ("max_distance", np.float32),
            ("_rsv", np.uint32),
        ]
    )


class PodInstanceArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("geometry_id", np.uint32),
            ("material_id", np.uint32),
            ("_pad0", np.uint32),
            ("_pad1", np.uint32),
            ("transform", np.float32, (4, 4)),  # row-major 4x4 matrix
            ("inv_transform", np.float32, (4, 4)),  # row-major 4x4 inverse matrix
        ]
    )


#
# BVH construction
#
