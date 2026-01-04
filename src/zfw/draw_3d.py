__all__ = [
    "Draw3dFrame",
    "Draw3dRenderer",
    "Draw3dScene",
]

import math
from dataclasses import dataclass, field

import numpy as np
import jaxtyping as jt
import wgpu

from .basic import BaseDisposable, StructuredNDArray
from .bvh import Bvh, build_bvh
from .resources import GeometryResource, MaterialResource

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

    renderer_bind_group_layout: wgpu.GPUBindGroupLayout
    per_frame_bind_group_layout: wgpu.GPUBindGroupLayout
    pipeline_layout: wgpu.GPUPipelineLayout

    draw_shader: wgpu.GPUShaderModule
    draw_pipeline: wgpu.GPUComputePipeline

    geometry_heap_device_buffer: wgpu.GPUBuffer
    bvh_node_heap_device_buffer: wgpu.GPUBuffer
    triangle_heap_device_buffer: wgpu.GPUBuffer

    renderer_bind_group: wgpu.GPUBindGroup

    allocated_geometry_count: int
    allocated_bvh_node_count: int
    allocated_triangle_count: int

    geometry_cache: dict["GeometryResource", "Draw3dGeometry"]
    material_cache: dict["MaterialResource", "Draw3dMaterial"]

    def __init__(
        self,
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
        target_size_wh_px: tuple[int, int],
        instance_capacity: int = 1 << 10,
        geometry_capacity: int = 1 << 8,
        bvh_node_capacity: int = 1 << 20,
        triangle_capacity: int = 1 << 22,
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
                        format=wgpu.TextureFormat.rgba16float,
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

        self.draw_shader = device.create_shader_module(code=shader_source)
        self.draw_pipeline = device.create_compute_pipeline(
            label="Draw3dRenderer.DrawPipeline",
            layout=pipeline_layout,
            compute=wgpu.ProgrammableStage(
                module=self.draw_shader,
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
        self.triangle_heap_device_buffer = device.create_buffer(
            label="Draw3dRenderer.TriangleHeapDeviceBuffer",
            size=PodVertexArray.array_size(shape=(self.triangle_capacity, 3)),
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
                        buffer=self.triangle_heap_device_buffer,
                        offset=0,
                        size=self.triangle_heap_device_buffer.size,
                    ),
                ),
            ],
        )

        self.allocated_geometry_count = 0
        self.allocated_bvh_node_count = 0
        self.allocated_triangle_count = 0

        # Cache for converting GeometryResource/MaterialResource to Draw3d objects
        # Maps resource objects to their Draw3d counterparts
        self._geometry_cache = {}
        self._material_cache = {}

        # Initialization: clear device buffers to zero
        encoder = device.create_command_encoder(
            label="Draw3dRenderer.InitializationEncoder"
        )
        encoder.clear_buffer(self.geometry_heap_device_buffer, offset=0)
        encoder.clear_buffer(self.bvh_node_heap_device_buffer, offset=0)
        encoder.clear_buffer(self.triangle_heap_device_buffer, offset=0)
        queue.submit([encoder.finish()])

    def _add_triangles(self, vertices: PodVertexArray) -> int:
        assert vertices.ndim == 1 and vertices.dtype == PodVertexArray.DTYPE

        vertex_count = vertices.shape[0]
        triangle_count = vertex_count // 3

        # Allocate:
        allocation_offset_in_triangles = self.allocated_triangle_count
        allocation_offset_in_bytes = (
            allocation_offset_in_triangles * PodVertexArray.DTYPE.itemsize * 3
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
            destination=self.triangle_heap_device_buffer,
            destination_offset=allocation_offset_in_bytes,
            size=vertices.nbytes,
        )
        self.queue.submit([encoder.finish()])

        staging_buffer.destroy()

        # Return offset in triangles:
        return allocation_offset_in_triangles

    def _add_bvh_nodes(self, bvh_nodes: PodBvhNodeArray) -> int:
        assert bvh_nodes.ndim == 1 and bvh_nodes.dtype == PodBvhNodeArray.DTYPE

        node_count = bvh_nodes.shape[0]

        # Allocate:
        allocation_offset = self.allocated_bvh_node_count
        self.allocated_bvh_node_count += node_count
        if self.allocated_bvh_node_count > self.bvh_node_capacity:
            raise RuntimeError("Draw3dRenderer BVH node heap capacity exceeded.")

        # Upload via staging buffer:
        staging_buffer = self.device.create_buffer(
            label="Draw3dRenderer.BvhNodeUploadStagingBuffer",
            size=bvh_nodes.nbytes,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )
        staging_buffer.map_sync(mode=wgpu.MapMode.WRITE)
        staging_buffer.write_mapped(data=bvh_nodes)
        staging_buffer.unmap()

        encoder = self.device.create_command_encoder(
            label="Draw3dRenderer.BvhNodeUploadEncoder"
        )
        encoder.copy_buffer_to_buffer(
            source=staging_buffer,
            source_offset=0,
            destination=self.bvh_node_heap_device_buffer,
            destination_offset=(
                allocation_offset * PodBvhNodeArray.array_size(shape=(1,))
            ),
            size=bvh_nodes.nbytes,
        )
        self.queue.submit([encoder.finish()])

        staging_buffer.destroy()

        # Return offset in BVH nodes:
        return allocation_offset

    def _add_geometry(
        self,
        *,
        triangle_span_begin: int,
        triangle_count: int,
        bvh_node_span_begin: int,
        bvh_node_count: int,
    ):
        data = PodGeometryArray.empty(shape=(1,))
        data["triangle_span"][0]["begin"] = triangle_span_begin
        data["triangle_span"][0]["end"] = triangle_span_begin + triangle_count
        data["bvh_node_span"][0]["begin"] = bvh_node_span_begin
        data["bvh_node_span"][0]["end"] = bvh_node_span_begin + bvh_node_count

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
            destination_offset=(
                allocation_offset * PodGeometryArray.array_size(shape=(1,))
            ),
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
            self.draw_pipeline,
            self.renderer_bind_group,
            self.target_size_wh_px,
            command_encoder,
            scene,
        )

    def get_geometry(self, resource: "GeometryResource") -> "Draw3dGeometry":
        """
        Convert a GeometryResource to a Draw3dGeometry, using a cache to avoid
        recreating the same geometry multiple times.

        :param resource: The GeometryResource to convert.
        :return: A Draw3dGeometry instance.
        """

        if cached_entry := self._geometry_cache.get(resource):
            return cached_entry

        new_draw_3d_geometry = Draw3dGeometry(
            renderer=self,
            v_p_array=resource.v_p_array,
            v_n_array=resource.v_n_array,
            v_t_array=resource.v_t_array,
            t_indices=resource.t_indices,
        )
        self._geometry_cache[resource] = new_draw_3d_geometry

        return new_draw_3d_geometry

    def get_material(self, resource: "MaterialResource") -> "Draw3dMaterial":
        """
        Convert a MaterialResource to a Draw3dMaterial, using a cache to avoid
        recreating the same material multiple times.

        :param resource: The MaterialResource to convert.
        :return: A Draw3dMaterial instance.
        """

        if cached_entry := self._material_cache.get(resource):
            return cached_entry

        new_draw_3d_material = Draw3dMaterial(
            renderer=self,
            color_map=resource.color_map,
            color_factor=resource.color_factor,
            normal_map=resource.normal_map,
            metalness_map=resource.metalness_map,
            metalness_factor=resource.metalness_factor,
            roughness_map=resource.roughness_map,
            roughness_factor=resource.roughness_factor,
        )
        self._material_cache[resource] = new_draw_3d_material

        return new_draw_3d_material


class Draw3dFrame:
    renderer: Draw3dRenderer

    def __init__(self, renderer: Draw3dRenderer) -> None:
        self.renderer = renderer
        self._debug_flags = 0

        self.output_image = self._device.create_texture(
            label="Draw3dFrame.OutputImage",
            size=(renderer.target_size_wh_px[0], renderer.target_size_wh_px[1], 1),
            format=wgpu.TextureFormat.rgba16float,
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
        debug_bvh_traversal: bool = False,
    ) -> None:
        """Set debug visualization flags.

        Args:
            emit_primary_ray_direction: If True, output normalized ray direction as RGB.
            emit_closest_hit_depth_in_r: If True, output normalized hit depth in red channel.
            emit_hit_world_position: If True, output world-space hit position as RGB.
            debug_bvh_traversal: If True, visualize BVH leaf AABBs instead of actual triangles.
        """
        self._debug_flags = 0
        if emit_primary_ray_direction:
            self._debug_flags |= _FRAME_FLAG_EMIT_PRIMARY_RAY_DIRECTION
        if emit_closest_hit_depth_in_r:
            self._debug_flags |= _FRAME_FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R
        if emit_hit_world_position:
            self._debug_flags |= _FRAME_FLAG_EMIT_HIT_WORLD_POSITION
        if debug_bvh_traversal:
            self._debug_flags |= _FRAME_FLAG_DEBUG_BVH_TRAVERSAL

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
    """
    IMPORTANT: do not call this constructor directly: use Draw3dRenderer.get_geometry() instead.
    """

    renderer: Draw3dRenderer

    triangle_count: int
    vertex_count: int

    geometry_id: int
    geometry_heap_offset_in_triangles: int

    def __init__(
        self,
        renderer: Draw3dRenderer,
        *,
        v_p_array: jt.Float32[jt.Array, "nv 3"],
        v_n_array: jt.Float32[jt.Array, "nv 3"],
        v_t_array: jt.Float32[jt.Array, "nv 2"],
        t_indices: jt.UInt32[jt.Array, "nt 3"],
    ) -> None:
        super().__init__()

        self.renderer = renderer
        self.triangle_count = t_indices.shape[0]

        # Construct the BVH first.
        # This produces an updated `t_indices` array with a different triangle order.
        # We need to use this reordered index array for all subsequent uploads.
        bvh = build_bvh(t=t_indices, v=v_p_array)
        t_indices = bvh.t

        # Upload triangles, using the reordered t array:
        pod_vertices = Draw3dGeometry._marshall_triangles(
            v_p_array=v_p_array,
            v_n_array=v_n_array,
            v_t_array=v_t_array,
            t_indices=t_indices,
        )
        triangle_span_begin = renderer._add_triangles(vertices=pod_vertices)
        triangle_span_count = self.triangle_count

        # Offset the BVH triangle spans to point to the global triangle heap instead of the per-geometry triangles list.
        bvh.tri_span += triangle_span_begin

        # Upload BVH:
        pod_bvh = Draw3dGeometry._marshall_bvh(bvh)
        bvh_node_span_begin = renderer._add_bvh_nodes(pod_bvh)
        bvh_node_span_count = bvh.node_count

        # Upload geometry record:
        self.geometry_id = renderer._add_geometry(
            triangle_span_begin=triangle_span_begin,
            triangle_count=triangle_span_count,
            bvh_node_span_begin=bvh_node_span_begin,
            bvh_node_count=bvh_node_span_count,
        )

    @staticmethod
    def _marshall_bvh(bvh: Bvh) -> PodBvhNodeArray:
        # TODO: Each node will either have children or a triangle span. We can save memory by
        # using a union-like structure here. Maybe negative values refer to triangle spans?

        pod_bvh = PodBvhNodeArray.empty((bvh.node_count,))
        for i in range(bvh.node_count):
            pod_bvh["tri_span"]["begin"][i] = bvh.tri_span[i, 0]
            pod_bvh["tri_span"]["end"][i] = bvh.tri_span[i, 1]
            pod_bvh["children"][i][0] = bvh.children[i, 0]
            pod_bvh["children"][i][1] = bvh.children[i, 1]
            pod_bvh["aabb"]["min"][i] = bvh.aabb[i, 0]
            pod_bvh["aabb"]["max"][i] = bvh.aabb[i, 1]
        return pod_bvh

    @staticmethod
    def _marshall_triangles(
        v_p_array: jt.Float32[jt.Array, "nv 3"],
        v_n_array: jt.Float32[jt.Array, "nv 3"],
        v_t_array: jt.Float32[jt.Array, "nv 2"],
        t_indices: jt.UInt32[jt.Array, "nt 3"],
    ) -> PodVertexArray:
        # Interleave the vertex arrays:
        v = np.concatenate([v_p_array, v_n_array, v_t_array], axis=-1)
        v = v.view(dtype=PodVertexArray.DTYPE).squeeze()
        assert v.shape == (v_p_array.shape[0],) and v.dtype == PodVertexArray.DTYPE

        # Get rid of the index buffer: load the vertices for each triangle:
        v = PodVertexArray(v[t_indices])
        assert v.shape == (t_indices.shape[0], 3) and v.dtype == PodVertexArray.DTYPE

        # Flatten to 1D array for _add_geometry: groups of 3 consecutive vertices form a triangle:
        v = v.reshape(-1).view(PodVertexArray)
        assert v.shape == (t_indices.shape[0] * 3,) and v.dtype == PodVertexArray.DTYPE

        # Done:
        return v


class Draw3dMaterial(BaseDisposable):
    """
    IMPORTANT: do not call this constructor directly: use Draw3dRenderer.get_material() instead.
    """

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


class PodVertexArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("position", np.float32, (3,)),
            ("normal", np.float32, (3,)),
            ("uv", np.float32, (2,)),
        ]
    )


class PodBvhNodeArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("tri_span", POD_SPAN_DTYPE),
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
_FRAME_FLAG_DEBUG_BVH_TRAVERSAL = 1 << 3


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
