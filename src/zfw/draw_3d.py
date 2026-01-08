__all__ = [
    "Draw3dFrame",
    "Draw3dRenderer",
    "Draw3dScene",
]

from contextlib import contextmanager
import math
from dataclasses import dataclass, field
from typing import Generator, Literal
import time

import numpy as np
import jaxtyping as jt
import wgpu


from .basic import BaseDisposable, StructuredNDArray, logger
from .excepts import LogicError
from .bvh import Bvh, build_bvh
from .resources import GeometryResource, ImageResource, MaterialResource
from .images import encode_bc1, encode_bc4, encode_bc5

#
# Renderer
#


class Draw3dRenderer(BaseDisposable):
    """
    A 3D renderer using ray tracing implemented with wgpu compute shaders.

    Usage:
    -   Create an instance of Draw3dRenderer.
    -   Create Draw3dFrame instances for each frame to be rendered concurrently. You can
        and should reuse Draw3dFrame instances across multiple frames.
    -   Create Draw3dScene instances representing the scenes to be rendered. These
        should be created anew for each frame.
    -   For each frame, call `Draw3dRenderer.record()` with the scene, frame, and a GPU
        command encoder (to record commands into).
    """

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

    geometry_heap: "LinearHeap[PodGeometryArray]"
    bvh_node_heap: "LinearHeap[PodBvhNodeArray]"
    triangle_heap: "LinearHeap[PodVertexArray]"
    material_heap: "LinearHeap[PodMaterialArray]"
    color_texture_heap: "TextureHeap"
    normal_texture_heap: "TextureHeap"
    metalness_texture_heap: "TextureHeap"
    roughness_texture_heap: "TextureHeap"
    environment_texture_heap: "TextureHeap"
    linear_sampler: wgpu.GPUSampler

    renderer_bind_group: wgpu.GPUBindGroup

    allocated_geometry_count: int
    allocated_bvh_node_count: int
    allocated_triangle_count: int
    allocated_material_count: int
    allocated_image_count: int
    allocated_subpixel_count: int

    geometry_cache: dict["GeometryResource", "Draw3dGeometry"]
    material_cache: dict["MaterialResource", "Draw3dMaterial"]
    texture_cache: dict[tuple["ImageResource", "Draw3dTextureUsage"], "Draw3dTexture"]

    def __init__(
        self,
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
        target_size_wh_px: tuple[int, int],
        instance_capacity: int = 1 << 10,
        geometry_capacity: int = 1 << 7,
        material_capacity: int = 1 << 7,
        bvh_node_capacity: int = 1 << 20,
        triangle_capacity: int = 1 << 22,
        image_capacity: int = 1 << 8,
        subpixel_capacity: int = 1 << 29,
    ):
        self.device = device
        self.queue = queue

        self.target_size_wh_px = target_size_wh_px

        self.instance_capacity = instance_capacity
        self.geometry_capacity = geometry_capacity
        self.material_capacity = material_capacity
        self.bvh_node_capacity = bvh_node_capacity
        self.triangle_capacity = triangle_capacity
        self.image_capacity = image_capacity
        self.subpixel_capacity = subpixel_capacity

        self.renderer_bind_group_layout = device.create_bind_group_layout(
            label="Draw3dRenderer.RendererBindGroupLayout",
            entries=[
                # Geometry heap:
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                # BVH node heap:
                wgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                # Triangle heap:
                wgpu.BindGroupLayoutEntry(
                    binding=2,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                # Material heap:
                wgpu.BindGroupLayoutEntry(
                    binding=3,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                # Color texture heap:
                wgpu.BindGroupLayoutEntry(
                    binding=4,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.float,
                        view_dimension=wgpu.TextureViewDimension.d2_array,
                        multisampled=False,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=5,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                # Normal texture heap:
                wgpu.BindGroupLayoutEntry(
                    binding=6,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.float,
                        view_dimension=wgpu.TextureViewDimension.d2_array,
                        multisampled=False,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=7,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                # Metalness texture heap:
                wgpu.BindGroupLayoutEntry(
                    binding=8,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.float,
                        view_dimension=wgpu.TextureViewDimension.d2_array,
                        multisampled=False,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=9,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                # Roughness texture heap:
                wgpu.BindGroupLayoutEntry(
                    binding=10,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.float,
                        view_dimension=wgpu.TextureViewDimension.d2_array,
                        multisampled=False,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=11,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                # Environment texture heap:
                wgpu.BindGroupLayoutEntry(
                    binding=12,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.float,
                        view_dimension=wgpu.TextureViewDimension.d2_array,
                        multisampled=False,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=13,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                # Linear sampler:
                wgpu.BindGroupLayoutEntry(
                    binding=14,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    sampler=wgpu.SamplerBindingLayout(
                        type=wgpu.SamplerBindingType.filtering
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

        self.geometry_heap = LinearHeap[PodGeometryArray](
            device=device,
            label="Draw3dRenderer.GeometryHeap",
            structured_array_cls=PodGeometryArray,
            element_capacity=self.geometry_capacity,
            persistent_staging_buffer_element_capacity=1,
        )
        self.bvh_node_heap = LinearHeap[PodBvhNodeArray](
            device=device,
            label="Draw3dRenderer.BvhNodeHeap",
            structured_array_cls=PodBvhNodeArray,
            element_capacity=self.bvh_node_capacity,
        )
        self.triangle_heap = LinearHeap[PodVertexArray](
            device=device,
            label="Draw3dRenderer.TriangleHeap",
            structured_array_cls=PodVertexArray,
            element_capacity=self.triangle_capacity,
        )
        self.material_heap = LinearHeap[PodMaterialArray](
            device=device,
            label="Draw3dRenderer.MaterialHeap",
            element_capacity=self.material_capacity,
            structured_array_cls=PodMaterialArray,
            persistent_staging_buffer_element_capacity=1,
        )

        self.color_texture_heap = TextureHeap(
            device=device,
            label="Draw3dRenderer.ColorTextureHeap",
            usage="color",
            page_count=16,
        )
        self.normal_texture_heap = TextureHeap(
            device=device,
            label="Draw3dRenderer.NormalTextureHeap",
            usage="normal",
            page_count=16,
        )
        self.metalness_texture_heap = TextureHeap(
            device=device,
            label="Draw3dRenderer.MetalnessTextureHeap",
            usage="metalness",
            page_count=16,
        )
        self.roughness_texture_heap = TextureHeap(
            device=device,
            label="Draw3dRenderer.RoughnessTextureHeap",
            usage="roughness",
            page_count=16,
        )
        self.environment_texture_heap = TextureHeap(
            device=device,
            label="Draw3dRenderer.EnvironmentTextureHeap",
            usage="environment",
            page_count=2,
        )

        self.linear_sampler = device.create_sampler(
            label="Draw3dRenderer.LinearSampler",
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
        )

        self.renderer_bind_group = device.create_bind_group(
            label="Draw3dRenderer.RendererBindGroup",
            layout=self.renderer_bind_group_layout,
            entries=[
                # Geometry heap
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=wgpu.BufferBinding(
                        buffer=self.geometry_heap.device_buffer,
                        offset=0,
                        size=self.geometry_heap.device_buffer.size,
                    ),
                ),
                # BVH node heap
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=wgpu.BufferBinding(
                        buffer=self.bvh_node_heap.device_buffer,
                        offset=0,
                        size=self.bvh_node_heap.device_buffer.size,
                    ),
                ),
                # Triangle heap
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=wgpu.BufferBinding(
                        buffer=self.triangle_heap.device_buffer,
                        offset=0,
                        size=self.triangle_heap.device_buffer.size,
                    ),
                ),
                # Material heap
                wgpu.BindGroupEntry(
                    binding=3,
                    resource=wgpu.BufferBinding(
                        buffer=self.material_heap.device_buffer,
                        offset=0,
                        size=self.material_heap.device_buffer.size,
                    ),
                ),
                # Color texture heap
                wgpu.BindGroupEntry(
                    binding=4,
                    resource=self.color_texture_heap.texture.create_view(
                        dimension=wgpu.TextureViewDimension.d2_array,
                    ),
                ),
                # Color texture allocations
                wgpu.BindGroupEntry(
                    binding=5,
                    resource=wgpu.BufferBinding(
                        buffer=self.color_texture_heap.allocation_heap.device_buffer,
                        offset=0,
                        size=self.color_texture_heap.allocation_heap.device_buffer.size,
                    ),
                ),
                # Normal texture heap
                wgpu.BindGroupEntry(
                    binding=6,
                    resource=self.normal_texture_heap.texture.create_view(
                        dimension=wgpu.TextureViewDimension.d2_array,
                    ),
                ),
                # Normal texture allocations
                wgpu.BindGroupEntry(
                    binding=7,
                    resource=wgpu.BufferBinding(
                        buffer=self.normal_texture_heap.allocation_heap.device_buffer,
                        offset=0,
                        size=self.normal_texture_heap.allocation_heap.device_buffer.size,
                    ),
                ),
                # Metalness texture heap
                wgpu.BindGroupEntry(
                    binding=8,
                    resource=self.metalness_texture_heap.texture.create_view(
                        dimension=wgpu.TextureViewDimension.d2_array,
                    ),
                ),
                # Metalness texture allocations
                wgpu.BindGroupEntry(
                    binding=9,
                    resource=wgpu.BufferBinding(
                        buffer=self.metalness_texture_heap.allocation_heap.device_buffer,
                        offset=0,
                        size=self.metalness_texture_heap.allocation_heap.device_buffer.size,
                    ),
                ),
                # Roughness texture heap
                wgpu.BindGroupEntry(
                    binding=10,
                    resource=self.roughness_texture_heap.texture.create_view(
                        dimension=wgpu.TextureViewDimension.d2_array,
                    ),
                ),
                # Roughness texture allocations
                wgpu.BindGroupEntry(
                    binding=11,
                    resource=wgpu.BufferBinding(
                        buffer=self.roughness_texture_heap.allocation_heap.device_buffer,
                        offset=0,
                        size=self.roughness_texture_heap.allocation_heap.device_buffer.size,
                    ),
                ),
                # Environment texture heap
                wgpu.BindGroupEntry(
                    binding=12,
                    resource=self.environment_texture_heap.texture.create_view(
                        dimension=wgpu.TextureViewDimension.d2_array,
                    ),
                ),
                # Environment texture allocations
                wgpu.BindGroupEntry(
                    binding=13,
                    resource=wgpu.BufferBinding(
                        buffer=self.environment_texture_heap.allocation_heap.device_buffer,
                        offset=0,
                        size=self.environment_texture_heap.allocation_heap.device_buffer.size,
                    ),
                ),
                # Linear sampler
                wgpu.BindGroupEntry(
                    binding=14,
                    resource=self.linear_sampler,
                ),
            ],
        )

        self.allocated_geometry_count = 0
        self.allocated_bvh_node_count = 0
        self.allocated_triangle_count = 0
        self.allocated_material_count = 0
        self.allocated_image_count = 0
        self.allocated_subpixel_count = 0

        # Cache for converting GeometryResource/MaterialResource to Draw3d objects
        # Maps resource objects to their Draw3d counterparts
        self._geometry_cache = {}
        self._material_cache = {}
        self._texture_cache = {}

    def _on_dispose(self) -> None:
        self.geometry_heap.dispose()
        self.bvh_node_heap.dispose()
        self.triangle_heap.dispose()
        self.material_heap.dispose()

        self.color_texture_heap.dispose()
        self.normal_texture_heap.dispose()
        self.metalness_texture_heap.dispose()
        self.roughness_texture_heap.dispose()
        self.environment_texture_heap.dispose()

        return super()._on_dispose()

    def _add_geometry(
        self,
        *,
        triangle_span_begin: int,
        triangle_count: int,
        bvh_node_span_begin: int,
        bvh_node_count: int,
    ) -> int:
        data = PodGeometryArray.empty(shape=(1,))
        data["triangle_span"][0]["begin"] = triangle_span_begin
        data["triangle_span"][0]["end"] = triangle_span_begin + triangle_count
        data["bvh_node_span"][0]["begin"] = bvh_node_span_begin
        data["bvh_node_span"][0]["end"] = bvh_node_span_begin + bvh_node_count
        return self.geometry_heap.insert(data)

    def _add_bvh_nodes(self, bvh_nodes: PodBvhNodeArray) -> int:
        assert bvh_nodes.ndim == 1 and bvh_nodes.dtype == PodBvhNodeArray.DTYPE
        return self.bvh_node_heap.insert(bvh_nodes)

    def _add_triangles(self, vertices: PodVertexArray) -> int:
        assert vertices.ndim == 1 and vertices.dtype == PodVertexArray.DTYPE
        assert vertices.shape[0] % 3 == 0
        return self.triangle_heap.insert(vertices) // 3

    def _add_texture(
        self,
        *,
        data: np.ndarray,
        usage: "Draw3dTextureUsage",
    ) -> TextureHeapAllocation:
        return self._get_texture_heap(usage).insert(data=data)

    def _get_texture_heap(self, usage: "Draw3dTextureUsage") -> "TextureHeap":
        return {
            "color": self.color_texture_heap,
            "normal": self.normal_texture_heap,
            "metalness": self.metalness_texture_heap,
            "roughness": self.roughness_texture_heap,
            "environment": self.environment_texture_heap,
        }[usage]

    def _add_material(
        self,
        *,
        color_map_id: int,
        color_factor: tuple[float, float, float],
        normal_map_id: int,
        metalness_map_id: int,
        metalness_factor: float,
        roughness_map_id: int,
        roughness_factor: float,
    ) -> int:
        data = PodMaterialArray.empty(shape=(1,))
        data["color_map_id"][0] = np.uint32(color_map_id)
        data["color_factor"][0] = np.array(color_factor, dtype=np.float32)
        data["normal_map_id"][0] = np.uint32(normal_map_id)
        data["metalness_map_id"][0] = np.uint32(metalness_map_id)
        data["metalness_factor"][0] = np.float32(metalness_factor)
        data["roughness_map_id"][0] = np.uint32(roughness_map_id)
        data["roughness_factor"][0] = np.float32(roughness_factor)
        return self.material_heap.insert(data)

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

    def get_texture(
        self,
        resource: "ImageResource",
        usage: "Draw3dTextureUsage",
    ) -> "Draw3dTexture":
        """
        Convert an ImageResource to a Draw3dTexture, using a cache to avoid
        recreating the same texture multiple times.

        :param resource: The ImageResource to convert.
        :return: A Draw3dTexture instance.
        """

        cache_key = (resource, usage)

        if cached_entry := self._texture_cache.get(cache_key):
            return cached_entry

        new_draw_3d_texture = Draw3dTexture(
            renderer=self,
            resource=resource,
            usage=usage,
        )
        self._texture_cache[cache_key] = new_draw_3d_texture

        return new_draw_3d_texture

    def get_material(self, resource: "MaterialResource") -> "Draw3dMaterial":
        """
        Convert a MaterialResource to a Draw3dMaterial, using a cache to avoid
        recreating the same material multiple times.

        :param resource: The MaterialResource to convert.
        :return: A Draw3dMaterial instance.
        """

        if cached_entry := self._material_cache.get(resource):
            return cached_entry

        # Convert ImageResource objects to Draw3dTexture objects
        color_texture = (
            self.get_texture(resource.color_map, usage="color")
            if resource.color_map
            else None
        )
        normal_texture = (
            self.get_texture(resource.normal_map, usage="normal")
            if resource.normal_map
            else None
        )
        metalness_texture = (
            self.get_texture(resource.metalness_map, usage="metalness")
            if resource.metalness_map
            else None
        )
        roughness_texture = (
            self.get_texture(resource.roughness_map, usage="roughness")
            if resource.roughness_map
            else None
        )

        new_draw_3d_material = Draw3dMaterial(
            renderer=self,
            color_texture=color_texture,
            color_factor=resource.color_factor,
            normal_texture=normal_texture,
            metalness_texture=metalness_texture,
            metalness_factor=resource.metalness_factor,
            roughness_texture=roughness_texture,
            roughness_factor=resource.roughness_factor,
        )
        self._material_cache[resource] = new_draw_3d_material

        return new_draw_3d_material


class Draw3dFrame(BaseDisposable):
    renderer: Draw3dRenderer

    debug_flags: int

    output_image: wgpu.GPUTexture
    frame_info_buffer: "PerFrameBuffer[PodFrameInfoArray]"
    camera_buffer: "PerFrameBuffer[PodCameraArray]"
    instance_buffer: "PerFrameBuffer[PodInstanceArray]"

    def __init__(self, renderer: Draw3dRenderer) -> None:
        self.renderer = renderer
        self.debug_flags = 0

        self.output_image = self._device.create_texture(
            label="Draw3dFrame.OutputImage",
            size=(renderer.target_size_wh_px[0], renderer.target_size_wh_px[1], 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self.frame_info_buffer = PerFrameBuffer[PodFrameInfoArray](
            device=self._device,
            label="Draw3dFrame.FrameInfoHeap",
            structured_array_cls=PodFrameInfoArray,
            element_capacity=1,
            device_buffer_usages=wgpu.BufferUsage.UNIFORM,
        )
        self.camera_buffer = PerFrameBuffer[PodCameraArray](
            device=self._device,
            label="Draw3dFrame.CameraHeap",
            structured_array_cls=PodCameraArray,
            element_capacity=1,
            device_buffer_usages=wgpu.BufferUsage.UNIFORM,
        )
        self.instance_buffer = PerFrameBuffer[PodInstanceArray](
            device=self._device,
            label="Draw3dFrame.InstanceHeap",
            structured_array_cls=PodInstanceArray,
            element_capacity=renderer.instance_capacity,
            device_buffer_usages=wgpu.BufferUsage.STORAGE,
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
                        buffer=self.frame_info_buffer.device_buffer,
                        offset=0,
                        size=self.frame_info_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=wgpu.BufferBinding(
                        buffer=self.camera_buffer.device_buffer,
                        offset=0,
                        size=self.camera_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=3,
                    resource=wgpu.BufferBinding(
                        buffer=self.instance_buffer.device_buffer,
                        offset=0,
                        size=self.instance_buffer.device_buffer.size,
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
        emit_closest_hit_bvh_depth_in_r: bool = False,
        emit_color: bool = False,
        emit_hit_normal: bool = False,
        emit_orm: bool = False,
    ) -> None:
        """
        Set debug visualization modes.

        :param emit_primary_ray_direction: If True, output normalized ray direction as RGB.
        :param emit_closest_hit_depth_in_r: If True, output normalized hit depth in red channel.
        :param emit_hit_world_position: If True, output world-space hit position as RGB.
        :param emit_closest_hit_bvh_depth_in_r: If True, output normalized hit depth to BVH leaf in red channel.
        :param emit_primary_ray_color: If True, output sampled texture color at hit point.
        :param emit_hit_normal: If True, output world-space hit normal as RGB.
        :param emit_orm: If True, output ORM (Opacity, Roughness, Metalness) as RGB.
        """
        self.debug_flags = 0
        if emit_primary_ray_direction:
            self.debug_flags |= _FRAME_FLAG_EMIT_PRIMARY_RAY_DIRECTION
        if emit_closest_hit_depth_in_r:
            self.debug_flags |= _FRAME_FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R
        if emit_hit_world_position:
            self.debug_flags |= _FRAME_FLAG_EMIT_HIT_WORLD_POSITION
        if emit_closest_hit_bvh_depth_in_r:
            self.debug_flags |= _FRAME_FLAG_EMIT_CLOSEST_BVH_HIT_DEPTH_IN_R
        if emit_color:
            self.debug_flags |= _FRAME_FLAG_EMIT_PRIMARY_RAY_COLOR
        if emit_hit_normal:
            self.debug_flags |= _FRAME_FLAG_EMIT_HIT_NORMAL
        if emit_orm:
            self.debug_flags |= _FRAME_FLAG_EMIT_ORM

    def record(
        self,
        pipeline: wgpu.GPUComputePipeline,
        renderer_bind_group: wgpu.GPUBindGroup,
        target_size_wh: tuple[int, int],
        encoder: wgpu.GPUCommandEncoder,
        scene: Draw3dScene,
    ) -> None:
        instance_count = sum(len(transforms) for transforms in scene.meshes.values())

        self._upload_frame_info(
            instance_count,
            encoder,
            self.debug_flags,
            environment_map_texture_id=(
                scene.environment_map.allocation.texture_id
                if scene.environment_map
                else -1
            ),
        )
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
        environment_map_texture_id: int = -1,
    ) -> None:
        frame_info_data = PodFrameInfoArray.empty(shape=(1,))
        frame_info_data["instance_count"] = instance_count
        frame_info_data["target_size_w_px"] = self.renderer.target_size_wh_px[0]
        frame_info_data["target_size_h_px"] = self.renderer.target_size_wh_px[1]
        frame_info_data["debug_flags"] = debug_flags
        frame_info_data["environment_map_texture_id"] = environment_map_texture_id

        self.frame_info_buffer.write(frame_info_data, command_encoder)

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

        self.camera_buffer.write(camera_data, command_encoder)

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

        data = PodInstanceArray.empty(shape=(total_instance_count,))
        offset = 0
        for (geometry, material), transforms in instances.items():
            n = transforms.shape[0]
            data["geometry_id"][offset : offset + n] = geometry.geometry_id
            data["material_id"][offset : offset + n] = material.material_id
            data["transform"][offset : offset + n] = transforms

            # Compute inverse transforms
            for i in range(n):
                transform_4x4 = transforms[i]  # Shape (4, 4)
                inv_transform_4x4 = np.linalg.inv(transform_4x4)
                data["inv_transform"][offset + i] = inv_transform_4x4

            offset += n

        self.instance_buffer.write(data, command_encoder)


class Draw3dGeometry(BaseDisposable):
    """
    IMPORTANT: do not call this constructor directly: use
    `Draw3dRenderer.get_geometry()` instead.
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

        # Offset the BVH triangle spans to point to the global triangle heap instead of
        # the per-geometry triangles list.
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


class Draw3dTexture(BaseDisposable):
    """
    IMPORTANT: do not call this constructor directly: use `Draw3dRenderer.get_texture()`
    instead.

    Represents a GPU texture created from an ImageResource.
    """

    renderer: Draw3dRenderer
    allocation: TextureHeapAllocation

    def __init__(
        self,
        renderer: Draw3dRenderer,
        *,
        resource: ImageResource,
        usage: "Draw3dTextureUsage",
    ) -> None:
        super().__init__()

        self.renderer = renderer
        self.allocation = renderer._add_texture(data=resource.data, usage=usage)


class Draw3dMaterial(BaseDisposable):
    """
    IMPORTANT: do not call this constructor directly: use
    `Draw3dRenderer.get_material()` instead.
    """

    renderer: Draw3dRenderer
    material_id: int

    color_texture: "Draw3dTexture | None"
    color_factor: tuple[float, float, float]
    normal_texture: "Draw3dTexture | None"
    metalness_texture: "Draw3dTexture | None"
    metalness_factor: float
    roughness_texture: "Draw3dTexture | None"
    roughness_factor: float

    def __init__(
        self,
        renderer: Draw3dRenderer,
        *,
        color_texture: "Draw3dTexture | None" = None,
        color_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        normal_texture: "Draw3dTexture | None" = None,
        metalness_texture: "Draw3dTexture | None" = None,
        metalness_factor: float = 1.0,
        roughness_texture: "Draw3dTexture | None" = None,
        roughness_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.renderer = renderer

        self.color_texture = color_texture
        self.color_factor = color_factor
        self.normal_texture = normal_texture
        self.metalness_texture = metalness_texture
        self.metalness_factor = metalness_factor
        self.roughness_texture = roughness_texture
        self.roughness_factor = roughness_factor

        # Upload material data to GPU
        color_map_id = color_texture.allocation.texture_id if color_texture else 0
        normal_map_id = normal_texture.allocation.texture_id if normal_texture else 0
        metalness_map_id = (
            metalness_texture.allocation.texture_id if metalness_texture else 0
        )
        roughness_map_id = (
            roughness_texture.allocation.texture_id if roughness_texture else 0
        )

        self.material_id = renderer._add_material(
            color_map_id=color_map_id,
            color_factor=color_factor,
            normal_map_id=normal_map_id,
            metalness_map_id=metalness_map_id,
            metalness_factor=metalness_factor,
            roughness_map_id=roughness_map_id,
            roughness_factor=roughness_factor,
        )


@dataclass(kw_only=True)
class Draw3dScene:
    """
    Represents a 3D scene to be rendered.
    """

    camera: Draw3dCamera

    meshes: dict[
        tuple[Draw3dGeometry, Draw3dMaterial],
        jt.Float32[np.ndarray, "n 4 4"],
    ] = field(default_factory=dict)

    environment_map: Draw3dTexture | None = None


@dataclass(kw_only=True)
class Draw3dCamera:
    """
    Represents a camera in the 3D scene.
    """

    transform: jt.Float32[np.ndarray, "4 4"]
    fov_y_rad: float
    aspect_ratio: float
    max_distance: float = 1e3


#
# PerFrameBuffer: data written to GPU for each frame
#


class PerFrameBuffer[T: StructuredNDArray](BaseDisposable):
    device: wgpu.GPUDevice
    label: str
    structured_array_cls: type[StructuredNDArray]
    element_capacity: int

    device_buffer: wgpu.GPUBuffer
    staging_buffer: wgpu.GPUBuffer

    def __init__(
        self,
        device: wgpu.GPUDevice,
        label: str,
        structured_array_cls: type[T],
        element_capacity: int,
        device_buffer_usages: int,
    ) -> None:
        self.device = device
        self.label = label
        self.structured_array_cls = structured_array_cls
        self.element_capacity = element_capacity

        self.device_buffer = device.create_buffer(
            label=f"{label}.DeviceBuffer",
            size=self.structured_array_cls.array_size(shape=(element_capacity,)),
            usage=device_buffer_usages | wgpu.BufferUsage.COPY_DST,
        )
        self.staging_buffer = device.create_buffer(
            label=f"{label}.StagingBuffer",
            size=self.structured_array_cls.array_size(shape=(element_capacity,)),
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )

    def _on_dispose(self) -> None:
        self.device_buffer.destroy()
        return super()._on_dispose()

    def write(self, data: T, command_encoder: wgpu.GPUCommandEncoder) -> None:
        assert data.__class__ is self.structured_array_cls
        assert data.shape[0] <= self.element_capacity

        # Early return if there's no data to write
        if data.shape[0] == 0:
            return

        self.staging_buffer.map_sync(mode=wgpu.MapMode.WRITE)
        self.staging_buffer.write_mapped(data=data)
        self.staging_buffer.unmap()

        command_encoder.copy_buffer_to_buffer(
            source=self.staging_buffer,
            source_offset=0,
            destination=self.device_buffer,
            destination_offset=0,
            size=self.structured_array_cls.array_size(shape=data.shape),
        )


#
# LinearHeap: helper for linear allocation in GPU storage buffers
#


class LinearHeap[T: StructuredNDArray](BaseDisposable):
    device: wgpu.GPUDevice
    label: str
    structured_array_cls: type[StructuredNDArray]
    element_capacity: int
    persistent_staging_buffer_element_capacity: int

    device_buffer: wgpu.GPUBuffer
    persistent_staging_buffer: wgpu.GPUBuffer | None
    allocated_element_count: int

    def __init__(
        self,
        device: wgpu.GPUDevice,
        label: str,
        structured_array_cls: type[T],
        element_capacity: int,
        persistent_staging_buffer_element_capacity: int = 0,
    ) -> None:
        assert persistent_staging_buffer_element_capacity <= element_capacity

        self.device = device
        self.label = label
        self.structured_array_cls = structured_array_cls
        self.element_capacity = element_capacity
        self.persistent_staging_buffer_element_capacity = (
            persistent_staging_buffer_element_capacity
        )

        self.device_buffer = device.create_buffer(
            label=f"{label}.DeviceBuffer",
            size=self.structured_array_cls.array_size(shape=(element_capacity,)),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.persistent_staging_buffer = (
            device.create_buffer(
                label=f"{label}.StagingBuffer",
                size=self.structured_array_cls.array_size(
                    shape=(persistent_staging_buffer_element_capacity,)
                ),
                usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
            )
            if persistent_staging_buffer_element_capacity > 0
            else None
        )
        self.allocated_element_count = 0

        self._clear_device_buffer()

    def _clear_device_buffer(self) -> None:
        command_encoder = self.device.create_command_encoder(
            label=f"{self.label}.InitializationEncoder"
        )
        command_encoder.clear_buffer(buffer=self.device_buffer)
        self.device.queue.submit([command_encoder.finish()])

    def _on_dispose(self) -> None:
        self.device_buffer.destroy()
        if self.persistent_staging_buffer:
            self.persistent_staging_buffer.destroy()
        return super()._on_dispose()

    @contextmanager
    def _acquire_staging_buffer(
        self,
        element_count: int,
    ) -> Generator[wgpu.GPUBuffer, None, None]:
        if self.persistent_staging_buffer:
            assert self.persistent_staging_buffer_element_capacity >= element_count
            yield self.persistent_staging_buffer
        else:
            # Create a temporary staging buffer
            staging_buffer = self.device.create_buffer(
                label=f"{self.label}.TempStagingBuffer",
                size=self.structured_array_cls.array_size(shape=(element_count,)),
                usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
            )
            yield staging_buffer
            staging_buffer.destroy()

    @contextmanager
    def _acquire_command_encoder(
        self,
        label_suffix: str,
        user_command_encoder: wgpu.GPUCommandEncoder | None,
    ) -> Generator[wgpu.GPUCommandEncoder, None, None]:
        if user_command_encoder:
            assert self.persistent_staging_buffer, (
                "No persistent staging buffer available when using user-provided "
                "command encoder: the transient staging buffer created for this "
                "allocator will be destroyed before the command encoder is submitted."
            )
            yield user_command_encoder
        else:
            encoder = self.device.create_command_encoder(
                label=f"{self.label}.{label_suffix}"
            )
            yield encoder
            self.device.queue.submit([encoder.finish()])

    def insert(
        self,
        data: T,
        *,
        command_encoder: wgpu.GPUCommandEncoder | None = None,
    ) -> int:
        """
        Uploads data to the heap, allocating space for it.

        :param data: The structured array data to upload.
        :param command_encoder: Optional command encoder to use for the copy operation.
            If not provided, a temporary encoder will be created and the command will be
            submitted immediately, introducing a GPU synchronization point.
        :return: The offset (in elements) where the data was uploaded.
        """

        assert data.__class__ is self.structured_array_cls

        n = data.shape[0]

        # If n == 0, do nothing and return early.
        if n == 0:
            return self.allocated_element_count

        # Allocate:
        allocation_offset = self.allocated_element_count
        if self.allocated_element_count + n > self.element_capacity:
            raise RuntimeError(f"{self.label} heap capacity exceeded.")
        self.allocated_element_count += n

        # Upload via staging buffer:
        with self._acquire_staging_buffer(n) as staging_buffer:
            staging_buffer.map_sync(mode=wgpu.MapMode.WRITE)
            staging_buffer.write_mapped(data=data)
            staging_buffer.unmap()

            with self._acquire_command_encoder(
                label_suffix="UploadEncoder",
                user_command_encoder=command_encoder,
            ) as encoder:
                encoder.copy_buffer_to_buffer(
                    source=staging_buffer,
                    source_offset=0,
                    destination=self.device_buffer,
                    destination_offset=(
                        allocation_offset
                        * self.structured_array_cls.array_size(shape=(1,))
                    ),
                    size=self.structured_array_cls.array_size(shape=(n,)),
                )

        # Return offset in elements:
        return allocation_offset


#
# TextureHeap: helper for texture allocation in GPU texture arrays
#


type Draw3dTextureUsage = Literal[
    "color",
    "normal",
    "metalness",
    "roughness",
    "environment",
]


def _texture_format_for_draw_3d_usage(usage: Draw3dTextureUsage) -> wgpu.TextureFormat:
    mapping: dict[Draw3dTextureUsage, str] = {
        "color": "bc1_rgba_unorm",
        "normal": "bc5_rg_unorm",
        "metalness": "bc4_r_unorm",
        "roughness": "bc4_r_unorm",
        "environment": "rgba16float",
    }
    return wgpu.TextureFormat[mapping[usage]]


class TextureHeap(BaseDisposable):
    """
    Manages a GPU texture array for storing multiple textures.

    Supports 1-channel, 2-channel, or 3-channel textures, using BC4, BC5, or BC1
    compression respectively.
    """

    device: wgpu.GPUDevice
    label: str
    usage: Draw3dTextureUsage
    page_size_px: int
    page_count: int

    texture_format: wgpu.TextureFormat
    texture: wgpu.GPUTexture

    allocation_list: list[TextureHeapAllocation]
    allocation_heap: LinearHeap[PodTextureAllocationArray]

    cursor_x_px: int
    cursor_y_px: int
    cursor_h_px: int
    cursor_page: int

    def __init__(
        self,
        device: wgpu.GPUDevice,
        label: str,
        usage: Draw3dTextureUsage,
        page_count: int,
        page_size_px: int = 8192,
        allocation_capacity: int = 1024,
    ) -> None:
        # Only compressed formats need multiple-of-4 constraint
        if usage != "environment" and page_size_px % 4 != 0:
            raise ValueError(
                "TextureHeap page_size_px must be a multiple of 4 for compressed formats."
            )

        super().__init__()

        self.device = device
        self.label = label
        self.usage = usage
        self.page_size_px = page_size_px
        self.page_count = page_count

        self.texture_format = _texture_format_for_draw_3d_usage(self.usage)
        self.texture = device.create_texture(
            label=f"{label}.TextureArray",
            size=(page_size_px, page_size_px, page_count),
            dimension="2d",
            format=str(self.texture_format),
            usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST),
        )
        self.allocation_heap = LinearHeap[PodTextureAllocationArray](
            device=device,
            label=f"{label}.TextureMetadataHeap",
            structured_array_cls=PodTextureAllocationArray,
            element_capacity=allocation_capacity,
        )

        self.allocation_list = []

        self.cursor_x_px = 0
        self.cursor_y_px = 0
        self.cursor_h_px = 0
        self.cursor_page = 0

    def _on_dispose(self) -> None:
        self.texture.destroy()
        super()._on_dispose()

    def _encode_texture(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 3:
            raise LogicError(f"Texture has invalid ndim: expected 3: {data.ndim=}")

        match self.usage:
            case "color":
                return TextureHeap._encode_color_texture(data)
            case "normal":
                return TextureHeap._encode_normal_texture(data)
            case "metalness":
                return TextureHeap._encode_metalness_texture(data)
            case "roughness":
                return TextureHeap._encode_roughness_texture(data)
            case "environment":
                return TextureHeap._encode_environment_texture(data)
            case _:
                raise NotImplementedError()

    @staticmethod
    def _encode_color_texture(data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3
        if data.shape[0] % 4 != 0 or data.shape[1] % 4 != 0:
            raise LogicError(f"Texture size must be multiple of 4: {data.shape=}")
        if data.shape[2] != 3:
            raise LogicError(f"Color texture must have 3 channels: {data.shape[2]=}")
        return encode_bc1(data)

    @staticmethod
    def _encode_normal_texture(data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3
        if data.shape[0] % 4 != 0 or data.shape[1] % 4 != 0:
            raise LogicError(f"Texture size must be multiple of 4: {data.shape=}")
        if data.shape[2] != 3:
            raise LogicError(f"Normal texture must have 3 channels: {data.shape[2]=}")

        # Expect normal components in [0, 1] range
        assert np.all((data >= 0.0) & (data <= 1.0))

        # Convert from [0, 1] to [-1, 1]
        data = data * 2.0 - 1.0

        # Normalize all vectors to ensure unit length
        norms = np.linalg.norm(data, axis=2, keepdims=True)
        normalized_data = data / norms

        # Expect normal vectors to always have Z>=0
        assert np.all(normalized_data[:, :, 2] >= 0.0), (
            "Expected normal texture Z component to be non-negative: "
            f"{normalized_data[:, :, 2].min()=}, {normalized_data[:, :, 2].max()=}"
        )

        # Rescale back to [0, 1] range after normalization, keeping only X and Y
        # channels:
        normalized_data = (normalized_data + 1.0) * 0.5
        return encode_bc5(input_=normalized_data[:, :, 0:2])

    @staticmethod
    def _encode_metalness_texture(data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3
        if data.shape[0] % 4 != 0 or data.shape[1] % 4 != 0:
            raise LogicError(f"Texture size must be multiple of 4: {data.shape=}")
        if data.shape[2] != 1:
            raise LogicError(f"Metalness texture must have 1 channel: {data.shape[2]=}")
        return encode_bc4(input_=data)

    @staticmethod
    def _encode_roughness_texture(data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3
        if data.shape[0] % 4 != 0 or data.shape[1] % 4 != 0:
            raise LogicError(f"Texture size must be multiple of 4: {data.shape=}")
        if data.shape[2] != 1:
            raise LogicError(f"Roughness texture must have 1 channel: {data.shape[2]=}")
        return encode_bc4(input_=data)

    @staticmethod
    def _encode_environment_texture(data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3
        if data.shape[2] not in (3, 4):
            raise LogicError(
                f"Environment texture must have 3 or 4 channels: {data.shape[2]=}"
            )
        # Convert to RGBA if needed
        if data.shape[2] == 3:
            alpha = np.ones((data.shape[0], data.shape[1], 1), dtype=data.dtype)
            data = np.concatenate([data, alpha], axis=2)
        # Convert to float16 and return as-is (no block compression)
        return data.astype(np.float16)

    def _allocate(self, width_px: int, height_px: int) -> TextureHeapAllocation:
        # Move to next row?
        if self.cursor_x_px + width_px > self.page_size_px:
            self.cursor_x_px = 0
            self.cursor_y_px += self.cursor_h_px
            self.cursor_h_px = 0

        # Move to next page?
        if self.cursor_y_px + height_px > self.page_size_px:
            self.cursor_x_px = 0
            self.cursor_y_px = 0
            self.cursor_h_px = 0
            self.cursor_page += 1

        # Out of space?
        if self.cursor_page >= self.page_count:
            raise RuntimeError("TextureHeap capacity exceeded.")

        # Finalize allocation
        allocation = TextureHeapAllocation(
            usage=self.usage,
            texture_id=len(self.allocation_list),
            page=self.cursor_page,
            texture_x_px=self.cursor_x_px,
            texture_y_px=self.cursor_y_px,
            texture_w_px=width_px,
            texture_h_px=height_px,
        )
        self.cursor_x_px += width_px
        self.cursor_h_px = max(self.cursor_h_px, height_px)
        self.allocation_list.append(allocation)

        # Done:
        return allocation

    def _get_texture_data_layout(
        self, encoded_data: np.ndarray
    ) -> wgpu.TexelCopyBufferLayout:
        """Get the texture data layout based on the texture format."""
        match self.usage:
            case "color" | "normal" | "metalness" | "roughness":
                # Compressed formats: data is organized in blocks
                # Each "pixel" in encoded_data is actually a compressed block
                return wgpu.TexelCopyBufferLayout(
                    offset=0,
                    bytes_per_row=(
                        encoded_data.shape[1]  # blocks per row
                        * encoded_data.shape[2]  # bytes per block
                        * encoded_data.dtype.itemsize
                    ),
                    rows_per_image=encoded_data.shape[0],  # block rows
                )
            case "environment":
                # Uncompressed format: data is regular pixels
                return wgpu.TexelCopyBufferLayout(
                    offset=0,
                    bytes_per_row=(
                        encoded_data.shape[1]  # pixels per row
                        * encoded_data.shape[2]  # channels per pixel
                        * encoded_data.dtype.itemsize
                    ),
                    rows_per_image=encoded_data.shape[0],  # pixel rows
                )
            case _:
                raise NotImplementedError(f"Unsupported usage: {self.usage}")

    def _upload_record_to_gpu(
        self,
        allocation: TextureHeapAllocation,
    ) -> None:
        # Upload to the GPU buffer:
        data = PodTextureAllocationArray.empty((1,))
        data["x"][0] = allocation.texture_x_px / self.page_size_px
        data["y"][0] = allocation.texture_y_px / self.page_size_px + allocation.page
        data["w"][0] = allocation.texture_w_px / self.page_size_px
        data["h"][0] = allocation.texture_h_px / self.page_size_px
        heap_index = self.allocation_heap.insert(data)
        assert heap_index == allocation.texture_id

    def _upload_texels_to_gpu(
        self,
        encoded_data: np.ndarray,
        allocation: TextureHeapAllocation,
    ) -> None:
        self.device.queue.write_texture(
            destination=wgpu.TexelCopyTextureInfo(
                texture=self.texture,
                mip_level=0,
                origin=(
                    allocation.texture_x_px,
                    allocation.texture_y_px,
                    allocation.page,
                ),
            ),
            data=encoded_data.tobytes(),
            data_layout=self._get_texture_data_layout(encoded_data),
            size=(allocation.texture_w_px, allocation.texture_h_px, 1),
        )

    def insert(self, data: np.ndarray) -> TextureHeapAllocation:
        t0 = time.perf_counter()
        encoded_data = self._encode_texture(data)
        t1 = time.perf_counter()
        LOG.debug(
            f"Encoded {self.usage} texture {data.shape} -> {encoded_data.shape} in {(t1 - t0) * 1000:.2f}ms"
        )
        # For compressed formats, encoded_data shape is in blocks, need to convert to pixels
        # For uncompressed formats, encoded_data shape is already in pixels
        if self.usage == "environment":
            width_px = encoded_data.shape[1]
            height_px = encoded_data.shape[0]
        else:
            # Compressed: each element in encoded_data represents a 4x4 block
            width_px = encoded_data.shape[1] * 4
            height_px = encoded_data.shape[0] * 4

        allocation = self._allocate(width_px=width_px, height_px=height_px)
        self._upload_record_to_gpu(allocation)
        self._upload_texels_to_gpu(encoded_data, allocation)
        return allocation


@dataclass
class TextureHeapAllocation:
    usage: Draw3dTextureUsage
    texture_id: int
    page: int
    texture_x_px: int
    texture_y_px: int
    texture_w_px: int
    texture_h_px: int

    def __post_init__(self):
        assert 0 <= self.texture_x_px <= 0xFFFF
        assert 0 <= self.texture_y_px <= 0xFFFF


#
# POD Types (NumPy structured dtypes)
#

_FRAME_FLAG_EMIT_PRIMARY_RAY_DIRECTION = 1 << 0
_FRAME_FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R = 1 << 1
_FRAME_FLAG_EMIT_HIT_WORLD_POSITION = 1 << 2
_FRAME_FLAG_EMIT_CLOSEST_BVH_HIT_DEPTH_IN_R = 1 << 3
_FRAME_FLAG_EMIT_PRIMARY_RAY_COLOR = 1 << 4
_FRAME_FLAG_EMIT_HIT_NORMAL = 1 << 5
_FRAME_FLAG_EMIT_ORM = 1 << 6


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


class PodGeometryArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("bvh_node_span", POD_SPAN_DTYPE),
            ("triangle_span", POD_SPAN_DTYPE),
        ]
    )


class PodMaterialArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("color_map_id", np.uint32),
            ("color_factor", np.float32, (3,)),
            ("normal_map_id", np.uint32),
            ("metalness_map_id", np.uint32),
            ("metalness_factor", np.float32),
            ("roughness_map_id", np.uint32),
            ("roughness_factor", np.float32),
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


class PodVertexArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("position", np.float32, (3,)),
            ("normal", np.float32, (3,)),
            ("uv", np.float32, (2,)),
        ]
    )


class PodSubpixelArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("subpixel", np.float16),
        ]
    )


class PodFrameInfoArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("instance_count", np.uint32),
            ("target_size_w_px", np.uint32),
            ("target_size_h_px", np.uint32),
            ("debug_flags", np.uint32),
            ("environment_map_texture_id", np.int32),
            ("_pad0", np.uint32),
            ("_pad1", np.uint32),
            ("_pad2", np.uint32),
        ]
    )


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


class PodTextureAllocationArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),  # upper 16 bits stores page index
            ("w", np.float32),
            ("h", np.float32),
        ]
    )


LOG = logger(__name__)
