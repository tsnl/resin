__all__ = [
    "Draw3dAov",
    "Draw3dRenderer",
    "Draw3dScene",
]

import logging
from contextlib import contextmanager
import math
from dataclasses import dataclass, field
from typing import Generator, Literal
import time

import numpy as np
import numpy.typing as npt
import wgpu


from .basic import BaseDisposable, StructuredNDArray
from . import trace
from .excepts import LogicError
from .bvh import Blas, build_blas_bvh, build_tlas_bvh, transform_aabb
from .resources import GeometryResource, MaterialResource
from .images import encode_bc1, encode_bc4, encode_bc5

#
# AOV type alias
#

type Draw3dAov = Literal[
    "default",
    "per-pixel-radiance",
    "primary-ray-direction",
    "surface-depth",
    "surface-position",
    "surface-color",
    "surface-normal",
    "surface-orm",
    "surface-emissive",
    "bvh-depth",
]

#
# Renderer
#


class Draw3dRenderer(BaseDisposable):
    """
    A 3D renderer using ray tracing implemented with wgpu compute shaders.

    Usage:
    -   Create an instance of Draw3dRenderer.
    -   Create Draw3dScene instances representing the scenes to be rendered.
    -   For each frame, call `Draw3dRenderer.record()` with the scene and a GPU
        command encoder.
    -   Call `reset()` when changing scenes to clear all loaded resources (geometry,
        materials, textures) and reset the accumulator.

    The renderer supports temporal accumulation for path tracing convergence. The
    `accumulator_frame_count` parameter controls how many frames are kept in the
    history buffer and averaged together. Default is 4 frames.
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
    rgb_texture_heap: "TextureHeap"
    rg_texture_heap: "TextureHeap"
    mono_texture_heap: "TextureHeap"
    hdr_texture_heap: "TextureHeap"
    linear_sampler: wgpu.GPUSampler

    renderer_bind_group: wgpu.GPUBindGroup

    allocated_geometry_count: int
    allocated_bvh_node_count: int
    allocated_triangle_count: int
    allocated_material_count: int
    allocated_image_count: int
    allocated_subpixel_count: int
    _frame_index: int
    _construction_time: float

    # Frame profiling
    num_frame_profiling_samples: int
    _profiling_enabled: bool
    _profiling_query_set: wgpu.GPUQuerySet | None
    _profiling_resolve_buffer: wgpu.GPUBuffer | None
    _profiling_staging_buffer: wgpu.GPUBuffer | None
    _profiling_write_index: int
    _profiling_sample_count: int
    _profiling_last_log_time: float

    def __init__(
        self,
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
        target_size_wh_px: tuple[int, int],
        samples_per_pixel: int = 1,
        accumulator_frame_count: int = 4,
        render_scale: float = 1.0,
        instance_capacity: int = 1 << 10,
        geometry_capacity: int = 1 << 7,
        material_capacity: int = 1 << 7,
        bvh_node_capacity: int = 1 << 20,
        triangle_capacity: int = 1 << 22,
        image_capacity: int = 1 << 8,
        subpixel_capacity: int = 1 << 29,
        num_frame_profiling_samples: int = 5 * 3600,
    ):
        super().__init__()

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
                # RGB texture heap (BC1):
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
                # RG texture heap (BC5):
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
                # Mono texture heap (BC4):
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
                # HDR texture heap (rgba16float):
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
                # Linear sampler:
                wgpu.BindGroupLayoutEntry(
                    binding=12,
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
                # @group(1) @binding(0) var output_image: texture_storage_2d<rgba16float, write>;
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(1) var accum_image: texture_storage_2d<rgba16float, read_write>;
                wgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.read_write,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(2) var frame_per_pixel_radiance: texture_storage_2d<rgba16float, write>;
                wgpu.BindGroupLayoutEntry(
                    binding=2,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(3) var frame_primary_ray_direction_image: texture_storage_2d<rgba16float, write>;
                wgpu.BindGroupLayoutEntry(
                    binding=3,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(4) var frame_surface_depth_image: texture_storage_2d<rgba16float, write>;
                wgpu.BindGroupLayoutEntry(
                    binding=4,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(5) var frame_surface_position_image: texture_storage_2d<rgba16float, write>;
                wgpu.BindGroupLayoutEntry(
                    binding=5,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(6) var frame_surface_color_image: texture_storage_2d<rgba16float, write>;
                wgpu.BindGroupLayoutEntry(
                    binding=6,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(7) var frame_surface_normal_image: texture_storage_2d<rgba16float, write>;
                wgpu.BindGroupLayoutEntry(
                    binding=7,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(8) var frame_surface_orm_image: texture_storage_2d<rgba16float, write>;
                wgpu.BindGroupLayoutEntry(
                    binding=8,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(9) var frame_surface_emissive_image: texture_storage_2d<rgba16float, write>;
                wgpu.BindGroupLayoutEntry(
                    binding=9,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    storage_texture=wgpu.StorageTextureBindingLayout(
                        access=wgpu.StorageTextureAccess.write_only,
                        format=wgpu.TextureFormat.rgba16float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                ),
                # @group(1) @binding(10) var<uniform> frame_info: PodFrameInfo;
                wgpu.BindGroupLayoutEntry(
                    binding=10,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="uniform"),
                ),
                # @group(1) @binding(11) var<uniform> camera: PodCamera;
                wgpu.BindGroupLayoutEntry(
                    binding=11,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="uniform"),
                ),
                # @group(1) @binding(12) var<storage, read> instances: array<PodInstance>;
                wgpu.BindGroupLayoutEntry(
                    binding=12,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="read-only-storage"),
                ),
                # @group(1) @binding(13) var<storage, read> tlas_nodes: array<PodTlasNode>;
                wgpu.BindGroupLayoutEntry(
                    binding=13,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="read-only-storage"),
                ),
                # @group(1) @binding(14) var<storage, read> tlas_instance_indices: array<u32>;
                wgpu.BindGroupLayoutEntry(
                    binding=14,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="read-only-storage"),
                ),
                # @group(1) @binding(15) var<storage, read_write> wf_rays: array<PodRay>;
                wgpu.BindGroupLayoutEntry(
                    binding=15,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="storage"),
                ),
                # @group(1) @binding(16) var<storage, read_write> wf_path_states: array<PodPathState>;
                wgpu.BindGroupLayoutEntry(
                    binding=16,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="storage"),
                ),
                # @group(1) @binding(17) var<storage, read_write> wf_hits: array<PodHit>;
                wgpu.BindGroupLayoutEntry(
                    binding=17,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="storage"),
                ),
                # @group(1) @binding(18) var<storage, read_write> wf_compact_state
                wgpu.BindGroupLayoutEntry(
                    binding=18,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(type="storage"),
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

        # Wavefront path tracing pipelines
        self._wf_gen_primary_rays_pipeline = device.create_compute_pipeline(
            label="Draw3dRenderer.WfGenPrimaryRaysPipeline",
            layout=pipeline_layout,
            compute=wgpu.ProgrammableStage(
                module=self.draw_shader,
                entry_point="wf_gen_primary_rays",
            ),
        )
        self._wf_trace_rays_pipeline = device.create_compute_pipeline(
            label="Draw3dRenderer.WfTraceRaysPipeline",
            layout=pipeline_layout,
            compute=wgpu.ProgrammableStage(
                module=self.draw_shader,
                entry_point="wf_trace_rays",
            ),
        )
        self._wf_shade_and_extend_pipeline = device.create_compute_pipeline(
            label="Draw3dRenderer.WfShadeAndExtendPipeline",
            layout=pipeline_layout,
            compute=wgpu.ProgrammableStage(
                module=self.draw_shader,
                entry_point="wf_shade_and_extend",
            ),
        )
        self._wf_finalize_pipeline = device.create_compute_pipeline(
            label="Draw3dRenderer.WfFinalizePipeline",
            layout=pipeline_layout,
            compute=wgpu.ProgrammableStage(
                module=self.draw_shader,
                entry_point="wf_finalize",
            ),
        )
        self._wf_swap_buffers_pipeline = device.create_compute_pipeline(
            label="Draw3dRenderer.WfSwapBuffersPipeline",
            layout=pipeline_layout,
            compute=wgpu.ProgrammableStage(
                module=self.draw_shader,
                entry_point="wf_swap_buffers",
            ),
        )

        # Postprocess pipeline for tonemapping and upscaling
        # Postprocess shader is now in the same .wgsl file
        self.postprocess_shader = device.create_shader_module(code=shader_source)
        self.postprocess_bind_group_layout = device.create_bind_group_layout(
            label="Draw3dRenderer.PostprocessBindGroupLayout",
            entries=[
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                        multisampled=False,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    sampler=wgpu.SamplerBindingLayout(
                        type=wgpu.SamplerBindingType.filtering
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=2,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.uniform,
                    ),
                ),
            ],
        )
        postprocess_pipeline_layout = device.create_pipeline_layout(
            label="Draw3dRenderer.PostprocessPipelineLayout",
            bind_group_layouts=[self.postprocess_bind_group_layout],
        )
        self.postprocess_pipeline = device.create_render_pipeline(
            label="Draw3dRenderer.PostprocessPipeline",
            layout=postprocess_pipeline_layout,
            vertex=wgpu.VertexState(
                module=self.postprocess_shader,
                entry_point="vs_postprocess",
            ),
            fragment=wgpu.FragmentState(
                module=self.postprocess_shader,
                entry_point="fs_postprocess",
                targets=[
                    wgpu.ColorTargetState(
                        format=wgpu.TextureFormat.rgba16float,
                    )
                ],
            ),
            primitive=wgpu.PrimitiveState(
                topology=wgpu.PrimitiveTopology.triangle_list,
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

        self.rgb_texture_heap = TextureHeap(
            device=device,
            label="Draw3dRenderer.RgbTextureHeap",
            texture_format="rgb",
            page_count=16,
        )
        self.rg_texture_heap = TextureHeap(
            device=device,
            label="Draw3dRenderer.RgTextureHeap",
            texture_format="rg",
            page_count=16,
        )
        self.mono_texture_heap = TextureHeap(
            device=device,
            label="Draw3dRenderer.MonoTextureHeap",
            texture_format="mono",
            page_count=16,
        )
        self.hdr_texture_heap = TextureHeap(
            device=device,
            label="Draw3dRenderer.HdrTextureHeap",
            texture_format="hdr",
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
                # RGB texture heap
                wgpu.BindGroupEntry(
                    binding=4,
                    resource=self.rgb_texture_heap.texture.create_view(
                        dimension=wgpu.TextureViewDimension.d2_array,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=5,
                    resource=wgpu.BufferBinding(
                        buffer=self.rgb_texture_heap.allocation_heap.device_buffer,
                        offset=0,
                        size=self.rgb_texture_heap.allocation_heap.device_buffer.size,
                    ),
                ),
                # RG texture heap
                wgpu.BindGroupEntry(
                    binding=6,
                    resource=self.rg_texture_heap.texture.create_view(
                        dimension=wgpu.TextureViewDimension.d2_array,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=7,
                    resource=wgpu.BufferBinding(
                        buffer=self.rg_texture_heap.allocation_heap.device_buffer,
                        offset=0,
                        size=self.rg_texture_heap.allocation_heap.device_buffer.size,
                    ),
                ),
                # Mono texture heap
                wgpu.BindGroupEntry(
                    binding=8,
                    resource=self.mono_texture_heap.texture.create_view(
                        dimension=wgpu.TextureViewDimension.d2_array,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=9,
                    resource=wgpu.BufferBinding(
                        buffer=self.mono_texture_heap.allocation_heap.device_buffer,
                        offset=0,
                        size=self.mono_texture_heap.allocation_heap.device_buffer.size,
                    ),
                ),
                # HDR texture heap
                wgpu.BindGroupEntry(
                    binding=10,
                    resource=self.hdr_texture_heap.texture.create_view(
                        dimension=wgpu.TextureViewDimension.d2_array,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=11,
                    resource=wgpu.BufferBinding(
                        buffer=self.hdr_texture_heap.allocation_heap.device_buffer,
                        offset=0,
                        size=self.hdr_texture_heap.allocation_heap.device_buffer.size,
                    ),
                ),
                # Linear sampler
                wgpu.BindGroupEntry(
                    binding=12,
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
        self._frame_index = 0
        self._construction_time = time.monotonic()

        # Per-frame resources
        self._debug_flags = 0
        self._max_bounces = 3
        self._samples_per_pixel = samples_per_pixel
        self.accumulator_frame_count = accumulator_frame_count
        self._render_scale = render_scale

        # Internal render target at reduced resolution
        internal_w = max(1, int(target_size_wh_px[0] * self._render_scale))
        internal_h = max(1, int(target_size_wh_px[1] * self._render_scale))
        self._internal_size_wh_px = (internal_w, internal_h)

        # Create all output textures at internal resolution
        # Main output image (averaged accumulated result)
        self._output_image = device.create_texture(
            label="Draw3dRenderer.OutputImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        # Accumulator image for temporal accumulation
        self._accum_image = device.create_texture(
            label="Draw3dRenderer.AccumImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_DST),
        )
        # Debug output textures (created unconditionally)
        self._frame_per_pixel_radiance_image = device.create_texture(
            label="Draw3dRenderer.FramePerPixelRadianceImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_primary_ray_direction_image = device.create_texture(
            label="Draw3dRenderer.FramePrimaryRayDirectionImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_depth_image = device.create_texture(
            label="Draw3dRenderer.FrameSurfaceDepthImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_position_image = device.create_texture(
            label="Draw3dRenderer.FrameSurfacePositionImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_color_image = device.create_texture(
            label="Draw3dRenderer.FrameSurfaceColorImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_normal_image = device.create_texture(
            label="Draw3dRenderer.FrameSurfaceNormalImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_orm_image = device.create_texture(
            label="Draw3dRenderer.FrameSurfaceOrmImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_emissive_image = device.create_texture(
            label="Draw3dRenderer.FrameSurfaceEmissiveImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        # Full-resolution final output (postprocessed from _output_image)
        self.output_image = device.create_texture(
            label="Draw3dRenderer.FinalOutputImage",
            size=(target_size_wh_px[0], target_size_wh_px[1], 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_info_buffer = PerFrameBuffer[PodFrameInfoArray](
            device=device,
            label="Draw3dRenderer.FrameInfoHeap",
            structured_array_cls=PodFrameInfoArray,
            element_capacity=1,
            device_buffer_usages=wgpu.BufferUsage.UNIFORM,
        )
        self._camera_buffer = PerFrameBuffer[PodCameraArray](
            device=device,
            label="Draw3dRenderer.CameraHeap",
            structured_array_cls=PodCameraArray,
            element_capacity=1,
            device_buffer_usages=wgpu.BufferUsage.UNIFORM,
        )
        self._instance_buffer = PerFrameBuffer[PodInstanceArray](
            device=device,
            label="Draw3dRenderer.InstanceHeap",
            structured_array_cls=PodInstanceArray,
            element_capacity=instance_capacity,
            device_buffer_usages=wgpu.BufferUsage.STORAGE,
        )
        self._tlas_node_buffer = PerFrameBuffer[PodTlasNodeArray](
            device=device,
            label="Draw3dRenderer.TlasNodeBuffer",
            structured_array_cls=PodTlasNodeArray,
            element_capacity=instance_capacity * 2,
            device_buffer_usages=wgpu.BufferUsage.STORAGE,
        )
        self._tlas_instance_index_buffer = PerFrameBuffer[PodTlasInstanceIndexArray](
            device=device,
            label="Draw3dRenderer.TlasInstanceIndexBuffer",
            structured_array_cls=PodTlasInstanceIndexArray,
            element_capacity=instance_capacity,
            device_buffer_usages=wgpu.BufferUsage.STORAGE,
        )

        # Wavefront path tracing buffers
        wf_ray_count = internal_w * internal_h
        self._wf_ray_count = wf_ray_count

        # Ray buffer is double-sized for ping-pong compaction
        self._wf_ray_buffer = device.create_buffer(
            label="Draw3dRenderer.RayBuffer",
            size=PodRayArray.array_size(shape=(wf_ray_count * 2,)),
            usage=wgpu.BufferUsage.STORAGE,
        )
        # Path state is indexed by pixel (not compacted)
        self._wf_path_state_buffer = device.create_buffer(
            label="Draw3dRenderer.PathStateBuffer",
            size=PodPathStateArray.array_size(shape=(wf_ray_count,)),
            usage=wgpu.BufferUsage.STORAGE,
        )
        # Hit buffer is single-buffered (used within each bounce)
        self._wf_hit_buffer = device.create_buffer(
            label="Draw3dRenderer.HitBuffer",
            size=PodHitArray.array_size(shape=(wf_ray_count,)),
            usage=wgpu.BufferUsage.STORAGE,
        )
        # Compact state: [active_count, current_buf, next_count] - 3 atomic u32s
        self._wf_compact_state_buffer = device.create_buffer(
            label="Draw3dRenderer.CompactStateBuffer",
            size=3 * 4,  # 3 x u32
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        self._per_frame_bind_group = device.create_bind_group(
            label="Draw3dRenderer.PerFrameBindGroup",
            layout=self.per_frame_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=self._output_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=self._accum_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=self._frame_per_pixel_radiance_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=3,
                    resource=self._frame_primary_ray_direction_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=4,
                    resource=self._frame_surface_depth_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=5,
                    resource=self._frame_surface_position_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=6,
                    resource=self._frame_surface_color_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=7,
                    resource=self._frame_surface_normal_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=8,
                    resource=self._frame_surface_orm_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=9,
                    resource=self._frame_surface_emissive_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=10,
                    resource=wgpu.BufferBinding(
                        buffer=self._frame_info_buffer.device_buffer,
                        offset=0,
                        size=self._frame_info_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=11,
                    resource=wgpu.BufferBinding(
                        buffer=self._camera_buffer.device_buffer,
                        offset=0,
                        size=self._camera_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=12,
                    resource=wgpu.BufferBinding(
                        buffer=self._instance_buffer.device_buffer,
                        offset=0,
                        size=self._instance_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=13,
                    resource=wgpu.BufferBinding(
                        buffer=self._tlas_node_buffer.device_buffer,
                        offset=0,
                        size=self._tlas_node_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=14,
                    resource=wgpu.BufferBinding(
                        buffer=self._tlas_instance_index_buffer.device_buffer,
                        offset=0,
                        size=self._tlas_instance_index_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=15,
                    resource=wgpu.BufferBinding(
                        buffer=self._wf_ray_buffer,
                        offset=0,
                        size=self._wf_ray_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=16,
                    resource=wgpu.BufferBinding(
                        buffer=self._wf_path_state_buffer,
                        offset=0,
                        size=self._wf_path_state_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=17,
                    resource=wgpu.BufferBinding(
                        buffer=self._wf_hit_buffer,
                        offset=0,
                        size=self._wf_hit_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=18,
                    resource=wgpu.BufferBinding(
                        buffer=self._wf_compact_state_buffer,
                        offset=0,
                        size=self._wf_compact_state_buffer.size,
                    ),
                ),
            ],
        )

        # Postprocess uniform buffer for debug flags
        self._postprocess_uniform_buffer = device.create_buffer(
            label="Draw3dRenderer.PostprocessUniformBuffer",
            size=16,  # vec4<u32> alignment
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Postprocess bind group (reads _output_image, writes to output_image)
        self._postprocess_bind_group = device.create_bind_group(
            label="Draw3dRenderer.PostprocessBindGroup",
            layout=self.postprocess_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=self._output_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=self.linear_sampler,
                ),
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=wgpu.BufferBinding(
                        buffer=self._postprocess_uniform_buffer,
                        offset=0,
                        size=16,
                    ),
                ),
            ],
        )

        # Per-AOV postprocess bind groups for viewport display selection
        self._display_aov: Draw3dAov = "default"
        self._aov_postprocess_bind_groups = self._create_aov_postprocess_bind_groups()

        # Frame profiling resources
        self.num_frame_profiling_samples = num_frame_profiling_samples
        self._profiling_write_index = 0
        self._profiling_sample_count = 0
        self._profiling_last_log_time = 0.0

        # Check if timestamp-query feature is available
        self._profiling_enabled = "timestamp-query" in device.features
        if self._profiling_enabled:
            # Query set stores 2 timestamps per frame (begin and end of compute pass)
            self._profiling_query_set = device.create_query_set(
                label="Draw3dRenderer.ProfilingQuerySet",
                type=wgpu.QueryType.timestamp,
                count=2,
            )
            # Resolve buffer receives raw timestamp values from query set
            self._profiling_resolve_buffer = device.create_buffer(
                label="Draw3dRenderer.ProfilingResolveBuffer",
                size=16,  # 2 * uint64
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
            )
            # Staging buffer stores all profiling samples for CPU readback
            self._profiling_staging_buffer = device.create_buffer(
                label="Draw3dRenderer.ProfilingStagingBuffer",
                size=PodFrameTimingArray.array_size(
                    shape=(num_frame_profiling_samples,)
                ),
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
        else:
            self._profiling_query_set = None
            self._profiling_resolve_buffer = None
            self._profiling_staging_buffer = None

        # Initialize frame state
        self.reset()

    def _on_dispose(self) -> None:
        self.geometry_heap.dispose()
        self.bvh_node_heap.dispose()
        self.triangle_heap.dispose()
        self.material_heap.dispose()

        self.rgb_texture_heap.dispose()
        self.rg_texture_heap.dispose()
        self.mono_texture_heap.dispose()
        self.hdr_texture_heap.dispose()

        if self._profiling_resolve_buffer is not None:
            self._profiling_resolve_buffer.destroy()
        if self._profiling_staging_buffer is not None:
            self._profiling_staging_buffer.destroy()

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

    def _add_bvh_nodes(self, bvh_nodes: "PodBvhNodeArray") -> int:
        assert bvh_nodes.ndim == 1 and bvh_nodes.dtype == PodBvhNodeArray.DTYPE
        return self.bvh_node_heap.insert(bvh_nodes)

    def _add_triangles(self, vertices: "PodVertexArray") -> int:
        assert vertices.ndim == 1 and vertices.dtype == PodVertexArray.DTYPE
        assert vertices.shape[0] % 3 == 0
        return self.triangle_heap.insert(vertices) // 3

    def _add_texture(
        self,
        *,
        data: np.ndarray,
        usage: "Draw3dTextureUsage",
    ) -> "TextureHeapAllocation":
        return self._get_texture_heap(usage).insert(data=data, usage=usage)

    def _get_texture_heap(self, usage: "Draw3dTextureUsage") -> "TextureHeap":
        return {
            "color": self.rgb_texture_heap,
            "normal": self.rg_texture_heap,
            "metalness": self.mono_texture_heap,
            "roughness": self.mono_texture_heap,
            "environment": self.hdr_texture_heap,
            "emissive": self.rgb_texture_heap,
            "diffuse_f0": self.rgb_texture_heap,
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
        emissive_map_id: int,
        emissive_factor: tuple[float, float, float],
        diffuse_f0_map_id: int,
        diffuse_f0_factor: tuple[float, float, float],
    ) -> int:
        data = PodMaterialArray.empty(shape=(1,))
        data["color_map_id"][0] = np.uint32(color_map_id)
        data["color_factor"][0] = np.array(color_factor, dtype=np.float32)
        data["normal_map_id"][0] = np.uint32(normal_map_id)
        data["metalness_map_id"][0] = np.uint32(metalness_map_id)
        data["metalness_factor"][0] = np.float32(metalness_factor)
        data["roughness_map_id"][0] = np.uint32(roughness_map_id)
        data["roughness_factor"][0] = np.float32(roughness_factor)
        data["emissive_map_id"][0] = np.uint32(emissive_map_id)
        data["emissive_factor"][0] = np.array(emissive_factor, dtype=np.float32)
        data["diffuse_f0_map_id"][0] = np.uint32(diffuse_f0_map_id)
        data["diffuse_f0_factor"][0] = np.array(diffuse_f0_factor, dtype=np.float32)
        return self.material_heap.insert(data)

    def record(
        self,
        scene: "Draw3dScene",
        command_encoder: wgpu.GPUCommandEncoder,
        timestamp: float | None = None,
    ) -> None:
        self._validate_scene(scene)

        timestamp = (
            timestamp
            if timestamp is not None
            else time.monotonic() - self._construction_time
        )

        self._record(
            command_encoder,
            scene,
            timestamp,
            self._frame_index,
        )
        self._frame_index += 1
        self._log_frame_timing_stats_if_due()

    def get_output_image(self, aov: Draw3dAov | None = None) -> wgpu.GPUTexture:
        """Get the final output image (full resolution, postprocessed).

        Args:
            aov: Which AOV to get output for. If None, uses the current display_aov.
                 Note: This returns the same postprocessed output texture, but the
                 content depends on which AOV was selected via set_display_aov()
                 before calling record().
        """
        return self.output_image

    def get_display_aov(self) -> Draw3dAov:
        """Get the currently selected AOV for viewport display."""
        return self._display_aov

    def set_display_aov(self, aov: Draw3dAov) -> None:
        """Set which AOV to display in the viewport.

        This determines which AOV texture is used as the source for postprocessing.
        The AOV must be enabled via set_render_settings(enabled_aov_list=...) for
        meaningful output.
        """
        self._display_aov = aov

    def _create_aov_postprocess_bind_groups(self) -> dict[Draw3dAov, wgpu.GPUBindGroup]:
        """Create postprocess bind groups for each AOV texture."""
        aov_textures: dict[Draw3dAov, wgpu.GPUTexture] = {
            "default": self._output_image,
            "per-pixel-radiance": self._frame_per_pixel_radiance_image,
            "primary-ray-direction": self._frame_primary_ray_direction_image,
            "surface-depth": self._frame_surface_depth_image,
            "surface-position": self._frame_surface_position_image,
            "surface-color": self._frame_surface_color_image,
            "surface-normal": self._frame_surface_normal_image,
            "surface-orm": self._frame_surface_orm_image,
            "surface-emissive": self._frame_surface_emissive_image,
            "bvh-depth": self._frame_surface_depth_image,  # Reuses depth texture
        }

        bind_groups: dict[Draw3dAov, wgpu.GPUBindGroup] = {}
        for aov, texture in aov_textures.items():
            bind_groups[aov] = self.device.create_bind_group(
                label=f"Draw3dRenderer.PostprocessBindGroup.{aov}",
                layout=self.postprocess_bind_group_layout,
                entries=[
                    wgpu.BindGroupEntry(
                        binding=0,
                        resource=texture.create_view(),
                    ),
                    wgpu.BindGroupEntry(
                        binding=1,
                        resource=self.linear_sampler,
                    ),
                    wgpu.BindGroupEntry(
                        binding=2,
                        resource=wgpu.BufferBinding(
                            buffer=self._postprocess_uniform_buffer,
                            offset=0,
                            size=16,
                        ),
                    ),
                ],
            )
        return bind_groups

    def resize(self, target_size_wh_px: tuple[int, int]) -> None:
        """
        Resize the renderer's output textures.

        Call this when the window size changes to avoid recreating the entire renderer.
        All loaded geometry, materials, and textures are preserved.
        """
        if target_size_wh_px == self.target_size_wh_px:
            return

        self.target_size_wh_px = target_size_wh_px

        # Calculate internal resolution
        internal_w = max(1, int(target_size_wh_px[0] * self._render_scale))
        internal_h = max(1, int(target_size_wh_px[1] * self._render_scale))
        self._internal_size_wh_px = (internal_w, internal_h)

        # Recreate all size-dependent textures
        self._output_image = self.device.create_texture(
            label="Draw3dRenderer.OutputImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._accum_image = self.device.create_texture(
            label="Draw3dRenderer.AccumImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_DST),
        )
        self._frame_per_pixel_radiance_image = self.device.create_texture(
            label="Draw3dRenderer.FramePerPixelRadianceImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_primary_ray_direction_image = self.device.create_texture(
            label="Draw3dRenderer.FramePrimaryRayDirectionImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_depth_image = self.device.create_texture(
            label="Draw3dRenderer.FrameSurfaceDepthImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_position_image = self.device.create_texture(
            label="Draw3dRenderer.FrameSurfacePositionImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_color_image = self.device.create_texture(
            label="Draw3dRenderer.FrameSurfaceColorImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_normal_image = self.device.create_texture(
            label="Draw3dRenderer.FrameSurfaceNormalImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_orm_image = self.device.create_texture(
            label="Draw3dRenderer.FrameSurfaceOrmImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._frame_surface_emissive_image = self.device.create_texture(
            label="Draw3dRenderer.FrameSurfaceEmissiveImage",
            size=(internal_w, internal_h, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        # Full-resolution final output
        self.output_image = self.device.create_texture(
            label="Draw3dRenderer.FinalOutputImage",
            size=(target_size_wh_px[0], target_size_wh_px[1], 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=(
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )

        # Recreate wavefront buffers for new resolution
        wf_ray_count = internal_w * internal_h
        self._wf_ray_count = wf_ray_count

        self._wf_ray_buffer = self.device.create_buffer(
            label="Draw3dRenderer.RayBuffer",
            size=PodRayArray.array_size(shape=(wf_ray_count * 2,)),
            usage=wgpu.BufferUsage.STORAGE,
        )
        self._wf_path_state_buffer = self.device.create_buffer(
            label="Draw3dRenderer.PathStateBuffer",
            size=PodPathStateArray.array_size(shape=(wf_ray_count,)),
            usage=wgpu.BufferUsage.STORAGE,
        )
        self._wf_hit_buffer = self.device.create_buffer(
            label="Draw3dRenderer.HitBuffer",
            size=PodHitArray.array_size(shape=(wf_ray_count,)),
            usage=wgpu.BufferUsage.STORAGE,
        )
        self._wf_compact_state_buffer = self.device.create_buffer(
            label="Draw3dRenderer.CompactStateBuffer",
            size=3 * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        # Recreate per-frame bind group with new texture views
        self._per_frame_bind_group = self.device.create_bind_group(
            label="Draw3dRenderer.PerFrameBindGroup",
            layout=self.per_frame_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=self._output_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=self._accum_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=self._frame_per_pixel_radiance_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=3,
                    resource=self._frame_primary_ray_direction_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=4,
                    resource=self._frame_surface_depth_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=5,
                    resource=self._frame_surface_position_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=6,
                    resource=self._frame_surface_color_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=7,
                    resource=self._frame_surface_normal_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=8,
                    resource=self._frame_surface_orm_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=9,
                    resource=self._frame_surface_emissive_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=10,
                    resource=wgpu.BufferBinding(
                        buffer=self._frame_info_buffer.device_buffer,
                        offset=0,
                        size=self._frame_info_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=11,
                    resource=wgpu.BufferBinding(
                        buffer=self._camera_buffer.device_buffer,
                        offset=0,
                        size=self._camera_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=12,
                    resource=wgpu.BufferBinding(
                        buffer=self._instance_buffer.device_buffer,
                        offset=0,
                        size=self._instance_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=13,
                    resource=wgpu.BufferBinding(
                        buffer=self._tlas_node_buffer.device_buffer,
                        offset=0,
                        size=self._tlas_node_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=14,
                    resource=wgpu.BufferBinding(
                        buffer=self._tlas_instance_index_buffer.device_buffer,
                        offset=0,
                        size=self._tlas_instance_index_buffer.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=15,
                    resource=wgpu.BufferBinding(
                        buffer=self._wf_ray_buffer,
                        offset=0,
                        size=self._wf_ray_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=16,
                    resource=wgpu.BufferBinding(
                        buffer=self._wf_path_state_buffer,
                        offset=0,
                        size=self._wf_path_state_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=17,
                    resource=wgpu.BufferBinding(
                        buffer=self._wf_hit_buffer,
                        offset=0,
                        size=self._wf_hit_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=18,
                    resource=wgpu.BufferBinding(
                        buffer=self._wf_compact_state_buffer,
                        offset=0,
                        size=self._wf_compact_state_buffer.size,
                    ),
                ),
            ],
        )

        # Recreate postprocess bind group
        self._postprocess_bind_group = self.device.create_bind_group(
            label="Draw3dRenderer.PostprocessBindGroup",
            layout=self.postprocess_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=self._output_image.create_view(),
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=self.linear_sampler,
                ),
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=wgpu.BufferBinding(
                        buffer=self._postprocess_uniform_buffer,
                        offset=0,
                        size=16,
                    ),
                ),
            ],
        )

        # Recreate per-AOV postprocess bind groups
        self._aov_postprocess_bind_groups = self._create_aov_postprocess_bind_groups()

        # Reset per-frame state (accumulator needs to restart)
        # Explicitly preserve geometry, materials, and textures
        self.reset(
            geometry_heap=False,
            material_heap=False,
            texture_heap=False,
            per_frame_state=True,
        )

    def get_frame_per_pixel_radiance_image(self) -> wgpu.GPUTexture:
        """Get the per-pixel radiance debug output (internal resolution)."""
        return self._frame_per_pixel_radiance_image

    def get_frame_primary_ray_direction_image(self) -> wgpu.GPUTexture:
        """Get the primary ray direction debug output (internal resolution)."""
        return self._frame_primary_ray_direction_image

    def get_frame_surface_depth_image(self) -> wgpu.GPUTexture:
        """Get the surface depth debug output (internal resolution)."""
        return self._frame_surface_depth_image

    def get_frame_surface_position_image(self) -> wgpu.GPUTexture:
        """Get the surface position debug output (internal resolution)."""
        return self._frame_surface_position_image

    def get_frame_surface_color_image(self) -> wgpu.GPUTexture:
        """Get the surface color debug output (internal resolution)."""
        return self._frame_surface_color_image

    def get_frame_surface_normal_image(self) -> wgpu.GPUTexture:
        """Get the surface normal debug output (internal resolution)."""
        return self._frame_surface_normal_image

    def get_frame_surface_orm_image(self) -> wgpu.GPUTexture:
        """Get the surface ORM (occlusion/roughness/metalness) debug output (internal resolution)."""
        return self._frame_surface_orm_image

    def get_frame_surface_emissive_image(self) -> wgpu.GPUTexture:
        """Get the surface emissive debug output (internal resolution)."""
        return self._frame_surface_emissive_image

    @property
    def is_profiling_enabled(self) -> bool:
        """
        Check if GPU frame profiling is enabled.

        Profiling requires the 'timestamp-query' feature to be available on the
        GPU device. If not available, `get_frame_timing_stats()` will return an
        empty array.
        """
        return self._profiling_enabled

    def get_frame_timing_stats(self) -> npt.NDArray[np.float32]:
        """
        Get GPU frame timing statistics for recent frames.

        Returns an array of frame times in seconds, one entry per frame. The array
        contains up to `num_frame_profiling_samples` entries (configured in constructor),
        ordered from oldest to newest.

        Note: This method blocks while mapping the staging buffer for CPU read access.
        Call this infrequently (e.g., once per second) to avoid stalling the GPU pipeline.

        Returns an empty array if profiling is not enabled (timestamp-query feature
        not available on the GPU device).

        :return: Array of frame times in seconds (float32).
        """
        if not self._profiling_enabled or self._profiling_staging_buffer is None:
            return np.array([], dtype=np.float32)

        if self._profiling_sample_count == 0:
            return np.array([], dtype=np.float32)

        self._profiling_staging_buffer.map_sync(mode=wgpu.MapMode.READ)
        raw_memoryview = self._profiling_staging_buffer.read_mapped()
        assert isinstance(raw_memoryview, memoryview)
        raw_data = bytes(raw_memoryview)
        self._profiling_staging_buffer.unmap()

        timing_data = np.frombuffer(raw_data, dtype=PodFrameTimingArray.DTYPE)

        # Extract valid samples in chronological order
        if self._profiling_sample_count < self.num_frame_profiling_samples:
            # Buffer not yet full - samples are at indices 0..sample_count-1
            valid_data = timing_data[: self._profiling_sample_count]
        else:
            # Buffer is full and wrapping - reorder from oldest to newest
            valid_data = np.concatenate(
                [
                    timing_data[self._profiling_write_index :],
                    timing_data[: self._profiling_write_index],
                ]
            )

        # Compute frame times in seconds from nanosecond timestamps
        begin_ns = valid_data["begin_ns"].astype(np.float64)
        end_ns = valid_data["end_ns"].astype(np.float64)
        frame_times_s = ((end_ns - begin_ns) / 1e9).astype(np.float32)

        return frame_times_s

    def _log_frame_timing_stats_if_due(self) -> None:
        """Log frame timing stats every 1 second (debug level)."""
        current_time = time.monotonic()
        if current_time - self._profiling_last_log_time < 1.0:
            return

        self._profiling_last_log_time = current_time

        parts: list[str] = []

        # GPU kernel timing (from timestamp queries)
        if self._profiling_enabled:
            frame_times = self.get_frame_timing_stats()
            if len(frame_times) > 0:
                gpu_mean = float(frame_times.mean()) * 1000.0
                gpu_p5 = float(np.percentile(frame_times, 5)) * 1000.0
                gpu_p50 = float(np.percentile(frame_times, 50)) * 1000.0
                gpu_p95 = float(np.percentile(frame_times, 95)) * 1000.0
                parts.append(
                    f"GPU p5/p50/p95/mean={gpu_p5:.2f}/{gpu_p50:.2f}/"
                    f"{gpu_p95:.2f}/{gpu_mean:.2f}ms"
                )

        # TLAS build timing (from global trace hub)
        tlas_stats = trace.compute_execution_time(
            "Draw3dRenderer/record/tlas_build", truncate_window_sec=1.0
        )
        if tlas_stats is not None:
            tlas_mean = tlas_stats.mean_sec * 1000.0
            tlas_p5 = tlas_stats.p5_sec * 1000.0
            tlas_p50 = tlas_stats.p50_sec * 1000.0
            tlas_p95 = tlas_stats.p95_sec * 1000.0
            parts.append(
                f"TLAS p5/p50/p95/mean={tlas_p5:.2f}/{tlas_p50:.2f}/"
                f"{tlas_p95:.2f}/{tlas_mean:.2f}ms"
            )

        if parts:
            LOG.debug(" | ".join(parts))

    def set_debug_flags(
        self,
        *,
        emit_primary_ray_direction: bool = False,
        emit_closest_hit_depth_in_r: bool = False,
        emit_hit_world_position: bool = False,
        emit_closest_hit_bvh_depth_in_r: bool = False,
        emit_surface_color: bool = False,
        emit_surface_normal: bool = False,
        emit_surface_orm: bool = False,
        emit_surface_emissive: bool = False,
        disable_jitter: bool = False,
    ) -> None:
        """
        Set debug visualization modes.

        :param emit_primary_ray_direction: If True, output normalized ray direction as RGB.
        :param emit_closest_hit_depth_in_r: If True, output normalized hit depth in red channel.
        :param emit_hit_world_position: If True, output world-space hit position as RGB.
        :param emit_closest_hit_bvh_depth_in_r: If True, output normalized hit depth to BVH leaf in red channel.
        :param emit_surface_color: If True, output sampled texture color at hit point.
        :param emit_surface_normal: If True, output world-space hit normal as RGB.
        :param emit_surface_orm: If True, output ORM (Opacity, Roughness, Metalness) as RGB.
        :param emit_surface_emissive: If True, output sampled emissive color at hit point.
        :param disable_jitter: If True, disable jitter for primary ray generation.
        """
        self._debug_flags = 0
        if emit_primary_ray_direction:
            self._debug_flags |= _FRAME_FLAG_EMIT_PRIMARY_RAY_DIRECTION
        if emit_closest_hit_depth_in_r:
            self._debug_flags |= _FRAME_FLAG_EMIT_SURFACE_DEPTH
        if emit_hit_world_position:
            self._debug_flags |= _FRAME_FLAG_EMIT_HIT_WORLD_POSITION
        if emit_closest_hit_bvh_depth_in_r:
            self._debug_flags |= _FRAME_FLAG_EMIT_BVH_DEPTH
        if emit_surface_color:
            self._debug_flags |= _FRAME_FLAG_EMIT_SURFACE_COLOR
        if emit_surface_normal:
            self._debug_flags |= _FRAME_FLAG_EMIT_SURFACE_NORMAL
        if emit_surface_orm:
            self._debug_flags |= _FRAME_FLAG_EMIT_SURFACE_ORM
        if emit_surface_emissive:
            self._debug_flags |= _FRAME_FLAG_EMIT_SURFACE_EMISSIVE
        if disable_jitter:
            self._debug_flags |= _FRAME_FLAG_DISABLE_JITTER

    def set_render_settings(
        self,
        *,
        samples_per_pixel: int | None = None,
        max_bounces: int | None = None,
        render_scale: float | None = None,
        accumulator_frame_count: int | None = None,
        enabled_aov_list: list[Draw3dAov] | None = None,
        disable_jitter: bool | None = None,
    ) -> None:
        """Update render settings dynamically.

        Args:
            samples_per_pixel: Number of samples per pixel per frame. If None, unchanged.
            max_bounces: Maximum ray bounces for path tracing. If None, unchanged.
            render_scale: Internal render resolution scale (0.0-1.0). If None, unchanged.
            accumulator_frame_count: Number of frames to accumulate. If None, unchanged.
            enabled_aov_list: List of AOVs to enable for rendering. When set, replaces
                the current AOV flags. Use ["default"] for standard path tracing output.
            disable_jitter: If True, disable jitter for primary ray generation.
                If False, enable jitter. If None, unchanged.
        """
        if samples_per_pixel is not None:
            self._samples_per_pixel = samples_per_pixel
        if max_bounces is not None:
            self._max_bounces = max_bounces
        if accumulator_frame_count is not None:
            self.accumulator_frame_count = accumulator_frame_count
        if render_scale is not None and render_scale != self._render_scale:
            self._render_scale = render_scale
            # Force resize to recreate render targets at new internal resolution
            current_size = self.target_size_wh_px
            self.target_size_wh_px = (0, 0)  # Bypass early-return check
            self.resize(current_size)
        if enabled_aov_list is not None:
            # Clear AOV flags and set new ones based on the list
            self._debug_flags &= _FRAME_FLAG_DISABLE_JITTER  # Preserve jitter flag
            for aov in enabled_aov_list:
                self._debug_flags |= _AOV_TO_FLAG[aov]
        if disable_jitter is not None:
            if disable_jitter:
                self._debug_flags |= _FRAME_FLAG_DISABLE_JITTER
            else:
                self._debug_flags &= ~_FRAME_FLAG_DISABLE_JITTER

    def reset(
        self,
        encoder: wgpu.GPUCommandEncoder | None = None,
        *,
        geometry_heap: bool = True,
        material_heap: bool = True,
        texture_heap: bool = True,
        per_frame_state: bool = True,
    ) -> None:
        """Reset renderer state, with selective control over what gets cleared.

        Args:
            encoder: If provided, uses clear_buffer for proper GPU synchronization.
                Otherwise, uses write_buffer which is asynchronous.
            geometry_heap: Clear geometry, BVH nodes, and triangles.
            material_heap: Clear materials.
            texture_heap: Clear all texture heaps (color, normal, metalness,
                roughness, environment).
            per_frame_state: Reset accumulator buffer, frame index, timestamp,
                and debug flags.
        """
        if per_frame_state:
            self._frame_index = 0
            self._construction_time = time.monotonic()
            self._debug_flags = 0
            self._reset_accumulator_texture()

        if geometry_heap:
            self.geometry_heap.clear()
            self.bvh_node_heap.clear()
            self.triangle_heap.clear()

        if material_heap:
            self.material_heap.clear()

        if texture_heap:
            self.rgb_texture_heap.clear()
            self.rg_texture_heap.clear()
            self.mono_texture_heap.clear()
            self.hdr_texture_heap.clear()

    def _reset_accumulator_texture(self) -> None:
        w, h = self._internal_size_wh_px
        zeros = np.zeros((h * w * 4 * 2,), dtype=np.uint8)  # rgba16float
        self.device.queue.write_texture(
            destination=wgpu.TexelCopyTextureInfo(
                texture=self._accum_image,
                mip_level=0,
                origin=(0, 0, 0),
            ),
            data=zeros,
            data_layout=wgpu.TexelCopyBufferLayout(
                bytes_per_row=w * 4 * 2,
                rows_per_image=h,
            ),
            size=(w, h, 1),
        )

    def _validate_scene(self, scene: "Draw3dScene") -> None:
        for geometry, material in scene.meshes:
            if not geometry.is_valid:
                raise LogicError(
                    f"Draw3dGeometry {geometry.geometry_id} is no longer valid "
                    "(geometry heap was cleared since it was created)"
                )
            if not material.is_valid:
                raise LogicError(
                    f"Draw3dMaterial {material.material_id} is no longer valid "
                    "(material heap was cleared since it was created)"
                )

        if scene.environment_map is not None:
            env_gen = scene.environment_map.allocation.heap_generation
            heap_gen = self.hdr_texture_heap.generation_count
            if env_gen != heap_gen:
                raise LogicError(
                    "Draw3dTexture (environment map) is no longer valid "
                    "(texture heap was cleared since it was created)"
                )

    def _record(
        self,
        encoder: wgpu.GPUCommandEncoder,
        scene: "Draw3dScene",
        timestamp: float,
        frame_index: int,
    ) -> None:
        with trace.span("Draw3dRenderer/record", "render", args={"frame": frame_index}):
            self._record_impl(encoder, scene, timestamp, frame_index)

    def _record_impl(
        self,
        encoder: wgpu.GPUCommandEncoder,
        scene: "Draw3dScene",
        timestamp: float,
        frame_index: int,
    ) -> None:
        instance_count = sum(len(transforms) for transforms in scene.meshes.values())

        # Upload instances and build TLAS first to get tlas_node_count
        with trace.span("Draw3dRenderer/record/upload_instances", "render"):
            tlas_node_count = self._upload_instances_info(scene.meshes, encoder)

        with trace.span("Draw3dRenderer/record/upload_frame_info", "render"):
            self._upload_frame_info(
                instance_count,
                tlas_node_count,
                encoder,
                self._debug_flags,
                timestamp,
                frame_index,
                environment_map_texture_id=(
                    scene.environment_map.allocation.texture_id
                    if scene.environment_map
                    else -1
                ),
            )
            self._upload_camera_info(scene.camera, encoder)

        # Path tracing compute passes (wavefront architecture)
        t_compute_start = time.perf_counter()
        timestamp_writes: wgpu.ComputePassTimestampWrites | None = None
        if self._profiling_enabled and self._profiling_query_set is not None:
            timestamp_writes = wgpu.ComputePassTimestampWrites(
                query_set=self._profiling_query_set,
                beginning_of_pass_write_index=0,
                end_of_pass_write_index=1,
            )

        self._record_wavefront_passes(encoder, timestamp_writes)

        t_compute_end = time.perf_counter()
        trace.add_time_span(
            "Draw3dRenderer/record/compute_pass_encode",
            t_compute_start,
            t_compute_end,
        )

        # Resolve timestamp queries and copy to staging buffer (if profiling enabled)
        if (
            self._profiling_enabled
            and self._profiling_query_set is not None
            and self._profiling_resolve_buffer is not None
            and self._profiling_staging_buffer is not None
        ):
            encoder.resolve_query_set(
                query_set=self._profiling_query_set,
                first_query=0,
                query_count=2,
                destination=self._profiling_resolve_buffer,
                destination_offset=0,
            )
            staging_offset = (
                self._profiling_write_index * PodFrameTimingArray.DTYPE.itemsize
            )
            encoder.copy_buffer_to_buffer(
                source=self._profiling_resolve_buffer,
                source_offset=0,
                destination=self._profiling_staging_buffer,
                destination_offset=staging_offset,
                size=PodFrameTimingArray.DTYPE.itemsize,
            )
            self._profiling_write_index = (
                self._profiling_write_index + 1
            ) % self.num_frame_profiling_samples
            self._profiling_sample_count = min(
                self._profiling_sample_count + 1, self.num_frame_profiling_samples
            )

        # Postprocess render pass (upscales and tonemaps to output_image)
        with trace.span("Draw3dRenderer/record/postprocess_encode", "render"):
            render_pass = encoder.begin_render_pass(
                label="Draw3dRenderer.PostprocessPass",
                color_attachments=[
                    wgpu.RenderPassColorAttachment(
                        view=self.output_image.create_view(),
                        load_op=wgpu.LoadOp.clear,
                        store_op=wgpu.StoreOp.store,
                        clear_value=(0.0, 0.0, 0.0, 1.0),
                    )
                ],
            )
            render_pass.set_pipeline(self.postprocess_pipeline)
            # Select bind group based on display AOV
            bind_group = self._aov_postprocess_bind_groups[self._display_aov]
            render_pass.set_bind_group(0, bind_group, [], 0, 0)
            render_pass.draw(3, 1, 0, 0)
            render_pass.end()

    def _record_wavefront_passes(
        self,
        encoder: wgpu.GPUCommandEncoder,
        timestamp_writes: "wgpu.ComputePassTimestampWrites | None",
    ) -> None:
        """Record wavefront path tracing passes.

        This dispatches separate compute passes for:
        1. Primary ray generation
        2. Ray tracing (BVH traversal) - repeated for each bounce
        3. Shading and path extension - repeated for each bounce
        4. Buffer swap (for compaction) - repeated for each bounce
        5. Finalization (write to output)
        """
        w, h = self._internal_size_wh_px
        total_rays = w * h

        # Workgroup sizes for different kernels
        wg_2d_x = math.ceil(w / 8)
        wg_2d_y = math.ceil(h / 8)
        wg_1d = math.ceil(total_rays / 64)

        # Initialize compact state: [active_count=total_rays, current_buf=0, next_count=0]
        compact_init = np.array([total_rays, 0, 0], dtype=np.uint32)
        self.queue.write_buffer(
            self._wf_compact_state_buffer, 0, compact_init.tobytes()
        )

        # Pass 1: Generate primary rays
        with trace.span("Draw3dRenderer/wavefront/gen_primary_rays", "render"):
            compute_pass = encoder.begin_compute_pass(
                label="Draw3dRenderer.WfGenPrimaryRays",
                timestamp_writes=timestamp_writes,
            )
            compute_pass.set_pipeline(self._wf_gen_primary_rays_pipeline)
            compute_pass.set_bind_group(0, self.renderer_bind_group, [], 0, 0)
            compute_pass.set_bind_group(1, self._per_frame_bind_group, [], 0, 0)
            compute_pass.dispatch_workgroups(wg_2d_x, wg_2d_y, 1)
            compute_pass.end()

        # Bounce loop: trace and shade for each bounce
        for bounce in range(self._max_bounces):
            # Pass 2: Trace rays (BVH traversal)
            with trace.span(f"Draw3dRenderer/wavefront/trace_rays/{bounce}", "render"):
                compute_pass = encoder.begin_compute_pass(
                    label=f"Draw3dRenderer.WfTraceRays.{bounce}",
                )
                compute_pass.set_pipeline(self._wf_trace_rays_pipeline)
                compute_pass.set_bind_group(0, self.renderer_bind_group, [], 0, 0)
                compute_pass.set_bind_group(1, self._per_frame_bind_group, [], 0, 0)
                compute_pass.dispatch_workgroups(wg_1d, 1, 1)
                compute_pass.end()

            # Pass 3: Shade hits and extend paths
            with trace.span(
                f"Draw3dRenderer/wavefront/shade_and_extend/{bounce}", "render"
            ):
                compute_pass = encoder.begin_compute_pass(
                    label=f"Draw3dRenderer.WfShadeAndExtend.{bounce}",
                )
                compute_pass.set_pipeline(self._wf_shade_and_extend_pipeline)
                compute_pass.set_bind_group(0, self.renderer_bind_group, [], 0, 0)
                compute_pass.set_bind_group(1, self._per_frame_bind_group, [], 0, 0)
                compute_pass.dispatch_workgroups(wg_1d, 1, 1)
                compute_pass.end()

            # Pass 4: Swap buffers (prepare for next bounce)
            with trace.span(
                f"Draw3dRenderer/wavefront/swap_buffers/{bounce}", "render"
            ):
                compute_pass = encoder.begin_compute_pass(
                    label=f"Draw3dRenderer.WfSwapBuffers.{bounce}",
                )
                compute_pass.set_pipeline(self._wf_swap_buffers_pipeline)
                compute_pass.set_bind_group(0, self.renderer_bind_group, [], 0, 0)
                compute_pass.set_bind_group(1, self._per_frame_bind_group, [], 0, 0)
                compute_pass.dispatch_workgroups(1, 1, 1)
                compute_pass.end()

        # Pass 5: Finalize and write to output
        with trace.span("Draw3dRenderer/wavefront/finalize", "render"):
            compute_pass = encoder.begin_compute_pass(
                label="Draw3dRenderer.WfFinalize",
            )
            compute_pass.set_pipeline(self._wf_finalize_pipeline)
            compute_pass.set_bind_group(0, self.renderer_bind_group, [], 0, 0)
            compute_pass.set_bind_group(1, self._per_frame_bind_group, [], 0, 0)
            compute_pass.dispatch_workgroups(wg_2d_x, wg_2d_y, 1)
            compute_pass.end()

    def _upload_frame_info(
        self,
        instance_count: int,
        tlas_node_count: int,
        command_encoder: wgpu.GPUCommandEncoder,
        debug_flags: int,
        timestamp: float,
        frame_index: int,
        environment_map_texture_id: int = -1,
    ) -> None:
        frame_info_data = PodFrameInfoArray.empty(shape=(1,))
        frame_info_data["instance_count"] = instance_count
        frame_info_data["target_size_w_px"] = self._internal_size_wh_px[0]
        frame_info_data["target_size_h_px"] = self._internal_size_wh_px[1]
        frame_info_data["debug_flags"] = debug_flags
        frame_info_data["environment_map_texture_id"] = environment_map_texture_id
        frame_info_data["timestamp"] = np.uint32(timestamp * 65536.0)  # 16.16 fixed pt
        frame_info_data["frame_index"] = np.uint32(frame_index)
        frame_info_data["max_bounces"] = np.uint32(self._max_bounces)
        frame_info_data["samples_per_pixel"] = np.uint32(self._samples_per_pixel)
        frame_info_data["accumulated_frame_index"] = np.uint32(frame_index)
        frame_info_data["tlas_node_count"] = np.uint32(tlas_node_count)

        self._frame_info_buffer.write(frame_info_data, command_encoder)

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

        self._camera_buffer.write(camera_data, command_encoder)

    def _upload_instances_info(
        self,
        instances: dict[
            tuple["Draw3dGeometry", "Draw3dMaterial"],
            npt.NDArray[np.float32],
        ],
        command_encoder: wgpu.GPUCommandEncoder,
    ) -> int:
        total_instance_count = sum(len(t) for t in instances.values())
        if total_instance_count > self.instance_capacity:
            raise RuntimeError("Draw3dRenderer instance heap capacity exceeded.")

        # Build instance data and compute world-space AABBs for TLAS
        t_instance_prep_start = time.perf_counter()
        data = PodInstanceArray.empty(shape=(total_instance_count,))
        instance_aabbs = np.empty((total_instance_count, 2, 3), dtype=np.float32)
        offset = 0
        for (geometry, material), transforms in instances.items():
            n = transforms.shape[0]
            data["geometry_id"][offset : offset + n] = geometry.geometry_id
            data["material_id"][offset : offset + n] = material.material_id
            data["transform"][offset : offset + n] = transforms

            # Compute inverse transforms and world-space AABBs
            for i in range(n):
                transform_4x4 = transforms[i]  # Shape (4, 4)
                inv_transform_4x4 = np.linalg.inv(transform_4x4)
                data["inv_transform"][offset + i] = inv_transform_4x4
                # Transform BLAS AABB to world space for TLAS construction
                instance_aabbs[offset + i] = transform_aabb(
                    geometry.blas_aabb, transform_4x4
                )

            offset += n
        t_instance_prep_end = time.perf_counter()
        trace.add_time_span(
            "Draw3dRenderer/record/instance_prep",
            t_instance_prep_start,
            t_instance_prep_end,
        )

        t_instance_upload_start = time.perf_counter()
        self._instance_buffer.write(data, command_encoder)
        t_instance_upload_end = time.perf_counter()
        trace.add_time_span(
            "Draw3dRenderer/record/instance_upload",
            t_instance_upload_start,
            t_instance_upload_end,
        )

        # Build TLAS from world-space instance AABBs
        if total_instance_count > 0:
            tlas_t0 = time.perf_counter()
            tlas = build_tlas_bvh(instance_aabbs)
            tlas_t1 = time.perf_counter()
            trace.add_time_span("Draw3dRenderer/record/tlas_build", tlas_t0, tlas_t1)

            # Marshall and upload TLAS nodes
            t_tlas_marshal_start = time.perf_counter()
            tlas_node_data = PodTlasNodeArray.empty(shape=(tlas.node_count,))
            for i in range(tlas.node_count):
                tlas_node_data["instance_span"]["begin"][i] = tlas.instance_span[i, 0]
                tlas_node_data["instance_span"]["end"][i] = tlas.instance_span[i, 1]
                tlas_node_data["children"][i][0] = tlas.children[i, 0]
                tlas_node_data["children"][i][1] = tlas.children[i, 1]
                tlas_node_data["aabb"]["min"][i] = tlas.aabb[i, 0]
                tlas_node_data["aabb"]["max"][i] = tlas.aabb[i, 1]
            t_tlas_marshal_end = time.perf_counter()
            trace.add_time_span(
                "Draw3dRenderer/record/tlas_marshal",
                t_tlas_marshal_start,
                t_tlas_marshal_end,
            )

            t_tlas_upload_start = time.perf_counter()
            self._tlas_node_buffer.write(tlas_node_data, command_encoder)

            # Upload reordered instance indices
            tlas_instance_index_data = PodTlasInstanceIndexArray.empty(
                shape=(total_instance_count,)
            )
            tlas_instance_index_data["index"] = tlas.instance_indices
            self._tlas_instance_index_buffer.write(
                tlas_instance_index_data, command_encoder
            )
            t_tlas_upload_end = time.perf_counter()
            trace.add_time_span(
                "Draw3dRenderer/record/tlas_upload",
                t_tlas_upload_start,
                t_tlas_upload_end,
            )

            return tlas.node_count
        else:
            return 0


class Draw3dGeometry(BaseDisposable):
    renderer: Draw3dRenderer

    triangle_count: int
    vertex_count: int

    geometry_id: int
    geometry_heap_offset_in_triangles: int
    heap_generation: int

    blas_aabb: npt.NDArray[np.float32]  # (2, 3) root BLAS AABB for TLAS construction

    def __init__(
        self,
        renderer: Draw3dRenderer,
        *,
        v_p_array: npt.NDArray[np.float32],
        v_n_array: npt.NDArray[np.float32],
        v_t_array: npt.NDArray[np.float32],
        t_indices: npt.NDArray[np.uint32],
    ) -> None:
        super().__init__()

        self.renderer = renderer
        self.triangle_count = t_indices.shape[0]
        self.heap_generation = renderer.geometry_heap.generation_count

        # Construct the BVH first.
        # This produces an updated `t_indices` array with a different triangle order.
        # We need to use this reordered index array for all subsequent uploads.
        bvh = build_blas_bvh(t=t_indices, v=v_p_array)
        t_indices = bvh.t

        # Store root BLAS AABB for TLAS construction
        self.blas_aabb = bvh.aabb[0].copy()

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
    def _marshall_bvh(bvh: Blas) -> "PodBvhNodeArray":
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
        v_p_array: npt.NDArray[np.float32],
        v_n_array: npt.NDArray[np.float32],
        v_t_array: npt.NDArray[np.float32],
        t_indices: npt.NDArray[np.uint32],
    ) -> "PodVertexArray":
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

    @property
    def is_valid(self) -> bool:
        """Check if this geometry is still valid (heap hasn't been cleared)."""
        return self.heap_generation == self.renderer.geometry_heap.generation_count

    @staticmethod
    def from_resource(
        geometry: "GeometryResource",
        renderer: Draw3dRenderer,
    ) -> "Draw3dGeometry":
        return Draw3dGeometry(
            renderer,
            v_p_array=geometry.v_p_array,
            v_n_array=geometry.v_n_array,
            v_t_array=geometry.v_t_array,
            t_indices=geometry.t_indices,
        )


class Draw3dTexture(BaseDisposable):
    renderer: Draw3dRenderer
    allocation: "TextureHeapAllocation"

    def __init__(
        self,
        renderer: Draw3dRenderer,
        *,
        data: np.ndarray,
        usage: "Draw3dTextureUsage",
    ) -> None:
        super().__init__()

        self.renderer = renderer
        self.allocation = renderer._add_texture(data=data, usage=usage)


class Draw3dMaterial(BaseDisposable):
    renderer: Draw3dRenderer
    material_id: int
    heap_generation: int

    color_texture: "Draw3dTexture | None"
    color_factor: tuple[float, float, float]
    normal_texture: "Draw3dTexture | None"
    metalness_texture: "Draw3dTexture | None"
    metalness_factor: float
    roughness_texture: "Draw3dTexture | None"
    roughness_factor: float
    emissive_texture: "Draw3dTexture | None"
    emissive_factor: tuple[float, float, float]
    diffuse_f0_texture: "Draw3dTexture | None"
    diffuse_f0_factor: tuple[float, float, float]

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
        emissive_texture: "Draw3dTexture | None" = None,
        emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0),
        diffuse_f0_texture: "Draw3dTexture | None" = None,
        diffuse_f0_factor: tuple[float, float, float] = (0.04, 0.04, 0.04),
    ) -> None:
        super().__init__()

        self.renderer = renderer
        self.heap_generation = renderer.material_heap.generation_count

        self.color_texture = color_texture
        self.color_factor = color_factor
        self.normal_texture = normal_texture
        self.metalness_texture = metalness_texture
        self.metalness_factor = metalness_factor
        self.roughness_texture = roughness_texture
        self.roughness_factor = roughness_factor
        self.emissive_texture = emissive_texture
        self.emissive_factor = emissive_factor
        self.diffuse_f0_texture = diffuse_f0_texture
        self.diffuse_f0_factor = diffuse_f0_factor

        # Upload material data to GPU
        color_map_id = color_texture.allocation.texture_id if color_texture else 0
        normal_map_id = normal_texture.allocation.texture_id if normal_texture else 0
        metalness_map_id = (
            metalness_texture.allocation.texture_id if metalness_texture else 0
        )
        roughness_map_id = (
            roughness_texture.allocation.texture_id if roughness_texture else 0
        )
        emissive_map_id = (
            emissive_texture.allocation.texture_id if emissive_texture else 0xFFFFFFFF
        )
        diffuse_f0_map_id = (
            diffuse_f0_texture.allocation.texture_id
            if diffuse_f0_texture
            else 0xFFFFFFFF
        )

        self.material_id = renderer._add_material(
            color_map_id=color_map_id,
            color_factor=color_factor,
            normal_map_id=normal_map_id,
            metalness_map_id=metalness_map_id,
            metalness_factor=metalness_factor,
            roughness_map_id=roughness_map_id,
            roughness_factor=roughness_factor,
            emissive_map_id=emissive_map_id,
            emissive_factor=emissive_factor,
            diffuse_f0_map_id=diffuse_f0_map_id,
            diffuse_f0_factor=diffuse_f0_factor,
        )

    @property
    def is_valid(self) -> bool:
        """Check if this material is still valid (heap hasn't been cleared)."""
        return self.heap_generation == self.renderer.material_heap.generation_count

    @staticmethod
    def from_resource(
        material: "MaterialResource",
        renderer: Draw3dRenderer,
    ) -> "Draw3dMaterial":
        color_texture = (
            Draw3dTexture(
                renderer,
                data=material.color_map,
                usage="color",
            )
            if material.color_map is not None
            else None
        )
        normal_texture = (
            Draw3dTexture(
                renderer,
                data=material.normal_map,
                usage="normal",
            )
            if material.normal_map is not None
            else None
        )
        metalness_texture = (
            Draw3dTexture(
                renderer,
                data=material.metalness_map,
                usage="metalness",
            )
            if material.metalness_map is not None
            else None
        )
        roughness_texture = (
            Draw3dTexture(
                renderer,
                data=material.roughness_map,
                usage="roughness",
            )
            if material.roughness_map is not None
            else None
        )
        emissive_texture = (
            Draw3dTexture(
                renderer,
                data=material.emissive_map,
                usage="emissive",
            )
            if material.emissive_map is not None
            else None
        )
        diffuse_f0_texture = (
            Draw3dTexture(
                renderer,
                data=material.diffuse_f0_map,
                usage="diffuse_f0",
            )
            if material.diffuse_f0_map is not None
            else None
        )
        return Draw3dMaterial(
            renderer,
            color_texture=color_texture,
            color_factor=material.color_factor,
            normal_texture=normal_texture,
            metalness_texture=metalness_texture,
            metalness_factor=material.metalness_factor,
            roughness_texture=roughness_texture,
            roughness_factor=material.roughness_factor,
            emissive_texture=emissive_texture,
            emissive_factor=material.emissive_factor,
            diffuse_f0_texture=diffuse_f0_texture,
            diffuse_f0_factor=material.diffuse_f0_factor,
        )


@dataclass(kw_only=True)
class Draw3dScene:
    """
    Represents a 3D scene to be rendered.
    """

    camera: "Draw3dCamera"

    meshes: dict[
        tuple[Draw3dGeometry, Draw3dMaterial],
        npt.NDArray[np.float32],
    ] = field(default_factory=dict)

    environment_map: Draw3dTexture | None = None


@dataclass(kw_only=True)
class Draw3dCamera:
    """
    Represents a camera in the 3D scene.
    """

    transform: npt.NDArray[np.float32]
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

    def _on_dispose(self) -> None:
        self.device_buffer.destroy()
        return super()._on_dispose()

    def write(self, data: T, command_encoder: wgpu.GPUCommandEncoder) -> None:
        assert data.__class__ is self.structured_array_cls
        assert data.shape[0] <= self.element_capacity

        # Early return if there's no data to write
        if data.shape[0] == 0:
            return

        # Use write_buffer for async upload (no GPU sync required)
        # The command_encoder parameter is kept for API compatibility but unused
        _ = command_encoder
        self.device.queue.write_buffer(
            buffer=self.device_buffer,
            buffer_offset=0,
            data=data,
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
    generation_count: int

    def __init__(
        self,
        device: wgpu.GPUDevice,
        label: str,
        structured_array_cls: type[T],
        element_capacity: int,
        persistent_staging_buffer_element_capacity: int = 0,
    ) -> None:
        super().__init__()

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
        self.generation_count = 0

        self._clear_device_buffer()

    def _clear_device_buffer(self) -> None:
        command_encoder = self.device.create_command_encoder(
            label=f"{self.label}.InitializationEncoder"
        )
        command_encoder.clear_buffer(buffer=self.device_buffer)
        self.device.queue.submit([command_encoder.finish()])

    def clear(self) -> None:
        """Reset the heap, deallocating all elements."""
        self.allocated_element_count = 0
        self.generation_count += 1
        self._clear_device_buffer()

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
    "emissive",
    "diffuse_f0",
]

type Draw3dTextureFormat = Literal["rgb", "rg", "mono", "hdr"]


def _wgpu_texture_format_for_draw_3d_texture_format(
    texture_format: Draw3dTextureFormat,
) -> wgpu.TextureFormat:
    mapping: dict[Draw3dTextureFormat, str] = {
        "rgb": "bc1_rgba_unorm",
        "rg": "bc5_rg_unorm",
        "mono": "bc4_r_unorm",
        "hdr": "rgba16float",
    }
    return wgpu.TextureFormat[mapping[texture_format]]


class TextureHeap(BaseDisposable):
    """
    Manages a GPU texture array for storing multiple textures.

    Supports 1-channel, 2-channel, 3-channel, or HDR textures, using BC4, BC5, BC1,
    or rgba16float respectively.
    """

    device: wgpu.GPUDevice
    label: str
    texture_format: Draw3dTextureFormat
    page_size_px: int
    page_count: int

    wgpu_texture_format: wgpu.TextureFormat
    texture: wgpu.GPUTexture

    allocation_list: list["TextureHeapAllocation"]
    allocation_heap: LinearHeap["PodTextureAllocationArray"]

    cursor_x_px: int
    cursor_y_px: int
    cursor_h_px: int
    cursor_page: int
    generation_count: int

    def __init__(
        self,
        device: wgpu.GPUDevice,
        label: str,
        texture_format: Draw3dTextureFormat,
        page_count: int,
        page_size_px: int = 8192,
        allocation_capacity: int = 1024,
    ) -> None:
        # Only compressed formats need multiple-of-4 constraint
        if texture_format != "hdr" and page_size_px % 4 != 0:
            raise ValueError(
                "TextureHeap page_size_px must be a multiple of 4 for compressed formats."
            )

        super().__init__()

        self.device = device
        self.label = label
        self.texture_format = texture_format
        self.page_size_px = page_size_px
        self.page_count = page_count

        self.wgpu_texture_format = _wgpu_texture_format_for_draw_3d_texture_format(
            self.texture_format
        )
        self.texture = device.create_texture(
            label=f"{label}.TextureArray",
            size=(page_size_px, page_size_px, page_count),
            dimension="2d",
            format=str(self.wgpu_texture_format),
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
        self.generation_count = 0

    def _on_dispose(self) -> None:
        self.texture.destroy()
        super()._on_dispose()

    def clear(self) -> None:
        """Reset the texture heap, deallocating all textures."""
        self.allocation_list = []
        self.allocation_heap.clear()
        self.cursor_x_px = 0
        self.cursor_y_px = 0
        self.cursor_h_px = 0
        self.cursor_page = 0
        self.generation_count += 1

    def _encode_texture(
        self, data: np.ndarray, usage: "Draw3dTextureUsage"
    ) -> np.ndarray:
        if data.ndim != 3:
            raise LogicError(f"Texture has invalid ndim: expected 3: {data.ndim=}")

        match usage:
            case "color":
                return TextureHeap._encode_rgb_texture(data)
            case "normal":
                return TextureHeap._encode_rg_texture(data)
            case "metalness":
                return TextureHeap._encode_mono_texture(data)
            case "roughness":
                return TextureHeap._encode_mono_texture(data)
            case "environment":
                return TextureHeap._encode_hdr_texture(data)
            case "emissive":
                return TextureHeap._encode_rgb_texture(data)
            case "diffuse_f0":
                return TextureHeap._encode_rgb_texture(data)
            case _:
                raise NotImplementedError()

    @staticmethod
    def _encode_rgb_texture(data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3
        if data.shape[0] % 4 != 0 or data.shape[1] % 4 != 0:
            raise LogicError(f"Texture size must be multiple of 4: {data.shape=}")
        if data.shape[2] != 3:
            raise LogicError(f"RGB texture must have 3 channels: {data.shape[2]=}")
        return encode_bc1(data)

    @staticmethod
    def _encode_rg_texture(data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3
        if data.shape[0] % 4 != 0 or data.shape[1] % 4 != 0:
            raise LogicError(f"Texture size must be multiple of 4: {data.shape=}")
        if data.shape[2] != 3:
            raise LogicError(f"RG texture must have 3 channels: {data.shape[2]=}")

        # Expect normal components in [0, 1] range
        assert np.all((data >= 0.0) & (data <= 1.0))

        # Convert from [0, 1] to [-1, 1]
        data = data * 2.0 - 1.0

        # Normalize all vectors to ensure unit length
        norms = np.linalg.norm(data, axis=2, keepdims=True)
        data_unit_length = data / norms

        # Expect normal vectors to always have Z>=0
        if np.any(data_unit_length[:, :, 2] < 0.0):
            LOG.debug(
                "Normal texture contains invalid normals with negative Z component. "
                "This is usually really subtle and is caused by compression artifacts. "
                "This may cause visual artifacts."
            )
            data_unit_length[:, :, 2] = data_unit_length[:, :, 2].clip(0.0, 1.0)

        # Rescale back to [0, 1] range after normalization, keeping only X and Y
        # channels:
        data_unit_length = (data_unit_length + 1.0) * 0.5
        return encode_bc5(input_=data_unit_length[:, :, 0:2])

    @staticmethod
    def _encode_mono_texture(data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3
        if data.shape[0] % 4 != 0 or data.shape[1] % 4 != 0:
            raise LogicError(f"Texture size must be multiple of 4: {data.shape=}")
        if data.shape[2] != 1:
            raise LogicError(f"Mono texture must have 1 channel: {data.shape[2]=}")
        return encode_bc4(input_=data)

    @staticmethod
    def _encode_hdr_texture(data: np.ndarray) -> np.ndarray:
        assert data.ndim == 3
        if data.shape[2] not in (3, 4):
            raise LogicError(f"HDR texture must have 3 or 4 channels: {data.shape[2]=}")
        # Convert to RGBA if needed
        if data.shape[2] == 3:
            alpha = np.ones((data.shape[0], data.shape[1], 1), dtype=data.dtype)
            data = np.concatenate([data, alpha], axis=2)
        # Convert to float16 and return as-is (no block compression)
        return data.astype(np.float16)

    def _allocate(self, width_px: int, height_px: int) -> "TextureHeapAllocation":
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
            texture_format=self.texture_format,
            texture_id=len(self.allocation_list),
            page=self.cursor_page,
            texture_x_px=self.cursor_x_px,
            texture_y_px=self.cursor_y_px,
            texture_w_px=width_px,
            texture_h_px=height_px,
            heap_generation=self.generation_count,
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
        match self.texture_format:
            case "rgb" | "rg" | "mono":
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
            case "hdr":
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
                raise NotImplementedError(
                    f"Unsupported texture format: {self.texture_format}"
                )

    def _upload_record_to_gpu(
        self,
        allocation: "TextureHeapAllocation",
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
        allocation: "TextureHeapAllocation",
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

    def insert(
        self, data: np.ndarray, usage: "Draw3dTextureUsage"
    ) -> "TextureHeapAllocation":
        t0 = time.perf_counter()
        encoded_data = self._encode_texture(data, usage)
        t1 = time.perf_counter()
        LOG.debug(
            f"Encoded {usage} texture {data.shape} -> {encoded_data.shape} in {(t1 - t0) * 1000:.2f}ms"
        )
        # For compressed formats, encoded_data shape is in blocks, need to convert to pixels
        # For uncompressed formats, encoded_data shape is already in pixels
        if self.texture_format == "hdr":
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
    texture_format: Draw3dTextureFormat
    texture_id: int
    page: int
    texture_x_px: int
    texture_y_px: int
    texture_w_px: int
    texture_h_px: int
    heap_generation: int

    def __post_init__(self) -> None:
        assert 0 <= self.texture_x_px <= 0xFFFF
        assert 0 <= self.texture_y_px <= 0xFFFF


#
# POD Types (NumPy structured dtypes)
#

_FRAME_FLAG_EMIT_PRIMARY_RAY_DIRECTION = 1 << 0
_FRAME_FLAG_EMIT_SURFACE_DEPTH = 1 << 1
_FRAME_FLAG_EMIT_HIT_WORLD_POSITION = 1 << 2
_FRAME_FLAG_EMIT_BVH_DEPTH = 1 << 3
_FRAME_FLAG_EMIT_SURFACE_COLOR = 1 << 4
_FRAME_FLAG_EMIT_SURFACE_NORMAL = 1 << 5
_FRAME_FLAG_EMIT_SURFACE_ORM = 1 << 6
_FRAME_FLAG_DISABLE_JITTER = 1 << 7
_FRAME_FLAG_EMIT_PER_PIXEL_RADIANCE = 1 << 8
_FRAME_FLAG_EMIT_SURFACE_EMISSIVE = 1 << 9

_AOV_TO_FLAG: dict[Draw3dAov, int] = {
    "default": 0,
    "per-pixel-radiance": _FRAME_FLAG_EMIT_PER_PIXEL_RADIANCE,
    "primary-ray-direction": _FRAME_FLAG_EMIT_PRIMARY_RAY_DIRECTION,
    "surface-depth": _FRAME_FLAG_EMIT_SURFACE_DEPTH,
    "surface-position": _FRAME_FLAG_EMIT_HIT_WORLD_POSITION,
    "surface-color": _FRAME_FLAG_EMIT_SURFACE_COLOR,
    "surface-normal": _FRAME_FLAG_EMIT_SURFACE_NORMAL,
    "surface-orm": _FRAME_FLAG_EMIT_SURFACE_ORM,
    "surface-emissive": _FRAME_FLAG_EMIT_SURFACE_EMISSIVE,
    "bvh-depth": _FRAME_FLAG_EMIT_BVH_DEPTH,
}


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
            ("emissive_map_id", np.uint32),
            ("emissive_factor", np.float32, (3,)),
            ("diffuse_f0_map_id", np.uint32),
            ("diffuse_f0_factor", np.float32, (3,)),
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


class PodTlasNodeArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("instance_span", POD_SPAN_DTYPE),
            ("children", np.uint32, (2,)),
            ("aabb", POD_AABB_DTYPE),
        ]
    )


class PodTlasInstanceIndexArray(StructuredNDArray):
    DTYPE = np.dtype([("index", np.uint32)])


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
            ("timestamp", np.uint32),
            ("frame_index", np.uint32),
            ("max_bounces", np.uint32),
            ("samples_per_pixel", np.uint32),
            ("accumulated_frame_index", np.uint32),
            ("tlas_node_count", np.uint32),
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


class PodFrameTimingArray(StructuredNDArray):
    """Stores GPU timestamp pairs (begin, end) for frame profiling."""

    DTYPE = np.dtype(
        [
            ("begin_ns", np.uint64),
            ("end_ns", np.uint64),
        ]
    )


class PodRayArray(StructuredNDArray):
    """Per-ray state for wavefront path tracing."""

    DTYPE = np.dtype(
        [
            ("origin_x", np.float32),
            ("origin_y", np.float32),
            ("origin_z", np.float32),
            ("pixel_id", np.uint32),  # x | (y << 16)
            ("direction_x", np.float32),
            ("direction_y", np.float32),
            ("direction_z", np.float32),
            ("bounce", np.uint32),
        ]
    )


class PodPathStateArray(StructuredNDArray):
    """Per-path state for wavefront path tracing."""

    DTYPE = np.dtype(
        [
            ("throughput_r", np.float32),
            ("throughput_g", np.float32),
            ("throughput_b", np.float32),
            ("rng_seed", np.uint32),
            ("accumulated_r", np.float32),
            ("accumulated_g", np.float32),
            ("accumulated_b", np.float32),
            ("flags", np.uint32),  # bit 0: terminated
        ]
    )


class PodHitArray(StructuredNDArray):
    """Per-ray hit record for wavefront path tracing."""

    DTYPE = np.dtype(
        [
            ("barycentric_u", np.float32),
            ("barycentric_v", np.float32),
            ("distance", np.float32),
            ("triangle_id", np.uint32),
            ("instance_id", np.uint32),
            ("geometry_id", np.uint32),
            ("_pad0", np.uint32),
            ("_pad1", np.uint32),
        ]
    )


LOG = logging.getLogger(__name__)
