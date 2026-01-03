__all__ = [
    "Draw3dFrame",
    "Draw3dRenderer",
    "Draw3dScene",
]

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import jaxtyping as jt
import wgpu

from .basic import BaseDisposable, StructuredNDArray

#
# Renderer
#


class Draw3dRenderer:
    device: wgpu.GPUDevice

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
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.uniform
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=2,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.uniform
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=3,
                    visibility=wgpu.ShaderStage.COMPUTE,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
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
            compute=wgpu.ProgrammableStage(module=draw_shader, entry_point="main"),
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

        # Initialization: clear device buffers to zero
        encoder = device.create_command_encoder(
            label="Draw3dRenderer.InitializationEncoder"
        )
        encoder.clear_buffer(self.geometry_heap_device_buffer, offset=0)
        encoder.clear_buffer(self.bvh_node_heap_device_buffer, offset=0)
        encoder.clear_buffer(self.triangle_heap_device_buffer, offset=0)
        queue.submit([encoder.finish()])

    def _add_geometry(self, vertices: PodVertexArray) -> int:
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

        # Upload:
        self.triangle_heap_device_buffer.map_sync(
            mode=wgpu.MapMode.WRITE,
            offset=allocation_offset_in_bytes,
            size=vertices.nbytes,
        )
        self.triangle_heap_device_buffer.write_mapped(
            data=vertices,
            buffer_offset=allocation_offset_in_bytes,
        )

        # Return offset in triangles:
        return allocation_offset_in_triangles

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

        self.output_image = self._device.create_texture(
            label="Draw3dFrame.OutputImage",
            size=(renderer.target_size_wh_px[0], renderer.target_size_wh_px[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
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

    def record(
        self,
        pipeline: wgpu.GPUComputePipeline,
        renderer_bind_group: wgpu.GPUBindGroup,
        target_size_wh: tuple[int, int],
        command_encoder: wgpu.GPUCommandEncoder,
        scene: Draw3dScene,
    ) -> None:
        instance_count = sum(len(transforms) for transforms in scene.instances.values())

        self._upload_frame_info(instance_count, command_encoder)
        self._upload_camera_info(scene.camera, command_encoder)
        # TODO: upload more data as needed

        compute_pass = command_encoder.begin_compute_pass(
            label="Draw3dFrame.ComputePass"
        )
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
    ) -> None:
        frame_info_data = PodFrameInfoArray.empty(shape=(1,))
        frame_info_data["instance_count"] = instance_count
        frame_info_data["target_size_w_px"] = self.renderer.target_size_wh_px[0]
        frame_info_data["target_size_h_px"] = self.renderer.target_size_wh_px[1]

        self.frame_info_staging_buffer.map_sync(wgpu.MapMode.WRITE)
        self.frame_info_staging_buffer.write_mapped(data=frame_info_data)
        self.frame_info_staging_buffer.unmap()

        command_encoder = self._device.create_command_encoder(
            label="Draw3dFrame.UploadFrameInfoEncoder"
        )
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


class Draw3dGeometry(BaseDisposable):
    renderer: Draw3dRenderer

    triangle_count: int
    vertex_count: int

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
        self.vertex_count = self.triangle_count * 3

        # Interleave the vertex arrays:
        v = np.concatenate([v_p_array, v_n_array, v_t_array], axis=-1)
        v = v.view(dtype=PodVertexArray.DTYPE).squeeze()
        assert v.shape == (self.vertex_count,) and v.dtype == PodVertexArray.DTYPE

        # Compute BVH, reordering indices as needed:
        # TODO

        # Get rid of the index buffer: load the vertices for each triangle:
        v = PodVertexArray(v[t_indices])
        assert v.shape == (self.triangle_count, 3) and v.dtype == PodVertexArray.DTYPE

        # Allocate space in renderer heaps, upload data
        self.geometry_heap_offset_in_triangles = renderer._add_geometry(v)


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

    instances: dict[
        tuple[Draw3dGeometry, Draw3dMaterial],
        jt.Float32[np.ndarray, "n 3 4"],
    ] = field(default_factory=dict)

    environment_map: jt.Float32[np.ndarray, "eh ew 3"] | None = None


@dataclass
class Draw3dCamera:
    transform: jt.Float32[np.ndarray, "3 4"]
    fov_y_rad: float
    aspect_ratio: float


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
            ("_rsv", np.uint32),
        ]
    )


class PodCameraArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("transform", np.float32, (3, 4)),  # row-major 3x4 matrix
            ("fov_y_rad", np.float32),
            ("aspect_ratio", np.float32),
            ("_rsv0", np.uint32),
            ("_rsv1", np.uint32),
        ]
    )


class PodInstanceArray(StructuredNDArray):
    DTYPE = np.dtype(
        [
            ("geometry_id", np.uint32),
            ("material_id", np.uint32),
            ("transform", np.float32, (3, 4)),  # row-major 3x4 matrix
        ]
    )


#
# BVH Construction
#

type Aabb = tuple[np.ndarray, np.ndarray]  # (min, max) as float32 arrays of shape (3,)


def aabb_min_max(arr: np.ndarray) -> Aabb:
    """Compute AABB from array of points (N, 3)."""
    return (
        np.min(arr, axis=0).astype(np.float32),
        np.max(arr, axis=0).astype(np.float32),
    )


def aabb_union(*aabbs: Aabb) -> Aabb:
    """Union multiple AABBs."""
    mins = np.stack([a[0] for a in aabbs], axis=0)
    maxs = np.stack([a[1] for a in aabbs], axis=0)
    return (
        np.min(mins, axis=0).astype(np.float32),
        np.max(maxs, axis=0).astype(np.float32),
    )


def aabb_extent_sum(aabb: Aabb) -> float:
    """Sum of extent dimensions."""
    extent = aabb[1] - aabb[0]
    return float(np.sum(extent))


def evaluate_sah(
    centroids: np.ndarray, aabbs: list[Aabb], dim: int, split: float
) -> float:
    """Evaluate SAH cost for a split."""
    mask_lt = centroids[:, dim] < split

    if not np.any(mask_lt) or not np.any(~mask_lt):
        return float("inf")

    lt_aabb = aabb_union(*[aabbs[i] for i in np.where(mask_lt)[0]])
    rt_aabb = aabb_union(*[aabbs[i] for i in np.where(~mask_lt)[0]])

    lt_count = float(np.sum(mask_lt))
    rt_count = float(np.sum(~mask_lt))

    return aabb_extent_sum(lt_aabb) * lt_count + aabb_extent_sum(rt_aabb) * rt_count


@dataclass
class BestPartitionParams:
    dim: int
    pivot: float
    cost: float


def find_best_partition_params(
    centroids: np.ndarray, aabbs: list[Aabb]
) -> BestPartitionParams:
    """Find best split plane."""
    best_dim = 0
    best_val = float("nan")
    best_cost = float("inf")

    for dim in [0, 1, 2]:
        # Test split at each centroid coordinate
        for split_val in np.unique(centroids[:, dim]):
            cost = evaluate_sah(centroids, aabbs, dim, float(split_val))
            if cost < best_cost:
                best_dim = dim
                best_val = float(split_val)
                best_cost = cost

    return BestPartitionParams(best_dim, best_val, best_cost)


@dataclass
class TrianglesPartitionResult:
    lt_indices: np.ndarray  # Indices of left triangles
    lt_aabb: Aabb
    rt_indices: np.ndarray  # Indices of right triangles
    rt_aabb: Aabb


def partition_triangles(
    indices: np.ndarray,
    centroids: np.ndarray,
    aabbs: list[Aabb],
    dim: int,
    pivot: float,
) -> TrianglesPartitionResult:
    """Partition triangles by split plane."""
    mask_lt = centroids[indices, dim] < pivot

    lt_indices = indices[mask_lt]
    rt_indices = indices[~mask_lt]

    lt_aabb = (
        aabb_union(*[aabbs[i] for i in lt_indices])
        if len(lt_indices) > 0
        else (
            np.array([0, 0, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
        )
    )
    rt_aabb = (
        aabb_union(*[aabbs[i] for i in rt_indices])
        if len(rt_indices) > 0
        else (
            np.array([0, 0, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
        )
    )

    return TrianglesPartitionResult(lt_indices, lt_aabb, rt_indices, rt_aabb)


def try_partition_bvh_node(
    root_node_idx: int,
    indices: np.ndarray,
    centroids: np.ndarray,
    aabbs: list[Aabb],
    nodes: list,
) -> None:
    """Recursively partition BVH node."""
    # Assert leaf
    assert (
        nodes[root_node_idx]["children"][0] == 0
        and nodes[root_node_idx]["children"][1] == 0
    )

    if len(indices) <= 1:
        return

    params = find_best_partition_params(centroids[indices], aabbs)

    # SAH cost for current node
    node_aabb = (
        nodes[root_node_idx]["aabb"][0].astype(np.float32),
        nodes[root_node_idx]["aabb"][1].astype(np.float32),
    )
    node_cost = aabb_extent_sum(node_aabb) * len(indices)

    if params.cost >= node_cost:
        return

    res = partition_triangles(indices, centroids, aabbs, params.dim, params.pivot)

    if len(res.lt_indices) == 0 or len(res.rt_indices) == 0:
        return

    lt_index = emplace_bvh_node(nodes, res.lt_indices, res.lt_aabb)
    rt_index = emplace_bvh_node(nodes, res.rt_indices, res.rt_aabb)

    # Update children of root node
    nodes[root_node_idx]["children"][0] = lt_index
    nodes[root_node_idx]["children"][1] = rt_index

    try_partition_bvh_node(lt_index, res.lt_indices, centroids, aabbs, nodes)
    try_partition_bvh_node(rt_index, res.rt_indices, centroids, aabbs, nodes)
