__all__ = [
    "Draw3dFrame",
    "Draw3dRenderer",
    "Draw3dScene",
    "Draw3dVertex",
]

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import wgpu

#
# Renderer
#


class Draw3dRenderer:
    INSTANCE_CAPACITY = 1 << 10
    GEOMETRY_CAPACITY = 1 << 8
    BVH_NODE_CAPACITY = 1 << 18
    TRIANGLE_CAPACITY = 1 << 20

    def __init__(
        self,
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
        target_size_wh_px: tuple[int, int],
    ):
        self.device = device
        self.target_size_wh_px = target_size_wh_px

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
            size=self.GEOMETRY_CAPACITY * POD_GEOMETRY_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.bvh_node_heap_device_buffer = device.create_buffer(
            label="Draw3dRenderer.BvhNodeHeapDeviceBuffer",
            size=self.BVH_NODE_CAPACITY * POD_BVH_NODE_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.triangle_heap_device_buffer = device.create_buffer(
            label="Draw3dRenderer.TriangleHeapDeviceBuffer",
            size=self.TRIANGLE_CAPACITY * POD_TRIANGLE_NODE_DTYPE.itemsize,
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

    def add_geometry(
        self,
        vertex_buffer: list[Draw3dVertex],
        index_buffer: list[tuple[int, int, int]],
    ) -> int:
        """Add geometry to renderer and build BVH."""
        # Create triangles
        triangles = []
        for triangle_indices in index_buffer:
            vertices = [
                vertex_buffer[triangle_indices[0]].to_pod(),
                vertex_buffer[triangle_indices[1]].to_pod(),
                vertex_buffer[triangle_indices[2]].to_pod(),
            ]
            triangles.append(create_pod_triangle(vertices))

        # Collect centroids and compute AABBs for each triangle
        centroids = np.zeros((len(triangles), 3), dtype=np.float32)
        aabbs: list[Aabb] = []
        for i, tri in enumerate(triangles):
            centroids[i] = tri["centroid"].astype(np.float32)
            # Compute AABB for this triangle
            positions = np.array(
                [v["position"].astype(np.float32) for v in tri["vertices"]],
                dtype=np.float32,
            )
            aabbs.append(aabb_min_max(positions))

        # Build BVH
        bvh_nodes: list["POD_BVH_NODE_DTYPE"] = []
        indices = np.arange(len(triangles), dtype=np.uint32)
        root_aabb = aabb_union(*aabbs)
        emplace_bvh_node(bvh_nodes, indices, root_aabb)
        try_partition_bvh_node(0, indices, centroids, aabbs, bvh_nodes)

        # TODO: Allocate space in heaps and upload
        raise NotImplementedError("TODO")

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
    def __init__(self, renderer: Draw3dRenderer) -> None:
        self.device = renderer.device

        self.output_image = self.device.create_texture(
            label="Draw3dFrame.OutputImage",
            size=(renderer.target_size_wh_px[0], renderer.target_size_wh_px[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
        )
        self.frame_info_device_buffer = self.device.create_buffer(
            label="Draw3dFrame.FrameInfoDeviceBuffer",
            size=POD_FRAME_INFO_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.frame_info_staging_buffer = self.device.create_buffer(
            label="Draw3dFrame.FrameInfoStagingBuffer",
            size=POD_FRAME_INFO_DTYPE.itemsize,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )
        self.camera_device_buffer = self.device.create_buffer(
            label="Draw3dFrame.CameraDeviceBuffer",
            size=POD_CAMERA_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.camera_staging_buffer = self.device.create_buffer(
            label="Draw3dFrame.CameraStagingBuffer",
            size=POD_CAMERA_DTYPE.itemsize,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )
        self.instances_list_device_buffer = self.device.create_buffer(
            label="Draw3dFrame.InstanceHeapDeviceBuffer",
            size=Draw3dRenderer.INSTANCE_CAPACITY * POD_INSTANCE_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.instances_list_staging_buffer = self.device.create_buffer(
            label="Draw3dFrame.InstanceHeapStagingBuffer",
            size=Draw3dRenderer.INSTANCE_CAPACITY * POD_INSTANCE_DTYPE.itemsize,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )

        self.bind_group = self.device.create_bind_group(
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
        frame_info = np.array(
            (0, target_size_wh[0], target_size_wh[1], 0),
            dtype=POD_FRAME_INFO_DTYPE,
        )

        self.frame_info_staging_buffer.map_sync(wgpu.MapMode.WRITE)
        self.frame_info_staging_buffer.write_mapped(
            data=np.array([frame_info], dtype=POD_FRAME_INFO_DTYPE)
        )
        self.frame_info_staging_buffer.unmap()

        command_encoder.copy_buffer_to_buffer(
            source=self.frame_info_staging_buffer,
            source_offset=0,
            destination_offset=0,
            destination=self.frame_info_device_buffer,
            size=frame_info.nbytes,
        )

        # TODO: Marshall and write camera data, instances, etc
        _ = scene

        compute_pass = command_encoder.begin_compute_pass(
            label="Draw3dFrame.ComputePass"
        )
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, renderer_bind_group, [], 0, 0)
        compute_pass.set_bind_group(1, self.bind_group, [], 0, 0)
        compute_pass.dispatch_workgroups(
            math.ceil(target_size_wh[0] / 8), math.ceil(target_size_wh[1] / 8), 1
        )
        compute_pass.end()


#
# POD Types (NumPy structured dtypes)
#

POD_SPAN_DTYPE = np.dtype([("begin", np.uint32), ("end", np.uint32)])

POD_AABB_DTYPE = np.dtype([("min", np.float32, (3,)), ("max", np.float32, (3,))])

POD_TRANSFORM_DTYPE = np.dtype([("matrix", np.float32, (3, 4))])

POD_VERTEX_DTYPE = np.dtype(
    [
        ("position", np.float32, (3,)),
        ("normal", np.float32, (3,)),
        ("uv", np.float32, (2,)),
    ]
)

POD_TRIANGLE_NODE_DTYPE = np.dtype(
    [
        ("vertices", POD_VERTEX_DTYPE, (3,)),
        ("centroid", np.float32, (3,)),
    ]
)

POD_BVH_NODE_DTYPE = np.dtype(
    [
        ("span", POD_SPAN_DTYPE),
        ("children", np.uint32, (2,)),
        ("aabb", np.float32, (2, 3)),
    ]
)

POD_GEOMETRY_DTYPE = np.dtype(
    [
        ("bvh_node_span_in_heap", POD_SPAN_DTYPE),
        ("triangle_span_in_heap", POD_SPAN_DTYPE),
    ]
)

POD_FRAME_INFO_DTYPE = np.dtype(
    [
        ("count", np.uint32),
        ("target_size_w_px", np.uint32),
        ("target_size_h_px", np.uint32),
        ("_rsv", np.uint32),
    ]
)

POD_CAMERA_DTYPE = np.dtype(
    [
        ("transform", POD_TRANSFORM_DTYPE),
        ("fov_y_rad", np.float32),
        ("aspect_ratio", np.float32),
        ("target_size_w_px", np.uint32),
        ("target_size_h_px", np.uint32),
    ]
)

POD_INSTANCE_DTYPE = np.dtype(
    [
        ("geometry_id", np.uint32),
        ("material_id", np.uint32),
        ("transform", POD_TRANSFORM_DTYPE),
    ]
)


#
# High Level Types
#


@dataclass
class Draw3dVertex:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    uv: tuple[float, float]

    def to_pod(self) -> np.ndarray:
        record = np.array(
            (
                np.array(self.position, dtype=np.float32),
                np.array(self.normal, dtype=np.float32),
                np.array(self.uv, dtype=np.float32),
            ),
            dtype=POD_VERTEX_DTYPE,
        )
        return record


@dataclass
class Draw3dScene:
    # meshes: Dict[mesh_handle] -> List of transforms as (N, 3, 4) arrays
    meshes: Dict[int, np.ndarray] = field(default_factory=dict)
    environment_map: Optional[wgpu.GPUTexture] = None


#
# Helper functions for creating Pod records
#


def create_pod_triangle(vertices: list[np.ndarray]) -> np.ndarray:
    """Create PodTriangle from vertices and compute centroid."""
    centroid = np.zeros(3, dtype=np.float32)
    for v in vertices:
        centroid += v["position"]
    centroid /= 3.0

    vertices_array = np.array([v for v in vertices], dtype=POD_VERTEX_DTYPE)
    record = np.array(
        (vertices_array, centroid),
        dtype=POD_TRIANGLE_NODE_DTYPE,
    )
    return record


def create_pod_bvh_node(
    begin: int, end: int, aabb_min: np.ndarray, aabb_max: np.ndarray
) -> np.ndarray:
    """Create PodBvhNode record."""
    record = np.array(
        (
            np.array((begin, end), dtype=POD_SPAN_DTYPE),
            np.array([0, 0], dtype=np.uint32),
            np.array([aabb_min, aabb_max], dtype=np.float32),
        ),
        dtype=POD_BVH_NODE_DTYPE,
    )
    return record


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


def emplace_bvh_node(nodes: list, indices: np.ndarray, aabb: Aabb) -> int:
    """Add a BVH node."""
    index = len(nodes)

    node = create_pod_bvh_node(0, len(indices), aabb[0], aabb[1])
    nodes.append(node)
    return index


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
