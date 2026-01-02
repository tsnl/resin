import wgpu
import ctypes
import math
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from .gpu_util import (
    Rgba32FloatTexture,
    StorageBuffer,
    UniformBuffer,
    StagingBuffer,
    TextureWrapper,
)

#
# Math Helpers
#


class SimdVec3:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)

    @staticmethod
    def splat(v):
        return SimdVec3([v, v, v])

    @staticmethod
    def zero():
        return SimdVec3([0.0, 0.0, 0.0])

    def __add__(self, other):
        return SimdVec3(self.data + other.data)

    def __truediv__(self, other):
        return SimdVec3(self.data / other.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimdRect3:
    def __init__(self, min_val, max_val):
        self.min_val = np.array(min_val, dtype=np.float32)
        self.max_val = np.array(max_val, dtype=np.float32)

    @staticmethod
    def union_identity():
        return SimdRect3([float("inf")] * 3, [float("-inf")] * 3)

    def union(self, other):
        if isinstance(other, SimdRect3):
            return SimdRect3(
                np.minimum(self.min_val, other.min_val),
                np.maximum(self.max_val, other.max_val),
            )
        elif isinstance(other, SimdVec3):
            return SimdRect3(
                np.minimum(self.min_val, other.data),
                np.maximum(self.max_val, other.data),
            )
        return self

    def __or__(self, other):
        return self.union(other)

    def extent(self):
        return SimdVec3(self.max_val - self.min_val)

    def reduce_sum(self):
        # This seems to be what extent().reduce_sum() does in Rust (sum of components)
        # Wait, extent returns a vector. reduce_sum on vector sums components.
        # But for AABB surface area, it's usually 2 * (w*h + h*d + d*w).
        # The Rust code says `area = self.aabb().extent().reduce_sum()`.
        # If `reduce_sum` just sums x+y+z, that's not area.
        # But let's follow the Rust code's method name.
        # simd_math::SimdVec3::reduce_sum sums the components.
        # So I will do the same.
        e = self.max_val - self.min_val
        return np.sum(e)


class SimdTransform:
    def __init__(self, position, rotation):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)  # Quat


#
# POD Types
#


class PodSpan(ctypes.Structure):
    _fields_ = [("begin", ctypes.c_uint32), ("end", ctypes.c_uint32)]

    def __len__(self):
        return self.end - self.begin


class PodAabb(ctypes.Structure):
    _fields_ = [("min", ctypes.c_float * 3), ("max", ctypes.c_float * 3)]


class PodTransform(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("_rsv", ctypes.c_uint32),
        ("rotation", ctypes.c_float * 4),
    ]


class PodBvhNode(ctypes.Structure):
    _fields_ = [
        ("span", PodSpan),
        ("children", ctypes.c_uint32 * 2),
        ("aabb", (ctypes.c_float * 3) * 2),
    ]

    def get_aabb(self) -> SimdRect3:
        return SimdRect3(self.aabb[0], self.aabb[1])

    def sah_cost(self) -> float:
        area = self.get_aabb().reduce_sum()  # Following Rust code
        tri_count = float(len(self.span))
        return area * tri_count


class PodVertex(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("normal", ctypes.c_float * 3),
        ("uv", ctypes.c_float * 2),
    ]


class PodTriangle(ctypes.Structure):
    _fields_ = [
        ("vertices", PodVertex * 3),
        ("centroid", ctypes.c_float * 3),
    ]

    @staticmethod
    def new(vertices):
        # vertices is list of PodVertex
        acc = np.zeros(3, dtype=np.float32)
        for v in vertices:
            acc += np.array(v.position, dtype=np.float32)
        centroid = acc / 3.0

        t = PodTriangle()
        t.vertices = (PodVertex * 3)(*vertices)
        t.centroid = (ctypes.c_float * 3)(*centroid)
        return t

    def get_aabb(self) -> SimdRect3:
        aabb = SimdRect3.union_identity()
        for v in self.vertices:
            aabb = aabb | SimdVec3(v.position)
        return aabb


class PodGeometry(ctypes.Structure):
    _fields_ = [
        ("bvh_node_span_in_heap", PodSpan),
        ("triangle_span_in_heap", PodSpan),
    ]


class PodFrameInfo(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("target_size_w_px", ctypes.c_uint32),
        ("target_size_h_px", ctypes.c_uint32),
        ("_rsv", ctypes.c_uint32),
    ]


class PodCamera(ctypes.Structure):
    _fields_ = [
        ("transform", PodTransform),
        ("fov_y_rad", ctypes.c_float),
        ("aspect_ratio", ctypes.c_float),
        ("target_size_w_px", ctypes.c_uint32),
        ("target_size_h_px", ctypes.c_uint32),
    ]


class PodInstance(ctypes.Structure):
    _fields_ = [
        ("geometry_id", ctypes.c_uint32),
        ("material_id", ctypes.c_uint32),
        ("transform", PodTransform),
    ]


#
# High Level Types
#


@dataclass
class Draw3dVertex:
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    uv: Tuple[float, float]

    def to_pod(self) -> PodVertex:
        return PodVertex(
            position=(ctypes.c_float * 3)(*self.position),
            normal=(ctypes.c_float * 3)(*self.normal),
            uv=(ctypes.c_float * 2)(*self.uv),
        )


@dataclass
class Draw3dScene:
    meshes: Dict[int, List[SimdTransform]] = field(default_factory=dict)
    environment_map: Optional[Rgba32FloatTexture] = None


#
# BVH Construction
#


def evaluate_sah(triangles: List[PodTriangle], dim: int, split: float) -> float:
    lt_aabb = SimdRect3.union_identity()
    lt_count = 0.0
    rt_aabb = SimdRect3.union_identity()
    rt_count = 0.0

    for triangle in triangles:
        c = triangle.centroid[dim]
        if c < split:
            lt_aabb = lt_aabb | triangle.get_aabb()
            lt_count += 1.0
        else:
            rt_aabb = rt_aabb | triangle.get_aabb()
            rt_count += 1.0

    if lt_count == 0.0 or rt_count == 0.0:
        return float("inf")  # f32::MAX

    lt_cost = lt_aabb.reduce_sum() * lt_count
    rt_cost = rt_aabb.reduce_sum() * rt_count
    return lt_cost + rt_cost


@dataclass
class BestPartitionParams:
    dim: int
    pivot: float
    cost: float


def find_best_partition_params(
    triangles: List[PodTriangle], node: PodBvhNode
) -> BestPartitionParams:
    best_dim = 0
    best_val = float("nan")
    best_cost = float("inf")

    span_start = node.span.begin
    span_end = node.span.end

    for dim in [0, 1, 2]:
        for i in range(span_start, span_end):
            val = triangles[i].centroid[dim]
            cost = evaluate_sah(triangles[span_start:span_end], dim, val)
            if cost < best_cost:
                best_dim = dim
                best_val = val
                best_cost = cost

    return BestPartitionParams(best_dim, best_val, best_cost)


@dataclass
class TrianglesPartitionResult:
    lt_span: range
    lt_aabb: SimdRect3
    rt_span: range
    rt_aabb: SimdRect3


def partition_triangles_in_place(
    triangles: List[PodTriangle], span: range, dim: int, pivot: float
) -> TrianglesPartitionResult:
    lt_count = 0
    lt_aabb = SimdRect3.union_identity()
    rt_count = 0
    rt_aabb = SimdRect3.union_identity()
    total_count = len(span)

    # We need to modify triangles list in place.
    # We can simulate the swap logic.

    # Working on the slice of the list
    # But we need to swap elements in the main list.

    start = span.start

    while lt_count + rt_count < total_count:
        o = start + lt_count
        d = triangles[o].centroid[dim]
        if d < pivot:
            lt_aabb = lt_aabb | triangles[o].get_aabb()
            lt_count += 1
        else:
            rt_aabb = rt_aabb | triangles[o].get_aabb()
            # swap
            idx1 = o
            idx2 = start + total_count - 1 - rt_count
            triangles[idx1], triangles[idx2] = triangles[idx2], triangles[idx1]
            rt_count += 1

    lt_span = range(start, start + lt_count)
    rt_span = range(lt_span.stop, lt_span.stop + rt_count)

    return TrianglesPartitionResult(lt_span, lt_aabb, rt_span, rt_aabb)


def emplace_bvh_node(nodes: List[PodBvhNode], span: range, aabb: SimdRect3) -> int:
    index = len(nodes)

    node = PodBvhNode()
    node.span.begin = span.start
    node.span.end = span.stop
    node.children = (ctypes.c_uint32 * 2)(0, 0)
    node.aabb[0] = (ctypes.c_float * 3)(*aabb.min_val)
    node.aabb[1] = (ctypes.c_float * 3)(*aabb.max_val)

    nodes.append(node)
    return index


def try_partition_bvh_node(
    root_node_idx: int, triangles: List[PodTriangle], nodes: List[PodBvhNode]
):
    # Assert leaf
    assert (
        nodes[root_node_idx].children[0] == 0 and nodes[root_node_idx].children[1] == 0
    )

    params = find_best_partition_params(triangles, nodes[root_node_idx])

    if params.cost >= nodes[root_node_idx].sah_cost():
        return

    span = range(nodes[root_node_idx].span.begin, nodes[root_node_idx].span.end)
    res = partition_triangles_in_place(triangles, span, params.dim, params.pivot)

    lt_index = emplace_bvh_node(nodes, res.lt_span, res.lt_aabb)
    rt_index = emplace_bvh_node(nodes, res.rt_span, res.rt_aabb)

    # Update children of root node
    nodes[root_node_idx].children[0] = lt_index
    nodes[root_node_idx].children[1] = rt_index

    try_partition_bvh_node(lt_index, triangles, nodes)
    try_partition_bvh_node(rt_index, triangles, nodes)


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
        target_size_wh_px: Tuple[int, int],
    ):
        self.device = device
        self.target_size_wh_px = target_size_wh_px

        self.renderer_bind_group_layout = device.create_bind_group_layout(
            label="Draw3dRenderer.RendererBindGroupLayout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
            ],
        )

        self.per_frame_bind_group_layout = device.create_bind_group_layout(
            label="Draw3dRenderer.PerFrameBindGroupLayout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "storage_texture": {
                        "access": wgpu.StorageTextureAccess.write_only,
                        "format": wgpu.TextureFormat.rgba32float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
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
            compute={"module": draw_shader, "entry_point": "main"},
        )

        self.geometry_heap_device_buffer = StorageBuffer(
            device,
            self.GEOMETRY_CAPACITY,
            "Draw3dRenderer.GeometryHeapDeviceBuffer",
            PodGeometry,
        )
        self.bvh_node_heap_device_buffer = StorageBuffer(
            device,
            self.BVH_NODE_CAPACITY,
            "Draw3dRenderer.BvhNodeHeapDeviceBuffer",
            PodBvhNode,
        )
        self.triangle_heap_device_buffer = StorageBuffer(
            device,
            self.TRIANGLE_CAPACITY,
            "Draw3dRenderer.TriangleHeapDeviceBuffer",
            PodTriangle,
        )

        self.renderer_bind_group = device.create_bind_group(
            label="Draw3dRenderer.RendererBindGroup",
            layout=self.renderer_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.geometry_heap_device_buffer.wgpu_buffer(),
                        "offset": 0,
                        "size": self.geometry_heap_device_buffer.size_in_bytes,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.bvh_node_heap_device_buffer.wgpu_buffer(),
                        "offset": 0,
                        "size": self.bvh_node_heap_device_buffer.size_in_bytes,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.triangle_heap_device_buffer.wgpu_buffer(),
                        "offset": 0,
                        "size": self.triangle_heap_device_buffer.size_in_bytes,
                    },
                },
            ],
        )

        self.allocated_geometry_count = 0
        self.allocated_bvh_node_count = 0
        self.allocated_triangle_count = 0

        # Initialization
        encoder = device.create_command_encoder(
            label="Draw3dRenderer.InitializationEncoder"
        )
        self.geometry_heap_device_buffer.clear(encoder)
        self.bvh_node_heap_device_buffer.clear(encoder)
        self.triangle_heap_device_buffer.clear(encoder)
        queue.submit([encoder.finish()])

    def add_geometry(
        self,
        vertex_buffer: List[Draw3dVertex],
        index_buffer: List[Tuple[int, int, int]],
    ) -> int:
        triangles = []
        for triangle_indices in index_buffer:
            vertices = [
                vertex_buffer[triangle_indices[0]].to_pod(),
                vertex_buffer[triangle_indices[1]].to_pod(),
                vertex_buffer[triangle_indices[2]].to_pod(),
            ]
            triangles.append(PodTriangle.new(vertices))

        bvh_nodes: List[PodBvhNode] = []
        root_span = range(0, len(triangles))

        aabb = SimdRect3.union_identity()
        for triangle in triangles:
            aabb = aabb | triangle.get_aabb()

        emplace_bvh_node(bvh_nodes, root_span, aabb)
        try_partition_bvh_node(0, triangles, bvh_nodes)

        # TODO: Allocate space in heaps and upload
        raise NotImplementedError("TODO")

    def record(
        self,
        scene: Draw3dScene,
        frame: "Draw3dFrame",
        command_encoder: wgpu.GPUCommandEncoder,
    ):
        frame.record(
            self.pipeline,
            self.renderer_bind_group,
            self.target_size_wh_px,
            command_encoder,
            scene,
        )


class Draw3dFrame:
    def __init__(self, renderer: Draw3dRenderer):
        self.device = renderer.device

        self.output_image = Rgba32FloatTexture(
            self.device, renderer.target_size_wh_px, "Draw3dFrame.OutputImage"
        )
        self.frame_info_device_buffer = UniformBuffer(
            self.device, 1, "Draw3dFrame.FrameInfoDeviceBuffer", PodFrameInfo
        )
        self.frame_info_staging_buffer = StagingBuffer(
            self.device, 1, "Draw3dFrame.FrameInfoStagingBuffer", PodFrameInfo
        )
        self.camera_device_buffer = UniformBuffer(
            self.device, 1, "Draw3dFrame.CameraDeviceBuffer", PodCamera
        )
        self.camera_staging_buffer = StagingBuffer(
            self.device, 1, "Draw3dFrame.CameraStagingBuffer", PodCamera
        )
        self.instances_list_device_buffer = StorageBuffer(
            self.device,
            Draw3dRenderer.INSTANCE_CAPACITY,
            "Draw3dFrame.InstanceHeapDeviceBuffer",
            PodInstance,
        )
        self.instances_list_staging_buffer = StagingBuffer(
            self.device,
            Draw3dRenderer.INSTANCE_CAPACITY,
            "Draw3dFrame.InstanceHeapStagingBuffer",
            PodInstance,
        )

        self.bind_group = self.device.create_bind_group(
            label="Draw3dFrame.BindGroup",
            layout=renderer.per_frame_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": self.output_image.wgpu_texture().create_view(),
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.frame_info_device_buffer.wgpu_buffer(),
                        "offset": 0,
                        "size": self.frame_info_device_buffer.size_in_bytes,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.camera_device_buffer.wgpu_buffer(),
                        "offset": 0,
                        "size": self.camera_device_buffer.size_in_bytes,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self.instances_list_device_buffer.wgpu_buffer(),
                        "offset": 0,
                        "size": self.instances_list_device_buffer.size_in_bytes,
                    },
                },
            ],
        )

    def get_output_image(self) -> Rgba32FloatTexture:
        return self.output_image

    def record(
        self,
        pipeline: wgpu.GPUComputePipeline,
        renderer_bind_group: wgpu.GPUBindGroup,
        target_size_wh: Tuple[int, int],
        command_encoder: wgpu.GPUCommandEncoder,
        scene: Draw3dScene,
    ):
        frame_info = PodFrameInfo(
            count=0,  # TODO
            target_size_w_px=target_size_wh[0],
            target_size_h_px=target_size_wh[1],
        )

        self.frame_info_staging_buffer.write([frame_info])
        self.frame_info_staging_buffer.copy_to_buffer(
            self.frame_info_device_buffer, command_encoder
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
