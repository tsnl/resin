use simd_math::SimdUnitQuat;

use super::*;

pub struct Draw3dRenderer {
    device: wgpu::Device,
    target_size_wh_px: [u16; 2],

    renderer_bind_group_layout: wgpu::BindGroupLayout,
    per_frame_bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,

    draw_shader: wgpu::ShaderModule,
    draw_pipeline: wgpu::ComputePipeline,

    geometry_heap_device_buffer: StorageBuffer<PodGeometry>,
    bvh_node_heap_device_buffer: StorageBuffer<PodBvhNode>,
    triangle_heap_device_buffer: StorageBuffer<PodTriangle>,
    renderer_bind_group: wgpu::BindGroup,

    allocated_geometry_count: usize,
    allocated_bvh_node_count: usize,
    allocated_triangle_count: usize,
}
impl Draw3dRenderer {
    const INSTANCE_CAPACITY: usize = 1 << 10;
    const GEOMETRY_CAPACITY: usize = 1 << 8;
    const BVH_NODE_CAPACITY: usize = 1 << 18;
    const TRIANGLE_CAPACITY: usize = 1 << 20;

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_size_wh_px: [u16; 2]) -> Self {
        let device = device.clone();

        let renderer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Draw3dRenderer.RendererBindGroupLayout"),
                entries: &[
                    // geometry heap buffer:
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // bvh node heap buffer:
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // triangle heap buffer:
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let per_frame_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Draw3d3Renderer.PerFrameBindGroupLayout"),
                entries: &[
                    // output image:
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // camera buffer:
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // instance heap buffer:
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Draw3dRenderer.PipelineLayout"),
            bind_group_layouts: &[&renderer_bind_group_layout, &per_frame_bind_group_layout],
            immediate_size: 0,
        });

        let draw_shader = device.create_shader_module(wgpu::include_wgsl!("draw_3d.wgsl"));
        let draw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Draw3dRenderer.DrawPipeline"),
            layout: Some(&pipeline_layout),
            module: &draw_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let geometry_heap_device_buffer = StorageBuffer::<PodGeometry>::new(
            &device,
            Self::GEOMETRY_CAPACITY,
            "Draw3dRenderer.GeometryHeapDeviceBuffer",
        );
        let bvh_node_heap_device_buffer = StorageBuffer::<PodBvhNode>::new(
            &device,
            Self::BVH_NODE_CAPACITY,
            "Draw3dRenderer.BvhNodeHeapDeviceBuffer",
        );
        let triangle_heap_device_buffer = StorageBuffer::<PodTriangle>::new(
            &device,
            Self::TRIANGLE_CAPACITY,
            "Draw3dRenderer.TriangleHeapDeviceBuffer",
        );

        let renderer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Draw3dRenderer.RendererBindGroup"),
            layout: &renderer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: geometry_heap_device_buffer
                        .wgpu_buffer()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bvh_node_heap_device_buffer
                        .wgpu_buffer()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: triangle_heap_device_buffer
                        .wgpu_buffer()
                        .as_entire_binding(),
                },
            ],
        });

        let allocated_geometry_count = 0;
        let allocated_bvh_node_count = 0;
        let allocated_triangle_count = 0;

        // Initialization:
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Draw3dRenderer.InitializationEncoder"),
            });
            geometry_heap_device_buffer.clear(&mut encoder);
            bvh_node_heap_device_buffer.clear(&mut encoder);
            triangle_heap_device_buffer.clear(&mut encoder);
            let sub_index = queue.submit(iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(sub_index),
                    timeout: None,
                })
                .unwrap();
        }

        // Done:
        Self {
            device,
            target_size_wh_px,
            renderer_bind_group_layout,
            per_frame_bind_group_layout,
            pipeline_layout,
            draw_shader,
            draw_pipeline,
            geometry_heap_device_buffer,
            bvh_node_heap_device_buffer,
            triangle_heap_device_buffer,
            renderer_bind_group,
            allocated_geometry_count,
            allocated_bvh_node_count,
            allocated_triangle_count,
        }
    }

    pub fn add_geometry(
        &mut self,
        vertex_buffer: &[Draw3dVertex],
        index_buffer: &[[u32; 3]],
    ) -> Draw3dGeometryHandle {
        // Build triangles:
        let mut triangles = Vec::with_capacity(index_buffer.len() / 3);
        for triangle_indices in index_buffer {
            let vertices = [
                vertex_buffer[triangle_indices[0] as usize].into(),
                vertex_buffer[triangle_indices[1] as usize].into(),
                vertex_buffer[triangle_indices[2] as usize].into(),
            ];
            triangles.push(PodTriangle::new(vertices));
        }

        // Build BVH:
        let mut bvh_nodes = Vec::new();
        let root_span = 0..triangles.len();
        emplace_bvh_node(&mut bvh_nodes, root_span.clone(), {
            let mut aabb = SimdRect3::union_identity();
            for triangle in &triangles {
                aabb |= triangle.aabb();
            }
            aabb
        });
        try_partition_bvh_node(0, &mut triangles, &mut bvh_nodes);

        // Allocate space in the heaps and upload data to GPU buffers:
        todo!();
    }

    pub fn record(
        &self,
        scene: &Draw3dScene,
        frame: &mut Draw3dFrame,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        frame.record(command_encoder, scene);
    }
}

pub struct Draw3dFrame {
    device: wgpu::Device,

    output_image: Rgba32FloatTexture,
    camera_device_buffer: UniformBuffer<PodCamera>,
    camera_staging_buffer: StagingBuffer<PodCamera>,
    instances_meta_device_buffer: UniformBuffer<PodInstancesMeta>,
    instances_meta_staging_buffer: StagingBuffer<PodInstancesMeta>,
    instances_list_device_buffer: StorageBuffer<PodInstance>,
    instances_list_staging_buffer: StagingBuffer<PodInstance>,
}
impl Draw3dFrame {
    pub fn new(renderer: &Draw3dRenderer) -> Self {
        let device = renderer.device.clone();

        let output_image = Rgba32FloatTexture::new(
            &device,
            renderer.target_size_wh_px,
            "Draw3dFrame.OutputImage",
        );
        let camera_device_buffer = UniformBuffer::new(&device, 1, "Draw3dFrame.CameraDeviceBuffer");
        let camera_staging_buffer =
            StagingBuffer::new(&device, 1, "Draw3dFrame.CameraStagingBuffer");
        let instances_meta_device_buffer =
            UniformBuffer::new(&device, 1, "Draw3dFrame.InstancesMetaDeviceBuffer");
        let instances_meta_staging_buffer =
            StagingBuffer::new(&device, 1, "Draw3dFrame.InstancesMetaStagingBuffer");
        let instances_list_device_buffer = StorageBuffer::new(
            &device,
            Draw3dRenderer::INSTANCE_CAPACITY,
            "Draw3dFrame.InstanceHeapDeviceBuffer",
        );
        let instances_list_staging_buffer = StagingBuffer::new(
            &device,
            Draw3dRenderer::INSTANCE_CAPACITY,
            "Draw3dFrame.InstanceHeapStagingBuffer",
        );

        Self {
            device,
            output_image,
            camera_device_buffer,
            camera_staging_buffer,
            instances_meta_device_buffer,
            instances_meta_staging_buffer,
            instances_list_device_buffer,
            instances_list_staging_buffer,
        }
    }
    pub fn output_image(&self) -> &Rgba32FloatTexture {
        &self.output_image
    }
    fn record(&self, command_encoder: &mut wgpu::CommandEncoder, scene: &Draw3dScene) {
        todo!();
    }
}

#[derive(Default)]
pub struct Draw3dScene {
    meshes: HashMap<Draw3dMeshHandle, Vec<SimdTransform>>,
    environment_map: Option<Rgba32FloatTexture>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Draw3dMeshHandle(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Draw3dMaterialHandle(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Draw3dGeometryHandle(usize);

#[derive(Clone, Copy, Debug)]
pub struct Draw3dVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

//
// Camera
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodCamera {
    transform: PodTransform,
    fov_y_rad: f32,
    aspect_ratio: f32,
    target_size_w_px: u32,
    target_size_h_px: u32,
}

//
// Instances:
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodInstancesMeta {
    count: u32,
    _rsv: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodInstance {
    geometry_id: u32,
    material_id: u32,
    transform: PodTransform,
}

//
// Geometry
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodGeometry {
    // BVH nodes and triangles are stored in contiguous arrays within a large, contiguous heap
    // buffer. BVH and triangle indices are offsets within these smaller arrays, not the heaps.
    bvh_node_span_in_heap: PodSpan,
    triangle_span_in_heap: PodSpan,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodBvhNode {
    span: PodSpan,
    children: [u32; 2],
    aabb: [[f32; 3]; 2],
}
impl PodBvhNode {
    fn aabb(&self) -> SimdRect3 {
        SimdRect3::new(SimdVec3::from(self.aabb[0]), SimdVec3::from(self.aabb[1]))
    }
    fn sah_cost(&self) -> f32 {
        let area = self.aabb().extent().reduce_sum();
        let tri_count = self.span.len() as f32;
        area * tri_count
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodTriangle {
    vertices: [PodVertex; 3],
    centroid: [f32; 3],
}
impl PodTriangle {
    fn new(vertices: [PodVertex; 3]) -> Self {
        let centroid = {
            let mut acc = SimdVec3::ZERO;
            for v in &vertices {
                acc += SimdVec3::from(v.position);
            }
            let res = acc / SimdVec3::splat(3.0);
            res.into()
        };
        Self { vertices, centroid }
    }
    fn aabb(&self) -> SimdRect3 {
        let mut aabb = SimdRect3::union_identity();
        for v in &self.vertices {
            aabb |= SimdVec3::from(v.position);
        }
        aabb
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}
impl From<Draw3dVertex> for PodVertex {
    fn from(v: Draw3dVertex) -> Self {
        Self {
            position: v.position,
            normal: v.normal,
            uv: v.uv,
        }
    }
}

fn try_partition_bvh_node(
    root_node: usize,
    triangles: &mut [PodTriangle],
    nodes: &mut Vec<PodBvhNode>,
) {
    // Assert that the node is currently a leaf node.
    debug_assert!(nodes[root_node].children == [0, 0]);

    // Find the best partition
    let BestPartitionParams {
        dim: best_dim,
        pivot: best_val,
        cost: best_cost,
    } = find_best_partition_params(triangles, &nodes[root_node]);

    // If the best cost is not better than the current node's cost, do not partition further.
    if best_cost >= nodes[root_node].sah_cost() {
        return;
    }

    // Partition the triangles based on the best split into left and right subspans.
    let TrianglesPartitionResult {
        lt_span,
        lt_aabb,
        rt_span,
        rt_aabb,
    } = partition_triangles_in_place(triangles, nodes[root_node].span.into(), best_dim, best_val);

    // Emplace two new child nodes.
    let lt_index = emplace_bvh_node(nodes, lt_span, lt_aabb);
    let rt_index = emplace_bvh_node(nodes, rt_span, rt_aabb);

    // Recurse on the two new child nodes.
    {
        try_partition_bvh_node(lt_index, triangles, nodes);
        try_partition_bvh_node(rt_index, triangles, nodes);
    }
}

/// Searches for the best split position along a given dimension and evaluates the SAH cost.
fn find_best_partition_params(triangles: &[PodTriangle], node: &PodBvhNode) -> BestPartitionParams {
    let mut best_dim = 0;
    let mut best_val = f32::NAN;
    let mut best_cost = f32::INFINITY;

    for dim in [0, 1, 2] {
        for i in node.span.into_range() {
            let val = triangles[i].centroid[dim];
            let cost = evaluate_sah(triangles, dim, val);
            if cost < best_cost {
                best_dim = dim;
                best_val = val;
                best_cost = cost;
            }
        }
    }

    BestPartitionParams {
        dim: best_dim,
        pivot: best_val,
        cost: best_cost,
    }
}
struct BestPartitionParams {
    dim: usize,
    pivot: f32,
    cost: f32,
}

/// Calculates the surface area heuristic cost for a given AABB and triangle count.
/// * https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
/// * https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#TheSurfaceAreaHeuristic
fn evaluate_sah(triangles: &[PodTriangle], dim: usize, split: f32) -> f32 {
    let mut lt_aabb = SimdRect3::union_identity();
    let mut lt_count = 0.0;
    let mut rt_aabb = SimdRect3::union_identity();
    let mut rt_count = 0.0;

    for triangle in triangles.iter() {
        let c = triangle.centroid[dim];
        if c < split {
            // left
            lt_aabb |= triangle.aabb();
            lt_count += 1.0;
        } else {
            // right
            rt_aabb |= triangle.aabb();
            rt_count += 1.0;
        }
    }

    if lt_count == 0.0 || rt_count == 0.0 {
        return f32::MAX;
    }

    let lt_cost = lt_aabb.extent().reduce_sum() * lt_count;
    let rt_cost = rt_aabb.extent().reduce_sum() * rt_count;
    lt_cost + rt_cost
}

/// Partitions a span of triangles into two consecutive sub-spans in place based on a pivot value and a pivot dimension.
fn partition_triangles_in_place(
    triangles: &mut [PodTriangle],
    span: Range<usize>,
    dim: usize,
    pivot: f32,
) -> TrianglesPartitionResult {
    let mut lt_count = 0;
    let mut lt_aabb = SimdRect3::union_identity();
    let mut rt_count = 0;
    let mut rt_aabb = SimdRect3::union_identity();
    let total_count = span.len();
    while lt_count + rt_count < total_count {
        let o = span.start + lt_count;
        let d = triangles[o].centroid[dim];
        if d < pivot {
            lt_aabb |= triangles[o].aabb();
            lt_count += 1;
        } else {
            rt_aabb |= triangles[o].aabb();
            triangles.swap(o, span.start + total_count - 1 - rt_count);
            rt_count += 1;
        }
    }
    let lt_span = (span.start)..(span.start + lt_count);
    let rt_span = (lt_span.end)..(lt_span.end + rt_count);
    TrianglesPartitionResult {
        lt_span,
        lt_aabb,
        rt_span,
        rt_aabb,
    }
}
struct TrianglesPartitionResult {
    lt_span: Range<usize>,
    lt_aabb: SimdRect3,
    rt_span: Range<usize>,
    rt_aabb: SimdRect3,
}

/// Appends a new node to the BVH nodes list, returning the index of the newly added node.
fn emplace_bvh_node(nodes: &mut Vec<PodBvhNode>, span: Range<usize>, aabb: SimdRect3) -> usize {
    let index = nodes.len();
    let span = span.into();
    let aabb = [aabb.min().into(), aabb.max().into()];
    nodes.push(PodBvhNode {
        span,
        aabb,
        children: [0, 0],
    });
    index
}

//
// Common
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodSpan {
    begin: u32,
    end: u32,
}
impl PodSpan {
    fn len(&self) -> u32 {
        self.end - self.begin
    }
    fn into_range(&self) -> Range<usize> {
        self.begin as usize..self.end as usize
    }
}
impl Into<Range<usize>> for PodSpan {
    fn into(self) -> Range<usize> {
        self.into_range()
    }
}
impl From<Range<usize>> for PodSpan {
    fn from(range: Range<usize>) -> Self {
        Self {
            begin: range.start as u32,
            end: range.end as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodAabb {
    min: [f32; 3],
    max: [f32; 3],
}
impl Into<SimdRect3> for PodAabb {
    fn into(self) -> SimdRect3 {
        SimdRect3::new(SimdVec3::from(self.min), SimdVec3::from(self.max))
    }
}
impl From<SimdRect3> for PodAabb {
    fn from(aabb: SimdRect3) -> Self {
        Self {
            min: aabb.min().into(),
            max: aabb.max().into(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodTransform {
    position: [f32; 3],
    _rsv: u32,
    rotation: [f32; 4],
}
impl From<SimdTransform> for PodTransform {
    fn from(t: SimdTransform) -> Self {
        Self {
            position: t.position().into(),
            _rsv: 0,
            rotation: t.rotation().into(),
        }
    }
}
impl Into<SimdTransform> for PodTransform {
    fn into(self) -> SimdTransform {
        SimdTransform::new(
            SimdVec3::from(self.position),
            SimdUnitQuat::from(self.rotation),
        )
    }
}
