//! Geometry resources contain 3D triangle meshes without any specific material applied.

use crate::{
    res::generic::{GenericBackend, GenericManager, GenericResource},
    util::{RangeAllocationError, RangeAllocator, StructuredBuffer},
};
use std::ops::Range;

//
// API:
//

pub type GeometryManager = GenericManager<GeometryBackend>;
pub type Geometry = GenericResource<GeometryBackend>;

pub struct GeometryManagerArgs {
    pub resource_capacity: usize,
    pub vertex_capacity: usize,
    pub index_capacity: usize,
}

pub struct GeometryArgs<'a> {
    pub vertex_position_data: &'a [[f32; 3]],
    pub vertex_normal_data: &'a [[f32; 3]],
    pub vertex_texcoord0_data: &'a [[f32; 2]],
    pub index_data: &'a [u32],
    pub queue: &'a wgpu::Queue,
}

pub struct GeometryInfo {
    pub id: usize,
    pub vertex_range: Range<usize>,
    pub index_range: Range<usize>,
}

#[repr(C)]
#[derive(Default, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GeometryDrawArgs {
    pub first_index: u32,
    pub index_count: u32,
    pub base_vertex: u32,
}

//
// Implementation
//

struct GeometryBackend {
    id_allocator: RangeAllocator,
    vertex_allocator: RangeAllocator,
    index_allocator: RangeAllocator,
    vertex_position_buffer: StructuredBuffer<[f32; 3]>,
    vertex_normal_buffer: StructuredBuffer<[f32; 3]>,
    vertex_texcoord0_buffer: StructuredBuffer<[f32; 2]>,
    index_buffer: StructuredBuffer<u32>,
    allocation_buffer: StructuredBuffer<GeometryDrawArgs>,
}
impl GenericBackend for GeometryBackend {
    type ManagerCreateArgs = GeometryManagerArgs;
    type ResourceCreateArgs<'a> = GeometryArgs<'a>;
    type ResourceCreateError = RangeAllocationError;
    type ResourceInfo = GeometryInfo;

    fn new<'a>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        create_info: Self::ManagerCreateArgs,
    ) -> Self {
        let GeometryManagerArgs {
            resource_capacity,
            vertex_capacity,
            index_capacity,
        } = create_info;
        _ = queue;

        Self {
            id_allocator: RangeAllocator::new(
                "GeometryResourceManager::id_allocator",
                0..resource_capacity,
            ),
            vertex_allocator: RangeAllocator::new(
                "GeometryResourceManager::vertex_allocator",
                0..vertex_capacity,
            ),
            index_allocator: RangeAllocator::new(
                "GeometryResourceManager::index_allocator",
                0..index_capacity,
            ),
            vertex_position_buffer: StructuredBuffer::new(
                device,
                vertex_capacity,
                wgpu::BufferUsages::STORAGE,
                "GeometryResourceManager::vertex_position_buffer",
            ),
            vertex_normal_buffer: StructuredBuffer::new(
                device,
                vertex_capacity,
                wgpu::BufferUsages::STORAGE,
                "GeometryResourceManager::vertex_normal_buffer",
            ),
            vertex_texcoord0_buffer: StructuredBuffer::new(
                device,
                vertex_capacity,
                wgpu::BufferUsages::STORAGE,
                "GeometryResourceManager::vertex_texcoord0_buffer",
            ),
            index_buffer: StructuredBuffer::new(
                device,
                index_capacity,
                wgpu::BufferUsages::INDEX,
                "GeometryResourceManager::index_buffer",
            ),
            allocation_buffer: StructuredBuffer::new(
                device,
                resource_capacity,
                wgpu::BufferUsages::STORAGE,
                "GeometryResourceManager::allocation_buffer",
            ),
        }
    }

    fn add_impl<'a>(
        &mut self,
        create_info: GeometryArgs<'a>,
    ) -> Result<GeometryInfo, Self::ResourceCreateError> {
        let GeometryArgs {
            vertex_position_data,
            vertex_normal_data,
            vertex_texcoord0_data,
            index_data,
            queue,
        } = create_info;

        // Allocate and write vertex data:
        let vertex_count = vertex_position_data.len();
        assert_eq!(vertex_count, vertex_normal_data.len());
        assert_eq!(vertex_count, vertex_texcoord0_data.len());
        let vertex_range = self.vertex_allocator.allocate(vertex_count)?;
        self.vertex_position_buffer
            .write(vertex_range.start, vertex_position_data, queue);
        self.vertex_normal_buffer
            .write(vertex_range.start, vertex_normal_data, queue);
        self.vertex_texcoord0_buffer
            .write(vertex_range.start, vertex_texcoord0_data, queue);

        // Allocate and write index data:
        let index_range = self.index_allocator.allocate(index_data.len())?;
        self.index_buffer
            .write(index_range.start, index_data, queue);

        // Write the allocated ranges to a GPU-accessible allocation table.
        // This can be used by a compute shader to set up DrawIndexedIndirectArgs for
        // rendering.
        // https://docs.rs/wgpu/latest/wgpu/util/struct.DrawIndexedIndirectArgs.html
        let id = self.id_allocator.allocate(1)?.start;
        self.allocation_buffer.write(
            id,
            &[GeometryDrawArgs::new(&vertex_range, &index_range)],
            queue,
        );

        // Done:
        Ok(GeometryInfo {
            id,
            vertex_range,
            index_range,
        })
    }

    fn del_impl(&mut self, resource: &Self::ResourceInfo) {
        self.vertex_allocator
            .deallocate(resource.vertex_range.clone());
        self.index_allocator
            .deallocate(resource.index_range.clone());
        self.id_allocator.deallocate(resource.id..resource.id + 1)
    }
}

impl<'a> Default for GeometryManagerArgs {
    fn default() -> Self {
        Self {
            resource_capacity: 512,
            vertex_capacity: 1 << 20,
            index_capacity: 1 << 20,
        }
    }
}

impl GeometryDrawArgs {
    fn new(vertex_range: &Range<usize>, index_range: &Range<usize>) -> Self {
        Self {
            first_index: index_range.start as u32,
            index_count: index_range.len() as u32,
            base_vertex: vertex_range.start as u32,
        }
    }
}
