//! Geometry resources model 3D triangle meshes without any specific material applied.

use crate::res::core::{GenericManager, GenericResource};
use std::ops::Range;

//
// Interface:
//

pub type GeometryManager = GenericManager<details::GeometryBackend>;
pub type Geometry = GenericResource<details::GeometryBackend>;

pub struct GeometryBackendCreateArgs<'a> {
    device: &'a wgpu::Device,
    resource_capacity: usize,
    vertex_capacity: usize,
    index_capacity: usize,
}

pub struct GeometryCreateArgs<'a> {
    vertex_position_data: &'a [[f32; 3]],
    vertex_normal_data: &'a [[f32; 3]],
    vertex_texcoord0_data: &'a [[f32; 2]],
    index_data: &'a [u32],
    queue: &'a wgpu::Queue,
}

pub struct GeometryInfo {
    pub id: usize,
    pub vertex_range: Range<usize>,
    pub index_range: Range<usize>,
}

#[derive(Default, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct Allocation {
    pub first_index: u32,
    pub index_count: u32,
    pub base_vertex: u32,
}

//
// Implementation
//

mod details {
    use super::*;
    use crate::res::{core::GenericBackend, heap::StructuredBuffer};
    use crate::util::{RangeAllocationError, RangeAllocator};

    pub(super) struct GeometryBackend {
        id_allocator: RangeAllocator,
        vertex_allocator: RangeAllocator,
        index_allocator: RangeAllocator,
        vertex_position_buffer: StructuredBuffer<[f32; 3]>,
        vertex_normal_buffer: StructuredBuffer<[f32; 3]>,
        vertex_texcoord0_buffer: StructuredBuffer<[f32; 2]>,
        index_buffer: StructuredBuffer<u32>,
        allocation_buffer: StructuredBuffer<Allocation>,
    }
    impl GenericBackend for GeometryBackend {
        type CreateInfo<'a> = GeometryBackendCreateArgs<'a>;
        type ResourceCreateInfo<'a> = GeometryCreateArgs<'a>;
        type ResourceCreateError = RangeAllocationError;
        type ResourceInfo = GeometryInfo;

        fn new<'a>(create_info: Self::CreateInfo<'a>) -> Self {
            let GeometryBackendCreateArgs {
                device,
                resource_capacity,
                vertex_capacity,
                index_capacity,
            } = create_info;

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
            create_info: GeometryCreateArgs<'a>,
        ) -> Result<GeometryInfo, Self::ResourceCreateError> {
            let GeometryCreateArgs {
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
                .write(index_range.start, &index_data, queue);

            // Write the allocated ranges to a GPU-accessible allocation table.
            // This can be used by a compute shader to set up DrawIndexedIndirectArgs for
            // rendering.
            // https://docs.rs/wgpu/latest/wgpu/util/struct.DrawIndexedIndirectArgs.html
            let id = self.id_allocator.allocate(1)?.start;
            self.allocation_buffer.write(
                id,
                &[Allocation::new(&vertex_range, &index_range)],
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

    impl Allocation {
        fn new(vertex_range: &Range<usize>, index_range: &Range<usize>) -> Self {
            Self {
                first_index: index_range.start as u32,
                index_count: index_range.len() as u32,
                base_vertex: vertex_range.start as u32,
            }
        }
    }
}
