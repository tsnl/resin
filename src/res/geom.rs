use parking_lot::RwLock;

use crate::util::RangeAllocator;

use super::heap::StructuredBuffer;
use std::{
    ops::Range,
    sync::{Arc, Weak},
};

pub struct Manager {
    state: RwLock<ManagerState>,
}
impl Manager {
    pub fn add(
        self: &Arc<Self>,
        vertex_position_data: &[[f32; 3]],
        vertex_normal_data: &[[f32; 3]],
        vertex_texcoord0_data: &[[f32; 2]],
        index_data: &[u32],
        queue: &wgpu::Queue,
    ) -> Result<Resource, crate::util::RangeAllocationError> {
        let mut state = self.state.write();

        let manager = Arc::downgrade(self);

        // Allocate and write vertex data:
        let vertex_count = vertex_position_data.len();
        assert_eq!(vertex_count, vertex_normal_data.len());
        assert_eq!(vertex_count, vertex_texcoord0_data.len());
        let vertex_range = state.vertex_allocator.allocate(vertex_count)?;
        state
            .vertex_position_buffer
            .write(vertex_range.start, vertex_position_data, queue);
        state
            .vertex_normal_buffer
            .write(vertex_range.start, vertex_normal_data, queue);
        state
            .vertex_texcoord0_buffer
            .write(vertex_range.start, vertex_texcoord0_data, queue);

        // Allocate and write index data:
        let index_range = state.index_allocator.allocate(index_data.len())?;
        state
            .index_buffer
            .write(index_range.start, &index_data, queue);

        // Write the allocated ranges to a GPU-accessible allocation table.
        // This can be used by a compute shader to set up DrawIndexedIndirectArgs for
        // rendering.
        // https://docs.rs/wgpu/latest/wgpu/util/struct.DrawIndexedIndirectArgs.html
        let id = state.id_allocator.allocate(1)?.start;
        state
            .allocation_buffer
            .write(id, &[Allocation::new(&vertex_range, &index_range)], queue);

        Ok(Resource {
            manager,
            id,
            vertex_range,
            index_range,
        })
    }
}

struct ManagerState {
    id_allocator: RangeAllocator,
    vertex_allocator: RangeAllocator,
    index_allocator: RangeAllocator,
    vertex_position_buffer: StructuredBuffer<[f32; 3]>,
    vertex_normal_buffer: StructuredBuffer<[f32; 3]>,
    vertex_texcoord0_buffer: StructuredBuffer<[f32; 2]>,
    index_buffer: StructuredBuffer<u32>,
    allocation_buffer: StructuredBuffer<Allocation>,
}
impl ManagerState {
    fn new(
        device: &wgpu::Device,
        resource_capacity: usize,
        vertex_capacity: usize,
        index_capacity: usize,
    ) -> Self {
        Self {
            id_allocator: RangeAllocator::new(
                "resin::res::geom::Manager::id_allocator",
                0..resource_capacity,
            ),
            vertex_allocator: RangeAllocator::new(
                "resin::res::geom::Manager::vertex_allocator",
                0..vertex_capacity,
            ),
            index_allocator: RangeAllocator::new(
                "resin::res::geom::Manager::index_allocator",
                0..index_capacity,
            ),
            vertex_position_buffer: StructuredBuffer::new(
                device,
                vertex_capacity,
                wgpu::BufferUsages::STORAGE,
                "resin::res::geom::Manager::vertex_position_buffer",
            ),
            vertex_normal_buffer: StructuredBuffer::new(
                device,
                vertex_capacity,
                wgpu::BufferUsages::STORAGE,
                "resin::res::geom::Manager::vertex_normal_buffer",
            ),
            vertex_texcoord0_buffer: StructuredBuffer::new(
                device,
                vertex_capacity,
                wgpu::BufferUsages::STORAGE,
                "resin::res::geom::Manager::vertex_texcoord0_buffer",
            ),
            index_buffer: StructuredBuffer::new(
                device,
                index_capacity,
                wgpu::BufferUsages::INDEX,
                "resin::res::geom::Manager::index_buffer",
            ),
            allocation_buffer: StructuredBuffer::new(
                device,
                resource_capacity,
                wgpu::BufferUsages::STORAGE,
                "resin::res::geom::Manager::allocation_buffer",
            ),
        }
    }
}

#[derive(Default, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct Allocation {
    first_index: u32,
    index_count: u32,
    base_vertex: u32,
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

pub struct Resource {
    manager: Weak<Manager>,
    id: usize,
    vertex_range: Range<usize>,
    index_range: Range<usize>,
}
impl Resource {
    pub fn id(&self) -> usize {
        self.id
    }
    pub fn vertex_range(&self) -> Range<usize> {
        self.vertex_range.clone()
    }
    pub fn index_range(&self) -> Range<usize> {
        self.index_range.clone()
    }
}
impl Drop for Resource {
    fn drop(&mut self) {
        if let Some(manager) = self.manager.upgrade() {
            let mut state = manager.state.write();
            state.vertex_allocator.deallocate(self.vertex_range.clone());
            state.index_allocator.deallocate(self.index_range.clone());
        }
    }
}
