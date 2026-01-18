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
        vertex_texcoord_data: &[[f32; 2]],
        index_data: &[u32],
        queue: &wgpu::Queue,
    ) -> Result<Resource, crate::util::RangeAllocationError> {
        let mut state = self.state.write();

        let manager = Arc::downgrade(self);

        let vertex_count = vertex_position_data.len();
        assert_eq!(vertex_count, vertex_normal_data.len());
        assert_eq!(vertex_count, vertex_texcoord_data.len());
        let vertex_range = state.vertex_allocator.allocate(vertex_count)?;
        state
            .vertex_position_buffer
            .write(vertex_range.start, vertex_position_data, queue);
        state
            .vertex_normal_buffer
            .write(vertex_range.start, vertex_normal_data, queue);
        state
            .vertex_texcoord_buffer
            .write(vertex_range.start, vertex_texcoord_data, queue);

        let index_range = state.index_allocator.allocate(index_data.len())?;
        state
            .index_buffer
            .write(index_range.start, index_data, queue);

        Ok(Resource {
            manager,
            vertex_range,
            index_range,
        })
    }
}

struct ManagerState {
    vertex_allocator: RangeAllocator,
    index_allocator: RangeAllocator,
    vertex_position_buffer: StructuredBuffer<[f32; 3]>,
    vertex_normal_buffer: StructuredBuffer<[f32; 3]>,
    vertex_texcoord_buffer: StructuredBuffer<[f32; 2]>,
    index_buffer: StructuredBuffer<u32>,
}
impl ManagerState {
    fn new(device: &wgpu::Device, vertex_capacity: usize, index_capacity: usize) -> Self {
        Self {
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
            vertex_texcoord_buffer: StructuredBuffer::new(
                device,
                vertex_capacity,
                wgpu::BufferUsages::STORAGE,
                "resin::res::geom::Manager::vertex_texcoord_buffer",
            ),
            index_buffer: StructuredBuffer::new(
                device,
                index_capacity,
                wgpu::BufferUsages::INDEX,
                "resin::res::geom::Manager::index_buffer",
            ),
        }
    }
}

pub struct Resource {
    manager: Weak<Manager>,
    vertex_range: Range<usize>,
    index_range: Range<usize>,
}
impl Resource {
    pub fn with_buffer_slices(&self, callback: impl FnOnce(BufferSlices)) {
        match self.manager.upgrade() {
            None => panic!("Attempted to use a Resource after its Manager was dropped."),
            Some(manager) => {
                let state = manager.state.read();
                callback(BufferSlices {
                    v_p_buffer: state
                        .vertex_position_buffer
                        .buffer_slice(self.vertex_range.clone()),
                    v_n_buffer: state
                        .vertex_normal_buffer
                        .buffer_slice(self.vertex_range.clone()),
                    v_t_buffer: state
                        .vertex_texcoord_buffer
                        .buffer_slice(self.vertex_range.clone()),
                    index_buffer: state.index_buffer.buffer_slice(self.index_range.clone()),
                })
            }
        }
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

pub struct BufferSlices<'a> {
    pub v_p_buffer: wgpu::BufferSlice<'a>,
    pub v_n_buffer: wgpu::BufferSlice<'a>,
    pub v_t_buffer: wgpu::BufferSlice<'a>,
    pub index_buffer: wgpu::BufferSlice<'a>,
}
