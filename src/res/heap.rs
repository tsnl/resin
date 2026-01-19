use std::{alloc::Layout, marker::PhantomData, ops::Range};

/// StructuredBuffer is a GPU buffer that stores structured data of type T.
/// It does not manage memory allocation itself; instead, it provides methods to write data
/// to specific ranges within the buffer. Use RangeAllocator to manage allocations.
pub struct StructuredBuffer<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    _marker: PhantomData<T>,
}
impl<T: bytemuck::Pod> StructuredBuffer<T> {
    pub fn new(
        device: &wgpu::Device,
        capacity: usize,
        usages: wgpu::BufferUsages,
        name: &'static str,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: (capacity * std::mem::size_of::<T>()) as wgpu::BufferAddress,
            usage: usages | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    /// Writes data to the buffer at the specified offset on the GPU queue with `wgpu::Queue::write_buffer`.
    /// The offset is specified in number of T elements.
    pub fn write(&mut self, offset: usize, data: &[T], queue: &wgpu::Queue) {
        let range = offset..offset + data.len();

        let offset = Layout::array::<T>(range.start).unwrap().size() as wgpu::BufferAddress;
        queue.write_buffer(&self.buffer, offset, bytemuck::cast_slice(data));
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn buffer_slice(&self, range: Range<usize>) -> wgpu::BufferSlice<'_> {
        let offset = Layout::array::<T>(range.start).unwrap().size() as wgpu::BufferAddress;
        let size = Layout::array::<T>(range.len()).unwrap().size() as wgpu::BufferAddress;
        self.buffer.slice(offset..offset + size)
    }
}
