use resin_ir::{IrBuffer, IrBufferView, IrDispatch};

/// Lowered CPU interpreter program (internal; not part of the public API).
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuProgram {
    pub buffers: Vec<IrBuffer>,
    pub buffer_views: Vec<IrBufferView>,
    pub queue: Vec<IrDispatch>,
    pub param_buffer_indices: Vec<usize>,
    pub sink_view_indices: Vec<usize>,
}

impl CpuProgram {
    pub fn dispatch_count(&self) -> usize {
        self.queue.len()
    }

    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn buffer_view_count(&self) -> usize {
        self.buffer_views.len()
    }
}