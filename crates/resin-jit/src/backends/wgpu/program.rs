//! Lowered WebGPU program: buffer table + per-kernel WGSL (no fused opts).

/// Spec for one device buffer (matches IR buffer layout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuBufferSpec {
    pub shape: Box<[u32]>,
    pub nbytes: u64,
    pub init: Option<Box<[u8]>>,
    pub readonly: bool,
}

/// Buffer + pitched view (matches IR buffer_views).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuBufferViewSpec {
    pub buffer_index: usize,
    pub offset: u32,
    pub shape: Box<[u32]>,
    pub pitch: Box<[u32]>,
}

/// One compiled compute pipeline (WGSL text + dispatch metadata).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuPipelineSpec {
    pub wgsl: String,
    pub entry_point: String,
    pub dispatch_size: [u32; 3],
    pub num_arg_bindings: u32,
    pub clear_output_before_dispatch: bool,
}

/// One queue op: run a pipeline with bound arg views into an output buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuDispatch {
    pub pipeline_index: usize,
    pub arg_view_indices: Vec<usize>,
    pub output_buffer_index: usize,
}

/// Artifact produced by naive IR → WGSL lowering.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WgpuProgram {
    pub buffers: Vec<WgpuBufferSpec>,
    pub buffer_views: Vec<WgpuBufferViewSpec>,
    pub pipelines: Vec<WgpuPipelineSpec>,
    pub queue: Vec<WgpuDispatch>,
    pub param_buffer_indices: Vec<usize>,
    pub sink_view_indices: Vec<usize>,
}

impl WgpuProgram {
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

impl WgpuBufferSpec {
    pub fn byte_len(&self) -> u64 {
        self.nbytes.max(4)
    }
}
