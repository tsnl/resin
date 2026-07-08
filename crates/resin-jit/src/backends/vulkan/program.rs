//! Lowered Vulkan program: buffer table + typed execution steps.
//!
//! Structurally identical to the wgpu artifact, but pipelines carry SPIR-V
//! words (translated from the shared WGSL emission by naga at lowering time).

/// Spec for one device buffer (matches IR buffer layout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VkBufferSpec {
    pub shape: Box<[u32]>,
    pub nbytes: u64,
    pub init: Option<Box<[u8]>>,
    pub readonly: bool,
}

impl VkBufferSpec {
    pub fn byte_len(&self) -> u64 {
        self.nbytes.max(4)
    }
}

/// Buffer + pitched view (matches IR buffer_views).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VkBufferViewSpec {
    pub buffer_index: usize,
    pub offset: u32,
    pub shape: Box<[u32]>,
    pub pitch: Box<[u32]>,
}

/// One compiled compute pipeline (SPIR-V words + dispatch metadata).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VkPipelineSpec {
    pub spirv: Vec<u32>,
    pub entry_point: String,
    pub dispatch_size: [u32; 3],
    pub num_arg_bindings: u32,
    pub clear_output_before_dispatch: bool,
}

/// One queue op: run a pipeline with bound arg views into an output buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VkDispatch {
    pub pipeline_index: usize,
    pub arg_view_indices: Vec<usize>,
    pub output_buffer_index: usize,
}

/// Closest-hit ray tracing step (dense argument buffers, in IR arg order:
/// origins, directions, t_min, t_max, vertices, triangles).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VkTraceStep {
    pub arg_buffer_indices: [usize; 6],
    pub output_buffer_index: usize,
    pub ray_count: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
}

impl VkTraceStep {
    pub fn vertices_buffer_index(&self) -> usize {
        self.arg_buffer_indices[4]
    }

    pub fn triangles_buffer_index(&self) -> usize {
        self.arg_buffer_indices[5]
    }
}

/// Visibility-buffer rasterization step (dense argument buffers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VkRasterStep {
    pub positions_buffer_index: usize,
    pub triangles_buffer_index: usize,
    pub output_buffer_index: usize,
    pub height: u32,
    pub width: u32,
    pub triangle_count: u32,
}

/// One execution step of a lowered program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VkStep {
    Compute(VkDispatch),
    TraceRays(VkTraceStep),
    Rasterize(VkRasterStep),
}

/// Artifact produced by IR → WGSL → SPIR-V lowering.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VulkanProgram {
    pub buffers: Vec<VkBufferSpec>,
    pub buffer_views: Vec<VkBufferViewSpec>,
    pub pipelines: Vec<VkPipelineSpec>,
    pub queue: Vec<VkStep>,
    pub param_buffer_indices: Vec<usize>,
    pub sink_view_indices: Vec<usize>,
}

impl VulkanProgram {
    pub fn dispatch_count(&self) -> usize {
        self.queue.len()
    }

    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn buffer_view_count(&self) -> usize {
        self.buffer_views.len()
    }

    /// Buffer indices used as ray-tracing geometry (acceleration-structure
    /// build inputs need device addresses + AS-build usage).
    pub fn trace_geometry_buffer_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.queue.iter().flat_map(|step| match step {
            VkStep::TraceRays(t) => {
                vec![t.vertices_buffer_index(), t.triangles_buffer_index()]
            }
            _ => Vec::new(),
        })
    }
}
