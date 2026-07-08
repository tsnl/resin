/// Lowered WebGPU program (internal; not part of the public API).
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WgpuProgram {
    pub dispatch_count: usize,
    pub buffer_count: usize,
    pub buffer_view_count: usize,
}