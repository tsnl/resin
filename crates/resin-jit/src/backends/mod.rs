//! JIT execution backends, enabled via crate features.

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
pub(crate) mod hwraster;
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
pub(crate) mod hwtrace;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "wgpu")]
pub mod wgpu;
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
pub(crate) mod wgsl;

/// Host densify of a pitched device view: the canonical sink-readback
/// semantics shared by the GPU runtimes (callers map the error into their
/// own enums).
#[cfg(any(feature = "wgpu", feature = "vulkan"))]
pub(crate) fn densify_view(
    buffer: &[u8],
    offset: u32,
    shape: &[u32],
    pitch: &[u32],
) -> Result<Vec<u8>, String> {
    let count: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().map(|&d| d as usize).product()
    };
    let mut out = vec![0u8; count * 4];
    let mut coords = vec![0u32; shape.len()];
    for linear in 0..count {
        let mut rem = linear;
        for axis in (0..shape.len()).rev() {
            let dim = shape[axis] as usize;
            coords[axis] = if dim == 0 { 0 } else { (rem % dim) as u32 };
            if dim != 0 {
                rem /= dim;
            }
        }
        let mut idx = offset as usize;
        for (c, p) in coords.iter().zip(pitch.iter()) {
            idx += (*c as usize) * (*p as usize);
        }
        let start = idx * 4;
        let end = start + 4;
        if end > buffer.len() {
            return Err(format!(
                "view densify OOB: index {idx}, buffer {} bytes",
                buffer.len()
            ));
        }
        out[linear * 4..linear * 4 + 4].copy_from_slice(&buffer[start..end]);
    }
    Ok(out)
}
