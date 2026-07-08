//! Minimal binary PPM output for the demo executables.

use std::io::Write;
use std::path::Path;

/// Write linear-RGB `[H*W*3]` values as an 8-bit `P6` PPM with a simple
/// gamma-2.2 transfer and clamping.
pub fn write_ppm(path: &Path, width: usize, height: usize, rgb: &[f32]) -> std::io::Result<()> {
    assert_eq!(rgb.len(), width * height * 3, "rgb buffer size");
    let mut out = Vec::with_capacity(width * height * 3 + 32);
    write!(out, "P6\n{width} {height}\n255\n")?;
    for &v in rgb {
        let v = v.max(0.0).powf(1.0 / 2.2).min(1.0);
        out.push((v * 255.0 + 0.5) as u8);
    }
    std::fs::write(path, out)
}
