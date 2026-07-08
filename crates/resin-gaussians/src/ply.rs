//! Minimal loader for standard 3DGS PLY checkpoints (INRIA layout).
//!
//! Parses `binary_little_endian` PLYs whose vertex element carries float
//! properties `x y z … f_dc_0..2 opacity scale_0..2 rot_0..3` (extra
//! properties like normals and `f_rest_*` SH bands are skipped). Checkpoint
//! parameters are stored pre-activation; loading applies the standard
//! transforms: `exp` on scales, `sigmoid` on opacity, and SH DC → RGB
//! (`0.5 + C0·f_dc`, clamped to [0, 1]). Rotations stay raw — the renderer
//! normalizes quaternions in-graph.

use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::cloud::CloudData;

/// `C0` spherical-harmonics constant (degree 0).
const SH_C0: f32 = 0.282_094_79;

#[derive(Debug, thiserror::Error)]
pub enum PlyError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a PLY file")]
    BadMagic,
    #[error("unsupported PLY format: {0} (need binary_little_endian 1.0)")]
    UnsupportedFormat(String),
    #[error("unsupported property type: {0} (only float supported)")]
    UnsupportedProperty(String),
    #[error("missing required property {0}")]
    MissingProperty(&'static str),
    #[error("no vertex element in header")]
    NoVertexElement,
}

/// Load a 3DGS checkpoint from a PLY file.
pub fn load_ply(path: impl AsRef<Path>) -> Result<CloudData, PlyError> {
    read_ply(&mut BufReader::new(std::fs::File::open(path)?))
}

/// Load a 3DGS checkpoint from any reader.
pub fn read_ply(reader: &mut impl BufRead) -> Result<CloudData, PlyError> {
    let (count, properties) = read_header(reader)?;

    let find = |name: &'static str| -> Result<usize, PlyError> {
        properties
            .iter()
            .position(|p| p == name)
            .ok_or(PlyError::MissingProperty(name))
    };
    let ix = find("x")?;
    let iy = find("y")?;
    let iz = find("z")?;
    let idc = [find("f_dc_0")?, find("f_dc_1")?, find("f_dc_2")?];
    let iop = find("opacity")?;
    let isc = [find("scale_0")?, find("scale_1")?, find("scale_2")?];
    let irot = [
        find("rot_0")?,
        find("rot_1")?,
        find("rot_2")?,
        find("rot_3")?,
    ];

    let stride = properties.len();
    let mut bytes = vec![0u8; count * stride * 4];
    reader.read_exact(&mut bytes)?;
    let row_f32 = |g: usize, p: usize| -> f32 {
        let at = (g * stride + p) * 4;
        f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    };

    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let mut data = CloudData {
        means: Vec::with_capacity(count),
        scales: Vec::with_capacity(count),
        quats: Vec::with_capacity(count),
        colors: Vec::with_capacity(count),
        opacities: Vec::with_capacity(count),
    };
    for g in 0..count {
        data.means.push([row_f32(g, ix), row_f32(g, iy), row_f32(g, iz)]);
        data.scales.push([
            row_f32(g, isc[0]).exp(),
            row_f32(g, isc[1]).exp(),
            row_f32(g, isc[2]).exp(),
        ]);
        data.quats.push([
            row_f32(g, irot[0]),
            row_f32(g, irot[1]),
            row_f32(g, irot[2]),
            row_f32(g, irot[3]),
        ]);
        data.colors.push([
            (0.5 + SH_C0 * row_f32(g, idc[0])).clamp(0.0, 1.0),
            (0.5 + SH_C0 * row_f32(g, idc[1])).clamp(0.0, 1.0),
            (0.5 + SH_C0 * row_f32(g, idc[2])).clamp(0.0, 1.0),
        ]);
        data.opacities.push(sigmoid(row_f32(g, iop)));
    }
    Ok(data)
}

fn read_header(reader: &mut impl BufRead) -> Result<(usize, Vec<String>), PlyError> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        return Err(PlyError::BadMagic);
    }
    let mut count: Option<usize> = None;
    let mut properties = Vec::new();
    let mut in_vertex_element = false;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(PlyError::NoVertexElement);
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["format", rest @ ..] => {
                if rest != ["binary_little_endian", "1.0"] {
                    return Err(PlyError::UnsupportedFormat(rest.join(" ")));
                }
            }
            ["comment", ..] => {}
            ["element", "vertex", n] => {
                in_vertex_element = true;
                count = Some(n.parse().map_err(|_| PlyError::NoVertexElement)?);
            }
            ["element", ..] => in_vertex_element = false,
            ["property", ptype, name] if in_vertex_element => {
                if *ptype != "float" {
                    return Err(PlyError::UnsupportedProperty(format!("{ptype} {name}")));
                }
                properties.push((*name).to_string());
            }
            ["property", ..] => {}
            ["end_header"] => break,
            _ => {}
        }
    }
    Ok((count.ok_or(PlyError::NoVertexElement)?, properties))
}

impl CloudData {
    /// Every `stride`-th gaussian (deterministic subsample for tests and
    /// CPU-scale runs).
    pub fn subsample(&self, stride: usize) -> CloudData {
        let pick = |i: &usize| i.is_multiple_of(stride);
        CloudData {
            means: keep(&self.means, pick),
            scales: keep(&self.scales, pick),
            quats: keep(&self.quats, pick),
            colors: keep(&self.colors, pick),
            opacities: keep(&self.opacities, pick),
        }
    }

    /// Axis-aligned bounding box of the means.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for m in &self.means {
            for a in 0..3 {
                lo[a] = lo[a].min(m[a]);
                hi[a] = hi[a].max(m[a]);
            }
        }
        (lo, hi)
    }
}

fn keep<T: Clone>(v: &[T], pick: impl Fn(&usize) -> bool) -> Vec<T> {
    v.iter()
        .enumerate()
        .filter(|(i, _)| pick(i))
        .map(|(_, x)| x.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_ply() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n\
              property float x\nproperty float y\nproperty float z\n\
              property float nx\nproperty float ny\nproperty float nz\n\
              property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n\
              property float opacity\n\
              property float scale_0\nproperty float scale_1\nproperty float scale_2\n\
              property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
              end_header\n",
        );
        for g in 0..2 {
            let base = g as f32;
            let row = [
                base, base + 0.5, -2.0, // xyz
                0.0, 0.0, 0.0, // normals (skipped)
                0.0, 1.0, -1.0, // f_dc
                0.0, // opacity logit → 0.5
                0.0, -1.0, 0.5, // log scales
                1.0, 0.0, 0.0, 0.0, // quat
            ];
            for v in row {
                out.extend_from_slice(&f32::to_le_bytes(v));
            }
        }
        out
    }

    #[test]
    fn parses_and_activates() {
        let bytes = tiny_ply();
        let data = read_ply(&mut std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(data.count(), 2);
        assert_eq!(data.means[1], [1.0, 1.5, -2.0]);
        assert!((data.opacities[0] - 0.5).abs() < 1e-6);
        assert!((data.scales[0][1] - (-1.0f32).exp()).abs() < 1e-6);
        assert!((data.colors[0][0] - 0.5).abs() < 1e-6);
        assert!((data.colors[0][1] - (0.5 + SH_C0)).abs() < 1e-6);
        assert_eq!(data.colors[0][2], 0.5 - SH_C0);
        assert_eq!(data.quats[0], [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn subsample_and_bounds() {
        let data = read_ply(&mut std::io::Cursor::new(tiny_ply())).unwrap();
        assert_eq!(data.subsample(2).count(), 1);
        let (lo, hi) = data.bounds();
        assert_eq!(lo, [0.0, 0.5, -2.0]);
        assert_eq!(hi, [1.0, 1.5, -2.0]);
    }
}
