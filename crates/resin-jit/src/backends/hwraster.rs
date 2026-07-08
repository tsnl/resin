//! Shared machinery for the `rasterize` node: the WGSL vertex/fragment
//! shaders (vertex pulling + flat prim id + barycentric varyings) used by
//! both GPU backends (`docs/hw-nodes.md`). wgpu compiles the text directly;
//! the Vulkan backend translates it to SPIR-V through naga per stage.

/// Vertex entry point of [`RASTER_WGSL`].
pub(crate) const VS_ENTRY: &str = "vs_main";

/// Fragment entry point of [`RASTER_WGSL`].
pub(crate) const FS_ENTRY: &str = "fs_main";

/// Visibility-buffer shaders. There are no vertex buffers: the vertex stage
/// pulls from the dense `[V, 4]` clip-position buffer (bound as
/// `array<vec4<f32>>` — a 16-byte stride matches the tensor layout exactly)
/// and the flattened `[T, 3]` index buffer, drawing `3 * T` non-indexed
/// vertices. Vertex indices are clamped in-bounds like `gather_rows` (and
/// the CPU reference). The default `perspective` interpolation of the
/// per-corner barycentric seeds yields perspective-correct `(u, v)` in the
/// fragment stage; `prim` is flat (all three corners carry the same id).
/// The fragment output is the `(prim, hit, u, v)` pixel record of
/// `resin_core::hw::raster_channel`, rendered to an RGBA32Float target with
/// blending off.
pub(crate) const RASTER_WGSL: &str = "\
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) prim: u32,
    @location(1) bary: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tri_indices: array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let tri = vi / 3u;
    let corner = vi % 3u;
    let index = min(tri_indices[tri * 3u + corner], arrayLength(&positions) - 1u);
    var out: VsOut;
    out.pos = positions[index];
    out.prim = tri;
    out.bary = vec2<f32>(
        select(0.0, 1.0, corner == 1u),
        select(0.0, 1.0, corner == 2u),
    );
    return out;
}

@fragment
fn fs_main(vin: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(vin.prim), 1.0, vin.bary.x, vin.bary.y);
}
";

#[cfg(test)]
mod tests {
    use super::{FS_ENTRY, RASTER_WGSL, VS_ENTRY};

    /// naga-validate the shared shaders (no GPU required); integration tests
    /// cannot reach this `pub(crate)` module, so the check lives here.
    #[test]
    fn raster_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(RASTER_WGSL)
            .unwrap_or_else(|e| panic!("parse: {e}\n---\n{RASTER_WGSL}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("validate: {e:?}\n---\n{RASTER_WGSL}"));

        for (entry, stage) in [
            (VS_ENTRY, naga::ShaderStage::Vertex),
            (FS_ENTRY, naga::ShaderStage::Fragment),
        ] {
            assert!(
                module
                    .entry_points
                    .iter()
                    .any(|ep| ep.name == entry && ep.stage == stage),
                "missing {stage:?} entry point {entry}"
            );
        }
    }
}
