"""WGSL sources for untiled 3DGS blend and gradient kernels."""


def gaussian_blend_wgsl(*, width: int, height: int, count: int) -> str:
    """Front-to-back alpha blending with transmittance and contrib tape."""
    return f"""
@group(0) @binding(0)
var<storage, read_write> image: array<f32>;
@group(0) @binding(1)
var<storage, read_write> final_T: array<f32>;
@group(0) @binding(2)
var<storage, read_write> n_contrib: array<u32>;
@group(0) @binding(3)
var<storage, read> means2d: array<f32>;
@group(0) @binding(4)
var<storage, read> conics: array<f32>;
@group(0) @binding(5)
var<storage, read> colors: array<f32>;
@group(0) @binding(6)
var<storage, read> opacities: array<f32>;
@group(0) @binding(7)
var<storage, read> order: array<u32>;

const WIDTH: u32 = {width}u;
const HEIGHT: u32 = {height}u;
const COUNT: u32 = {count}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let pixel = gid.x;
    if (pixel >= WIDTH * HEIGHT) {{
        return;
    }}
    let px = pixel % WIDTH;
    let py = pixel / WIDTH;
    let fx = f32(px) + 0.5;
    let fy = f32(py) + 0.5;

    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;
    var T: f32 = 1.0;
    var contrib: u32 = 0u;

    for (var i: u32 = 0u; i < COUNT; i += 1u) {{
        let gi = order[i];
        let base2 = gi * 2u;
        let mx = means2d[base2];
        let my = means2d[base2 + 1u];
        let base3 = gi * 3u;
        let c0 = conics[base3];
        let c1 = conics[base3 + 1u];
        let c2 = conics[base3 + 2u];
        let cr = colors[base3];
        let cg = colors[base3 + 1u];
        let cb = colors[base3 + 2u];
        let opacity = opacities[gi];

        let dx = fx - mx;
        let dy = fy - my;
        let power = -0.5 * (c0 * dx * dx + c2 * dy * dy) - c1 * dx * dy;
        if (power > 0.0) {{
            continue;
        }}
        let alpha = min(0.99, opacity * exp(power));
        if (alpha < (1.0 / 255.0)) {{
            continue;
        }}
        let weight = alpha * T;
        r += cr * weight;
        g += cg * weight;
        b += cb * weight;
        T = T * (1.0 - alpha);
        contrib += 1u;
        if (T < 1e-4) {{
            break;
        }}
    }}

    let out_base = pixel * 3u;
    image[out_base] = r;
    image[out_base + 1u] = g;
    image[out_base + 2u] = b;
    final_T[pixel] = T;
    n_contrib[pixel] = contrib;
}}
"""


