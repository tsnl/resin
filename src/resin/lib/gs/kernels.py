"""WGSL source generators for 3DGS custom kernels."""


def gaussian_blend_wgsl(*, width: int, height: int, count: int) -> str:
    """Front-to-back alpha splatting for a fixed image size and gaussian count."""
    return f"""
@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@group(0) @binding(1)
var<storage, read> means2d: array<f32>;

@group(0) @binding(2)
var<storage, read> conics: array<f32>;

@group(0) @binding(3)
var<storage, read> colors: array<f32>;

@group(0) @binding(4)
var<storage, read> opacities: array<f32>;

@group(0) @binding(5)
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
        r += cr * alpha;
        g += cg * alpha;
        b += cb * alpha;
    }}

    let out_base = pixel * 3u;
    output[out_base] = r;
    output[out_base + 1u] = g;
    output[out_base + 2u] = b;
}}
"""


def argsort_depths_wgsl(*, count: int) -> str:
    """Write permutation indices that sort depths ascending (front-to-back)."""
    # Simple selection-sort network executed by one thread for tiny N.
    loop_init = "\n".join(f"    sorted[{i}] = {i}u;" for i in range(count))
    compare_loops: list[str] = []
    for i in range(count):
        for j in range(i + 1, count):
            compare_loops.append(
                f"""
    if (depths[sorted[{j}]] < depths[sorted[{i}]]) {{
        let tmp = sorted[{i}];
        sorted[{i}] = sorted[{j}];
        sorted[{j}] = tmp;
    }}"""
            )
    compare_body = "\n".join(compare_loops)
    store_body = "\n".join(
        f"    output[{i}] = sorted[{i}];" for i in range(count)
    )
    return f"""
@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

@group(0) @binding(1)
var<storage, read> depths: array<f32>;

const COUNT: u32 = {count}u;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x > 0u) {{
        return;
    }}
    var sorted: array<u32, {count}>;
{loop_init}
{compare_body}
{store_body}
}}
"""