@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_postprocess(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var output: VertexOutput;
    output.position = compute_fullscreen_position(vertex_idx);
    output.uv = compute_fullscreen_uv(vertex_idx);
    return output;
}

fn compute_fullscreen_position(vertex_idx: u32) -> vec4<f32> {
    let x = f32(i32(vertex_idx & 1u) * 4 - 1);
    let y = f32(i32(vertex_idx & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn compute_fullscreen_uv(vertex_idx: u32) -> vec2<f32> {
    let x = f32(i32(vertex_idx & 1u) * 4 - 1);
    let y = f32(i32(vertex_idx & 2u) * 2 - 1);
    return vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
}

@fragment
fn fs_postprocess(input: VertexOutput) -> @location(0) vec4<f32> {
    let hdr_color = textureSample(input_texture, input_sampler, input.uv);
    return apply_tonemap(hdr_color);
}

fn apply_tonemap(hdr_color: vec4<f32>) -> vec4<f32> {
    return hdr_color;
}
