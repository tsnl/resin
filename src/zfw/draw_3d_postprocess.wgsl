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
    let exposed = apply_exposure(hdr_color.rgb, 1.5);
    let tonemapped = naughty_dog_tonemap(exposed);
    let gamma_corrected = linear_to_srgb(tonemapped);
    return vec4<f32>(gamma_corrected, 1.0);
}

fn apply_exposure(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    return color * exposure;
}

fn naughty_dog_tonemap(color: vec3<f32>) -> vec3<f32> {
    // Attempt to better capture Uncharted 2 / Naughty Dog filmic curve
    let A = 0.22;  // Shoulder strength
    let B = 0.30;  // Linear strength
    let C = 0.10;  // Linear angle
    let D = 0.20;  // Toe strength
    let E = 0.01;  // Toe numerator
    let F = 0.30;  // Toe denominator
    let white = 11.2;
    let num = naughty_dog_curve(color, A, B, C, D, E, F);
    let denom = naughty_dog_curve(vec3<f32>(white), A, B, C, D, E, F);
    return num / denom;
}

fn naughty_dog_curve(x: vec3<f32>, A: f32, B: f32, C: f32, D: f32, E: f32, F: f32) -> vec3<f32> {
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}
