@group(0) @binding(0) var main_texture: texture_2d<f32>;
@group(0) @binding(1) var main_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var output: VertexOutput;
    if vertex_idx == 0u {
        output.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        output.uv = vec2<f32>(0.0, 1.0);
    } else if (vertex_idx == 1u) {
        output.position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
        output.uv = vec2<f32>(2.0, 1.0);
    } else {
        output.position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
        output.uv = vec2<f32>(0.0, -1.0);
    }
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(main_texture, main_sampler, input.uv);
}
