// Full-screen blit shader for copying the draw2d frame output to the canvas texture.
// Uses a full-screen triangle technique (no vertex buffer needed).

@group(0) @binding(0) var tex_sampler: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_blit(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    // Full-screen triangle: vertices at (-1,-1), (3,-1), (-1,3)
    // This covers the entire clip space with a single triangle.
    var output: VertexOutput;

    // vertex 0: x = -1, y = -1
    // vertex 1: x =  3, y = -1
    // vertex 2: x = -1, y =  3
    let x = f32(i32(vertex_idx & 1u) * 4 - 1);
    let y = f32(i32(vertex_idx & 2u) * 2 - 1);

    output.position = vec4<f32>(x, y, 0.0, 1.0);

    // UV coordinates: map clip space to [0,1] range
    // Note: Y is flipped because NDC has Y pointing up, but texture has Y pointing down
    output.uv = vec2<f32>(
        (x + 1.0) * 0.5,
        (1.0 - y) * 0.5
    );

    return output;
}

@fragment
fn fs_blit(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(tex, tex_sampler, input.uv);
}
