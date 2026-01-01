// draw_2d.wgsl

struct PodQuad {
    dst_xy_ndc: vec2<f32>,
    dst_wh_ndc: vec2<f32>,
    src_xy_uv: vec2<f32>,
    src_wh_uv: vec2<f32>,
    fill_color_rgba: vec4<f32>,
    border_thickness_ndc: vec2<f32>,
    border_color_rgba: vec4<f32>,
    _rsv: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> quads: array<PodQuad>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) instance_idx: u32,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) fill_color: vec4<f32>,
    @location(3) @interpolate(flat) border_thickness_ndc: vec2<f32>,
    @location(4) @interpolate(flat) border_color: vec4<f32>,
}

fn vertex_array_index(vertex_idx: u32) -> u32
{
    // Corners: 0=TL, 1=TR, 2=BR, 3=BL
    // Vertices: (0=TL, 1=TR, 2=BR), (0=TL, 2=BR, 3=BL)
    // Mapping:
    // * vertexIdx: 0 -> cornerIdx: 0 = (v=0 % 3)=0 + (v=0 // 4)=0
    // * vertexIdx: 1 -> cornerIdx: 1 = (v=1 % 3)=1 + (v=1 // 4)=0
    // * vertexIdx: 2 -> cornerIdx: 2 = (v=2 % 3)=2 + (v=2 // 4)=0
    // * vertexIdx: 3 -> cornerIdx: 0 = (v=3 % 3)=0 + (v=3 // 4)=0
    // * vertexIdx: 4 -> cornerIdx: 2 = (v=4 % 3)=1 + (v=4 // 4)=1
    // * vertexIdx: 5 -> cornerIdx: 3 = (v=5 % 3)=2 + (v=5 // 4)=1
    return vertex_idx % 3 + vertex_idx / 4;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32, @builtin(instance_index) instance_idx: u32) -> VertexOutput {
    var output: VertexOutput;

    // Load quad and common vertex data:
    let quad = quads[instance_idx];
    let corner_idx = vertex_array_index(vertex_idx);

    // output.position:
    let vertex_array = array<vec2<f32>, 4>(
        vec2<f32>(quad.dst_xy_ndc) + vec2<f32>(0.0, 0.0) * quad.dst_wh_ndc,
        vec2<f32>(quad.dst_xy_ndc) + vec2<f32>(1.0, 0.0) * quad.dst_wh_ndc,
        vec2<f32>(quad.dst_xy_ndc) + vec2<f32>(1.0, 1.0) * quad.dst_wh_ndc,
        vec2<f32>(quad.dst_xy_ndc) + vec2<f32>(0.0, 1.0) * quad.dst_wh_ndc,
    );
    output.position = vec4<f32>(vertex_array[corner_idx], 0.0, 1.0);

    // output.instance_idx:
    output.instance_idx = instance_idx;

    // output.uv:
    let uv_array = array<vec2<f32>, 4>(
        vec2<f32>(quad.src_xy_uv) + vec2<f32>(0.0, 0.0) * quad.src_wh_uv,
        vec2<f32>(quad.src_xy_uv) + vec2<f32>(1.0, 0.0) * quad.src_wh_uv,
        vec2<f32>(quad.src_xy_uv) + vec2<f32>(1.0, 1.0) * quad.src_wh_uv,
        vec2<f32>(quad.src_xy_uv) + vec2<f32>(0.0, 1.0) * quad.src_wh_uv,
    );
    output.uv = uv_array[corner_idx];

    // output.fill_color:
    output.fill_color = quad.fill_color_rgba;

    // output.border_thickness_ndc:
    output.border_thickness_ndc = quad.border_thickness_ndc;

    // output.border_color:
    output.border_color = quad.border_color_rgba;

    // Done:
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(tex, tex_sampler, input.uv);
    return tex_color * input.fill_color;
}
