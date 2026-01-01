// draw_2d.wgsl

struct PodQuad {
    dst_xy_ndc: u32,         // Packed i16 [x2]
    dst_wh_ndc: u32,         // Packed i16 [x2]
    src_xy_uv: u32,          // Packed u16 [x2]
    src_wh_uv: u32,          // Packed u16 [x2]
    fill_color_rgba: u32,    // Packed u8 [x4]
    border_thickness_px: u32, // Packed u8 [x4]
    border_color_rgba: u32,  // Packed u8 [x4]
    _rsv: u32,               // Packed u8 [x4] padding
}

@group(0) @binding(0) var<storage, read> quads: array<PodQuad>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) fill_color: vec4<f32>,
}

// Unpack two i16 values from a u32
fn unpack_i16x2(packed: u32) -> vec2<i32> {
    let x = i32(packed & 0xFFFFu);
    let y = i32((packed >> 16u) & 0xFFFFu);
    return vec2<i32>(x, y);
}

// Unpack two u16 values from a u32
fn unpack_u16x2(packed: u32) -> vec2<u32> {
    let x = packed & 0xFFFFu;
    let y = (packed >> 16u) & 0xFFFFu;
    return vec2<u32>(x, y);
}

// Unpack four u8 values from a u32
fn unpack_u8x4(packed: u32) -> vec4<u32> {
    let x = packed & 0xFFu;
    let y = (packed >> 8u) & 0xFFu;
    let z = (packed >> 16u) & 0xFFu;
    let w = (packed >> 24u) & 0xFFu;
    return vec4<u32>(x, y, z, w);
}

// Convert fixed-point i16 to float
fn fx_i16_to_f32(v: i32) -> f32 {
    // Reinterpret as signed 16-bit by sign-extending
    let sign_extended = (v << 16) >> 16;
    return f32(sign_extended) / 32768.0;
}

// Convert fixed-point u16 to float
fn fx_u16_to_f32(v: u32) -> f32 {
    return f32(v) / 65535.0;
}

// Convert u8 to float [0..1]
fn u8_to_f32(v: u32) -> f32 {
    return f32(v) / 255.0;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32, @builtin(instance_index) instance_idx: u32) -> VertexOutput {
    let quad = quads[instance_idx];

    // Unpack values from packed u32 fields
    let dst_xy_packed = unpack_i16x2(quad.dst_xy_ndc);
    let dst_wh_packed = unpack_i16x2(quad.dst_wh_ndc);
    let src_xy_packed = unpack_u16x2(quad.src_xy_uv);
    let src_wh_packed = unpack_u16x2(quad.src_wh_uv);
    let fill_color_packed = unpack_u8x4(quad.fill_color_rgba);

    // Decode fixed-point values
    let dst_xy = vec2<f32>(
        fx_i16_to_f32(dst_xy_packed.x),
        fx_i16_to_f32(dst_xy_packed.y)
    );
    let dst_wh = vec2<f32>(
        fx_i16_to_f32(dst_wh_packed.x),
        fx_i16_to_f32(dst_wh_packed.y)
    );
    let src_xy = vec2<f32>(
        fx_u16_to_f32(src_xy_packed.x),
        fx_u16_to_f32(src_xy_packed.y)
    );
    let src_wh = vec2<f32>(
        fx_u16_to_f32(src_wh_packed.x),
        fx_u16_to_f32(src_wh_packed.y)
    );

    // Generate quad vertices (two triangles: 0,1,2 and 2,1,3)
    var local_pos: vec2<f32>;
    var local_uv: vec2<f32>;

    switch vertex_idx {
        case 0u: { local_pos = vec2<f32>(0.0, 0.0); local_uv = vec2<f32>(0.0, 0.0); }
        case 1u: { local_pos = vec2<f32>(1.0, 0.0); local_uv = vec2<f32>(1.0, 0.0); }
        case 2u: { local_pos = vec2<f32>(0.0, 1.0); local_uv = vec2<f32>(0.0, 1.0); }
        case 3u: { local_pos = vec2<f32>(0.0, 1.0); local_uv = vec2<f32>(0.0, 1.0); }
        case 4u: { local_pos = vec2<f32>(1.0, 0.0); local_uv = vec2<f32>(1.0, 0.0); }
        default: { local_pos = vec2<f32>(1.0, 1.0); local_uv = vec2<f32>(1.0, 1.0); }
    }

    // Calculate position in NDC
    let ndc_pos = dst_xy + local_pos * dst_wh;

    // Calculate UV coordinates
    let uv = src_xy + local_uv * src_wh;

    // Decode fill color
    let fill_color = vec4<f32>(
        u8_to_f32(fill_color_packed.x),
        u8_to_f32(fill_color_packed.y),
        u8_to_f32(fill_color_packed.z),
        u8_to_f32(fill_color_packed.w)
    );

    var output: VertexOutput;
    output.position = vec4<f32>(ndc_pos, 0.0, 1.0);
    output.uv = uv;
    output.fill_color = fill_color;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(tex, tex_sampler, input.uv);
    return tex_color * input.fill_color;
}
