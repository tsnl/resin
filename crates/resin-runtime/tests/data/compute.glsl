#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 8, local_size_y = 8) in;

layout(buffer_reference, std430, buffer_reference_align = 8) buffer Root {
    uint width;
    uint height;
    uint64_t pixels;
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Pixels {
    uint rgba[];
};

layout(push_constant) uniform Push {
    uint64_t root;
};

void main() {
    Root r = Root(root);
    uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= r.width || p.y >= r.height) return;
    vec3 color = vec3(float(p.x) / float(r.width), float(p.y) / float(r.height), 0.25);
    Pixels pix = Pixels(r.pixels);
    pix.rgba[p.y * r.width + p.x] = packUnorm4x8(vec4(color, 1.0));
}
