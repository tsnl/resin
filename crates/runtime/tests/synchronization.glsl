#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430, buffer_reference_align = 16) buffer Root {
    vec4 color;
};
layout(push_constant) uniform Push { uint64_t root; } pc;

#ifdef COMPUTE
layout(local_size_x = 1) in;
void main() { Root(pc.root).color = vec4(1.0, 0.0, 0.0, 1.0); }
#endif

#ifdef VERTEX
void main() {
    const vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
#endif

#ifdef FRAGMENT
layout(location = 0) out vec4 color;
void main() { color = Root(pc.root).color; }
#endif
