#version 460

#ifdef VERTEX
layout(location = 0) out vec3 v_color;

layout(push_constant) uniform Push {
    uvec2 root;
};

void main() {
    const vec2 pos[3] = vec2[](
        vec2(0.0, -0.75),
        vec2(-0.7, 0.6),
        vec2(0.7, 0.6)
    );
    const vec3 col[3] = vec3[](
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0)
    );
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
    v_color = col[gl_VertexIndex];
}
#endif

#ifdef FRAGMENT
layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 o_color;

layout(push_constant) uniform Push {
    uvec2 root;
};

void main() {
    o_color = vec4(v_color, 1.0);
}
#endif
