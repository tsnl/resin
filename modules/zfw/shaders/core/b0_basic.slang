//! Basic uniforms @ descriptor set 0.

struct BasicUniform
{
    int2 framebufferSizePx;
};

[[vk::binding(0, 0)]]
ConstantBuffer<BasicUniform> u;
