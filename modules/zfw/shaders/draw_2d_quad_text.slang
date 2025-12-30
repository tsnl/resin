//! Text quad shader for draw_2d.
//!
//! Binds a monochrome texture atlas where the texture gives per-pixel alpha.
//! The alpha is tinted by the quad's RGBA color.

import draw_2d_quad_base;

//
// Bindings (Descriptor Set 0)
//

[[vk::binding(0, 0)]]
ConstantBuffer<Draw2dUniform> uniform;

[[vk::binding(1, 0)]]
StructuredBuffer<Quad> quads;

[[vk::binding(2, 0)]]
Texture2D<float> atlasTexture;

[[vk::binding(3, 0)]]
SamplerState atlasSampler;

//
// Vertex Shader
//

[shader("vertex")]
VSOutput vertexMain(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID)
{
    Quad quad = quads[instanceId];
    return computeVertexOutput(uniform, quad, vertexId, instanceId);
}

//
// Fragment Shader
//

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    // Check if we're in the border region
    if (isInBorder(input))
    {
        return input.borderColor;
    }

    // Sample the monochrome atlas (gives alpha value)
    float alpha = atlasTexture.Sample(atlasSampler, input.uv).r;

    // The quad's tint color provides RGB, multiply alpha
    float4 color = input.tintColor;
    color.a *= alpha;

    return color;
}
