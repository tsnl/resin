//! RGBA quad shader for draw_2d.
//!
//! Binds an RGBA texture atlas where the texture gives per-pixel RGBA color.
//! The color is tinted (multiplied) by the quad's RGBA color.

import draw_2d_quad_base;

//
// Bindings (Descriptor Set 0)
//

[[vk::binding(0, 0)]]
ConstantBuffer<Draw2dUniform> uniform;

[[vk::binding(1, 0)]]
StructuredBuffer<Quad> quads;

[[vk::binding(2, 0)]]
Texture2D<float4> atlasTexture;

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

    // Sample the RGBA atlas
    float4 texColor = atlasTexture.Sample(atlasSampler, input.uv);

    // Tint with the quad's color (multiply)
    return texColor * input.tintColor;
}
