//! GUI present shader for GuiWindow.
//!
//! A simple fullscreen pass that copies a source image to the output 1:1.
//! Used to present the 2D render target to the swapchain.
//!
//! Bindings:
//! - Descriptor Set 0:
//!   - binding 0: Sampled image (source texture)
//!   - binding 1: Sampler

[[vk::binding(0, 0)]]
Texture2D<float4> sourceTexture;

[[vk::binding(1, 0)]]
SamplerState sourceSampler;

//
// Vertex Shader Output
//

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

//
// Vertex Shader
//
// Generates a fullscreen triangle without any vertex buffer.
// Uses vertex IDs 0, 1, 2 to generate positions that cover the screen.
//

[shader("vertex")]
VSOutput vertexMain(uint vertexId: SV_VertexID)
{
    VSOutput output;

    // Generate fullscreen triangle vertices:
    // vertexId 0: (-1, -1) -> uv (0, 0)
    // vertexId 1: ( 3, -1) -> uv (2, 0)
    // vertexId 2: (-1,  3) -> uv (0, 2)
    // This creates a triangle that covers the entire screen when clipped.
    float2 pos;
    pos.x = (vertexId == 1) ? 3.0 : -1.0;
    pos.y = (vertexId == 2) ? 3.0 : -1.0;

    output.position = float4(pos, 0.0, 1.0);

    // Convert from NDC [-1, 1] to UV [0, 1]
    output.uv.x = (pos.x + 1.0) * 0.5;
    output.uv.y = (pos.y + 1.0) * 0.5;

    return output;
}

//
// Fragment Shader
//

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return sourceTexture.Sample(sourceSampler, input.uv);
}
