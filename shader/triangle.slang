// Triangle shader with hardcoded vertex data
// Vertex shader

struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexID: SV_VertexID)
{
    // Hardcoded triangle vertices (NDC space: -1 to 1)
    float3 positions[3];
    positions[0] = float3(0.0, -0.5, 0.0); // Bottom center
    positions[1] = float3(0.5, 0.5, 0.0);  // Top right
    positions[2] = float3(-0.5, 0.5, 0.0); // Top left

    // Hardcoded colors (RGB)
    float3 colors[3];
    colors[0] = float3(1.0, 0.0, 0.0); // Red
    colors[1] = float3(0.0, 1.0, 0.0); // Green
    colors[2] = float3(0.0, 0.0, 1.0); // Blue

    VSOutput output;
    output.position = float4(positions[vertexID], 1.0);
    output.color = colors[vertexID];

    return output;
}

// Fragment shader

struct PSInput
{
    float3 color : COLOR0;
};

struct PSOutput
{
    float4 color : SV_Target0;
};

[shader("fragment")]
PSOutput fragmentMain(PSInput input)
{
    PSOutput output;
    output.color = float4(input.color, 1.0);
    return output;
}
