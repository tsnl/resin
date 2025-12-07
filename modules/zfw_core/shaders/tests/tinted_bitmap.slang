// Tinted quad shader - samples a texture and applies a color tint

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Uniform buffer for the tint color
struct TintParams
{
    float4 tintColor;
};

[[vk::binding(0, 0)]]
ConstantBuffer<TintParams> tintParams;

[[vk::binding(1, 0)]]
Sampler2D textureSampler;

[shader("vertex")]
VSOutput vertexMain(uint vertexID: SV_VertexID)
{
    // Full-screen quad using two triangles (6 vertices)
    // Triangle 1: (0, 1, 2), Triangle 2: (2, 1, 3)
    // Vertices: 0=top-left, 1=bottom-left, 2=top-right, 3=bottom-right
    float2 positions[6];
    positions[0] = float2(-1.0, -1.0); // top-left (NDC)
    positions[1] = float2(-1.0, 1.0);  // bottom-left
    positions[2] = float2(1.0, -1.0);  // top-right
    positions[3] = float2(1.0, -1.0);  // top-right
    positions[4] = float2(-1.0, 1.0);  // bottom-left
    positions[5] = float2(1.0, 1.0);   // bottom-right

    // UVs: map NDC (-1 to 1) to UV (0 to 1)
    float2 uvs[6];
    uvs[0] = float2(0.0, 0.0); // top-left
    uvs[1] = float2(0.0, 1.0); // bottom-left
    uvs[2] = float2(1.0, 0.0); // top-right
    uvs[3] = float2(1.0, 0.0); // top-right
    uvs[4] = float2(0.0, 1.0); // bottom-left
    uvs[5] = float2(1.0, 1.0); // bottom-right

    VSOutput output;
    output.position = float4(positions[vertexID], 0.0, 1.0);
    output.uv = uvs[vertexID];

    return output;
}

// Fragment shader

struct PSInput
{
    float2 uv : TEXCOORD0;
};

struct PSOutput
{
    float4 color : SV_Target0;
};

[shader("fragment")]
PSOutput fragmentMain(PSInput input)
{
    // Sample the texture
    float4 texColor = textureSampler.Sample(input.uv);

    // Apply tint by multiplying with tint color
    PSOutput output;
    output.color = texColor * tintParams.tintColor;

    return output;
}
