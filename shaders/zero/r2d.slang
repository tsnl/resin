//
// Bindings:
//

// Bind Group 0: common uniforms

struct R2dUniform
{
    uint2 framebufferSize;
    uint2 atlasSize;
}

[[vk::binding(0, 0)]]
ConstantBuffer<R2dUniform> uniform;

// Bind Group 1: per-batch data

struct R2dQuad
{
    int2x4 dstPx;           // 4 corners * 2 coords
    float2x4 srcUv;         // 4 corners * 2 coords
    float4 tintColor;       // RGBA
    float4 borderColor;     // [u8; 4]: RGBA
    int4 borderThicknessPx; // [u8; 4]: T, R, B, L
    int cornerRadius;       // corner radius
    int height;             // height
    int _rsv0;              // reserved for alignment
    int _rsv1;              // reserved for alignment
};

[[vk::binding(0, 1)]]
Sampler2D atlasTexture;

[[vk::binding(1, 1)]]
StructuredBuffer<R2dQuad> quads;

//
// Vertex Shader
//

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 tintColor : TEXCOORD1;
    float4 borderColor : TEXCOORD2;
    int4 borderThicknessPx : TEXCOORD3;
    int4 cornerRadius : TEXCOORD4;
    int height : TEXCOORD5;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID)
{
    VSOutput output;

    // TODO

    return output;
}
