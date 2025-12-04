//
// Bindings:
//

// Bind Group 0: common uniforms

[[vk::binding(0, 0)]]
ConstantBuffer<uint2> framebufferSize;

[[vk::binding(0, 1)]]
ConstantBuffer<uint2> atlasSize;

// Bind Group 1: per-batch data

struct Quad
{
    uint4x2 dstPx;          // 4 corners * 2 coords
    float4x2 srcUv;         // 4 corners * 2 coords
    uint tintColor;         // [u8; 4]: RGBA
    uint borderColor;       // [u8; 4]: RGBA
    uint borderThicknessPx; // [u8; 4]: T, R, B, L
    uint cornerRadius;      // corner radius
    uint height;            // height
};

[[vk::binding(1, 0)]]
StructuredBuffer<int4x2> dstPxArray;

[[vk::binding(1, 1)]]
StructuredBuffer<int4x2> srcUvArray;

[[vk::binding(1, 2)]]
StructuredBuffer<uint> tintColorArray;

[[vk::binding(1, 3)]]
StructuredBuffer<uint> borderColorArray;

[[vk::binding(1, 4)]]
StructuredBuffer<int4> borderThicknessPxArray;

[[vk::binding(1, 5)]]
StructuredBuffer<int4> cornerRadius;

[[vk::binding(1, 6)]]
StructuredBuffer<int> heightArray;

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
