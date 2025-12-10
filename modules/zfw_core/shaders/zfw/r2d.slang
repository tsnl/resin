//
// Bindings:
//

// Bind Group 0: common uniforms

struct R2dUniform
{
    int2 framebufferSizePx;
    float maxHeight;
    uint _rsv0; // reserved for alignment
}

[[vk::binding(0, 0)]]
ConstantBuffer<R2dUniform> uniform;

// Bind Group 1: per-batch data

struct R2dQuad
{
    Array<int2, 4> dstPx;   // 4 corners * 2 coords: TL, TR, BR, BL
    Array<float2, 4> srcUv; // 4 corners * 2 coords: TL, TR, BR, BL
    float4 tintColor;       // RGBA
    float4 borderColor;     // [u8; 4]: RGBA
    int4 borderThicknessPx; // [u8; 4]: T, R, B, L
    int cornerRadius;       // corner radius
    float height;           // height
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
};

[shader("vertex")]
VSOutput vertexMain(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID)
{
    R2dQuad quad = quads[instanceId];

    // TODO: handle non-0 border thickness.

    VSOutput output;
    uint cornerIdx = vertexCornerIdx(vertexId);
    output.position.xy = framebufferPxToNdcVec2(quad.dstPx[cornerIdx]);
    output.position.z = 1.0 - ((quad.height + 1.0) / (uniform.maxHeight + 1.0));
    output.position.w = 1.0;
    output.uv = quad.srcUv[cornerIdx];
    output.tintColor = quad.tintColor;
    output.borderColor = quad.borderColor;
    output.borderThicknessPx = quad.borderThicknessPx;
    output.cornerRadius = int4(quad.cornerRadius, quad.cornerRadius, quad.cornerRadius, quad.cornerRadius);
    return output;
}

uint vertexCornerIdx(uint vertexIdx)
{
    // Corners: 0=TL, 1=TR, 2=BR, 3=BL
    // Vertices: (0=TL, 1=TR, 2=BR), (0=TL, 2=BR, 3=BL)
    // Mapping:
    // * vertexIdx: 0 -> cornerIdx: 0 = (v=0 % 3)=0 + (v=0 // 4)=0
    // * vertexIdx: 1 -> cornerIdx: 1 = (v=1 % 3)=1 + (v=1 // 4)=0
    // * vertexIdx: 2 -> cornerIdx: 2 = (v=2 % 3)=2 + (v=2 // 4)=0
    // * vertexIdx: 3 -> cornerIdx: 0 = (v=3 % 3)=0 + (v=3 // 4)=0
    // * vertexIdx: 4 -> cornerIdx: 2 = (v=4 % 3)=1 + (v=4 // 4)=1
    // * vertexIdx: 5 -> cornerIdx: 3 = (v=5 % 3)=2 + (v=5 // 4)=1
    return vertexIdx % 3 + vertexIdx / 4;
}

float2 framebufferPxToNdcVec2(int2 posPx)
{
    float2 ndc;
    ndc.x = (float(posPx.x) / float(uniform.framebufferSizePx.x)) * 2.0 - 1.0;
    ndc.y = (float(posPx.y) / float(uniform.framebufferSizePx.y)) * 2.0 - 1.0;
    return ndc;
}

//
// Fragment Shader
//

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    // TODO: implement border rendering and corner radius

    float4 color = atlasTexture.SampleLevel(input.uv, 0);
    return color * input.tintColor;
}
