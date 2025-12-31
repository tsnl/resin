//! 2D quad shader for Draw2dRenderer.
//!
//! Renders textured 2D quads with support for borders and alpha blending.
//!
//! Bindings:
//! - Descriptor Set 0:
//!   - binding 0: Uniform buffer with viewport size
//! - Descriptor Set 1:
//!   - binding 0: Storage buffer with quad data
//!   - binding 1: Sampled image (texture)
//!   - binding 2: Sampler

//
// Uniform (Descriptor Set 0, Binding 0)
//

struct Uniform
{
    float2 viewportSizePx;
    float2 _rsv;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniform> uniform_;

//
// Quad (Descriptor Set 1, Binding 0)
//

struct Quad
{
    int4 dstXywhPx;
    float4 srcXywhUv;
    float4 fillColor;
    int4 borderThickness; // TRBL
    float4 borderColor;
    float depth;
    uint flags;
    int2 _rsv;
};

[[vk::binding(0, 1)]]
StructuredBuffer<Quad> quads;

[[vk::binding(1, 1)]]
Texture2D<float4> quadTexture;

[[vk::binding(2, 1)]]
SamplerState quadSampler;

//
// Vertex Shader Output
//

struct VSOutput
{
    float4 position : SV_Position;
    nointerpolation uint instanceId : TEXCOORD0;
    float2 uv : TEXCOORD1;
    nointerpolation float4 fillColor : TEXCOORD2;
    nointerpolation float4 borderColor : TEXCOORD3;
    nointerpolation int4 borderThickness : TEXCOORD4;
    float2 positionPx : TEXCOORD5;
    nointerpolation int4 dstXywhPx : TEXCOORD6;
};

//
// Helpers
//

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

int2 getCornerPosPx(int4 dstXywhPx, uint cornerIdx)
{
    // Corners: 0=TL, 1=TR, 2=BR, 3=BL
    int x = dstXywhPx.x;
    int y = dstXywhPx.y;
    int w = dstXywhPx.z;
    int h = dstXywhPx.w;

    switch (cornerIdx)
    {
    case 0:
        return int2(x, y); // TL
    case 1:
        return int2(x + w, y); // TR
    case 2:
        return int2(x + w, y + h); // BR
    case 3:
        return int2(x, y + h); // BL
    default:
        return int2(0, 0);
    }
}

float2 getCornerUv(float4 srcXywhUv, uint cornerIdx)
{
    // Corners: 0=TL, 1=TR, 2=BR, 3=BL
    float u = srcXywhUv.x;
    float v = srcXywhUv.y;
    float w = srcXywhUv.z;
    float h = srcXywhUv.w;

    switch (cornerIdx)
    {
    case 0:
        return float2(u, v); // TL
    case 1:
        return float2(u + w, v); // TR
    case 2:
        return float2(u + w, v + h); // BR
    case 3:
        return float2(u, v + h); // BL
    default:
        return float2(0.0, 0.0);
    }
}

float2 framebufferPxToNdc(int2 posPx, float2 framebufferSizePx)
{
    float2 ndc;
    ndc.x = (float(posPx.x) / framebufferSizePx.x) * 2.0 - 1.0;
    ndc.y = (float(posPx.y) / framebufferSizePx.y) * 2.0 - 1.0;
    return ndc;
}

bool isInBorder(VSOutput input)
{
    // Check if the fragment is in the border region.
    // Border is defined by borderThickness: T, R, B, L.
    int4 bt = input.borderThickness; // T, R, B, L
    int4 dst = input.dstXywhPx;

    // Inner rect bounds:
    int innerLeft = dst.x + bt.w;           // L
    int innerRight = dst.x + dst.z - bt.y;  // R
    int innerTop = dst.y + bt.x;            // T
    int innerBottom = dst.y + dst.w - bt.z; // B

    int2 pos = int2(input.positionPx);

    // If inside inner rect, not in border
    if (pos.x >= innerLeft && pos.x < innerRight &&
        pos.y >= innerTop && pos.y < innerBottom)
    {
        return false;
    }
    return true;
}

//
// Vertex Shader
//

[shader("vertex")]
VSOutput vertexMain(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID)
{
    Quad quad = quads[instanceId];

    uint cornerIdx = vertexCornerIdx(vertexId);
    int2 vertexPosPx = getCornerPosPx(quad.dstXywhPx, cornerIdx);
    float2 vertexUv = getCornerUv(quad.srcXywhUv, cornerIdx);

    VSOutput output;
    output.position.xy = framebufferPxToNdc(vertexPosPx, uniform_.viewportSizePx);
    output.position.z = quad.depth / 65536.0; // Normalize depth to [0, 1]
    output.position.w = 1.0;
    output.instanceId = instanceId;
    output.uv = vertexUv;
    output.fillColor = quad.fillColor;
    output.borderColor = quad.borderColor;
    output.borderThickness = quad.borderThickness;
    output.positionPx = float2(vertexPosPx);
    output.dstXywhPx = quad.dstXywhPx;

    return output;
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

    // Sample the texture
    float4 texColor = quadTexture.Sample(quadSampler, input.uv);

    // Tint with the quad's fill color (multiply)
    return texColor * input.fillColor;
}
