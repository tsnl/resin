//! Base module for draw_2d quad shaders.
//!
//! Provides common structures and functions used by both text and RGBA quad shaders.

//
// Uniform
//

public struct Draw2dUniform
{
    public int2 framebufferSizePx;
    public uint quadCount;
    public uint _rsv0;
};

//
// Quad
//

public struct Quad
{
    // Destination rectangle in pixels (x, y, w, h):
    public int4 dstXywhPx;
    // Source rectangle in UV coordinates (x, y, w, h):
    public float4 srcXywhUv;
    // Depth value for sorting:
    public float depth;
    // Border thickness (T, R, B, L) in pixels:
    public int borderThicknessT;
    public int borderThicknessR;
    public int borderThicknessB;
    // Padding to align color to 16-byte boundary:
    public int borderThicknessL;
    public int _pad0;
    public int _pad1;
    public int _pad2;
    // Tint color (R, G, B, A), 0.0-1.0 (linear space):
    public float4 color;
    // Border color (R, G, B, A), 0.0-1.0 (linear space):
    public float4 borderColor;
};

//
// Vertex Shader Output
//

public struct VSOutput
{
    public float4 position : SV_Position;
    public nointerpolation uint instanceId : SV_InstanceID;
    public float2 uv : TEXCOORD0;
    public float4 tintColor : TEXCOORD1;
    public float4 borderColor : TEXCOORD2;
    public int4 borderThickness : TEXCOORD3;
    public float2 positionPx : TEXCOORD4;
    public int4 dstXywhPx : TEXCOORD5;
};

//
// Common Functions
//

public uint vertexCornerIdx(uint vertexIdx)
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

public int2 getCornerPosPx(int4 dstXywhPx, uint cornerIdx)
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

public float2 getCornerUv(float4 srcXywhUv, uint cornerIdx)
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

public float2 framebufferPxToNdc(int2 posPx, int2 framebufferSizePx)
{
    float2 ndc;
    ndc.x = (float(posPx.x) / float(framebufferSizePx.x)) * 2.0 - 1.0;
    ndc.y = (float(posPx.y) / float(framebufferSizePx.y)) * 2.0 - 1.0;
    return ndc;
}

public VSOutput computeVertexOutput(
    Draw2dUniform uniform,
    Quad quad,
    uint vertexId,
    uint instanceId)
{
    uint cornerIdx = vertexCornerIdx(vertexId);
    int2 vertexPosPx = getCornerPosPx(quad.dstXywhPx, cornerIdx);
    float2 vertexUv = getCornerUv(quad.srcXywhUv, cornerIdx);

    VSOutput output;
    output.position.xy = framebufferPxToNdc(vertexPosPx, uniform.framebufferSizePx);
    output.position.z = (1.0 + quad.depth) / (1.0 + float(uniform.quadCount));
    output.position.w = 1.0;
    output.instanceId = instanceId;
    output.uv = vertexUv;
    output.tintColor = quad.color;
    output.borderColor = quad.borderColor;
    output.borderThickness = int4(
        quad.borderThicknessT,
        quad.borderThicknessR,
        quad.borderThicknessB,
        quad.borderThicknessL);
    output.positionPx = float2(vertexPosPx);
    output.dstXywhPx = quad.dstXywhPx;
    return output;
}

public bool isInBorder(VSOutput input)
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
