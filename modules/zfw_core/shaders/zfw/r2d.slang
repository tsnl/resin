//
// Bindings:
//

// Bind Group 0: common uniforms

struct R2dUniform
{
    int2 framebufferSizePx;
    float totalQuadCount;
    uint _rsv0; // reserved for alignment
}

[[vk::binding(0, 0)]]
ConstantBuffer<R2dUniform> commonUniform;

// Bind Group 1: per-batch data

struct R2dQuad
{
    Array<int2, 4> dstPx;   // 4 corners * 2 coords: TL, TR, BR, BL
    Array<float2, 4> srcUv; // 4 corners * 2 coords: TL, TR, BR, BL
    float4 tintColor;       // RGBA
    float4 borderColor;     // RGBA
    int4 borderThicknessPx; // T, R, B, L
    float height;           // height
    int _rsv0;              // reserved for alignment
    int _rsv1;              // reserved for alignment
    int _rsv2;              // reserved for alignment
};

struct R2dBatchUniform
{
    uint instanceCount;
    uint _rsv0; // reserved for alignment
    uint _rsv1; // reserved for alignment
    uint _rsv2; // reserved for alignment
};

[[vk::binding(0, 1)]]
Sampler2D atlasTexture;

[[vk::binding(1, 1)]]
StructuredBuffer<R2dQuad> quads;

[[vk::binding(2, 1)]]
ConstantBuffer<R2dBatchUniform> batchUniform;

//
// Vertex Shader
//

struct VSOutput
{
    float4 position : SV_Position;
    nointerpolation uint instanceId : SV_InstanceID;
    float2 uv : TEXCOORD0;
    float4 tintColor : TEXCOORD1;
    float4 borderColor : TEXCOORD2;
    int4 borderThicknessPx : TEXCOORD3;
    float2 positionPx : TEXCOORD4;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID)
{
    // Fetch quad data.
    R2dQuad quad = quads[instanceId];

    // Compute quad corner in pixels:
    uint cornerIdx = vertexCornerIdx(vertexId);
    int2 vertexPosPx = getCornerCoordsPxArray(quad, 1)[cornerIdx];

    // Emit vertex data.
    VSOutput output;
    output.position.xy = framebufferPxToNdcVec2(vertexPosPx);
    output.position.z = (1.0 + quad.height) / (1.0 + commonUniform.totalQuadCount);
    output.position.w = 1.0;
    output.instanceId = instanceId;
    output.uv = quad.srcUv[cornerIdx];
    output.tintColor = quad.tintColor;
    output.borderColor = quad.borderColor;
    output.borderThicknessPx = quad.borderThicknessPx;
    output.positionPx = float2(vertexPosPx);
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
    ndc.x = (float(posPx.x) / float(commonUniform.framebufferSizePx.x)) * 2.0 - 1.0;
    ndc.y = (float(posPx.y) / float(commonUniform.framebufferSizePx.y)) * 2.0 - 1.0;
    return ndc;
}

//
// Fragment Shader
//

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    R2dQuad quad = quads[input.instanceId];

    int2 quadHalfSizePx = (quad.dstPx[2] - quad.dstPx[0]) / 2;
    int2 quadCenterPx = quad.dstPx[0] + quadHalfSizePx;

    const int2 innerRectCoordsPx[4] = getCornerCoordsPxArray(quad, 0);

    // Fragments inside the inner rect are not in the border.
    // Fragments between inner and outer rects are in the border.
    if (intersectRectPoint(innerRectCoordsPx, int2(input.positionPx)))
    {
        float4 color = atlasTexture.SampleLevel(input.uv, 0);
        return color * input.tintColor;
    }
    else
    {
        return quad.borderColor;
    }
}

//
// Common:
//

int2[4] getCornerEgressPxArray(R2dQuad quad)
{
    // Precompute corner border egress in pixels.
    const int2 cornerBorderEgressPx[4] = {
        int2(-quad.borderThicknessPx[3], -quad.borderThicknessPx[0]), // TL
        int2(+quad.borderThicknessPx[1], -quad.borderThicknessPx[0]), // TR
        int2(+quad.borderThicknessPx[1], +quad.borderThicknessPx[2]), // BR
        int2(-quad.borderThicknessPx[3], +quad.borderThicknessPx[2]), // BL
    };
    return cornerBorderEgressPx;
}

int2[4] getCornerCoordsPxArray(R2dQuad quad, int egressScalar)
{
    const int2 egressPx[4] = getCornerEgressPxArray(quad);
    const int2 cornerCoordsPx[4] = {
        quad.dstPx[0] + egressScalar * egressPx[0], // TL
        quad.dstPx[1] + egressScalar * egressPx[1], // TR
        quad.dstPx[2] + egressScalar * egressPx[2], // BR
        quad.dstPx[3] + egressScalar * egressPx[3], // BL
    };
    return cornerCoordsPx;
}

bool intersectRectPoint(int2[4] corners, int2 point)
{
    return (point.x >= corners[0].x && point.x < corners[1].x &&
            point.y >= corners[0].y && point.y < corners[2].y);
}
