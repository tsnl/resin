//! Atlas texture sampling utilities @ descriptor set 1.
//!
//! Bindless texture arrays are used to store the atlas pages.
//! Each image in the atlas is represented by a rectangle in one of the pages.
//! The rectangle UV coordinates and dimensions are stored in structured buffers.

struct UvRect
{
    float x;
    float y;
    float w;
    float h;
};

[[vk::binding(0, 1)]]
Texture2D atlasMono[];

[[vk::binding(1, 1)]]
Texture2D atlasRgba[];

[[vk::binding(2, 1)]]
StructuredBuffer<UvRect> rectsMono;

[[vk::binding(3, 1)]]
StructuredBuffer<UvRect> rectsRgba;

[[vk::binding(4, 1)]]
SamplerState nearestSampler;

[[vk::binding(5, 1)]]
SamplerState linearSampler;

float sampleAtlasMono(uint imageId, float2 uv, bool useLinear)
{
    UvRect rect = rectsMono[imageId];
    uint pageIndex = uvToPageIndex(uv);
    float2 pageUv = uvToPageUv(rect, uv);
    if (useLinear)
    {
        return atlasMono[pageIndex].SampleLevel(linearSampler, pageUv, 0.0).r;
    }
    else
    {
        return atlasMono[pageIndex].SampleLevel(nearestSampler, pageUv, 0.0).r;
    }
}

float4 sampleAtlasRgba(uint imageId, float2 uv, bool useLinear)
{
    UvRect rect = rectsRgba[imageId];
    uint pageIndex = uvToPageIndex(uv);
    float2 pageUv = uvToPageUv(rect, uv);
    if (useLinear)
    {
        return atlasRgba[pageIndex].SampleLevel(linearSampler, pageUv, 0.0);
    }
    else
    {
        return atlasRgba[pageIndex].SampleLevel(nearestSampler, pageUv, 0.0);
    }
}

uint uvToPageIndex(float2 uv)
{
    return uint(uv.y);
}

float2 uvToPageUv(UvRect rect, float2 uv)
{
    float2 st = float2(uv.x, frac(uv.y));
    return float2(rect.x, rect.y) + st * float2(rect.w, rect.h);
}
