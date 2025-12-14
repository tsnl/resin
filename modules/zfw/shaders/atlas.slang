struct UvRect
{
    float x, y, w, h;
};

// Bind Group 1: Renderer Global Set

[[vk::binding(0, 1)]]
Texture2D atlasMono[];

[[vk::binding(1, 1)]]
Texture2D atlasRgba[];

[[vk::binding(2, 1)]]
StructuredBuffer<UvRect> rectsMono;

[[vk::binding(3, 1)]]
StructuredBuffer<UvRect> rectsRgba;

[[vk::binding(4, 1)]]
SamplerState sampler;

float sampleAtlasMono(uint imageId, float2 uv, uint atlasSizePx)
{
    UvRect rect = rectsMono[imageId];
    SamplePageAndUv pageAndUv = getSamplePageAndUv(rect, uv, atlasSizePx);
    return atlasMono[pageAndUv.pageIndex].SampleLevel(sampler, pageAndUv.uv, 0.0).r;
}

float4 sampleAtlasRgba(uint imageId, float2 uv, uint atlasSizePx)
{
    UvRect rect = rectsRgba[imageId];
    SamplePageAndUv pageAndUv = getSamplePageAndUv(rect, uv, atlasSizePx);
    return atlasRgba[pageAndUv.pageIndex].SampleLevel(sampler, pageAndUv.uv, 0.0);
}

struct SamplePageAndUv
{
    uint pageIndex;
    float2 uv;
};
SamplePageAndUv getSamplePageAndUv(UvRect rect, float2 uv, uint atlasSizePx)
{
    // rect.y contains page index in integer part
    uint pageIndex = uint(rect.y);
    float y_uv = frac(rect.y);

    // Compute final UV in the page
    // rect.x, y_uv is the top-left corner of the rect in the page
    // rect.w, rect.h is the size of the rect
    float2 finalUv = float2(rect.x, y_uv) + uv * float2(rect.w, rect.h);

    // Clamp UVs to avoid bleeding
    // We clamp to the center of the edge texels of the sprite.
    // The sprite in UV space is [rect.x, rect.x + rect.w] x [y_uv, y_uv + rect.h]
    // In texels, this is [rect.x * size, (rect.x + rect.w) * size]
    // We want to clamp to [min + 0.5, max - 0.5] in texel space.
    // In UV space, this is [min + 0.5/size, max - 0.5/size].

    float halfTexel = 0.5 / float(atlasSizePx);
    float2 minUv = float2(rect.x, y_uv) + halfTexel;
    float2 maxUv = float2(rect.x + rect.w, y_uv + rect.h) - halfTexel;

    finalUv = clamp(finalUv, minUv, maxUv);

    SamplePageAndUv result;
    result.pageIndex = pageIndex;
    result.uv = finalUv;
    return result;
}
