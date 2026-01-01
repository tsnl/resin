//! Ray tracing shader for draw_3d_v2
//!
//! Implements basic ray tracing with texture sampling.
//!
//! Descriptor Sets:
//!   Set 0: Output image (storage image)
//!   Set 1: Acceleration structure (TLAS)
//!   Set 2: Material (color texture, sampler)

//
// Structures
//

struct CameraUniform
{
    float4x4 projectionView;
    float4x4 inverseProjectionView;
    float3 cameraPosition;
    uint _pad0;
};

struct Payload
{
    float3 color;
    float3 barycentrics;
};

//
// Descriptor Set 0: Output image
//

[[vk::binding(0, 0)]]
RWTexture2D<float4> outputImage;

//
// Descriptor Set 1: Acceleration structure
//

[[vk::binding(0, 1)]]
RaytracingAccelerationStructure topLevelAS;

//
// Descriptor Set 2: Camera uniform
//

[[vk::binding(0, 2)]]
ConstantBuffer<CameraUniform> cameraUniform;

//
// Descriptor Set 3: Material texture
//

[[vk::binding(0, 3)]]
Texture2D colorTexture;

[[vk::binding(1, 3)]]
SamplerState colorSampler;

//
// Ray Generation Shader
//

[shader("raygeneration")]
void main()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    // Compute normalized device coordinates
    float2 ndc = (float2(launchID) + 0.5) / float2(launchSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y

    // Transform NDC to world space ray
    float4 origin = mul(cameraUniform.inverseProjectionView, float4(ndc, 0.0, 1.0));
    float4 target = mul(cameraUniform.inverseProjectionView, float4(ndc, 1.0, 1.0));
    origin.xyz /= origin.w;
    target.xyz /= target.w;

    float3 direction = normalize(target.xyz - origin.xyz);

    // Setup ray description
    RayDesc ray;
    ray.Origin = origin.xyz;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    // Trace ray
    Payload payload;
    payload.color = float3(0.0, 0.0, 0.0);
    payload.barycentrics = float3(0.0, 0.0, 0.0);

    TraceRay(
        topLevelAS,
        RAY_FLAG_NONE,
        0xFF, // cull mask
        0,    // sbt ray contribution offset
        0,    // sbt ray contribution stride
        0,    // miss index
        ray,
        payload);

    // Write output
    outputImage[launchID] = float4(payload.color, 1.0);
}

//
// Miss Shader
//

[shader("miss")]
void missMain(inout Payload payload)
{
    // Simple sky gradient background
    float t = DispatchRaysIndex().y / float(DispatchRaysDimensions().y);
    payload.color = lerp(float3(0.5, 0.7, 1.0), float3(1.0, 1.0, 1.0), t);
}

//
// Closest Hit Shader
//

[shader("closesthit")]
void closestHitMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    // Get barycentric coordinates
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                                 attribs.barycentrics.x,
                                 attribs.barycentrics.y);

    payload.barycentrics = barycentrics;

    // Simple texture coordinate computation from barycentrics
    // In a real application, you'd want to interpolate actual vertex UVs
    float2 uv = float2(attribs.barycentrics.x, attribs.barycentrics.y);

    // Sample color texture
    float3 color = colorTexture.SampleLevel(colorSampler, uv, 0).rgb;

    // Apply simple lighting based on barycentrics (placeholder for proper normals)
    float lighting = dot(barycentrics, float3(0.33, 0.33, 0.33)) * 0.5 + 0.5;

    payload.color = color * lighting;
}
