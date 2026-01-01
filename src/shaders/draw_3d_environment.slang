//! Environment map background rendering shader for draw_3d.
//!
//! Renders an equirectangular environment map as a background.
//! Uses the camera's inverse projection-view matrix to compute ray directions
//! for each pixel, then samples the environment map.
//!
//! Descriptor Sets:
//!   Set 0: Camera uniform (inverse projection-view matrix, camera position)
//!   Set 1: Environment map (texture, sampler)

//
// Structures
//

struct CameraUniform
{
    float4x4 inverseProjectionView;
    float3 cameraPosition;
    float _pad0;
};

struct VSOutput
{
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

//
// Descriptor Set 0: Camera
//

[[vk::binding(0, 0)]]
ConstantBuffer<CameraUniform> camera;

//
// Descriptor Set 1: Environment Map
//

[[vk::binding(0, 1)]]
Texture2D<float4> environmentMap;

[[vk::binding(1, 1)]]
SamplerState environmentSampler;

//
// Constants
//

static const float PI = 3.14159265359;

//
// Vertex Shader
//
// Renders a fullscreen quad using vertex ID:
// - Vertex 0: (-1, -1) bottom-left
// - Vertex 1: ( 3, -1) far right
// - Vertex 2: (-1,  3) far top
// This creates a triangle that covers the entire screen.
//

[shader("vertex")]
VSOutput vertexMain(uint vertexId: SV_VertexID)
{
    // Generate fullscreen triangle vertices
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    float2 ndc = uv * 2.0 - 1.0;

    VSOutput output;
    // Place at far plane (z=0 in reverse-Z, w=1)
    output.position = float4(ndc, 0.0, 1.0);
    output.ndc = ndc;
    return output;
}

//
// Fragment Shader
//

// Convert a 3D direction to equirectangular UV coordinates
float2 directionToEquirectangular(float3 dir)
{
    // Normalize direction
    float3 d = normalize(dir);

    // Compute spherical coordinates
    // phi: azimuthal angle around Y axis (0 to 2*PI)
    // theta: polar angle from Y axis (0 to PI)
    float phi = atan2(d.z, d.x);               // -PI to PI
    float theta = acos(clamp(d.y, -1.0, 1.0)); // 0 to PI

    // Map to UV coordinates
    // u: phi mapped from [-PI, PI] to [0, 1]
    // v: theta mapped from [0, PI] to [0, 1]
    float u = (phi + PI) / (2.0 * PI);
    float v = theta / PI;

    return float2(u, v);
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    // Reconstruct world-space ray direction from NDC coordinates
    // NDC is at the near plane (z=1 in reverse-Z after perspective divide)
    float4 nearNdc = float4(input.ndc, 1.0, 1.0);

    // Transform from NDC to world space using inverse projection-view
    float4 nearWorld = mul(camera.inverseProjectionView, nearNdc);
    nearWorld /= nearWorld.w;

    // Ray direction from camera position to near plane point
    float3 rayDir = normalize(nearWorld.xyz - camera.cameraPosition);

    // Sample environment map using equirectangular projection
    float2 uv = directionToEquirectangular(rayDir);
    float4 envColor = environmentMap.Sample(environmentSampler, uv);

    return float4(envColor.rgb, 1.0);
}
