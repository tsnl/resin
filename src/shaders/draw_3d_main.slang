//! 3D mesh rendering shader for draw_3d with PBR and IBL support.
//!
//! Implements physically-based rendering with image-based lighting.
//! Uses Cook-Torrance BRDF with GGX distribution for specular reflections.
//! IBL specular uses importance sampling with Hammersley sequence.
//!
//! Descriptor Sets:
//!   Set 0: Camera uniform (projection-view matrix, inverse PV, camera position, IBL samples)
//!   Set 1: Instance transforms (storage buffer of model matrices)
//!   Set 2: Material (tint, color/normal/metalness/roughness textures, sampler)
//!   Set 3: Environment map (texture, sampler)

//
// Constants
//

static const float PI = 3.14159265359;
static const float EPSILON = 1e-6;

//
// Structures
//

struct CameraUniform
{
    float4x4 projectionView;
    float4x4 inverseProjectionView;
    float3 cameraPosition;
    uint iblSamples;
};

struct MaterialTint
{
    float4 colorTint;
    float metalnessTint;
    float roughnessTint;
    float2 _pad0;
};

struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texcoord0 : TEXCOORD0;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 worldPosition : TEXCOORD0;
    float3 worldNormal : TEXCOORD1;
    float2 texcoord0 : TEXCOORD2;
};

//
// Descriptor Set 0: Camera
//

[[vk::binding(0, 0)]]
ConstantBuffer<CameraUniform> camera;

//
// Descriptor Set 1: Instance Transforms
//

[[vk::binding(0, 1)]]
StructuredBuffer<float4x4> instanceTransforms;

//
// Descriptor Set 2: Material
//

[[vk::binding(0, 2)]]
ConstantBuffer<MaterialTint> materialTint;

[[vk::binding(1, 2)]]
Texture2D<float4> colorTexture;

[[vk::binding(2, 2)]]
Texture2D<float4> normalTexture;

[[vk::binding(3, 2)]]
Texture2D<float4> metalnessTexture;

[[vk::binding(4, 2)]]
Texture2D<float4> roughnessTexture;

[[vk::binding(5, 2)]]
SamplerState materialSampler;

//
// Descriptor Set 3: Environment Map
//

[[vk::binding(0, 3)]]
Texture2D<float4> environmentMap;

[[vk::binding(1, 3)]]
SamplerState environmentSampler;

//
// Utility Functions
//

// Convert a 3D direction to equirectangular UV coordinates
float2 directionToEquirectangular(float3 dir)
{
    float3 d = normalize(dir);
    float phi = atan2(d.z, d.x);
    float theta = acos(clamp(d.y, -1.0, 1.0));
    float u = (phi + PI) / (2.0 * PI);
    float v = theta / PI;
    return float2(u, v);
}

// Sample environment map with direction
float3 sampleEnvironment(float3 dir)
{
    float2 uv = directionToEquirectangular(dir);
    return environmentMap.SampleLevel(environmentSampler, uv, 0).rgb;
}

//
// Low-discrepancy sequence functions
//

// Van Der Corput sequence for Hammersley
float radicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// Hammersley sequence for quasi-Monte Carlo sampling
float2 hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), radicalInverse_VdC(i));
}

//
// PBR Functions
//

// GGX/Trowbridge-Reitz normal distribution function
float distributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return a2 / max(denom, EPSILON);
}

// Schlick-GGX geometry function (single direction)
float geometrySchlickGGX(float NdotV, float roughness)
{
    // For IBL, use k = (roughness^2) / 2
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / max(denom, EPSILON);
}

// Smith's geometry function (both directions)
float geometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Fresnel-Schlick approximation
float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick with roughness for IBL
float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    float3 oneMinusRoughness = float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness);
    return F0 + (max(oneMinusRoughness, F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// GGX importance sampling - generates sample direction in tangent space
float3 importanceSampleGGX(float2 Xi, float3 N, float roughness)
{
    float a = roughness * roughness;

    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // Spherical to Cartesian (tangent space)
    float3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // Tangent space to world space
    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    float3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

// Approximate environment BRDF integration
// Based on curve fitting from Unreal Engine 4
float2 approximateEnvBRDF(float NdotV, float roughness)
{
    // Analytical approximation of the BRDF integration
    // From: "Real Shading in Unreal Engine 4" by Brian Karis
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
    return AB;
}

//
// IBL Functions
//

// Compute diffuse IBL contribution using simple hemisphere sampling
float3 computeDiffuseIBL(float3 N)
{
    // For diffuse, we'd ideally use a pre-convoluted irradiance map
    // For now, use a simple approximation by sampling in the normal direction
    // with some softening
    float3 irradiance = float3(0.0, 0.0, 0.0);

    // Sample environment in normal direction as rough approximation
    // A proper implementation would pre-compute an irradiance cubemap
    float3 envSample = sampleEnvironment(N);

    // Add some samples around the hemisphere for better approximation
    float3 up = abs(N.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    const int DIFFUSE_SAMPLES = 8;
    float totalWeight = 0.0;

    for (int i = 0; i < DIFFUSE_SAMPLES; i++)
    {
        float2 Xi = hammersley(uint(i), uint(DIFFUSE_SAMPLES));
        float phi = 2.0 * PI * Xi.x;
        float cosTheta = sqrt(1.0 - Xi.y);
        float sinTheta = sqrt(Xi.y);

        // Cosine-weighted hemisphere sample
        float3 sampleDir = tangent * (cos(phi) * sinTheta) +
                           bitangent * (sin(phi) * sinTheta) +
                           N * cosTheta;
        sampleDir = normalize(sampleDir);

        float NdotL = max(dot(N, sampleDir), 0.0);
        irradiance += sampleEnvironment(sampleDir) * NdotL;
        totalWeight += NdotL;
    }

    return irradiance / max(totalWeight, EPSILON);
}

// Compute specular IBL using importance sampling
float3 computeSpecularIBL(float3 N, float3 V, float3 F0, float roughness, uint numSamples)
{
    float3 R = reflect(-V, N);

    float3 prefilteredColor = float3(0.0, 0.0, 0.0);
    float totalWeight = 0.0;

    for (uint i = 0u; i < numSamples; i++)
    {
        float2 Xi = hammersley(i, numSamples);
        float3 H = importanceSampleGGX(Xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0)
        {
            // Sample environment
            float3 envSample = sampleEnvironment(L);

            prefilteredColor += envSample * NdotL;
            totalWeight += NdotL;
        }
    }

    prefilteredColor = prefilteredColor / max(totalWeight, EPSILON);

    // Compute BRDF integration approximation
    // Using an analytical approximation instead of LUT for simplicity
    float NdotV = max(dot(N, V), 0.0);
    float3 F = fresnelSchlickRoughness(NdotV, F0, roughness);

    // Approximate the BRDF integration
    // This is a simplification - a proper implementation uses a pre-computed BRDF LUT
    float2 envBRDF = approximateEnvBRDF(NdotV, roughness);
    float3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    return specular;
}

//
// Vertex Shader
//

[shader("vertex")]
VSOutput vertexMain(VSInput input, uint instanceId: SV_InstanceID)
{
    float4x4 model = instanceTransforms[instanceId];
    float4x4 pv = camera.projectionView;

    // Transform position to world and clip space
    float4 worldPos = mul(model, float4(input.position, 1.0));
    float4 clipPos = mul(pv, worldPos);

    // Transform normal to world space (assuming uniform scale)
    float3 worldNormal = normalize(mul(model, float4(input.normal, 0.0)).xyz);

    VSOutput output;
    output.position = clipPos;
    output.worldPosition = worldPos.xyz;
    output.worldNormal = worldNormal;
    output.texcoord0 = input.texcoord0;
    return output;
}

//
// Fragment Shader
//

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    // Sample material textures
    float4 baseColor = colorTexture.Sample(materialSampler, input.texcoord0);
    baseColor *= materialTint.colorTint;

    float metalness = metalnessTexture.Sample(materialSampler, input.texcoord0).r;
    metalness *= materialTint.metalnessTint;

    float roughness = roughnessTexture.Sample(materialSampler, input.texcoord0).r;
    roughness *= materialTint.roughnessTint;
    roughness = max(roughness, 0.04); // Prevent divide by zero in specular calculations

    // Surface vectors
    float3 N = normalize(input.worldNormal);
    float3 V = normalize(camera.cameraPosition - input.worldPosition);
    float3 R = reflect(-V, N);

    // Compute F0 (reflectance at normal incidence)
    // For dielectrics, use 0.04. For metals, use the base color.
    float3 F0 = float3(0.04, 0.04, 0.04);
    F0 = lerp(F0, baseColor.rgb, metalness);

    // Fresnel term
    float NdotV = max(dot(N, V), 0.0);
    float3 F = fresnelSchlickRoughness(NdotV, F0, roughness);

    // Energy conservation
    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;
    kD *= 1.0 - metalness; // Metals have no diffuse

    // Diffuse IBL
    float3 irradiance = computeDiffuseIBL(N);
    float3 diffuse = irradiance * baseColor.rgb;

    // Specular IBL with importance sampling
    uint numSamples = max(camera.iblSamples, 1u);
    float3 specular = computeSpecularIBL(N, V, F0, roughness, numSamples);

    // Combine lighting
    float3 ambient = kD * diffuse + specular;

    // Tone mapping can be added here if needed
    float3 color = ambient;

    return float4(color, baseColor.a);
}
