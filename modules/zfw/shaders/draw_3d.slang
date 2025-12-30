//! 3D mesh rendering shader for draw_3d.
//!
//! Uses a simple Lambertian shading model with a fixed overhead light direction (0, 0, 1).
//! Supports instanced rendering with per-instance model matrices stored in a storage buffer.
//!
//! Descriptor Sets:
//!   Set 0: Camera uniform (projection-view matrix)
//!   Set 1: Instance transforms (storage buffer of model matrices)
//!   Set 2: Material (tint, color/normal/metalness/roughness textures, sampler)

//
// Structures
//

struct CameraUniform
{
    float4x4 projectionView;
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
    float3 worldNormal : TEXCOORD0;
    float2 texcoord0 : TEXCOORD1;
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
// Vertex Shader
//

[shader("vertex")]
VSOutput vertexMain(VSInput input, uint instanceId: SV_InstanceID)
{
    float4x4 model = instanceTransforms[instanceId];
    float4x4 pv = camera.projectionView;

    // Transform position to clip space
    float4 worldPos = mul(model, float4(input.position, 1.0));
    float4 clipPos = mul(pv, worldPos);

    // Transform normal to world space (assuming uniform scale, otherwise use inverse transpose)
    // In HLSL/Slang, matrix[i] gives the i-th column, not row.
    // For mul(M, v) where v is a column vector, we need rows as: row0, row1, row2.
    // Since model[i] gives column i, we need to transpose by constructing from:
    //   row0 = (model[0][0], model[1][0], model[2][0])
    //   row1 = (model[0][1], model[1][1], model[2][1])
    //   row2 = (model[0][2], model[1][2], model[2][2])
    // Or equivalently, use mul(normal, normalMatrix) with columns.
    float3 worldNormal = normalize(
        float3(
            dot(input.normal, float3(model[0][0], model[1][0], model[2][0])),
            dot(input.normal, float3(model[0][1], model[1][1], model[2][1])),
            dot(input.normal, float3(model[0][2], model[1][2], model[2][2]))
        )
    );

    VSOutput output;
    output.position = clipPos;
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
    // Sample color texture and apply tint
    float4 baseColor = colorTexture.Sample(materialSampler, input.texcoord0);
    baseColor *= materialTint.colorTint;

    // Simple Lambertian shading with overhead light direction (0, 0, 1) in world space
    float3 lightDir = float3(0.0, 0.0, 1.0);
    float3 normal = normalize(input.worldNormal);

    // Compute diffuse factor (clamp to [0, 1])
    float NdotL = max(dot(normal, lightDir), 0.0);

    // Add ambient term to avoid completely black surfaces
    float ambient = 0.1;
    float diffuse = NdotL * 0.9 + ambient;

    // Final color
    float3 finalColor = baseColor.rgb * diffuse;

    return float4(finalColor, baseColor.a);
}
