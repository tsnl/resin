// 2D Quad Renderer Shader
//
// Renders quads with texture sampling, tint colors, and borders.
// Uses instanced rendering where each instance is a quad.
//
// Quad corner ordering: top-left (0), top-right (1), bottom-right (2), bottom-left (3) - like CSS
// Border side ordering: top (0), right (1), bottom (2), left (3) - like CSS

// Common uniform data
struct R2dUniforms
{
    uint2 framebuffer_size_px;  // (width, height)
    uint2 atlas_size_px;        // (width, height)
};

// Vertex shader output / Fragment shader input
struct VSOutput
{
    float4 position : SV_Position;
    float2 texcoord_px : TEXCOORD0;      // Interpolated texcoord in pixels
    nointerpolation uint instance_id : TEXCOORD1;
    float2 local_pos : TEXCOORD2;        // Position within quad (0-1 range for border calculation)
};

// Bindings
[[vk::binding(0, 0)]]
Texture2D<float4> atlas;

[[vk::binding(0, 0)]]
SamplerState atlasSampler;

[[vk::binding(1, 0)]]
ConstantBuffer<R2dUniforms> uniforms;

// Per-instance data (structured buffers)
// offset_px: 4 corners * 2 coords per instance
[[vk::binding(2, 0)]]
StructuredBuffer<uint> offset_px;

// texcoord_px: 4 corners * 2 coords per instance
[[vk::binding(3, 0)]]
StructuredBuffer<uint> texcoord_px;

// tint_color: 4 components (RGBA) per instance
[[vk::binding(4, 0)]]
StructuredBuffer<uint> tint_color;

// border_color: 4 components (RGBA) per instance
[[vk::binding(5, 0)]]
StructuredBuffer<uint> border_color;

// border_thickness_px: 4 values (top, right, bottom, left) per instance
[[vk::binding(6, 0)]]
StructuredBuffer<uint> border_thickness_px;


// Helper to get corner index from vertex ID
// Vertex ordering for 2 triangles (CCW):
// Triangle 1: TL(0), TR(1), BR(2)
// Triangle 2: TL(0), BR(2), BL(3)
// Vertex IDs: 0->TL, 1->TR, 2->BR, 3->TL, 4->BR, 5->BL
uint getCornerIndex(uint vertexID)
{
    uint cornerMap[6] = { 0, 1, 2, 0, 2, 3 };
    return cornerMap[vertexID];
}

// Get offset for a corner
float2 getOffset(uint instanceID, uint cornerIndex)
{
    uint baseIdx = instanceID * 8 + cornerIndex * 2;  // 8 uints per instance (4 corners * 2 coords)
    return float2(float(offset_px[baseIdx]), float(offset_px[baseIdx + 1]));
}

// Get texcoord for a corner
float2 getTexcoord(uint instanceID, uint cornerIndex)
{
    uint baseIdx = instanceID * 8 + cornerIndex * 2;
    return float2(float(texcoord_px[baseIdx]), float(texcoord_px[baseIdx + 1]));
}

// Get border thickness (top, right, bottom, left)
float4 getBorderThickness(uint instanceID)
{
    uint baseIdx = instanceID * 4;
    return float4(
        float(border_thickness_px[baseIdx]),
        float(border_thickness_px[baseIdx + 1]),
        float(border_thickness_px[baseIdx + 2]),
        float(border_thickness_px[baseIdx + 3])
    );
}

// Get tint color (RGBA, 0-255 -> 0-1)
float4 getTintColor(uint instanceID)
{
    uint baseIdx = instanceID * 4;
    return float4(
        float(tint_color[baseIdx]) / 255.0,
        float(tint_color[baseIdx + 1]) / 255.0,
        float(tint_color[baseIdx + 2]) / 255.0,
        float(tint_color[baseIdx + 3]) / 255.0
    );
}

// Get border color (RGBA, 0-255 -> 0-1)
float4 getBorderColor(uint instanceID)
{
    uint baseIdx = instanceID * 4;
    return float4(
        float(border_color[baseIdx]) / 255.0,
        float(border_color[baseIdx + 1]) / 255.0,
        float(border_color[baseIdx + 2]) / 255.0,
        float(border_color[baseIdx + 3]) / 255.0
    );
}

// Convert pixel position to NDC
float2 pixelToNDC(float2 pixel_pos, float2 framebuffer_size)
{
    // Map pixel [0, size] to NDC [-1, 1]
    return (pixel_pos / framebuffer_size) * 2.0 - 1.0;
}


[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    uint cornerIndex = getCornerIndex(vertexID);

    // Get corner position in pixels
    float2 pos_px = getOffset(instanceID, cornerIndex);
    float2 texcoord = getTexcoord(instanceID, cornerIndex);

    // Get border thickness
    float4 border = getBorderThickness(instanceID);  // top, right, bottom, left

    // Expand position outward by border thickness
    // Corner 0 (TL): expand left (-x) and up (-y)
    // Corner 1 (TR): expand right (+x) and up (-y)
    // Corner 2 (BR): expand right (+x) and down (+y)
    // Corner 3 (BL): expand left (-x) and down (+y)
    float2 expansion;
    switch (cornerIndex)
    {
        case 0:  // TL
            expansion = float2(-border.w, -border.x);  // left, top
            break;
        case 1:  // TR
            expansion = float2(border.y, -border.x);   // right, top
            break;
        case 2:  // BR
            expansion = float2(border.y, border.z);    // right, bottom
            break;
        case 3:  // BL
            expansion = float2(-border.w, border.z);   // left, bottom
            break;
        default:
            expansion = float2(0, 0);
            break;
    }

    pos_px += expansion;

    // Convert to NDC
    float2 framebuffer_size = float2(uniforms.framebuffer_size_px);
    float2 ndc = pixelToNDC(pos_px, framebuffer_size);

    // Z depth: negative of instance index (later quads appear on top)
    // Map to [0, 1] depth range where smaller values are closer to camera
    // We use a small epsilon to ensure proper ordering
    float depth = 1.0 - float(instanceID) * 0.0001;
    depth = clamp(depth, 0.0, 1.0);

    output.position = float4(ndc, depth, 1.0);
    output.texcoord_px = texcoord;
    output.instance_id = instanceID;

    // Compute local position for border calculation
    // Map corner to (0,0), (1,0), (1,1), (0,1) for TL, TR, BR, BL
    float2 localCorners[4] = {
        float2(0, 0),  // TL
        float2(1, 0),  // TR
        float2(1, 1),  // BR
        float2(0, 1)   // BL
    };
    output.local_pos = localCorners[cornerIndex];

    return output;
}


// Fragment shader

struct PSInput
{
    float2 texcoord_px : TEXCOORD0;
    nointerpolation uint instance_id : TEXCOORD1;
    float2 local_pos : TEXCOORD2;
};

struct PSOutput
{
    float4 color : SV_Target0;
};

[shader("fragment")]
PSOutput fragmentMain(PSInput input)
{
    PSOutput output;

    uint instanceID = input.instance_id;

    // Get colors
    float4 tint = getTintColor(instanceID);
    float4 borderCol = getBorderColor(instanceID);
    float4 border = getBorderThickness(instanceID);  // top, right, bottom, left

    // Sample atlas texture
    float2 atlas_size = float2(uniforms.atlas_size_px);
    float2 uv = input.texcoord_px / atlas_size;
    float4 texColor = atlas.Sample(atlasSampler, uv);

    // Apply tint
    float4 tintedColor = texColor * tint;

    // Calculate if we're in the border region
    // We need to know the quad size to properly calculate border regions
    // Using local_pos (0-1) and border thickness

    // Get quad dimensions from corner positions
    float2 tl = getOffset(instanceID, 0);
    float2 br = getOffset(instanceID, 2);
    float2 quad_size = br - tl;

    // Add border to get total size (border expands outward)
    float2 total_size = quad_size + float2(border.y + border.w, border.x + border.z);

    // Calculate pixel position within the expanded quad
    float2 pixel_in_quad = input.local_pos * total_size;

    // Check if in border region
    bool inTopBorder = pixel_in_quad.y < border.x;
    bool inBottomBorder = pixel_in_quad.y >= (total_size.y - border.z);
    bool inLeftBorder = pixel_in_quad.x < border.w;
    bool inRightBorder = pixel_in_quad.x >= (total_size.x - border.y);

    bool inBorder = inTopBorder || inBottomBorder || inLeftBorder || inRightBorder;

    // Choose color based on whether we're in border
    if (inBorder && (border.x > 0 || border.y > 0 || border.z > 0 || border.w > 0))
    {
        output.color = borderCol;
    }
    else
    {
        output.color = tintedColor;
    }

    return output;
}
