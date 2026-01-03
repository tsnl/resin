//
// Bindings:
//

// Renderer bind group:
@group(0) @binding(0) var<storage, read> geometry_heap: array<PodGeometry>;
@group(0) @binding(1) var<storage, read> bvh_node_heap: array<PodBvhNode>;
@group(0) @binding(2) var<storage, read> triangle_heap: array<PodTriangle>;

// Per-frame bind group:
@group(1) @binding(0) var output_image: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var<uniform> frame_info: PodFrameInfo;
@group(1) @binding(2) var<storage, read> camera: PodCamera;
@group(1) @binding(3) var<storage, read> instances: array<PodInstance>;

//
// Pod types: used for CPU-GPU data exchange.
//

struct PodFrameInfo {
    instance_count: u32,
    target_size_w_px: u32,
    target_size_h_px: u32,
    _rsv: u32,
}

struct PodInstance {
    geometry_id: u32,
    material_id: u32,
    transform: PodTransform,
}

struct PodGeometry {
    bvh_node_span_in_heap: PodSpan,
    triangle_span_in_heap: PodSpan,
}

struct PodBvhNode {
    span: PodSpan,
    children: array<u32, 2>,
    aabb: PodAabb,
}
struct PodTriangle {
    vertices: array<PodVertex, 3>,
}
struct PodVertex {
    position: array<f32, 3>,
    normal: array<f32, 3>,
    uv: array<f32, 2>,
}

struct PodCamera {
    transform: PodTransform,
    fov_y_rad: f32,
    aspect_ratio: f32,
    _rsv0: u32,
    _rsv1: u32,
}

struct PodSpan {
    begin: u32,
    end: u32,
}
struct PodAabb {
    min: array<f32, 3>,
    max: array<f32, 3>,
}

struct PodTransform {
    row0: array<f32, 4>,
    row1: array<f32, 4>,
    row2: array<f32, 4>,
}
fn mat4x4_from_pod_transform(t: PodTransform) -> mat4x4<f32> {
    let col0 = vec4<f32>(t.row0[0], t.row1[0], t.row2[0], 0.0);
    let col1 = vec4<f32>(t.row0[1], t.row1[1], t.row2[1], 0.0);
    let col2 = vec4<f32>(t.row0[2], t.row1[2], t.row2[2], 0.0);
    let col3 = vec4<f32>(t.row0[3], t.row1[3], t.row2[3], 1.0);
    return mat4x4<f32>(col0, col1, col2, col3);
}

//
// Entry point:
//

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= frame_info.target_size_w_px || global_id.y >= frame_info.target_size_h_px {
        return;
    }
    let r = f32(global_id.x) / f32(frame_info.target_size_w_px);
    let g = f32(global_id.y) / f32(frame_info.target_size_h_px);
    let b = 1.0 - r;
    textureStore(output_image, global_id.xy, vec4<f32>(r, g, b, 1.0));
}
