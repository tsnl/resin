struct PodFrameInfo {
    count: u32,
    target_w_px: u32,
    target_h_px: u32,
    _rsv2: u32,
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
    centroid: array<f32, 3>,
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
    target_size_w_px: u32,
    target_size_h_px: u32,
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
    position: array<f32, 4>,
    rotation: array<f32, 4>,
}

@group(0) @binding(0) var<storage, read> geometry_heap: array<PodGeometry>;
@group(0) @binding(1) var<storage, read> bvh_node_heap: array<PodBvhNode>;
@group(0) @binding(2) var<storage, read> triangle_heap: array<PodTriangle>;

@group(1) @binding(0) var output_image: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var<uniform> frame_info: PodFrameInfo;
@group(1) @binding(2) var<storage, read> camera: PodCamera;
@group(1) @binding(3) var<storage, read> instances: array<PodInstance>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= frame_info.target_w_px || global_id.y >= frame_info.target_h_px {
        return;
    }
    let r = f32(global_id.x) / f32(frame_info.target_w_px);
    let g = f32(global_id.y) / f32(frame_info.target_h_px);
    let b = 1.0 - r;
    textureStore(output_image, global_id.xy, vec4<f32>(r, g, b, 1.0));
}
