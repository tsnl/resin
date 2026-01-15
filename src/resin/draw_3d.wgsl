enable f16;

//
// Bindings:
//

const FLAG_EMIT_PRIMARY_RAY_DIRECTION: u32 = 1u;
const FLAG_EMIT_FRAME_SURFACE_DEPTH: u32 = 2u;
const FLAG_EMIT_FRAME_SURFACE_POSITION: u32 = 4u;
const FLAG_BVH_AABB_VIEW: u32 = 8u;
const FLAG_EMIT_FRAME_SURFACE_COLOR: u32 = 16u;
const FLAG_EMIT_FRAME_SURFACE_NORMAL: u32 = 32u;
const FLAG_EMIT_FRAME_SURFACE_ORM: u32 = 64u;
const FLAG_DISABLE_JITTER: u32 = 128u;
const FRAME_PER_PIXEL_RADIANCE: u32 = 256u;
const FLAG_EMIT_FRAME_SURFACE_EMISSIVE: u32 = 512u;

struct PodFrameInfo {
    instance_count: u32,
    target_size_w_px: u32,
    target_size_h_px: u32,
    debug_flags: u32,
    environment_map_texture_id: i32,
    timestamp: u32,
    frame_index: u32,
    max_bounces: u32,
    samples_per_pixel: u32,
    accumulated_frame_index: u32,
}
struct PodInstance {
    geometry_id: u32,
    material_id: u32,
    _pad0: u32,
    _pad1: u32,
    transform: PodTransform,
    inv_transform: PodTransform,
}
struct PodGeometry {
    bvh_node_span_in_heap: PodSpan,
    triangle_span_in_heap: PodSpan,
}
struct PodBvhNode {
    tri_span: PodSpan,
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
struct PodMaterial {
    color_map_id: u32,
    color_factor: array<f32, 3>,
    normal_map_id: u32,
    metalness_map_id: u32,
    metalness_factor: f32,
    roughness_map_id: u32,
    roughness_factor: f32,
    emissive_map_id: u32,
    emissive_factor: array<f32, 3>,
}
struct PodCamera {
    transform: PodTransform,
    fov_y_rad: f32,
    aspect_ratio: f32,
    clip_aabb_max: f32,
    _rsv: u32,
}
struct PodTextureAllocation {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
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
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    row3: vec4<f32>,
}

// Renderer bind group:
@group(0) @binding(0) var<storage, read> geometry_heap: array<PodGeometry>;
@group(0) @binding(1) var<storage, read> bvh_node_heap: array<PodBvhNode>;
@group(0) @binding(2) var<storage, read> triangle_heap: array<PodTriangle>;
@group(0) @binding(3) var<storage, read> material_heap: array<PodMaterial>;
@group(0) @binding(4) var color_texture_heap: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read> color_texture_allocations: array<PodTextureAllocation>;
@group(0) @binding(6) var normal_texture_heap: texture_2d_array<f32>;
@group(0) @binding(7) var<storage, read> normal_texture_allocations: array<PodTextureAllocation>;
@group(0) @binding(8) var metalness_texture_heap: texture_2d_array<f32>;
@group(0) @binding(9) var<storage, read> metalness_texture_allocations: array<PodTextureAllocation>;
@group(0) @binding(10) var roughness_texture_heap: texture_2d_array<f32>;
@group(0) @binding(11) var<storage, read> roughness_texture_allocations: array<PodTextureAllocation>;
@group(0) @binding(12) var environment_texture_heap: texture_2d_array<f32>;
@group(0) @binding(13) var<storage, read> environment_texture_allocations: array<PodTextureAllocation>;
@group(0) @binding(14) var emissive_texture_heap: texture_2d_array<f32>;
@group(0) @binding(15) var<storage, read> emissive_texture_allocations: array<PodTextureAllocation>;
@group(0) @binding(16) var linear_sampler: sampler;

// Per-frame bind group:
@group(1) @binding(0) var output_image: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var accum_image: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(2) var frame_per_pixel_radiance: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var frame_primary_ray_direction_image: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var frame_surface_depth_image: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var frame_surface_position_image: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var frame_surface_color_image: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var frame_surface_normal_image: texture_storage_2d<rgba16float, write>;
@group(1) @binding(8) var frame_surface_orm_image: texture_storage_2d<rgba16float, write>;
@group(1) @binding(9) var frame_surface_emissive_image: texture_storage_2d<rgba16float, write>;
@group(1) @binding(10) var<uniform> frame_info: PodFrameInfo;
@group(1) @binding(11) var<uniform> camera: PodCamera;
@group(1) @binding(12) var<storage, read> instances: array<PodInstance>;

//
// Constants and configuration:
//

const F32_INFINITY: f32 = 1e8;  // WGSL does not have f32::INFINITY?
const TRIANGLE_RAY_INTERSECTION_EPSILON: f32 = 1e-8;
const MAX_STACK_DEPTH: u32 = 64u;
const MAX_PATH_DEPTH: u32 = 8u;
const PI: f32 = 3.14159265359;

const POST_PRIMARY_RAY_GEN_DEBUG_MASK: u32 = FLAG_EMIT_PRIMARY_RAY_DIRECTION;
const POST_PRIMARY_RAY_HIT_DEBUG_MASK: u32 = FLAG_EMIT_FRAME_SURFACE_DEPTH | FLAG_EMIT_FRAME_SURFACE_POSITION | FLAG_BVH_AABB_VIEW;
const POST_PRIMARY_RAY_HIT_DETAILS_DEBUG_MASK: u32 = FLAG_EMIT_FRAME_SURFACE_COLOR | FLAG_EMIT_FRAME_SURFACE_NORMAL | FLAG_EMIT_FRAME_SURFACE_ORM;
const ALL_DEBUG_VISUALIZATION_MASK: u32 = POST_PRIMARY_RAY_GEN_DEBUG_MASK | POST_PRIMARY_RAY_HIT_DEBUG_MASK | POST_PRIMARY_RAY_HIT_DETAILS_DEBUG_MASK;

//
// Accessors
//

fn get_triangle_vertices_positions(triangle_id: u32) -> mat3x3<f32> {
    let pod_triangle = triangle_heap[triangle_id];
    let v0 = vec3<f32>(
        pod_triangle.vertices[0].position[0],
        pod_triangle.vertices[0].position[1],
        pod_triangle.vertices[0].position[2],
    );
    let v1 = vec3<f32>(
        pod_triangle.vertices[1].position[0],
        pod_triangle.vertices[1].position[1],
        pod_triangle.vertices[1].position[2],
    );
    let v2 = vec3<f32>(
        pod_triangle.vertices[2].position[0],
        pod_triangle.vertices[2].position[1],
        pod_triangle.vertices[2].position[2],
    );
    return mat3x3<f32>(v0, v1, v2);
}
fn get_triangle_vertices_normals(triangle_id: u32) -> mat3x3<f32> {
    let pod_triangle = triangle_heap[triangle_id];
    let v0 = vec3<f32>(
        pod_triangle.vertices[0].normal[0],
        pod_triangle.vertices[0].normal[1],
        pod_triangle.vertices[0].normal[2],
    );
    let v1 = vec3<f32>(
        pod_triangle.vertices[1].normal[0],
        pod_triangle.vertices[1].normal[1],
        pod_triangle.vertices[1].normal[2],
    );
    let v2 = vec3<f32>(
        pod_triangle.vertices[2].normal[0],
        pod_triangle.vertices[2].normal[1],
        pod_triangle.vertices[2].normal[2],
    );
    return mat3x3<f32>(v0, v1, v2);
}
fn get_triangle_vertices_texcoords(triangle_id: u32) -> mat3x2<f32> {
    let pod_triangle = triangle_heap[triangle_id];
    let v0 = vec2<f32>(
        pod_triangle.vertices[0].uv[0],
        pod_triangle.vertices[0].uv[1],
    );
    let v1 = vec2<f32>(
        pod_triangle.vertices[1].uv[0],
        pod_triangle.vertices[1].uv[1],
    );
    let v2 = vec2<f32>(
        pod_triangle.vertices[2].uv[0],
        pod_triangle.vertices[2].uv[1],
    );
    return mat3x2<f32>(v0, v1, v2);
}

//
// Random number generation and low-discrepancy sequences:
//

fn halton_base2(index: u32) -> f32 {
    var bits = index;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn halton_base3(index: u32) -> f32 {
    var result: f32 = 0.0;
    var f: f32 = 1.0 / 3.0;
    var i = index;
    while i > 0u {
        result += f * f32(i % 3u);
        i = i / 3u;
        f = f / 3.0;
    }
    return result;
}

fn halton_2d(index: u32) -> vec2<f32> {
    return vec2<f32>(halton_base2(index), halton_base3(index));
}

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_from_seed(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / f32(0xFFFFFFFFu);
}

fn rand2_from_seed(seed: ptr<function, u32>) -> vec2<f32> {
    return vec2<f32>(rand_from_seed(seed), rand_from_seed(seed));
}

fn rand3_from_seed(seed: ptr<function, u32>) -> vec3<f32> {
    return vec3<f32>(rand_from_seed(seed), rand_from_seed(seed), rand_from_seed(seed));
}

//
// Texture sampling:
//

fn sample_color_texture(
    color_texture_id: u32,
    uv: vec2<f32>,
) -> vec3<f32> {
    let alloc = color_texture_allocations[color_texture_id];
    let page = u32(alloc.y);
    let alloc_uv = vec2<f32>(alloc.x, fract(alloc.y));
    let alloc_size = vec2<f32>(alloc.w, alloc.h);
    let wrapped_uv = fract(uv);
    let sample_uv = alloc_uv + wrapped_uv * alloc_size;
    let sample_color = textureSampleLevel(color_texture_heap, linear_sampler, sample_uv, page, 0.0);
    return sample_color.rgb;
}

/// Samples a normal in [-1,+1]^3 from a normal map texture.
fn sample_normal_texture(
    normal_texture_id: u32,
    uv: vec2<f32>,
) -> vec3<f32> {
    let alloc = normal_texture_allocations[normal_texture_id];
    let page = u32(alloc.y);
    let alloc_uv = vec2<f32>(alloc.x, fract(alloc.y));
    let alloc_size = vec2<f32>(alloc.w, alloc.h);
    let wrapped_uv = fract(uv);
    let sample_uv = alloc_uv + wrapped_uv * alloc_size;
    let sample_rg = textureSampleLevel(normal_texture_heap, linear_sampler, sample_uv, page, 0.0);
    let xy = 2.0 * sample_rg.rg - vec2<f32>(1.0);
    let z = sqrt(max(0.0, 1.0 - dot(xy, xy)));
    return vec3<f32>(xy.x, xy.y, z);
}

fn sample_metalness_texture(
    metalness_texture_id: u32,
    uv: vec2<f32>,
) -> f32 {
    let alloc = metalness_texture_allocations[metalness_texture_id];
    let page = u32(alloc.y);
    let alloc_uv = vec2<f32>(alloc.x, fract(alloc.y));
    let alloc_size = vec2<f32>(alloc.w, alloc.h);
    let wrapped_uv = fract(uv);
    let sample_uv = alloc_uv + wrapped_uv * alloc_size;
    return textureSampleLevel(metalness_texture_heap, linear_sampler, sample_uv, page, 0.0).r;
}

fn sample_roughness_texture(
    roughness_texture_id: u32,
    uv: vec2<f32>,
) -> f32 {
    let alloc = roughness_texture_allocations[roughness_texture_id];
    let page = u32(alloc.y);
    let alloc_uv = vec2<f32>(alloc.x, fract(alloc.y));
    let alloc_size = vec2<f32>(alloc.w, alloc.h);
    let wrapped_uv = fract(uv);
    let sample_uv = alloc_uv + wrapped_uv * alloc_size;
    return textureSampleLevel(roughness_texture_heap, linear_sampler, sample_uv, page, 0.0).r;
}

fn sample_environment_texture(
    environment_texture_id: u32,
    uv: vec2<f32>,
) -> vec4<f32> {
    let alloc = environment_texture_allocations[environment_texture_id];
    let page = u32(alloc.y);
    let alloc_uv = vec2<f32>(alloc.x, fract(alloc.y));
    let alloc_size = vec2<f32>(alloc.w, alloc.h);
    let wrapped_uv = fract(uv);
    let sample_uv = alloc_uv + wrapped_uv * alloc_size;
    return textureSampleLevel(environment_texture_heap, linear_sampler, sample_uv, page, 0.0);
}

fn sample_emissive_texture(
    emissive_texture_id: u32,
    uv: vec2<f32>,
) -> vec3<f32> {
    let alloc = emissive_texture_allocations[emissive_texture_id];
    let page = u32(alloc.y);
    let alloc_uv = vec2<f32>(alloc.x, fract(alloc.y));
    let alloc_size = vec2<f32>(alloc.w, alloc.h);
    let wrapped_uv = fract(uv);
    let sample_uv = alloc_uv + wrapped_uv * alloc_size;
    let sample_color = textureSampleLevel(emissive_texture_heap, linear_sampler, sample_uv, page, 0.0);
    return sample_color.rgb;
}

//
// Linalg
//

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,   // Does not need to be normalized, but never 0
}

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
}

fn h_mat4x4_from_pod_transform(t: PodTransform) -> mat4x4<f32> {
    let col0 = vec4<f32>(t.row0[0], t.row1[0], t.row2[0], 0.0);
    let col1 = vec4<f32>(t.row0[1], t.row1[1], t.row2[1], 0.0);
    let col2 = vec4<f32>(t.row0[2], t.row1[2], t.row2[2], 0.0);
    let col3 = vec4<f32>(t.row0[3], t.row1[3], t.row2[3], 1.0);
    return mat4x4<f32>(col0, col1, col2, col3);
}
fn h_mat4x4_inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    // Assumes affine transform matrix.
    let r = mat3x3<f32>(
        m[0].xyz,
        m[1].xyz,
        m[2].xyz,
    );
    let t = m[3].xyz;

    let r_inv = transpose(r);
    let t_inv = -(r_inv * t);

    let col0 = vec4<f32>(r_inv[0], 0.0);
    let col1 = vec4<f32>(r_inv[1], 0.0);
    let col2 = vec4<f32>(r_inv[2], 0.0);
    let col3 = vec4<f32>(t_inv, 1.0);
    return mat4x4<f32>(col0, col1, col2, col3);
}
fn h_mat4x4_transform_position(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
    let p_h = vec4<f32>(p, 1.0);
    let transformed_p_h = m * p_h;
    return transformed_p_h.xyz / transformed_p_h.w;
}
fn h_mat4x4_transform_direction(m: mat4x4<f32>, d: vec3<f32>) -> vec3<f32> {
    let d_h = vec4<f32>(d, 0.0);
    let transformed_d_h = m * d_h;
    return transformed_d_h.xyz;
}
fn h_mat4x4_transform_ray(m: mat4x4<f32>, ray: Ray) -> Ray {
    let transformed_origin = h_mat4x4_transform_position(m, ray.origin);
    let transformed_direction = h_mat4x4_transform_direction(m, ray.direction);
    return Ray(transformed_origin, transformed_direction);
}

fn convert_direction_to_rgb(dir: vec3<f32>) -> vec3<f32> {
    return 0.5 * (normalize(dir) + vec3<f32>(1.0, 1.0, 1.0));
}
fn convert_rgb_to_direction(rgb: vec3<f32>) -> vec3<f32> {
    return normalize(2.0 * rgb - vec3<f32>(1.0, 1.0, 1.0));
}

//
// Raycast: Ray -> HitRecord
//

/// Record of a ray hit against an instance. Product of ray-world intersection tests.
struct HitRecord {
    world_hit_position: vec3<f32>,
    world_hit_distance: f32,
    barycentric_coordinates: vec3<f32>,
    ray: Ray,
    triangle_id: u32,
    geometry_id: u32,
    instance_id: u32,
}
fn new_invalid_hit_record(ray: Ray) -> HitRecord {
    var hit: HitRecord;
    hit.world_hit_distance = F32_INFINITY;
    hit.ray = ray;
    return hit;
}
fn is_hit_record_valid(hit: HitRecord) -> bool {
    return hit.world_hit_distance < F32_INFINITY;
}
fn hit(ray: Ray) -> HitRecord {
    // Raycast against all instances, find closest hit.
    // TODO: Use TLAS to accelerate this.
    var closest_hit = new_invalid_hit_record(ray);
    for (var instance_id = 0u; instance_id < frame_info.instance_count; instance_id++) {
        let hit_record = hit_instance(ray, instance_id);
        if hit_record.world_hit_distance < closest_hit.world_hit_distance {
            closest_hit = hit_record;
        }
    }
    return closest_hit;
}
fn hit_instance(ray: Ray, instance_id: u32) -> HitRecord {
    let instance = instances[instance_id];
    let instance_transform = h_mat4x4_from_pod_transform(instance.transform);
    let inv_instance_transform = h_mat4x4_from_pod_transform(instance.inv_transform);

    let geometry_id = instance.geometry_id;

    // Transform ray into model space by applying the inverse of the instance's transform.
    // This lets us raycast against the geometry without transforming all the vertices per-instance.
    let local_ray = h_mat4x4_transform_ray(inv_instance_transform, ray);
    let local_hit = hit_geometry_with_bvh(local_ray, geometry_id);

    // If hit is not valid, early out.
    if !is_geometry_hit_record_valid(local_hit) {
        return new_invalid_hit_record(ray);
    }

    // Local hit is valid. Transform local hit position back to world space.
    var hit_record: HitRecord;
    hit_record.world_hit_position = h_mat4x4_transform_position(
        instance_transform,
        local_ray.origin + local_hit.triangle_raycast_result.w * local_ray.direction,
    );
    hit_record.world_hit_distance = length(hit_record.world_hit_position - ray.origin);
    hit_record.barycentric_coordinates = local_hit.triangle_raycast_result.xyz;
    hit_record.ray = ray;
    hit_record.triangle_id = local_hit.triangle_id;
    hit_record.geometry_id = geometry_id;
    hit_record.instance_id = instance_id;
    return hit_record;
}

/// Record of a ray hit against geometry (no instance transform applied).
struct GeometryHitRecord {
    triangle_raycast_result: vec4<f32>, // XYZ = barycentric coords, W = distance
    triangle_id: u32,
}
fn new_invalid_geometry_hit_record() -> GeometryHitRecord {
    var hit: GeometryHitRecord;
    hit.triangle_raycast_result = vec4<f32>(0.0, 0.0, 0.0, F32_INFINITY);
    return hit;
}
fn is_geometry_hit_record_valid(hit: GeometryHitRecord) -> bool {
    return hit.triangle_raycast_result.w < F32_INFINITY;
}
fn hit_tri_list(ray: Ray, triangle_span: PodSpan) -> GeometryHitRecord {
    var closest_hit_record = new_invalid_geometry_hit_record();
    for (var triangle_id = triangle_span.begin; triangle_id < triangle_span.end; triangle_id++) {
        let triangle_vertices = get_triangle_vertices_positions(triangle_id);
        let hit_result = hit_triangle(ray, triangle_vertices);
        if hit_result.w > 0.0 && hit_result.w < closest_hit_record.triangle_raycast_result.w {
            closest_hit_record.triangle_raycast_result = hit_result;
            closest_hit_record.triangle_id = triangle_id;
        }
    }
    return closest_hit_record;
}
fn hit_geometry(ray: Ray, geometry_id: u32) -> GeometryHitRecord {
    let triangle_span = geometry_heap[geometry_id].triangle_span_in_heap;
    return hit_tri_list(ray, triangle_span);
}
fn hit_geometry_with_bvh(ray: Ray, geometry_id: u32) -> GeometryHitRecord {
    let geometry = geometry_heap[geometry_id];
    let bvh_node_span = geometry.bvh_node_span_in_heap;
    let debug_bvh_traversal = (frame_info.debug_flags & FLAG_BVH_AABB_VIEW) != 0u;

    // Stack-based BVH traversal
    // We use a fixed-size stack for iterative traversal instead of recursion
    var stack: array<u32, 64u>;
    var stack_ptr: u32 = 0u;

    // Start with root node (first node in the BVH span)
    if bvh_node_span.begin >= bvh_node_span.end {
        return new_invalid_geometry_hit_record();
    }
    stack[stack_ptr] = bvh_node_span.begin;
    stack_ptr += 1u;

    var closest_hit = new_invalid_geometry_hit_record();

    while stack_ptr > 0u {
        // Pop node from stack
        stack_ptr -= 1u;
        let node_id = stack[stack_ptr];
        let node = bvh_node_heap[node_id];

        // Convert PodAabb to Aabb
        let aabb = Aabb(
            vec3<f32>(node.aabb.min[0], node.aabb.min[1], node.aabb.min[2]),
            vec3<f32>(node.aabb.max[0], node.aabb.max[1], node.aabb.max[2]),
        );

        // Test ray against AABB
        let aabb_hit_dist = hit_aabb(ray, aabb);
        if aabb_hit_dist >= closest_hit.triangle_raycast_result.w {
            // AABB is further than current closest hit, skip this branch
            continue;
        }

        // Check if this is a leaf node (both children are 0)
        let is_leaf = node.children[0] == 0u && node.children[1] == 0u;

        if is_leaf {
            if debug_bvh_traversal {
                // Debug mode: return AABB hit instead of triangle hit
                if aabb_hit_dist < closest_hit.triangle_raycast_result.w {
                    closest_hit.triangle_raycast_result = vec4<f32>(0.0, 0.0, 0.0, aabb_hit_dist);
                    // Pick first triangle in the leaf span
                    if node.tri_span.begin < node.tri_span.end {
                        closest_hit.triangle_id = node.tri_span.begin;
                    }
                }
            } else {
                // Normal mode: test triangles in the span
                let leaf_hit = hit_tri_list(ray, node.tri_span);
                if leaf_hit.triangle_raycast_result.w < closest_hit.triangle_raycast_result.w {
                    closest_hit = leaf_hit;
                }
            }
        } else {
            // Internal node: push children onto stack
            // Child indices are relative to this geometry's BVH, so offset them by the span begin
            // Only fetch second child's AABB if first child intersects (lazy evaluation)
            let child0_idx = node.children[0];
            let child1_idx = node.children[1];
            let closest_dist = closest_hit.triangle_raycast_result.w;

            // Test first child
            var child0_dist = F32_INFINITY;
            if child0_idx != 0u {
                let child0_node = bvh_node_heap[bvh_node_span.begin + child0_idx];
                let child0_aabb = Aabb(
                    vec3<f32>(child0_node.aabb.min[0], child0_node.aabb.min[1], child0_node.aabb.min[2]),
                    vec3<f32>(child0_node.aabb.max[0], child0_node.aabb.max[1], child0_node.aabb.max[2]),
                );
                child0_dist = hit_aabb(ray, child0_aabb);
            }

            // Test second child only if it might be closer than current best
            var child1_dist = F32_INFINITY;
            if child1_idx != 0u {
                let child1_node = bvh_node_heap[bvh_node_span.begin + child1_idx];
                let child1_aabb = Aabb(
                    vec3<f32>(child1_node.aabb.min[0], child1_node.aabb.min[1], child1_node.aabb.min[2]),
                    vec3<f32>(child1_node.aabb.max[0], child1_node.aabb.max[1], child1_node.aabb.max[2]),
                );
                child1_dist = hit_aabb(ray, child1_aabb);
            }

            // Push children in reverse order of distance so closer child is popped first
            // Skip children that don't intersect or are further than current closest hit
            if child0_dist < child1_dist {
                // child0 is closer: push child1 first (farther), then child0 (closer)
                if child1_dist < closest_dist && stack_ptr < MAX_STACK_DEPTH {
                    stack[stack_ptr] = bvh_node_span.begin + child1_idx;
                    stack_ptr += 1u;
                }
                if child0_dist < closest_dist && stack_ptr < MAX_STACK_DEPTH {
                    stack[stack_ptr] = bvh_node_span.begin + child0_idx;
                    stack_ptr += 1u;
                }
            } else {
                // child1 is closer: push child0 first (farther), then child1 (closer)
                if child0_dist < closest_dist && stack_ptr < MAX_STACK_DEPTH {
                    stack[stack_ptr] = bvh_node_span.begin + child0_idx;
                    stack_ptr += 1u;
                }
                if child1_dist < closest_dist && stack_ptr < MAX_STACK_DEPTH {
                    stack[stack_ptr] = bvh_node_span.begin + child1_idx;
                    stack_ptr += 1u;
                }
            }
        }
    }

    return closest_hit;
}

/// Triangle-ray intersection test
/// Returns barycentric coordinates (XYZ) and distance (W) of closest hit. If no hit, (W) is infinity.
fn hit_triangle(ray: Ray, triangle_vertices: mat3x3<f32>) -> vec4<f32> {
    let v0 = triangle_vertices[0];
    let v1 = triangle_vertices[1];
    let v2 = triangle_vertices[2];

    // https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
    // Core idea: express ray hit (if exists) in barycentric coordinates.
    //  i.e. p_hit = (ray.o + t * ray.d) = (v0 + u (v1 - v0) + v (v2 - v0))
    // Now, we want to solve for u, v.
    // We are in the triangle (and have a hit) if
    // - u >= 0, v >= 0
    // - u + v <= 1
    // - ray.t > 0 (indicating a "forward" hit for the ray)

    // Calculate edges e1, e2 for barycentric coordinates:
    let e1 = v1 - v0;
    let e2 = v2 - v0;

    // Check if the ray is (almost) parallel to the triangle plane, and if so, early out.
    // Save 'a' for later.
    let h = cross(ray.direction, e2);
    let a = dot(e1, h);
    if abs(a) < TRIANGLE_RAY_INTERSECTION_EPSILON {
        // Ray parallel to triangle
        return vec4<f32>(0.0, 0.0, 0.0, F32_INFINITY);
    }

    // Let 's' be vector to the ray origin from vertex v0.
    //  s + td = u.e1 + v.e2
    // Rearranging, we get
    //  s = -t.d + u.e1 + v.e2
    // This is a system of linear equations where we solve for t, u, v.
    // We can now invert the 3x3 matrix [-d, e1, e2].
    // Note that 'a', which we already have, is the determinant.
    let s = ray.origin - v0;
    let f = 1.0 / a;

    // Evaluate 'u', early out if u is invalid:
    let v1_barycentric_coordinate = f * dot(s, h);
    if !(0.0 <= v1_barycentric_coordinate && v1_barycentric_coordinate <= 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, F32_INFINITY);
    }

    // Evaluate 'v', early out if v is invalid or if u + v > 1.0 (out of triangle):
    let q = cross(s, e1);
    let v2_barycentric_coordinate = f * dot(ray.direction, q);
    if v2_barycentric_coordinate < 0.0
        || v1_barycentric_coordinate + v2_barycentric_coordinate > 1.0
    {
        return vec4<f32>(0.0, 0.0, 0.0, F32_INFINITY);
    }

    // Compute 't' for distance travelled by the ray, filter out backward hits (negative t):
    let t = f * dot(e2, q);
    if t < TRIANGLE_RAY_INTERSECTION_EPSILON {
        return vec4<f32>(0.0, 0.0, 0.0, F32_INFINITY);
    }

    // Hit successful:
    let v0_barycentric_coordinate =
        1.0 - (v1_barycentric_coordinate + v2_barycentric_coordinate);
    return vec4<f32>(
        v0_barycentric_coordinate,
        v1_barycentric_coordinate,
        v2_barycentric_coordinate,
        t,
    );
}

/// Triangle-AABB intersection test.
/// Returns distance to nearest hit, or infinity if no hit.
fn hit_aabb(ray: Ray, aabb: Aabb) -> f32 {
    let lo = aabb.min;
    let hi = aabb.max;

    let inv_dir = 1.0 / ray.direction;

    let t_lo = (lo - ray.origin) * inv_dir;
    let t_hi = (hi - ray.origin) * inv_dir;

    let t_close_percoeff = min(t_lo, t_hi);
    let t_far_percoeff = max(t_lo, t_hi);

    // Vectorized reduce operations for better performance
    let t_close = max(t_close_percoeff.x, max(t_close_percoeff.y, t_close_percoeff.z));
    let t_far = min(t_far_percoeff.x, min(t_far_percoeff.y, t_far_percoeff.z));

    if t_close > t_far || t_far < 0.0 {
        // No intersection, or intersection is behind the ray origin.
        return F32_INFINITY;
    } else {
        // Intersection exists, return the distance to the nearest intersection point.
        return max(t_close, 0.0);
    }
}

//
// PBR shading and path tracing:
//

fn get_environment_radiance(ray: Ray) -> vec3<f32> {
    if frame_info.environment_map_texture_id >= 0 {
        return sample_environment_map(ray).rgb;
    }
    return sample_procedural_environment_map(ray).rgb;
}

fn sample_procedural_environment_map(ray: Ray) -> vec4<f32> {
    // Otherwise, use procedural sky gradient
    // Create a sky gradient with brightest spot at azimuth 45°, altitude 45°
    let dir = normalize(ray.direction);

    // Compute altitude (elevation angle from horizontal plane)
    let altitude = asin(dir.y);  // -π/2 to π/2

    // Compute azimuth (horizontal angle)
    let azimuth = atan2(dir.x, dir.z);  // -π to π

    // Target: azimuth 45° = π/4, altitude 45° = π/4
    let target_azimuth = 0.785398;  // π/4
    let target_altitude = 0.785398;  // π/4

    // Angular distance from brightest spot
    let azimuth_diff = azimuth - target_azimuth;
    let altitude_diff = altitude - target_altitude;
    let angular_distance = sqrt(azimuth_diff * azimuth_diff + altitude_diff * altitude_diff);

    // Brightness falloff from the sun spot
    let sun_brightness = exp(-angular_distance * 2.0);

    // Base sky color gradient (horizon to zenith)
    let horizon_color = vec3<f32>(0.6, 0.7, 0.9);  // Light blue
    let zenith_color = vec3<f32>(0.2, 0.4, 0.8);   // Deeper blue
    let t = clamp((dir.y + 1.0) * 0.5, 0.0, 1.0);
    let sky_color = mix(horizon_color, zenith_color, t);

    // Sun color (warm yellow-white)
    let sun_color = vec3<f32>(1.0, 0.95, 0.8);

    // Blend sun and sky
    let final_color = mix(sky_color, sun_color, sun_brightness * 0.8);

    // Done:
    return vec4<f32>(final_color, 1.0);
}

/// Sample an equirectangular environment map based on ray direction.
/// The environment map is assumed to be in latitude-longitude format.
/// Coordinate system: Z-up, Y-forward, X-right (right-handed)
fn sample_environment_map(ray: Ray) -> vec4<f32> {
    let dir = normalize(ray.direction);

    // Convert 3D direction to spherical coordinates for Z-up, Y-forward system
    // Longitude (θ): horizontal angle in XY plane from +Y axis, range [-π, π]
    // Latitude (φ): elevation angle from XY plane towards +Z, range [-π/2, π/2]
    let theta = atan2(dir.x, dir.y);  // Horizontal angle from +Y (forward)
    let phi = asin(dir.z);  // Elevation angle (+Z is up)

    // Convert to UV coordinates [0, 1]
    // U maps longitude: [-π, π] -> [0, 1]
    // V maps latitude: [-π/2, π/2] -> [0, 1] (flip so +Z up is at top of image)
    let u = (theta + 3.14159265359) / (2.0 * 3.14159265359);
    let v = 1.0 - ((phi + 1.5707963268) / 3.14159265359);  // Flip V so +Z is at top

    let uv = vec2<f32>(u, v);
    return sample_environment_texture(u32(frame_info.environment_map_texture_id), uv);
}



struct HitDetails {
    world_hit_position: vec3<f32>,
    world_hit_distance: f32,
    barycentric_coordinates: vec3<f32>,
    texcoords: vec2<f32>,
    tbn: mat3x3<f32>,
    surface_color: vec4<f32>,
    surface_normal: vec3<f32>,
    surface_metalness: f32,
    surface_roughness: f32,
    surface_emissive: vec3<f32>,
    instance_id: u32,
}
fn compute_hit_details(hit: HitRecord) -> HitDetails {
    var hit_details: HitDetails;
    hit_details.world_hit_position = hit.world_hit_position;
    hit_details.world_hit_distance = hit.world_hit_distance;
    hit_details.barycentric_coordinates = hit.barycentric_coordinates.xyz;
    hit_details.texcoords = compute_hit_details_texcoords(hit);
    hit_details.tbn = compute_hit_details_tbn_matrix(hit);
    hit_details.instance_id = hit.instance_id;
    hit_details.surface_color = compute_hit_details_surface_color(hit_details);
    hit_details.surface_normal = compute_hit_details_surface_normal(hit_details);
    hit_details.surface_metalness = compute_hit_details_surface_metalness(hit_details);
    hit_details.surface_roughness = compute_hit_details_surface_roughness(hit_details);
    hit_details.surface_emissive = compute_hit_details_surface_emissive(hit_details);
    return hit_details;
}
fn compute_hit_details_texcoords(hit: HitRecord) -> vec2<f32> {
    let uv = get_triangle_vertices_texcoords(hit.triangle_id);
    let bc = hit.barycentric_coordinates;
    let interpolated_uv = bc.x * uv[0] + bc.y * uv[1] + bc.z * uv[2];
    return interpolated_uv;
}
fn compute_hit_details_tbn_matrix(hit: HitRecord) -> mat3x3<f32> {
    let instance = instances[hit.instance_id];

    // Load vertex data:
    let pos = get_triangle_vertices_positions(hit.triangle_id);
    let normal = get_triangle_vertices_normals(hit.triangle_id);
    let uv = get_triangle_vertices_texcoords(hit.triangle_id);

    // Interpolate normal in model space using barycentric coordinates
    let bc = hit.barycentric_coordinates;
    let model_normal = normalize(normal * bc);

    // Compute tangent and bitangent in model space using edge vectors and UV deltas
    let edge1 = pos[1] - pos[0];
    let edge2 = pos[2] - pos[0];
    let delta_uv1 = uv[1] - uv[0];
    let delta_uv2 = uv[2] - uv[0];

    let uv_det = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;
    let r = 1.0 / (uv_det + 1e-6);
    let m = r * mat2x2<f32>(delta_uv2.y, -delta_uv1.y, -delta_uv2.x, delta_uv1.x);
    let e = mat2x3<f32>(edge1, edge2);
    let tb = e * m;

    // Transform to world space:
    // - Tangent uses regular transform (it's a direction in the surface plane)
    // - Normal uses inverse-transpose to handle non-uniform scale correctly
    let transform_4x4 = h_mat4x4_from_pod_transform(instance.transform);
    let transform = mat3x3<f32>(
        transform_4x4[0].xyz,
        transform_4x4[1].xyz,
        transform_4x4[2].xyz,
    );
    let inv_transform_4x4 = h_mat4x4_from_pod_transform(instance.inv_transform);
    let normal_transform = transpose(mat3x3<f32>(
        inv_transform_4x4[0].xyz,
        inv_transform_4x4[1].xyz,
        inv_transform_4x4[2].xyz,
    ));

    // Transform tangent and normal to world space, then orthonormalize
    let world_tangent_raw = transform * tb[0];
    let world_normal = normalize(normal_transform * model_normal);

    // Gram-Schmidt orthonormalize in world space: ensure T is perpendicular to N, then B perpendicular to both
    let world_tangent = normalize(world_tangent_raw - world_normal * dot(world_normal, world_tangent_raw));
    let world_bitangent = cross(world_normal, world_tangent);

    return mat3x3<f32>(world_tangent, world_bitangent, world_normal);

}
fn compute_hit_details_surface_color(hit_details: HitDetails) -> vec4<f32> {
    let instance = instances[hit_details.instance_id];
    let material = material_heap[instance.material_id];
    let color = sample_color_texture(material.color_map_id, hit_details.texcoords);
    let color_factor = vec3<f32>(material.color_factor[0], material.color_factor[1], material.color_factor[2]);
    return vec4<f32>(color * color_factor, 1.0);
}
fn compute_hit_details_surface_normal(hit_details: HitDetails) -> vec3<f32> {
    let instance = instances[hit_details.instance_id];
    let material = material_heap[instance.material_id];
    let texture_normal = sample_normal_texture(material.normal_map_id, hit_details.texcoords);
    return hit_details.tbn * texture_normal;
}
fn compute_hit_details_surface_metalness(hit_details: HitDetails) -> f32 {
    let instance = instances[hit_details.instance_id];
    let material = material_heap[instance.material_id];
    let metalness_texture = sample_metalness_texture(material.metalness_map_id, hit_details.texcoords);
    let metalness_factor = material.metalness_factor;
    return metalness_factor * metalness_texture;
}
fn compute_hit_details_surface_roughness(hit_details: HitDetails) -> f32 {
    let instance = instances[hit_details.instance_id];
    let material = material_heap[instance.material_id];
    let roughness_texture = sample_roughness_texture(material.roughness_map_id, hit_details.texcoords);
    let roughness_factor = material.roughness_factor;
    return roughness_factor * roughness_texture;
}
fn compute_hit_details_surface_emissive(hit_details: HitDetails) -> vec3<f32> {
    let instance = instances[hit_details.instance_id];
    let material = material_heap[instance.material_id];
    let emissive_factor = vec3<f32>(
        material.emissive_factor[0],
        material.emissive_factor[1],
        material.emissive_factor[2],
    );
    // If emissive_map_id is 0xFFFFFFFF, there's no emissive texture, just use the factor
    if material.emissive_map_id == 0xFFFFFFFFu {
        return emissive_factor;
    }
    let emissive_texture = sample_emissive_texture(material.emissive_map_id, hit_details.texcoords);
    return emissive_factor * emissive_texture;
}

//
// BRDF sampling and evaluation:
//

fn cosine_weighted_hemisphere_sample(normal: vec3<f32>, rand: vec2<f32>) -> vec3<f32> {
    let r = sqrt(rand.x);
    let theta = 2.0 * PI * rand.y;
    let local_dir = vec3<f32>(r * cos(theta), r * sin(theta), sqrt(1.0 - rand.x));
    return orient_to_normal(local_dir, normal);
}

fn orient_to_normal(local_dir: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let tangent = select_orthogonal(normal);
    let bitangent = cross(normal, tangent);
    return tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z;
}

fn select_orthogonal(v: vec3<f32>) -> vec3<f32> {
    let abs_v = abs(v);
    let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs_v.x > abs_v.y);
    return normalize(cross(v, axis));
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn ggx_sample_hemisphere(roughness: f32, rand: vec2<f32>) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * rand.y;
    let cos_theta = sqrt((1.0 - rand.x) / (1.0 + (a * a - 1.0) * rand.x));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

fn ggx_sample_direction(normal: vec3<f32>, view_dir: vec3<f32>, roughness: f32, rand: vec2<f32>) -> vec3<f32> {
    let local_h = ggx_sample_hemisphere(roughness, rand);
    let h = orient_to_normal(local_h, normal);
    return reflect(-view_dir, h);
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

//
// Path tracing core:
//

struct PathState {
    ray: Ray,
    throughput: vec3<f32>,
    accumulated_radiance: vec3<f32>,
    bounce: u32,
    terminated: bool,
}

fn create_initial_path_state(ray: Ray) -> PathState {
    var state: PathState;
    state.ray = ray;
    state.throughput = vec3<f32>(1.0);
    state.accumulated_radiance = vec3<f32>(0.0);
    state.bounce = 0u;
    state.terminated = false;
    return state;
}

fn trace_path(initial_ray: Ray, seed: ptr<function, u32>) -> vec3<f32> {
    var state = create_initial_path_state(initial_ray);

    for (var bounce = 0u; bounce < frame_info.max_bounces; bounce++) {
        if state.terminated { break; }
        state = process_path_bounce(state, seed);
    }

    return state.accumulated_radiance;
}

fn process_path_bounce(state: PathState, seed: ptr<function, u32>) -> PathState {
    var next_state = state;
    let hit_record = hit(state.ray);

    if !is_hit_record_valid(hit_record) {
        return handle_path_miss(next_state);
    }

    let hit_details = compute_hit_details(hit_record);
    return handle_path_hit(next_state, hit_details, seed);
}

fn handle_path_miss(state: PathState) -> PathState {
    var next_state = state;
    let env_radiance = get_environment_radiance(state.ray);
    next_state.accumulated_radiance += state.throughput * env_radiance;
    next_state.terminated = true;
    return next_state;
}

fn handle_path_hit(state: PathState, hit_details: HitDetails, seed: ptr<function, u32>) -> PathState {
    var next_state = state;

    // Add emissive contribution from the hit surface
    next_state.accumulated_radiance += state.throughput * hit_details.surface_emissive;

    let brdf_result = sample_brdf(hit_details, state.ray.direction, seed);

    next_state.throughput *= brdf_result.weight;
    next_state.ray = Ray(offset_ray_origin(hit_details, brdf_result.direction), brdf_result.direction);
    next_state.bounce += 1u;

    if should_terminate_russian_roulette(next_state.throughput, seed) {
        next_state.terminated = true;
    } else {
        // Compensate throughput for Russian Roulette to maintain unbiased estimate
        next_state.throughput = compensate_russian_roulette(next_state.throughput);
    }

    return next_state;
}

fn offset_ray_origin(hit_details: HitDetails, direction: vec3<f32>) -> vec3<f32> {
    let offset_sign = sign(dot(direction, hit_details.surface_normal));
    return hit_details.world_hit_position + offset_sign * hit_details.surface_normal * 0.001;
}

struct BrdfSample {
    direction: vec3<f32>,
    weight: vec3<f32>,
}

fn sample_brdf(hit_details: HitDetails, incoming_dir: vec3<f32>, seed: ptr<function, u32>) -> BrdfSample {
    let view_dir = normalize(-incoming_dir);
    let normal = hit_details.surface_normal;
    let roughness = max(hit_details.surface_roughness, 0.04);
    let metalness = hit_details.surface_metalness;

    let rand = rand2_from_seed(seed);
    let select_rand = rand_from_seed(seed);

    let base_color = hit_details.surface_color.rgb;
    let f0 = mix(vec3<f32>(0.04), base_color, metalness);

    if select_rand < 0.5 {
        return sample_diffuse_brdf(normal, base_color, metalness, rand);
    }
    return sample_specular_brdf(normal, view_dir, roughness, f0, rand);
}

fn sample_diffuse_brdf(normal: vec3<f32>, base_color: vec3<f32>, metalness: f32, rand: vec2<f32>) -> BrdfSample {
    var result: BrdfSample;
    result.direction = cosine_weighted_hemisphere_sample(normal, rand);
    result.weight = base_color * (1.0 - metalness) * 2.0;
    return result;
}

fn sample_specular_brdf(normal: vec3<f32>, view_dir: vec3<f32>, roughness: f32, f0: vec3<f32>, rand: vec2<f32>) -> BrdfSample {
    var result: BrdfSample;
    result.direction = ggx_sample_direction(normal, view_dir, roughness, rand);
    let n_dot_l = max(dot(normal, result.direction), 0.0);
    let fresnel = fresnel_schlick(n_dot_l, f0);
    result.weight = fresnel * 2.0;
    return result;
}

fn should_terminate_russian_roulette(throughput: vec3<f32>, seed: ptr<function, u32>) -> bool {
    let survival_prob = clamp(luminance(throughput), 0.1, 0.95);
    let rand = rand_from_seed(seed);
    return rand > survival_prob;
}

fn compensate_russian_roulette(throughput: vec3<f32>) -> vec3<f32> {
    let survival_prob = clamp(luminance(throughput), 0.1, 0.95);
    return throughput / survival_prob;
}

fn render_pixel_path_traced(pixel_xy: vec2<u32>, seed: ptr<function, u32>) -> vec3<f32> {
    var radiance = vec3<f32>(0.0);

    for (var sample_idx = 0u; sample_idx < frame_info.samples_per_pixel; sample_idx++) {
        var jitter = vec2<f32>(0.0);
        if (frame_info.debug_flags & FLAG_DISABLE_JITTER) == 0u {
            jitter = compute_primary_ray_jitter(sample_idx);
        }
        let ray = gen_primary_ray_jittered(pixel_xy, jitter);
        radiance += trace_path(ray, seed);
    }

    return radiance / f32(frame_info.samples_per_pixel);
}

fn compute_primary_ray_jitter(sample_idx: u32) -> vec2<f32> {
    let combined_idx = frame_info.frame_index * frame_info.samples_per_pixel + sample_idx;
    return halton_2d(combined_idx + 1u) - 0.5;
}

fn gen_primary_ray_jittered(pixel_coord_px: vec2<u32>, jitter: vec2<f32>) -> Ray {
    let target_size = vec2<f32>(f32(frame_info.target_size_w_px), f32(frame_info.target_size_h_px));
    // Add 0.5 to get pixel center, then apply jitter (which is in [-0.5, +0.5])
    let jittered_coord = vec2<f32>(pixel_coord_px) + vec2<f32>(0.5) + jitter;
    return gen_primary_ray_from_coord(jittered_coord, target_size);
}

fn gen_primary_ray_from_coord(pixel_coord: vec2<f32>, target_size: vec2<f32>) -> Ray {
    let pixel_coord_ndc = compute_ndc_from_pixel_coord(pixel_coord, target_size);
    let sensor_pixel = compute_sensor_pixel_camera_space(pixel_coord_ndc);
    let camera_space_ray = Ray(vec3<f32>(0.0), sensor_pixel);
    let camera_transform = h_mat4x4_from_pod_transform(camera.transform);
    return h_mat4x4_transform_ray(camera_transform, camera_space_ray);
}

fn compute_ndc_from_pixel_coord(pixel_coord: vec2<f32>, target_size: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        (pixel_coord.x / target_size.x) * 2.0 - 1.0,
        1.0 - (pixel_coord.y / target_size.y) * 2.0,
    );
}

fn compute_sensor_pixel_camera_space(pixel_coord_ndc: vec2<f32>) -> vec3<f32> {
    let hw = tan(camera.fov_y_rad / 2.0) * camera.aspect_ratio;
    let hh = tan(camera.fov_y_rad / 2.0);
    return vec3<f32>(pixel_coord_ndc.x * hw, 1.0, pixel_coord_ndc.y * hh);
}

fn accumulator_index(pixel_xy: vec2<u32>) -> u32 {
    return pixel_xy.y * frame_info.target_size_w_px + pixel_xy.x;
}

fn accumulator_index_for_frame(pixel_xy: vec2<u32>, frame_slot: u32) -> u32 {
    let pixels_per_frame = frame_info.target_size_w_px * frame_info.target_size_h_px;
    return frame_slot * pixels_per_frame + accumulator_index(pixel_xy);
}

//
// Debug shading:
//

fn debug_hook_post_primary_ray_gen(ray: Ray, pixel_coords: vec2<i32>) {
    if (frame_info.debug_flags & FLAG_EMIT_PRIMARY_RAY_DIRECTION) != 0u {
        emit_primary_ray_direction(ray, pixel_coords);
    }
}
fn emit_primary_ray_direction(ray: Ray, pixel_coords: vec2<i32>) {
    let rgb = vec4<f32>(convert_direction_to_rgb(ray.direction.xyz), 1.0);
    textureStore(frame_primary_ray_direction_image, pixel_coords, rgb);
}

fn debug_hook_post_primary_ray_hit(hit: HitRecord, pixel_coords: vec2<i32>) {
    if (frame_info.debug_flags & FLAG_EMIT_FRAME_SURFACE_DEPTH) != 0u {
        emit_frame_surface_depth(hit, pixel_coords);
    }
    if (frame_info.debug_flags & FLAG_EMIT_FRAME_SURFACE_POSITION) != 0u {
        emit_frame_surface_position(hit, pixel_coords);
    }
}
fn emit_frame_surface_depth(hit: HitRecord, pixel_coords: vec2<i32>) {
    let depth = clamp(hit.world_hit_distance / camera.clip_aabb_max, 0.0, 1.0);
    let alpha = f32(is_hit_record_valid(hit));
    let rgba = vec4<f32>(vec3<f32>(depth), alpha);
    textureStore(frame_surface_depth_image, pixel_coords, rgba);
}
fn emit_frame_surface_position(hit: HitRecord, pixel_coords: vec2<i32>) {
    let clip_aabb_min = vec3<f32>(-10.0);
    let clip_aabb_max = vec3<f32>(10.0);
    let rgb = (hit.world_hit_position - clip_aabb_min) / (clip_aabb_max - clip_aabb_min);
    let alpha = f32(is_hit_record_valid(hit));
    let color = vec4<f32>(rgb, alpha);
    textureStore(frame_surface_position_image, pixel_coords, color);
}

fn debug_hook_post_primary_ray_hit_details(hit_details: HitDetails, pixel_coords: vec2<i32>) {
    if (frame_info.debug_flags & FLAG_EMIT_FRAME_SURFACE_COLOR) != 0u {
        emit_frame_surface_color(hit_details, pixel_coords);
    }
    if (frame_info.debug_flags & FLAG_EMIT_FRAME_SURFACE_NORMAL) != 0u {
        emit_frame_surface_normal(hit_details, pixel_coords);
    }
    if (frame_info.debug_flags & FLAG_EMIT_FRAME_SURFACE_ORM) != 0u {
        emit_frame_surface_orm(hit_details, pixel_coords);
    }
    if (frame_info.debug_flags & FLAG_EMIT_FRAME_SURFACE_EMISSIVE) != 0u {
        emit_frame_surface_emissive(hit_details, pixel_coords);
    }
}
fn emit_frame_surface_color(hit_details: HitDetails, pixel_coords: vec2<i32>) {
    textureStore(frame_surface_color_image, pixel_coords, hit_details.surface_color);
}
fn emit_frame_surface_normal(hit_details: HitDetails, pixel_coords: vec2<i32>) {
    let normal_rgb = convert_direction_to_rgb(hit_details.surface_normal);
    textureStore(frame_surface_normal_image, pixel_coords, vec4<f32>(normal_rgb, 1.0));
}
fn emit_frame_surface_orm(hit_details: HitDetails, pixel_coords: vec2<i32>) {
    let o = 1.0;  // Opacity placeholder
    let r = hit_details.surface_roughness;
    let m = hit_details.surface_metalness;
    let orm = vec4<f32>(o, r, m, 1.0);
    textureStore(frame_surface_orm_image, pixel_coords, orm);
}
fn emit_frame_surface_emissive(hit_details: HitDetails, pixel_coords: vec2<i32>) {
    textureStore(frame_surface_emissive_image, pixel_coords, vec4<f32>(hit_details.surface_emissive, 1.0));
}

fn debug_hook_post_path_trace_complete(radiance: vec3<f32>, pixel_coords: vec2<i32>) {
    if (frame_info.debug_flags & FRAME_PER_PIXEL_RADIANCE) != 0u {
        emit_frame_pixel_radiance(radiance, pixel_coords);
    }
}
fn emit_frame_pixel_radiance(radiance: vec3<f32>, pixel_coords: vec2<i32>) {
    textureStore(frame_per_pixel_radiance, pixel_coords, vec4<f32>(radiance, 1.0));
}

//
// render_pixel():
//

fn render_pixel(pixel_xy: vec2<u32>, seed: ptr<function, u32>) -> vec4<f32> {
    let ray = gen_primary_ray(pixel_xy);
    debug_hook_post_primary_ray_gen(ray, vec2<i32>(pixel_xy));

    let closest_hit = hit(ray);
    debug_hook_post_primary_ray_hit(closest_hit, vec2<i32>(pixel_xy));

    if !is_hit_record_valid(closest_hit) {
        return vec4<f32>(get_environment_radiance(ray), 1.0);
    }

    let hit_details = compute_hit_details(closest_hit);
    debug_hook_post_primary_ray_hit_details(hit_details, vec2<i32>(pixel_xy));

    let radiance = render_pixel_path_traced(pixel_xy, seed);
    debug_hook_post_path_trace_complete(radiance, vec2<i32>(pixel_xy));

    return vec4<f32>(radiance, 1.0);
}

fn gen_primary_ray(pixel_coord_px: vec2<u32>) -> Ray {
    let target_size = vec2<f32>(f32(frame_info.target_size_w_px), f32(frame_info.target_size_h_px));
    // Use pixel center (add 0.5) for non-jittered rays
    return gen_primary_ray_from_coord(vec2<f32>(pixel_coord_px) + vec2<f32>(0.5), target_size);
}

fn compute_pixel_seed(pixel_xy: vec2<u32>) -> u32 {
    let pixel_idx = pixel_xy.y * frame_info.target_size_w_px + pixel_xy.x;
    return pcg_hash(pixel_idx ^ (frame_info.frame_index * 0x9E3779B9u));
}

//
// Bootstrap compute shader entry point:
//

@compute @workgroup_size(8, 8, 1)
fn main_wrapper(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= frame_info.target_size_w_px || global_id.y >= frame_info.target_size_h_px {
        return;
    }

    let pixel_xy = global_id.xy;
    var seed = compute_pixel_seed(pixel_xy);

    var pixel_color = render_pixel(pixel_xy, &seed);

    finish_frame(pixel_xy, pixel_color);
}

fn finish_frame(pixel_xy: vec2<u32>, pixel_color: vec4<f32>) {
    // Accumulate into the accumulator image:
    let prev = textureLoad(accum_image, vec2<i32>(pixel_xy)).xyz;
    let curr = prev + pixel_color.xyz;
    textureStore(accum_image, vec2<i32>(pixel_xy), vec4<f32>(curr, 1.0));

    // Average and store into the output image:
    let accumulated_frame_count = 1u + frame_info.accumulated_frame_index;
    let average_pixel_color = curr / f32(accumulated_frame_count);
    textureStore(output_image, vec2<i32>(pixel_xy), vec4<f32>(average_pixel_color, 1.0));
}

//
// Postprocess shader:
//

struct PostprocessUniforms {
    debug_flags: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> postprocess_uniforms: PostprocessUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_postprocess(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var output: VertexOutput;
    output.position = compute_fullscreen_position(vertex_idx);
    output.uv = compute_fullscreen_uv(vertex_idx);
    return output;
}

fn compute_fullscreen_position(vertex_idx: u32) -> vec4<f32> {
    let x = f32(i32(vertex_idx & 1u) * 4 - 1);
    let y = f32(i32(vertex_idx & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn compute_fullscreen_uv(vertex_idx: u32) -> vec2<f32> {
    let x = f32(i32(vertex_idx & 1u) * 4 - 1);
    let y = f32(i32(vertex_idx & 2u) * 2 - 1);
    return vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
}

@fragment
fn fs_postprocess(input: VertexOutput) -> @location(0) vec4<f32> {
    let hdr_color = textureSample(input_texture, input_sampler, input.uv);

    // Skip tonemapping when debug flags are active (pass through raw values)
    if postprocess_uniforms.debug_flags != 0u {
        return hdr_color;
    }

    let exposed = apply_exposure(hdr_color.rgb, 1.5);
    let tonemapped = naughty_dog_tonemap(exposed);
    let gamma_corrected = linear_to_srgb(tonemapped);
    return vec4<f32>(gamma_corrected, 1.0);
}

fn apply_exposure(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    return color * exposure;
}

fn naughty_dog_tonemap(color: vec3<f32>) -> vec3<f32> {
    // Attempt to better capture Uncharted 2 / Naughty Dog filmic curve
    let A = 0.22;  // Shoulder strength
    let B = 0.30;  // Linear strength
    let C = 0.10;  // Linear angle
    let D = 0.20;  // Toe strength
    let E = 0.01;  // Toe numerator
    let F = 0.30;  // Toe denominator
    let white = 11.2;
    let num = naughty_dog_curve(color, A, B, C, D, E, F);
    let denom = naughty_dog_curve(vec3<f32>(white), A, B, C, D, E, F);
    return num / denom;
}

fn naughty_dog_curve(x: vec3<f32>, A: f32, B: f32, C: f32, D: f32, E: f32, F: f32) -> vec3<f32> {
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}
