enable f16;

//
// Bindings:
//

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
@group(0) @binding(14) var linear_sampler: sampler;

// Per-frame bind group:
@group(1) @binding(0) var output_image: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> frame_info: PodFrameInfo;
@group(1) @binding(2) var<uniform> camera: PodCamera;
@group(1) @binding(3) var<storage, read> instances: array<PodInstance>;

//
// Constants and configuration:
//

/// A large finite value to represent "infinity" in ray intersection tests.
const F32_INFINITY: f32 = 1e8;  // WGSL does not have f32::INFINITY?

/// Epsilon value for triangle-ray intersection tests, used when ray is nearly parallel to triangle plane.
const TRIANGLE_RAY_INTERSECTION_EPSILON: f32 = 1e-8;

/// Max BVH traversal stack depth
const MAX_STACK_DEPTH: u32 = 64u;
    

//
// Pod types: used for CPU-GPU data exchange.
//

struct PodFrameInfo {
    instance_count: u32,
    target_size_w_px: u32,
    target_size_h_px: u32,
    debug_flags: u32,
    environment_map_texture_id: i32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const FLAG_EMIT_PRIMARY_RAY_DIRECTION: u32 = 1u;
const FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R: u32 = 2u;
const FLAG_EMIT_HIT_WORLD_POSITION: u32 = 4u;
const FLAG_EMIT_CLOSEST_BVH_HIT_DEPTH_IN_R: u32 = 8u;
const FLAG_EMIT_PRIMARY_RAY_COLOR: u32 = 16u;
const FLAG_EMIT_HIT_NORMAL: u32 = 32u;
const FLAG_EMIT_ORM: u32 = 64u;

const POST_PRIMARY_RAY_GEN_DEBUG_MASK: u32 = FLAG_EMIT_PRIMARY_RAY_DIRECTION;
const POST_PRIMARY_RAY_HIT_DEBUG_MASK: u32 = FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R | FLAG_EMIT_HIT_WORLD_POSITION | FLAG_EMIT_CLOSEST_BVH_HIT_DEPTH_IN_R;
const POST_PRIMARY_RAY_HIT_DETAILS_DEBUG_MASK: u32 = FLAG_EMIT_PRIMARY_RAY_COLOR | FLAG_EMIT_HIT_NORMAL | FLAG_EMIT_ORM;

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

struct PodMaterial {
    color_map_id: u32,
    color_factor: array<f32, 3>,
    normal_map_id: u32,
    metalness_map_id: u32,
    metalness_factor: f32,
    roughness_map_id: u32,
    roughness_factor: f32,
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
    let z = sqrt(1.0 - dot(xy, xy));
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
    let t_inv = -r_inv * t;

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
    let debug_bvh_traversal = (frame_info.debug_flags & FLAG_EMIT_CLOSEST_BVH_HIT_DEPTH_IN_R) != 0u;
    
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
            // Test both children's AABBs and push them in order of distance (closer first)
            let child0_idx = node.children[0];
            let child1_idx = node.children[1];
            
            var child0_dist = F32_INFINITY;
            var child1_dist = F32_INFINITY;
            
            if child0_idx != 0u {
                let child0_node = bvh_node_heap[bvh_node_span.begin + child0_idx];
                let child0_aabb = Aabb(
                    vec3<f32>(child0_node.aabb.min[0], child0_node.aabb.min[1], child0_node.aabb.min[2]),
                    vec3<f32>(child0_node.aabb.max[0], child0_node.aabb.max[1], child0_node.aabb.max[2]),
                );
                child0_dist = hit_aabb(ray, child0_aabb);
            }
            
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
                // Push child1 first (farther), then child0 (closer)
                if child1_dist < closest_hit.triangle_raycast_result.w && stack_ptr < MAX_STACK_DEPTH {
                    stack[stack_ptr] = bvh_node_span.begin + child1_idx;
                    stack_ptr += 1u;
                }
                if child0_dist < closest_hit.triangle_raycast_result.w && stack_ptr < MAX_STACK_DEPTH {
                    stack[stack_ptr] = bvh_node_span.begin + child0_idx;
                    stack_ptr += 1u;
                }
            } else {
                // Push child0 first (farther), then child1 (closer)
                if child0_dist < closest_hit.triangle_raycast_result.w && stack_ptr < MAX_STACK_DEPTH {
                    stack[stack_ptr] = bvh_node_span.begin + child0_idx;
                    stack_ptr += 1u;
                }
                if child1_dist < closest_hit.triangle_raycast_result.w && stack_ptr < MAX_STACK_DEPTH {
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
// PBR shading (WIP):
//

fn compute_hit_color(hit_details: HitDetails) -> vec4<f32> {
    let sun_direction = normalize(vec3<f32>(-1.0, -1.0, -0.5));
    let color = hit_details.surface_color;
    let normal = hit_details.surface_normal;
    let intensity = clamp(dot(normal, sun_direction), 0.05, 1.0);
    return vec4<f32>(color.xyz * intensity, 1.0);
}

fn compute_miss_color(ray: Ray) -> vec4<f32> {
    // If environment map is available, sample it
    if frame_info.environment_map_texture_id >= 0 {
        return sample_environment_map(ray);
    } else {
        return sample_procedural_environment_map(ray);
    }
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
    let model_normal = normalize(bc.x * normal[0] + bc.y * normal[1] + bc.z * normal[2]);
    
    // Compute tangent and bitangent in model space using edge vectors and UV deltas
    let edge1 = pos[1] - pos[0];
    let edge2 = pos[2] - pos[0];
    let delta_uv1 = uv[1] - uv[0];
    let delta_uv2 = uv[2] - uv[0];
    
    let uv_det = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;
    let r = 1.0 / uv_det;
    let model_tangent = (edge1 * delta_uv2.y - edge2 * delta_uv1.y) * r;
    let model_bitangent = (edge2 * delta_uv1.x - edge1 * delta_uv2.x) * r;
    
    // Transform to world space, accounting for non-uniform scale
    let transform = h_mat4x4_from_pod_transform(instance.transform);
    let inv_transform = h_mat4x4_from_pod_transform(instance.inv_transform);
    
    // For normals with non-uniform scale, use inverse transpose of the 3x3 rotation/scale part
    let normal_transform = transpose(mat3x3<f32>(inv_transform[0].xyz, inv_transform[1].xyz, inv_transform[2].xyz));
    
    // For tangent/bitangent (surface directions), use the regular 3x3 transform
    let tangent_transform = mat3x3<f32>(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    
    // Apply transforms
    var world_normal = normalize(normal_transform * model_normal);
    var world_tangent = normalize(tangent_transform * model_tangent);
    var world_bitangent = normalize(tangent_transform * model_bitangent);
    
    // Gram-Schmidt orthonormalization to ensure TBN is orthonormal after transformation
    // Re-orthogonalize tangent against normal
    world_tangent = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);

    // Re-orthogonalize bitangent against both normal and tangent
    world_bitangent = normalize(
        world_bitangent 
        - dot(world_bitangent, world_normal) * world_normal 
        - dot(world_bitangent, world_tangent) * world_tangent
    );
    
    // Return TBN matrix with T, B, N as column vectors
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
    let normal = hit_details.tbn * texture_normal;
    return normalize(normal);   // for good measure
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

//
// Debug shading:
//

fn post_primary_ray_gen_debug_output(ray: Ray) -> vec4<f32> {
    let debug_emit_primary_ray_direction = (frame_info.debug_flags & FLAG_EMIT_PRIMARY_RAY_DIRECTION) != 0u;
    if debug_emit_primary_ray_direction {
        return debug_visualize_primary_ray_direction(ray);
    }

    // No debug flag matched, return magenta to indicate error.
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
fn debug_visualize_primary_ray_direction(ray: Ray) -> vec4<f32> {
    let dir_normalized = normalize(ray.direction);
    return vec4<f32>(dir_normalized * 0.5 + 0.5, 1.0);
}


fn post_primary_ray_hit_debug_output(hit: HitRecord) -> vec4<f32> {
    let debug_emit_depth_in_r = (frame_info.debug_flags & FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R) != 0u;
    if debug_emit_depth_in_r {
        return debug_visualize_depth_in_r(hit);
    }

    let debug_emit_hit_world_pos = (frame_info.debug_flags & FLAG_EMIT_HIT_WORLD_POSITION) != 0u;
    if debug_emit_hit_world_pos {
        return debug_visualize_hit_world_position(hit);
    }

    let debug_bvh_traversal = (frame_info.debug_flags & FLAG_EMIT_CLOSEST_BVH_HIT_DEPTH_IN_R) != 0u;
    if debug_bvh_traversal {
        return debug_visualize_depth_in_r(hit);
    }

    // No debug flag matched, return magenta to indicate error.
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
fn debug_visualize_primary_ray_color(hit_details: HitDetails) -> vec4<f32> {
    // Get material from instance and sample its color map
    let instance = instances[hit_details.instance_id];
    let material = material_heap[instance.material_id];
    let color_map_id = material.color_map_id;
    let color = sample_color_texture(color_map_id, hit_details.texcoords);
    return vec4<f32>(color, 1.0);
}

fn debug_visualize_hit_normal(hit_details: HitDetails) -> vec4<f32> {
    return vec4<f32>((hit_details.surface_normal + 1.0) * 0.5, 1.0);
}
fn debug_visualize_depth_in_r(hit: HitRecord) -> vec4<f32> {
    if is_hit_record_valid(hit) {
        let depth_normalized = clamp(hit.world_hit_distance / camera.clip_aabb_max, 0.0, 1.0);
        return vec4<f32>(depth_normalized, 0.0, 0.0, 1.0);
    } else {
        return vec4<f32>(0.0);
    }
}

fn post_primary_ray_hit_details_debug_output(hit_details: HitDetails) -> vec4<f32> {
    let debug_emit_primary_ray_color = (frame_info.debug_flags & FLAG_EMIT_PRIMARY_RAY_COLOR) != 0u;
    if debug_emit_primary_ray_color {
        return debug_visualize_primary_ray_color(hit_details);
    }

    let debug_emit_hit_normal = (frame_info.debug_flags & FLAG_EMIT_HIT_NORMAL) != 0u;
    if debug_emit_hit_normal {
        return debug_visualize_hit_normal(hit_details);
    }

    let debug_emit_orm = (frame_info.debug_flags & FLAG_EMIT_ORM) != 0u;
    if debug_emit_orm {
        return debug_visualize_orm(hit_details);
    }

    // No debug flag matched, return magenta to indicate error.
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}

fn debug_visualize_hit_world_position(hit: HitRecord) -> vec4<f32> {
    let clip_aabb_min = vec3<f32>(-10.0);
    let clip_aabb_max = vec3<f32>(10.0);
    if is_hit_record_valid(hit) {
        let pos_normalized = (hit.world_hit_position - clip_aabb_min) / (clip_aabb_max - clip_aabb_min);
        return vec4<f32>(pos_normalized, 1.0);
    } else {
        return vec4<f32>(0.0);
    }
}
fn debug_visualize_orm(hit_details: HitDetails) -> vec4<f32> {
    // Visualize ORM: Opacity, Roughness, Metalness in RGB channels
    // For now, using placeholder values - will be replaced with actual material sampling
    // TODO: Sample from material maps based on hit_details
    let opacity = 1.0;  // Placeholder: sample from opacity/alpha texture
    let roughness = 0.5;  // Placeholder: sample from roughness texture
    let metalness = 0.5;  // Placeholder: sample from metalness texture
    return vec4<f32>(opacity, roughness, metalness, 1.0);
}

//
// Entry point:
//

fn main(pixel_xy: vec2<u32>) -> vec4<f32> {
    let ray = gen_primary_ray(pixel_xy);
    if (frame_info.debug_flags & POST_PRIMARY_RAY_GEN_DEBUG_MASK) != 0u {
        return post_primary_ray_gen_debug_output(ray);
    }

    let closest_hit = hit(ray);
    if (frame_info.debug_flags & POST_PRIMARY_RAY_HIT_DEBUG_MASK) != 0u {
        return post_primary_ray_hit_debug_output(closest_hit);
    }

    if !is_hit_record_valid(closest_hit) {
        return compute_miss_color(ray);
    } else {
        var closest_hit_details = compute_hit_details(closest_hit);
        if (frame_info.debug_flags & POST_PRIMARY_RAY_HIT_DETAILS_DEBUG_MASK) != 0u {
            return post_primary_ray_hit_details_debug_output(closest_hit_details);
        }
        return compute_hit_color(closest_hit_details);
    }
}

fn gen_primary_ray(pixel_coord_px: vec2<u32>) -> Ray {
    // Compute 2D NDC coordinates of the pixel in the output image:
    let target_size_wh_px_f = vec2<f32>(f32(frame_info.target_size_w_px), f32(frame_info.target_size_h_px));
    let pixel_coord_px_f = vec2<f32>(pixel_coord_px);
    let pixel_coord_ndc = vec2<f32>(
        (pixel_coord_px_f.x / target_size_wh_px_f.x) * 2.0 - 1.0,
        1.0 - (pixel_coord_px_f.y / target_size_wh_px_f.y) * 2.0,
    );

    // Compute 3D camera-space coordinates of the pixel on the sensor plane at unit focal length.
    // NOTE: Camera looks down +Y axis, with +X to the right and +Z up.
    // NOTE: In a pinhole camera, the sensor plane is behind the pinhole and the image is inverted. To simplify, we 
    // place the sensor plane in front of the pinhole. The ray still originates from origin in camera space, this is 
    // just used to calculate the ray direction.
    let sensor_hw_at_unit_focal_length = tan(camera.fov_y_rad / 2.0) * camera.aspect_ratio;
    let sensor_hh_at_unit_focal_length = tan(camera.fov_y_rad / 2.0);
    let sensor_pixel_camera_space = vec3<f32>(
        pixel_coord_ndc.x * sensor_hw_at_unit_focal_length,     // sensor right
        1.0,                                                    // sensor plane at unit focal length (forward)
        pixel_coord_ndc.y * sensor_hh_at_unit_focal_length,     // sensor up
    );

    // Create ray in camera space, then transform to world space.
    let camera_space_ray = Ray(vec3<f32>(0.0, 0.0, 0.0), sensor_pixel_camera_space);
    let camera_transform = h_mat4x4_from_pod_transform(camera.transform);
    return h_mat4x4_transform_ray(camera_transform, camera_space_ray);
}

//
// Bootstrap compute shader entry point:
//

@compute @workgroup_size(8, 8, 1)
fn main_wrapper(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= frame_info.target_size_w_px || global_id.y >= frame_info.target_size_h_px {
        return;
    }

    let pixel_color = main(global_id.xy);

    textureStore(output_image, vec2<i32>(global_id.xy), pixel_color);
}
