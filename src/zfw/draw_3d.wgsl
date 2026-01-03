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
@group(1) @binding(2) var<uniform> camera: PodCamera;
@group(1) @binding(3) var<storage, read> instances: array<PodInstance>;

//
// Pod types: used for CPU-GPU data exchange.
//

struct PodFrameInfo {
    instance_count: u32,
    target_size_w_px: u32,
    target_size_h_px: u32,
    flags: u32,
}

const FLAG_EMIT_PRIMARY_RAY_DIRECTION: u32 = 1u;
const FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R: u32 = 2u;
const FLAG_EMIT_HIT_WORLD_POSITION: u32 = 4u;

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
    max_distance: f32,
    _rsv: u32,
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

//
// Linalg
//

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

//
// Ray
//

const TRIANGLE_RAY_INTERSECTION_EPSILON: f32 = 1e-7;
const F32_INFINITY: f32 = 1e8;  // WGSL does not have f32::INFINITY?

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,   // Does not need to be normalized, but never 0
}

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
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

    let t_close = max(max(t_close_percoeff.x, t_close_percoeff.y), t_close_percoeff.z);
    let t_far = min(min(t_far_percoeff.x, t_far_percoeff.y), t_far_percoeff.z);

    if t_close > t_far || t_far < 0.0 {
        // No intersection, or intersection is behind the ray origin.
        return F32_INFINITY;
    } else {
        // Intersection exists, return the distance to the nearest intersection point.
        return max(t_close, 0.0);
    }
}

/// Triangle-ray intersection test
/// Returns barycentric coordinates (XYZ) and distance (W) of closest hit. If no hit, (W) is infinity.
fn hit_tri(ray: Ray, triangle_vertices: mat3x3<f32>) -> vec4<f32> {
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

fn transform_ray(ray: Ray, transform: mat4x4<f32>) -> Ray {
    let origin_h = vec4<f32>(ray.origin, 1.0);
    let direction_h = vec4<f32>(ray.direction, 0.0);
    let transformed_origin_h = transform * origin_h;
    let transformed_direction_h = transform * direction_h;
    return Ray(transformed_origin_h.xyz, transformed_direction_h.xyz);
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
    return transform_ray(camera_space_ray, camera_transform);
}

//
// Ray tracing:
//

fn load_triangle_vertices(triangle_id: u32) -> mat3x3<f32> {
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

//
// Entry point:
//

fn pixel_main(pixel_xy: vec2<u32>) -> vec4<f32> {
    let debug_emit_primary_ray_direction = (frame_info.flags & FLAG_EMIT_PRIMARY_RAY_DIRECTION) != 0u;
    let debug_emit_depth_in_r = (frame_info.flags & FLAG_EMIT_CLOSEST_HIT_DEPTH_IN_R) != 0u;
    let debug_emit_hit_world_pos = (frame_info.flags & FLAG_EMIT_HIT_WORLD_POSITION) != 0u;

    let ray = gen_primary_ray(pixel_xy);
    
    var closest_hit_distance = F32_INFINITY;
    var closest_hit_world_pos = vec3<f32>(0.0);
    var hit_count = 0u;
    
    for (var instance_id = 0u; instance_id < frame_info.instance_count; instance_id = instance_id + 1u) {
        let instance = instances[instance_id];
        let instance_transform = h_mat4x4_from_pod_transform(instance.transform);
        let inv_instance_transform = h_mat4x4_from_pod_transform(instance.inv_transform);
        
        // Transform ray to model space by applying inverse instance transform
        let local_ray = transform_ray(ray, inv_instance_transform);

        // Access geometry
        let geometry = geometry_heap[instance.geometry_id];
        
        // For simplicity, we just iterate over all triangles in the geometry.
        let triangle_span = geometry.triangle_span_in_heap;
        for (var tri_index = triangle_span.begin; tri_index < triangle_span.end; tri_index = tri_index + 1u) {
            let triangle_vertices = load_triangle_vertices(tri_index);

            // Test ray against triangle in local space
            let hit_result = hit_tri(local_ray, triangle_vertices);
            if hit_result.w > 0.0 {
                // Compute hit position in local space
                let local_hit_pos = local_ray.origin + hit_result.w * local_ray.direction;

                // Transform to world space
                let world_hit_pos_h = instance_transform * vec4<f32>(local_hit_pos, 1.0);
                let world_hit_pos = world_hit_pos_h.xyz / world_hit_pos_h.w;
                // Compute world-space distance from camera
                let world_distance = length(world_hit_pos - ray.origin);
                
                if world_distance < closest_hit_distance {
                    closest_hit_distance = world_distance;
                    closest_hit_world_pos = world_hit_pos;
                    hit_count = hit_count + 1u;
                }
            }
        }
    }

    // DEBUG: emit primary ray direction if flag is set
    if debug_emit_primary_ray_direction {
        let dir_normalized = normalize(ray.direction);
        return vec4<f32>(dir_normalized * 0.5 + 0.5, 1.0);
    }
    
    // DEBUG: emit hit world position
    if debug_emit_hit_world_pos {
        var alpha = 0.0;
        if closest_hit_distance < F32_INFINITY {
            alpha = 1.0;
        }
        // Map world position to [0,1] for visualization (assume scene is within [-10,10] cube)
        let pos_normalized = (closest_hit_world_pos + 10.0) / 20.0;
        return vec4<f32>(pos_normalized, alpha);
    }
    
    // DEBUG: Visualize hits
    if debug_emit_depth_in_r {
        var alpha = 0.0;
        if closest_hit_distance < F32_INFINITY {
            alpha = 1.0;
        }
        let depth_normalized = clamp(closest_hit_distance / camera.max_distance, 0.0, 1.0);
        return vec4<f32>(depth_normalized, 0.0, 0.0, alpha);
    }
    
    // If hit, return red
    if closest_hit_distance < F32_INFINITY {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }

    // No hit - show light grey
    return vec4<f32>(0.5, 0.5, 0.5, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= frame_info.target_size_w_px || global_id.y >= frame_info.target_size_h_px {
        return;
    }

    let pixel_color = pixel_main(global_id.xy);
    textureStore(output_image, vec2<i32>(global_id.xy), pixel_color);
}
