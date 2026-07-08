//! Shared machinery for the `trace_rays` compute fallback: host BVH build
//! over triangle geometry and the WGSL traversal kernel, used by both GPU
//! backends. The hardware ray-query kernel lives here too so the two
//! backends bind the same shader text (`docs/hw-nodes.md`).
//!
//! # BVH layout (mirrored by the WGSL in [`fallback_kernel`])
//!
//! Flat array of 32-byte nodes, 8 `u32` words each:
//!
//! ```text
//! word 0..2  AABB min xyz (f32 bits)     word 3  a
//! word 4..6  AABB max xyz (f32 bits)     word 7  b
//! ```
//!
//! Leaf when `b & LEAF_FLAG != 0`: `a` = first packed-triangle index,
//! `b & !LEAF_FLAG` = triangle count. Inner node: `a` = left child index and
//! the right child is `a + 1` (children are allocated adjacently), `b = 0`.
//! Node 0 is the root; an empty scene is a single zero-count leaf.
//!
//! Packed triangles are `(i0, i1, i2, prim)` u32×4 in BVH-leaf order, with
//! vertex indices clamped in-bounds (mirroring gather semantics and the CPU
//! reference). Packing by leaf order keeps the fallback at exactly the
//! 8-storage-buffer WebGPU minimum: output + origins + directions + t_min +
//! t_max + vertices + packed triangles + nodes.

/// Marks a leaf in a node's `b` word; low 31 bits hold the triangle count.
const LEAF_FLAG: u32 = 0x8000_0000;

/// Max triangles per leaf (small: leaves are tested brute-force in-shader).
const LEAF_SIZE: usize = 4;

/// Threads per workgroup for both trace kernels (one ray per thread).
pub(crate) const WORKGROUP_SIZE: u32 = 64;

/// Traversal stack depth in the fallback kernel. Median splits keep the tree
/// balanced (depth ≈ log2(T / LEAF_SIZE)), so 64 covers any `T < 2^24`.
const STACK_DEPTH: u32 = 64;

/// WebGPU/Vulkan guaranteed minimum for workgroups per dispatch dimension.
pub(crate) const MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

/// Workgroups along x for `ray_count` rays. Errors instead of tripping
/// driver validation when the naive 1-D dispatch exceeds the per-dimension
/// limit (~4.19M rays; splitting across y is a follow-up).
pub(crate) fn dispatch_x(ray_count: u32) -> Result<u32, String> {
    let x = ray_count.div_ceil(WORKGROUP_SIZE);
    if x > MAX_WORKGROUPS_PER_DIM {
        return Err(format!(
            "trace_rays: {ray_count} rays need {x} workgroups, exceeding the \
             {MAX_WORKGROUPS_PER_DIM} per-dimension dispatch limit"
        ));
    }
    Ok(x)
}

/// Testing override (`docs/hw-nodes.md`: mode selection is "overridable for
/// testing"): forces the compute fallback even on ray-tracing hardware.
pub(crate) fn force_fallback() -> bool {
    std::env::var_os("RESIN_TRACE_FORCE_FALLBACK").is_some_and(|v| v != "0")
}

/// Host-built BVH ready for upload (see the module docs for the layout).
pub(crate) struct TraceBvh {
    /// Flat nodes, 8 `u32` words each.
    pub nodes: Vec<u32>,
    /// `(i0, i1, i2, prim)` per triangle in leaf order; never empty (a zero
    /// dummy entry keeps GPU buffer creation trivial for empty scenes).
    pub packed_tris: Vec<u32>,
}

/// Per-triangle build input: bounds and centroid, indexed by original id.
struct BuildInput {
    boxes: Vec<([f32; 3], [f32; 3])>,
    centroids: Vec<[f32; 3]>,
}

/// Build a median-split BVH over `triangles` (`[T*3]` indices into
/// `vertices` `[V*3]`). Degenerate splits (coincident centroids) fall back
/// to larger leaves.
pub(crate) fn build_bvh(vertices: &[f32], triangles: &[u32]) -> TraceBvh {
    let triangle_count = triangles.len() / 3;
    let vertex_count = vertices.len() / 3;

    if triangle_count == 0 {
        return TraceBvh {
            nodes: flatten_node(&Node::leaf([0.0; 3], [0.0; 3], 0, 0)),
            packed_tris: vec![0; 4],
        };
    }

    // Per-triangle bounds and centroids (indices clamped like gathers).
    let clamp = vertex_count.saturating_sub(1) as u32;
    let corner = |tri: usize, slot: usize| -> [f32; 3] {
        let idx = triangles[tri * 3 + slot].min(clamp) as usize * 3;
        [vertices[idx], vertices[idx + 1], vertices[idx + 2]]
    };
    let mut input = BuildInput {
        boxes: Vec::with_capacity(triangle_count),
        centroids: Vec::with_capacity(triangle_count),
    };
    for tri in 0..triangle_count {
        let (a, b, c) = (corner(tri, 0), corner(tri, 1), corner(tri, 2));
        let mut lo = [0.0f32; 3];
        let mut hi = [0.0f32; 3];
        for axis in 0..3 {
            lo[axis] = a[axis].min(b[axis]).min(c[axis]);
            hi[axis] = a[axis].max(b[axis]).max(c[axis]);
        }
        input.centroids.push([
            (lo[0] + hi[0]) * 0.5,
            (lo[1] + hi[1]) * 0.5,
            (lo[2] + hi[2]) * 0.5,
        ]);
        input.boxes.push((lo, hi));
    }

    let mut order: Vec<u32> = (0..triangle_count as u32).collect();
    let mut nodes = vec![Node::leaf([0.0; 3], [0.0; 3], 0, 0)];
    build_node(&mut nodes, 0, &mut order, 0, triangle_count, &input);

    let mut packed_tris = Vec::with_capacity(triangle_count * 4);
    for &tri in &order {
        for slot in 0..3 {
            packed_tris.push(triangles[tri as usize * 3 + slot].min(clamp));
        }
        packed_tris.push(tri);
    }

    TraceBvh {
        nodes: nodes.iter().flat_map(flatten_node).collect(),
        packed_tris,
    }
}

#[derive(Clone, Copy)]
struct Node {
    min: [f32; 3],
    max: [f32; 3],
    a: u32,
    b: u32,
}

impl Node {
    fn leaf(min: [f32; 3], max: [f32; 3], first: u32, count: u32) -> Self {
        Self {
            min,
            max,
            a: first,
            b: count | LEAF_FLAG,
        }
    }
}

fn flatten_node(node: &Node) -> Vec<u32> {
    vec![
        node.min[0].to_bits(),
        node.min[1].to_bits(),
        node.min[2].to_bits(),
        node.a,
        node.max[0].to_bits(),
        node.max[1].to_bits(),
        node.max[2].to_bits(),
        node.b,
    ]
}

/// Fill `nodes[index]` from `order[start..start + count]`, recursing on
/// median splits. `order` is permuted in place so leaves reference
/// contiguous runs of the final packed-triangle array.
fn build_node(
    nodes: &mut Vec<Node>,
    index: usize,
    order: &mut [u32],
    start: usize,
    count: usize,
    input: &BuildInput,
) {
    let range = &order[start..start + count];

    // Node bounds: union of member boxes, padded so the conservative slab
    // test can never falsely reject a triangle sitting exactly on a face
    // (zero-thickness boxes from axis-aligned triangles).
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    let mut centroid_lo = [f32::INFINITY; 3];
    let mut centroid_hi = [f32::NEG_INFINITY; 3];
    for &tri in range {
        let (tlo, thi) = input.boxes[tri as usize];
        let centroid = input.centroids[tri as usize];
        for axis in 0..3 {
            lo[axis] = lo[axis].min(tlo[axis]);
            hi[axis] = hi[axis].max(thi[axis]);
            centroid_lo[axis] = centroid_lo[axis].min(centroid[axis]);
            centroid_hi[axis] = centroid_hi[axis].max(centroid[axis]);
        }
    }
    for axis in 0..3 {
        let pad = (hi[axis] - lo[axis]) * 1e-6 + 1e-6;
        lo[axis] -= pad;
        hi[axis] += pad;
    }

    // Split the widest centroid axis; a degenerate spread (all centroids
    // coincident) cannot partition, so keep an oversized leaf instead.
    let split_axis = (0..3)
        .max_by(|&a, &b| {
            (centroid_hi[a] - centroid_lo[a]).total_cmp(&(centroid_hi[b] - centroid_lo[b]))
        })
        .expect("three axes");
    let spread = centroid_hi[split_axis] - centroid_lo[split_axis];
    if count <= LEAF_SIZE || spread <= 0.0 {
        nodes[index] = Node::leaf(lo, hi, start as u32, count as u32);
        return;
    }

    let mid = count / 2;
    order[start..start + count].select_nth_unstable_by(mid, |&a, &b| {
        input.centroids[a as usize][split_axis]
            .total_cmp(&input.centroids[b as usize][split_axis])
    });

    let left = nodes.len();
    nodes.push(Node::leaf([0.0; 3], [0.0; 3], 0, 0));
    nodes.push(Node::leaf([0.0; 3], [0.0; 3], 0, 0));
    nodes[index] = Node {
        min: lo,
        max: hi,
        a: left as u32,
        b: 0,
    };
    build_node(nodes, left, order, start, mid, input);
    build_node(nodes, left + 1, order, start + mid, count - mid, input);
}

/// WGSL compute kernel: hardware closest-hit via `ray_query` against a bound
/// acceleration structure. Bindings: 0 = output records, 1..4 = origins /
/// directions / t_min / t_max, 5 = TLAS. Shared verbatim by wgpu (compiled
/// as WGSL) and Vulkan (naga → SPIR-V with the `RAY_QUERY` capability).
pub(crate) fn ray_query_kernel(ray_count: u32) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> origins: array<f32>;
@group(0) @binding(2) var<storage, read> directions: array<f32>;
@group(0) @binding(3) var<storage, read> t_min: array<f32>;
@group(0) @binding(4) var<storage, read> t_max: array<f32>;
@group(0) @binding(5) var tlas: acceleration_structure;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let ray = gid.x;
    if (ray >= {n}u) {{
        return;
    }}
    let o = vec3<f32>(origins[3u * ray], origins[3u * ray + 1u], origins[3u * ray + 2u]);
    let d = vec3<f32>(directions[3u * ray], directions[3u * ray + 1u], directions[3u * ray + 2u]);
    // Flags 0: no culling (the node has none) and opaque geometry as marked
    // in the BLAS. `t` stays in units of the (unnormalized) direction.
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xffu, t_min[ray], t_max[ray], o, d));
    while (rayQueryProceed(&rq)) {{
    }}
    let hit = rayQueryGetCommittedIntersection(&rq);
    let base = 5u * ray;
    if (hit.kind == RAY_QUERY_INTERSECTION_TRIANGLE) {{
        // Hardware barycentrics weight vertices 1 and 2, matching the record.
        output[base] = hit.t;
        output[base + 1u] = hit.barycentrics.x;
        output[base + 2u] = hit.barycentrics.y;
        output[base + 3u] = f32(hit.primitive_index);
        output[base + 4u] = 1.0;
    }} else {{
        output[base] = 0.0;
        output[base + 1u] = 0.0;
        output[base + 2u] = 0.0;
        output[base + 3u] = 0.0;
        output[base + 4u] = 0.0;
    }}
}}
"#,
        wg = WORKGROUP_SIZE,
        n = ray_count,
    )
}

/// WGSL compute kernel: BVH traversal fallback (runs on any adapter).
/// Bindings: 0 = output records, 1..4 = origins / directions / t_min /
/// t_max, 5 = vertices, 6 = packed triangles, 7 = nodes — exactly the
/// 8-storage-buffer WebGPU minimum. Node/triangle words match
/// [`build_bvh`]; the intersection math matches the CPU reference.
pub(crate) fn fallback_kernel(ray_count: u32) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> origins: array<f32>;
@group(0) @binding(2) var<storage, read> directions: array<f32>;
@group(0) @binding(3) var<storage, read> t_min: array<f32>;
@group(0) @binding(4) var<storage, read> t_max: array<f32>;
@group(0) @binding(5) var<storage, read> vertices: array<f32>;
@group(0) @binding(6) var<storage, read> tris: array<u32>;
@group(0) @binding(7) var<storage, read> nodes: array<u32>;

fn vertex_at(i: u32) -> vec3<f32> {{
    return vec3<f32>(vertices[3u * i], vertices[3u * i + 1u], vertices[3u * i + 2u]);
}}

// Möller–Trumbore without backface culling, mirroring the CPU reference.
// Returns (t, u, v, valid).
fn tri_intersect(o: vec3<f32>, d: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> vec4<f32> {{
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-12) {{
        return vec4<f32>(0.0);
    }}
    let inv_det = 1.0 / det;
    let s = o - v0;
    let u = dot(s, p) * inv_det;
    if (u < 0.0 || u > 1.0) {{
        return vec4<f32>(0.0);
    }}
    let q = cross(s, e1);
    let v = dot(d, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) {{
        return vec4<f32>(0.0);
    }}
    return vec4<f32>(dot(e2, q) * inv_det, u, v, 1.0);
}}

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let ray = gid.x;
    if (ray >= {n}u) {{
        return;
    }}
    let o = vec3<f32>(origins[3u * ray], origins[3u * ray + 1u], origins[3u * ray + 2u]);
    let d = vec3<f32>(directions[3u * ray], directions[3u * ray + 1u], directions[3u * ray + 2u]);
    let lo = t_min[ray];
    let hi = t_max[ray];
    // Finite stand-in for 1/0 keeps the slab test IEEE-free: with the ray
    // origin inside a zero-direction slab both bounds keep their signs, and
    // outside it both blow up the same way (no 0 * inf NaNs).
    let inv_d = select(1.0 / d, vec3<f32>(1e30), abs(d) < vec3<f32>(1e-30));

    var best = vec4<f32>(0.0, 0.0, 0.0, 0.0); // u, v, prim, hit
    var best_t = hi;
    var stack: array<u32, {stack}>;
    stack[0] = 0u;
    var sp = 1u;
    while (sp > 0u) {{
        sp = sp - 1u;
        let base = 8u * stack[sp];
        let bmin = vec3<f32>(
            bitcast<f32>(nodes[base]),
            bitcast<f32>(nodes[base + 1u]),
            bitcast<f32>(nodes[base + 2u]));
        let a = nodes[base + 3u];
        let bmax = vec3<f32>(
            bitcast<f32>(nodes[base + 4u]),
            bitcast<f32>(nodes[base + 5u]),
            bitcast<f32>(nodes[base + 6u]));
        let b = nodes[base + 7u];

        let t0 = (bmin - o) * inv_d;
        let t1 = (bmax - o) * inv_d;
        let near = min(t0, t1);
        let far = max(t0, t1);
        let enter = max(max(near.x, near.y), max(near.z, lo));
        let exit = min(min(far.x, far.y), min(far.z, best_t));
        if (enter > exit) {{
            continue;
        }}
        if ((b & {leaf_flag}u) != 0u) {{
            let count = b & {leaf_mask}u;
            for (var k = 0u; k < count; k = k + 1u) {{
                let tri = 4u * (a + k);
                let cand = tri_intersect(
                    o, d, vertex_at(tris[tri]), vertex_at(tris[tri + 1u]), vertex_at(tris[tri + 2u]));
                // First hit accepts t == t_max (like the CPU reference);
                // afterwards only strictly closer hits win.
                if (cand.w != 0.0 && cand.x >= lo
                    && ((best.w == 0.0 && cand.x <= best_t) || cand.x < best_t)) {{
                    best_t = cand.x;
                    best = vec4<f32>(cand.y, cand.z, f32(tris[tri + 3u]), 1.0);
                }}
            }}
        }} else {{
            stack[sp] = a;
            stack[sp + 1u] = a + 1u;
            sp = sp + 2u;
        }}
    }}
    let out = 5u * ray;
    if (best.w != 0.0) {{
        output[out] = best_t;
        output[out + 1u] = best.x;
        output[out + 2u] = best.y;
        output[out + 3u] = best.z;
        output[out + 4u] = 1.0;
    }} else {{
        output[out] = 0.0;
        output[out + 1u] = 0.0;
        output[out + 2u] = 0.0;
        output[out + 3u] = 0.0;
        output[out + 4u] = 0.0;
    }}
}}
"#,
        wg = WORKGROUP_SIZE,
        n = ray_count,
        stack = STACK_DEPTH,
        leaf_flag = LEAF_FLAG,
        leaf_mask = !LEAF_FLAG,
    )
}

/// Reinterpret little-endian bytes as f32 (device readback → host build).
pub(crate) fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Reinterpret little-endian bytes as u32 (device readback → host build).
pub(crate) fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Little-endian bytes for u32 words (host build → device upload).
pub(crate) fn u32s_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    //! BVH + kernel-text tests that need crate-private access (the fallback
    //! cannot be forced through the public API when ray-tracing hardware is
    //! present; see `tests/trace_gpu.rs` for the end-to-end parity suite).

    use super::*;

    /// Deterministic xorshift* for random scenes/rays.
    struct Rng(u64);

    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }

        fn in_range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_f32()
        }
    }

    fn random_scene(rng: &mut Rng, tris: usize) -> (Vec<f32>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for tri in 0..tris {
            let cx = rng.in_range(-1.0, 1.0);
            let cy = rng.in_range(-1.0, 1.0);
            let cz = rng.in_range(-1.0, 1.0);
            for _ in 0..3 {
                vertices.push(cx + rng.in_range(-0.4, 0.4));
                vertices.push(cy + rng.in_range(-0.4, 0.4));
                vertices.push(cz + rng.in_range(-0.4, 0.4));
            }
            triangles.extend([tri as u32 * 3, tri as u32 * 3 + 1, tri as u32 * 3 + 2]);
        }
        (vertices, triangles)
    }

    /// Host mirror of the WGSL traversal (same node decoding and accept
    /// rule), used to validate the builder against brute force.
    fn traverse(
        bvh: &TraceBvh,
        vertices: &[f32],
        o: [f32; 3],
        d: [f32; 3],
        lo: f32,
        hi: f32,
    ) -> Option<(f32, u32)> {
        let vertex_at = |i: u32| -> [f32; 3] {
            let i = i as usize * 3;
            [vertices[i], vertices[i + 1], vertices[i + 2]]
        };
        let inv_d = |axis: usize| {
            if d[axis].abs() < 1e-30 {
                1e30
            } else {
                1.0 / d[axis]
            }
        };
        let inv = [inv_d(0), inv_d(1), inv_d(2)];

        let mut best: Option<(f32, u32)> = None;
        let mut best_t = hi;
        let mut stack = vec![0u32];
        while let Some(node) = stack.pop() {
            let base = node as usize * 8;
            let words = &bvh.nodes[base..base + 8];
            let mut enter = lo;
            let mut exit = best_t;
            for axis in 0..3 {
                let bmin = f32::from_bits(words[axis]);
                let bmax = f32::from_bits(words[axis + 4]);
                let t0 = (bmin - o[axis]) * inv[axis];
                let t1 = (bmax - o[axis]) * inv[axis];
                enter = enter.max(t0.min(t1));
                exit = exit.min(t0.max(t1));
            }
            if enter > exit {
                continue;
            }
            let (a, b) = (words[3], words[7]);
            if b & LEAF_FLAG != 0 {
                for k in 0..(b & !LEAF_FLAG) {
                    let tri = (a + k) as usize * 4;
                    let words = &bvh.packed_tris[tri..tri + 4];
                    if let Some((t, _, _)) = intersect(
                        o,
                        d,
                        vertex_at(words[0]),
                        vertex_at(words[1]),
                        vertex_at(words[2]),
                    ) && t >= lo
                        && (best.is_none() && t <= best_t || t < best_t)
                    {
                        best_t = t;
                        best = Some((t, words[3]));
                    }
                }
            } else {
                stack.push(a);
                stack.push(a + 1);
            }
        }
        best
    }

    /// The CPU reference's Möller–Trumbore (same constants and order).
    fn intersect(
        o: [f32; 3],
        d: [f32; 3],
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
    ) -> Option<(f32, f32, f32)> {
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let e1 = sub(v1, v0);
        let e2 = sub(v2, v0);
        let p = cross(d, e2);
        let det = dot(e1, p);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let s = sub(o, v0);
        let u = dot(s, p) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = cross(s, e1);
        let v = dot(d, q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        Some((dot(e2, q) * inv_det, u, v))
    }

    fn brute_force(
        vertices: &[f32],
        triangles: &[u32],
        o: [f32; 3],
        d: [f32; 3],
        lo: f32,
        hi: f32,
    ) -> Option<(f32, u32)> {
        let vertex_at = |i: u32| -> [f32; 3] {
            let i = i as usize * 3;
            [vertices[i], vertices[i + 1], vertices[i + 2]]
        };
        let mut best: Option<(f32, u32)> = None;
        for (prim, tri) in triangles.chunks_exact(3).enumerate() {
            if let Some((t, _, _)) =
                intersect(o, d, vertex_at(tri[0]), vertex_at(tri[1]), vertex_at(tri[2]))
                && t >= lo
                && t <= hi
                && best.map(|(bt, _)| t < bt).unwrap_or(true)
            {
                best = Some((t, prim as u32));
            }
        }
        best
    }

    #[test]
    fn bvh_structure_is_consistent() {
        let mut rng = Rng(7);
        let (vertices, triangles) = random_scene(&mut rng, 100);
        let bvh = build_bvh(&vertices, &triangles);
        assert_eq!(bvh.packed_tris.len(), 100 * 4);

        // Every original prim id appears exactly once, and leaves partition
        // the packed range.
        let mut seen = [false; 100];
        for tri in bvh.packed_tris.chunks_exact(4) {
            assert!(!seen[tri[3] as usize], "duplicate prim {}", tri[3]);
            seen[tri[3] as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));

        let mut covered = [false; 100];
        let node_count = bvh.nodes.len() / 8;
        for node in bvh.nodes.chunks_exact(8) {
            let (a, b) = (node[3], node[7]);
            if b & LEAF_FLAG != 0 {
                for k in 0..(b & !LEAF_FLAG) {
                    let slot = (a + k) as usize;
                    assert!(!covered[slot], "leaf overlap at {slot}");
                    covered[slot] = true;
                }
            } else {
                assert!(((a + 1) as usize) < node_count, "child oob");
            }
        }
        assert!(covered.iter().all(|&c| c));
    }

    #[test]
    fn bvh_traversal_matches_brute_force() {
        let mut rng = Rng(1234);
        for scene_seed in 0..4 {
            let (vertices, triangles) = random_scene(&mut rng, 40 + scene_seed * 17);
            let bvh = build_bvh(&vertices, &triangles);
            for _ in 0..200 {
                let o = [
                    rng.in_range(-2.5, 2.5),
                    rng.in_range(-2.5, 2.5),
                    rng.in_range(-2.5, 2.5),
                ];
                let target = [
                    rng.in_range(-1.0, 1.0),
                    rng.in_range(-1.0, 1.0),
                    rng.in_range(-1.0, 1.0),
                ];
                let d = [target[0] - o[0], target[1] - o[1], target[2] - o[2]];
                let expected = brute_force(&vertices, &triangles, o, d, 1e-4, 1e4);
                let got = traverse(&bvh, &vertices, o, d, 1e-4, 1e4);
                match (expected, got) {
                    (None, None) => {}
                    (Some((et, ep)), Some((gt, gp))) => {
                        assert!((et - gt).abs() < 1e-5, "t mismatch: {et} vs {gt}");
                        assert_eq!(ep, gp, "prim mismatch at t = {et}");
                    }
                    other => panic!("hit disagreement: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn empty_scene_builds_a_zero_leaf() {
        let bvh = build_bvh(&[], &[]);
        assert_eq!(bvh.nodes.len(), 8);
        assert_eq!(bvh.nodes[7], LEAF_FLAG);
        assert!(traverse(&bvh, &[], [0.0; 3], [0.0, 0.0, 1.0], 0.0, 1e4).is_none());
    }

    fn validate(wgsl: &str, capabilities: naga::valid::Capabilities) {
        let module = naga::front::wgsl::parse_str(wgsl)
            .unwrap_or_else(|e| panic!("WGSL parse error: {e}\n---\n{wgsl}"));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .unwrap_or_else(|e| panic!("WGSL validation error: {e:?}\n---\n{wgsl}"));
    }

    #[test]
    fn fallback_kernel_validates_without_extensions() {
        validate(&fallback_kernel(1024), naga::valid::Capabilities::default());
    }

    #[test]
    fn ray_query_kernel_validates_with_ray_query() {
        validate(&ray_query_kernel(1024), naga::valid::Capabilities::RAY_QUERY);
    }
}
