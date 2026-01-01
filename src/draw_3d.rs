use std::ops::Range;

use simd_math::{SimdRect3, SimdVec3};

pub struct Draw3dRenderer {
    device: wgpu::Device,
    target_size_wh: [u16; 2],
}
impl Draw3dRenderer {
    pub fn new() -> Self {
        todo!()
    }
}

pub struct Draw3dFrame {}

pub struct Draw3dGeometry {}

pub struct Draw3dMaterial {}

//
// BVH
//

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodBvhNode {
    triangles_span: [u32; 2],
    children: [u32; 2],
    aabb: [[f32; 3]; 2],
}
impl PodBvhNode {
    fn aabb(&self) -> SimdRect3 {
        SimdRect3::new(SimdVec3::from(self.aabb[0]), SimdVec3::from(self.aabb[1]))
    }
    fn sah_cost(&self) -> f32 {
        let area = self.aabb().extent().reduce_sum();
        let tri_count = (self.triangles_span[1] - self.triangles_span[0]) as f32;
        area * tri_count
    }
    fn triangles_range(&self) -> Range<usize> {
        self.triangles_span[0] as usize..self.triangles_span[1] as usize
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodTriangle {
    vertices: [PodVertex; 3],
    centroid: [f32; 3],
    aabb: [[f32; 3]; 2],
}
impl PodTriangle {
    fn new(vertices: [PodVertex; 3]) -> Self {
        let centroid = {
            let mut acc = SimdVec3::ZERO;
            for v in &vertices {
                acc += SimdVec3::from(v.position);
            }
            let res = acc / SimdVec3::splat(3.0);
            res.into()
        };
        let aabb = {
            let mut aabb = SimdRect3::union_identity();
            for v in &vertices {
                aabb |= SimdVec3::from(v.position);
            }
            [aabb.min().into(), aabb.max().into()]
        };
        Self {
            vertices,
            centroid,
            aabb,
        }
    }
    fn aabb(&self) -> SimdRect3 {
        SimdRect3::new(SimdVec3::from(self.aabb[0]), SimdVec3::from(self.aabb[1]))
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PodVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

fn try_partition_bvh_node(
    root_node: usize,
    triangles: &mut [PodTriangle],
    nodes: &mut Vec<PodBvhNode>,
) {
    // Find the best partition
    let (best_dim, best_i, best_cost) = {
        let mut best_dim = 0;
        let mut best_i = 0;
        let mut best_cost = f32::INFINITY;

        for dim in [0, 1, 2] {
            for i in nodes[root_node].triangles_range() {
                let cost = evaluate_sah(triangles, dim, triangles[i].centroid[dim]);
                if cost < best_cost {
                    best_dim = dim;
                    best_i = i;
                    best_cost = cost;
                }
            }
        }

        (best_i, best_dim, best_cost)
    };

    // If the best cost is not better than the current node's cost, do not partition further.
    if best_cost >= nodes[root_node].sah_cost() {
        return;
    }

    // Partition the triangles based on the best split, moving them in place within the triangles slice.
    {
        todo!()
    }

    // Recurse on the two new child nodes.
    {
        todo!();
    }
}

/// Calculates the surface area heuristic cost for a given AABB and triangle count.
/// * https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
/// * https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#TheSurfaceAreaHeuristic
fn evaluate_sah(triangles: &[PodTriangle], dim: usize, split: f32) -> f32 {
    let mut lt_aabb = SimdRect3::union_identity();
    let mut lt_count = 0.0;
    let mut rt_aabb = SimdRect3::union_identity();
    let mut rt_count = 0.0;

    for triangle in triangles.iter() {
        let c = triangle.centroid[dim];
        if c < split {
            // left
            lt_aabb |= triangle.aabb();
            lt_count += 1.0;
        } else {
            // right
            rt_aabb |= triangle.aabb();
            rt_count += 1.0;
        }
    }

    (lt_aabb.extent().reduce_sum() * lt_count) + (rt_aabb.extent().reduce_sum() * rt_count)
}
