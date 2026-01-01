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

fn build_bvh(triangles: &mut [PodTriangle], out: &mut Vec<PodBvhNode>) {
    /// Searches all possible partitions and finds the best one according to the surface area heuristic metric.
    /// Returns the best split dimension, the best split triangle index, and the best (lowest) SAH cost.
    fn find_best_partition(triangles: &[PodTriangle]) -> (usize, usize, f32) {
        let mut best_dim = 0;
        let mut best_i = 0;
        let mut best_cost = f32::INFINITY;

        for dim in [0, 1, 2] {
            for i in 0..triangles.len() {
                let split = triangles[i].centroid[dim];

                let mut lt_aabb = SimdRect3::union_identity();
                let mut rt_aabb = SimdRect3::union_identity();

                for triangle in triangles.iter() {
                    let c = triangle.centroid[dim];
                    if c < split {
                        // left
                        lt_aabb |= triangle.aabb();
                    } else {
                        // right
                        rt_aabb |= triangle.aabb();
                    }
                }

                let cost = lt_aabb.extent().reduce_sum() * lt_aabb.extent().reduce_sum();

                if cost < best_cost {
                    best_dim = dim;
                    best_i = i;
                    best_cost = cost;
                }
            }
        }

        (best_i, best_dim, best_cost)
    }

    let (best_dim, best_i, best_cost) = find_best_partition(triangles);
    // TODO: compare best_cost to the current surface area to determine whether to terminate.

    todo!()
}
