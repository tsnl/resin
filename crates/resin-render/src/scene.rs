//! Host-side triangle scenes with per-triangle materials.

use resin_dsl::Tensor;

/// Triangle soup + per-triangle materials (albedo and emission as linear
/// RGB). Everything is host data; [`Scene::vertices_tensor`] & friends lift
/// it into graph constants.
#[derive(Debug, Clone)]
pub struct Scene {
    /// `[V, 3]` positions, row-major.
    pub vertices: Vec<f32>,
    /// `[T, 3]` vertex indices, row-major.
    pub triangles: Vec<u32>,
    /// `[T, 3]` diffuse albedo per triangle.
    pub albedo: Vec<f32>,
    /// `[T, 3]` emitted radiance per triangle.
    pub emission: Vec<f32>,
}

impl Scene {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / 3
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len() / 3
    }

    pub fn vertices_tensor(&self) -> Tensor {
        Tensor::constant_f32(&[self.vertex_count(), 3], &self.vertices)
    }

    pub fn triangles_tensor(&self) -> Tensor {
        Tensor::constant_u32(&[self.triangle_count(), 3], &self.triangles)
    }

    pub fn albedo_tensor(&self) -> Tensor {
        Tensor::constant_f32(&[self.triangle_count(), 3], &self.albedo)
    }

    pub fn emission_tensor(&self) -> Tensor {
        Tensor::constant_f32(&[self.triangle_count(), 3], &self.emission)
    }

    /// Append a quad (two triangles) with one material.
    ///
    /// `corners` in winding order; both triangles share `albedo`/`emission`.
    pub fn push_quad(&mut self, corners: [[f32; 3]; 4], albedo: [f32; 3], emission: [f32; 3]) {
        let base = self.vertex_count() as u32;
        for corner in corners {
            self.vertices.extend_from_slice(&corner);
        }
        for tri in [[0, 1, 2], [0, 2, 3]] {
            for offset in tri {
                self.triangles.push(base + offset);
            }
            self.albedo.extend_from_slice(&albedo);
            self.emission.extend_from_slice(&emission);
        }
    }

    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
            albedo: Vec::new(),
            emission: Vec::new(),
        }
    }
}

/// The classic test scene: a 2×2×2 box centered on the origin of the XZ
/// plane, open toward +z (the camera side), with a red left wall, green
/// right wall, white floor/ceiling/back, an emissive ceiling panel, and a
/// short white box inside.
pub fn cornell_box() -> Scene {
    let mut scene = Scene::empty();
    let white = [0.73, 0.73, 0.73];
    let red = [0.65, 0.05, 0.05];
    let green = [0.12, 0.45, 0.15];
    let none = [0.0, 0.0, 0.0];

    // Floor (y = 0), ceiling (y = 2), back wall (z = -1).
    scene.push_quad(
        [[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0]],
        white,
        none,
    );
    scene.push_quad(
        [[-1.0, 2.0, -1.0], [1.0, 2.0, -1.0], [1.0, 2.0, 1.0], [-1.0, 2.0, 1.0]],
        white,
        none,
    );
    scene.push_quad(
        [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]],
        white,
        none,
    );
    // Left (x = -1, red) and right (x = 1, green) walls.
    scene.push_quad(
        [[-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, 2.0, 1.0]],
        red,
        none,
    );
    scene.push_quad(
        [[1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, -1.0]],
        green,
        none,
    );
    // Emissive panel just below the ceiling.
    scene.push_quad(
        [
            [-0.4, 1.98, -0.4],
            [0.4, 1.98, -0.4],
            [0.4, 1.98, 0.4],
            [-0.4, 1.98, 0.4],
        ],
        none,
        [15.0, 15.0, 15.0],
    );
    // A short box (axis-aligned, 0.6³) sitting on the floor, left of center.
    let (x0, x1) = (-0.6, 0.0);
    let (y0, y1) = (0.0, 0.6);
    let (z0, z1) = (-0.55, 0.05);
    scene.push_quad(
        [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
        white,
        none,
    );
    scene.push_quad(
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        white,
        none,
    );
    scene.push_quad(
        [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
        white,
        none,
    );
    scene.push_quad(
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
        white,
        none,
    );
    scene.push_quad(
        [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
        white,
        none,
    );
    scene
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cornell_box_is_consistent() {
        let scene = cornell_box();
        let t = scene.triangle_count();
        assert_eq!(scene.albedo.len(), t * 3);
        assert_eq!(scene.emission.len(), t * 3);
        assert!(scene.triangles.iter().all(|&i| (i as usize) < scene.vertex_count()));
        assert!(scene.emission.iter().any(|&e| e > 0.0), "has a light");
    }
}
