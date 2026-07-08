//! Pinhole camera: host-side primary-ray generation (for the path tracer)
//! and a view-projection matrix (for the raster path, NDC depth 0..1).

/// Right-handed pinhole camera. `fov_y` in radians; image row 0 = top.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y: f32,
    pub width: usize,
    pub height: usize,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    /// A camera looking into the [`crate::scene::cornell_box`] from +z.
    pub fn cornell_default(width: usize, height: usize) -> Self {
        Self {
            eye: [0.0, 1.0, 3.4],
            target: [0.0, 1.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y: 40f32.to_radians(),
            width,
            height,
            near: 0.1,
            far: 20.0,
        }
    }

    /// Orthonormal camera basis: `(right, up, forward)` with `forward`
    /// pointing from the eye toward the target.
    pub fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let forward = normalize(sub(self.target, self.eye));
        let right = normalize(cross(forward, self.up));
        let up = cross(right, forward);
        (right, up, forward)
    }

    /// One primary ray per pixel through pixel centers, row-major
    /// (`origins`, `directions`), each `[H*W, 3]`. Directions are unit
    /// length.
    pub fn primary_rays(&self) -> (Vec<f32>, Vec<f32>) {
        let (right, up, forward) = self.basis();
        let (w, h) = (self.width, self.height);
        let aspect = w as f32 / h as f32;
        let tan_half = (self.fov_y * 0.5).tan();

        let mut origins = Vec::with_capacity(w * h * 3);
        let mut directions = Vec::with_capacity(w * h * 3);
        for r in 0..h {
            for c in 0..w {
                // NDC pixel center, y up.
                let x = ((c as f32 + 0.5) / w as f32 * 2.0 - 1.0) * tan_half * aspect;
                let y = (1.0 - (r as f32 + 0.5) / h as f32 * 2.0) * tan_half;
                let dir = normalize(add(
                    add(scale(right, x), scale(up, y)),
                    forward,
                ));
                origins.extend_from_slice(&self.eye);
                directions.extend_from_slice(&dir);
            }
        }
        (origins, directions)
    }

    /// Column-major-free 4×4 view-projection matrix in **row-major** order
    /// (`m[row * 4 + col]`), mapping world → clip with NDC depth `0..1` and
    /// y up (matching the `rasterize` node's conventions).
    pub fn view_proj(&self) -> [f32; 16] {
        let (right, up, forward) = self.basis();
        // View: world → camera (camera looks down -z... we use +forward as
        // view -z by negating forward row so depth increases away from eye).
        let view = [
            right[0], right[1], right[2], -dot(right, self.eye),
            up[0], up[1], up[2], -dot(up, self.eye),
            -forward[0], -forward[1], -forward[2], dot(forward, self.eye),
            0.0, 0.0, 0.0, 1.0,
        ];
        let aspect = self.width as f32 / self.height as f32;
        let f = 1.0 / (self.fov_y * 0.5).tan();
        let (n, fr) = (self.near, self.far);
        // Perspective with z mapped to [0, 1] (Vulkan/WebGPU style),
        // right-handed view space (camera looks toward -z).
        let proj = [
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, fr / (n - fr), n * fr / (n - fr),
            0.0, 0.0, -1.0, 0.0,
        ];
        mat_mul(proj, view)
    }

    /// Apply [`Camera::view_proj`] to host positions `[V, 3]` → clip `[V, 4]`.
    pub fn to_clip(&self, positions: &[f32]) -> Vec<f32> {
        let m = self.view_proj();
        let mut out = Vec::with_capacity(positions.len() / 3 * 4);
        for p in positions.chunks_exact(3) {
            for row in 0..4 {
                out.push(
                    m[row * 4] * p[0] + m[row * 4 + 1] * p[1] + m[row * 4 + 2] * p[2]
                        + m[row * 4 + 3],
                );
            }
        }
        out
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    // A zero vector means a degenerate camera (eye == target, or up parallel
    // to the view direction); failing loudly beats silently NaN images.
    assert!(len > 0.0, "camera basis is degenerate (zero-length vector)");
    scale(v, 1.0 / len)
}

/// Row-major 4×4 multiply `a * b`.
fn mat_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for row in 0..4 {
        for col in 0..4 {
            out[row * 4 + col] = (0..4).map(|k| a[row * 4 + k] * b[k * 4 + col]).sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_rays_center_points_forward() {
        let cam = Camera::cornell_default(3, 3);
        let (_, dirs) = cam.primary_rays();
        // Center pixel (1,1) of 3x3.
        let d = &dirs[(3 + 1) * 3..(3 + 1) * 3 + 3];
        let (_, _, forward) = cam.basis();
        let dp = d[0] * forward[0] + d[1] * forward[1] + d[2] * forward[2];
        assert!(dp > 0.999, "center ray ~ forward, dot = {dp}");
    }

    #[test]
    fn view_proj_maps_target_into_ndc() {
        let cam = Camera::cornell_default(16, 16);
        let clip = cam.to_clip(&cam.target);
        let w = clip[3];
        assert!(w > 0.0);
        let ndc: Vec<f32> = clip[..3].iter().map(|v| v / w).collect();
        assert!(ndc[0].abs() < 1e-4 && ndc[1].abs() < 1e-4, "target at center");
        assert!(ndc[2] > 0.0 && ndc[2] < 1.0, "depth in [0,1], got {}", ndc[2]);
    }

    #[test]
    fn clip_positions_land_in_expected_half() {
        let cam = Camera::cornell_default(8, 8);
        // A point above the target should project to NDC y > 0.
        let clip = cam.to_clip(&[0.0, 1.8, 0.0]);
        assert!(clip[1] / clip[3] > 0.0);
    }
}
