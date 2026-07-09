//! Scalar host reference renderer: the same math as [`crate::gaussians::render`], one
//! gaussian and one pixel at a time, in plain f32 Rust. Golden tests compare
//! the composed tensor graph against this.
//!
//! Semantics shared with the tensor path (deliberately, so the two agree
//! everywhere, not just on "easy" pixels): dense evaluation (no bounding-box
//! shortcut), no early transmittance termination, `power > 0` skipped, alpha
//! clamped to 0.99 and thresholded at 1/255, culling by near plane /
//! degenerate covariance / sub-pixel radius / screen overlap.

use super::camera::Camera;
use super::cloud::CloudData;

struct PreGaussian {
    px: f32,
    py: f32,
    depth: f32,
    conic: [f32; 3],
    color: [f32; 3],
    opacity: f32,
    valid: bool,
}

/// Render to a row-major `[H, W, 3]` RGB buffer.
pub fn render_reference(data: &CloudData, camera: &Camera) -> Vec<f32> {
    let (w, h) = (camera.width, camera.height);
    let pre: Vec<PreGaussian> = (0..data.count())
        .map(|i| preprocess_one(data, camera, i))
        .collect();
    // Stable front-to-back order by view depth.
    let mut order: Vec<usize> = (0..pre.len()).collect();
    order.sort_by(|&a, &b| pre[a].depth.partial_cmp(&pre[b].depth).unwrap());
    let mut image = vec![0.0f32; w * h * 3];
    let mut transmittance = vec![1.0f32; w * h];

    for &i in &order {
        let g = &pre[i];
        if !g.valid {
            continue;
        }
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 + 0.5 - g.px;
                let dy = y as f32 + 0.5 - g.py;
                let power = -0.5 * (g.conic[0] * dx * dx + g.conic[2] * dy * dy)
                    - g.conic[1] * dx * dy;
                if power > 0.0 {
                    continue;
                }
                let alpha = (g.opacity * power.exp()).min(0.99);
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let pix = y * w + x;
                let weight = alpha * transmittance[pix];
                for c in 0..3 {
                    image[pix * 3 + c] += g.color[c] * weight;
                }
                transmittance[pix] *= 1.0 - alpha;
            }
        }
    }
    image
}

fn preprocess_one(data: &CloudData, camera: &Camera, i: usize) -> PreGaussian {
    let (w, h) = (camera.width as f32, camera.height as f32);
    let m = data.means[i];
    let cam = mat4_mul_point(&camera.view, m);
    let clip = mat4_mul_point4(&camera.proj, cam);
    let cw = if clip[3].abs() > 1e-8 { clip[3] } else { 1.0 };
    let px = (clip[0] / cw * 0.5 + 0.5) * w;
    let py = (clip[1] / cw * 0.5 + 0.5) * h;
    let depth = cam[2];

    let mut valid = depth > 1e-4;
    let tz = if valid { depth } else { 1.0 };

    let cov3d = scale_rot_to_cov3d(data.scales[i], data.quats[i]);
    let (fx, fy) = (camera.focal_x(), camera.focal_y());
    let tz_inv = 1.0 / tz;
    let tz2_inv = tz_inv * tz_inv;
    let j00 = fx * tz_inv;
    let j02 = -fx * cam[0] * tz2_inv;
    let j11 = fy * tz_inv;
    let j12 = -fy * cam[1] * tz2_inv;

    let t00 = j00 * cov3d[0] + j02 * cov3d[2];
    let t01 = j00 * cov3d[1] + j02 * cov3d[4];
    let t02 = j00 * cov3d[2] + j02 * cov3d[5];
    let t12 = j11 * cov3d[3] + j12 * cov3d[4];

    let cov00 = t00 * j00 + t02 * j02 + 0.3;
    let cov01 = t01 * j00 + t12 * j02;
    let cov11 = t01 * j11 + t12 * j12 + 0.3;

    let det = cov00 * cov11 - cov01 * cov01;
    valid = valid && det > 1e-12;
    let det_safe = if valid { det } else { 1.0 };
    let conic = [cov11 / det_safe, -cov01 / det_safe, cov00 / det_safe];

    let radius = (3.0 * conic[0].max(conic[2]).sqrt()).ceil();
    valid = valid && radius >= 1.0;
    valid = valid
        && px + radius >= 0.0
        && py + radius >= 0.0
        && px - radius <= w
        && py - radius <= h;

    PreGaussian {
        px,
        py,
        depth,
        conic,
        color: data.colors[i],
        opacity: data.opacities[i],
        valid,
    }
}

/// Six unique entries `[c00, c01, c02, c11, c12, c22]` of `R diag(s²) Rᵀ`.
fn scale_rot_to_cov3d(scale: [f32; 3], quat: [f32; 4]) -> [f32; 6] {
    let norm =
        (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).sqrt();
    let (qw, qx, qy, qz) = (
        quat[0] / norm,
        quat[1] / norm,
        quat[2] / norm,
        quat[3] / norm,
    );
    let r = [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qw * qz),
            2.0 * (qx * qz + qw * qy),
        ],
        [
            2.0 * (qx * qy + qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qw * qx),
        ],
        [
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ];
    let v = [
        scale[0] * scale[0],
        scale[1] * scale[1],
        scale[2] * scale[2],
    ];
    let entry = |a: usize, b: usize| -> f32 {
        r[a][0] * v[0] * r[b][0] + r[a][1] * v[1] * r[b][1] + r[a][2] * v[2] * r[b][2]
    };
    [
        entry(0, 0),
        entry(0, 1),
        entry(0, 2),
        entry(1, 1),
        entry(1, 2),
        entry(2, 2),
    ]
}

fn mat4_mul_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for (r, out_r) in out.iter_mut().enumerate() {
        *out_r = m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3];
    }
    out
}

fn mat4_mul_point4(m: &[[f32; 4]; 4], p: [f32; 4]) -> [f32; 4] {
    // Callers only produce w = 1 view points (view row 3 is [0,0,0,1]).
    debug_assert!((p[3] - 1.0).abs() < 1e-6);
    mat4_mul_point(m, [p[0], p[1], p[2]])
}
