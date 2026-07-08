//! Host-side camera: view/projection matrices baked into the graph as
//! constants.

/// Pinhole camera with row-major `view` and `proj` matrices (OpenGL-style
/// perspective, right-handed, camera looking down −Z; view-space depth is
/// positive in front of the camera after the view row-2 flip).
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub width: usize,
    pub height: usize,
}

impl Camera {
    /// Look-at view + perspective projection (`fov_y` in degrees).
    #[allow(clippy::too_many_arguments)]
    pub fn look_at(
        eye: [f32; 3],
        center: [f32; 3],
        up: [f32; 3],
        fov_y_deg: f32,
        z_near: f32,
        z_far: f32,
        width: usize,
        height: usize,
    ) -> Self {
        let aspect = width as f32 / height as f32;
        let f = normalize([
            center[0] - eye[0],
            center[1] - eye[1],
            center[2] - eye[2],
        ]);
        let s = normalize(cross(f, up));
        let u = cross(s, f);

        // Row 2 holds +f (not −f): view-space z is positive depth in front of
        // the camera, matching the reference renderer this port follows.
        let view = [
            [s[0], s[1], s[2], -dot(s, eye)],
            [u[0], u[1], u[2], -dot(u, eye)],
            [f[0], f[1], f[2], -dot(f, eye)],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let fl = 1.0 / (fov_y_deg.to_radians() * 0.5).tan();
        let proj = [
            [fl / aspect, 0.0, 0.0, 0.0],
            [0.0, fl, 0.0, 0.0],
            [
                0.0,
                0.0,
                z_far / (z_near - z_far),
                (z_far * z_near) / (z_near - z_far),
            ],
            // w = +view_z: this view convention keeps depth positive in
            // front of the camera, so the perspective divide must not flip
            // signs (a -1 here mirrors both screen axes).
            [0.0, 0.0, 1.0, 0.0],
        ];
        Self {
            view,
            proj,
            width,
            height,
        }
    }

    /// Default camera at the origin looking down −Z (the gnomen test view).
    pub fn gnomen_default(width: usize, height: usize) -> Self {
        Self::look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            60.0,
            0.1,
            100.0,
            width,
            height,
        )
    }

    /// Orbit camera: eye on a sphere of `radius` around `center` at the given
    /// yaw/pitch (degrees), looking at `center`. COLMAP-convention 3DGS
    /// checkpoints are y-down; pass `up = [0.0, -1.0, 0.0]` for those.
    #[allow(clippy::too_many_arguments)]
    pub fn orbit(
        center: [f32; 3],
        radius: f32,
        yaw_deg: f32,
        pitch_deg: f32,
        up: [f32; 3],
        fov_y_deg: f32,
        width: usize,
        height: usize,
    ) -> Self {
        let (yaw, pitch) = (yaw_deg.to_radians(), pitch_deg.to_radians());
        let eye = [
            center[0] + radius * yaw.sin() * pitch.cos(),
            center[1] + radius * pitch.sin(),
            center[2] + radius * yaw.cos() * pitch.cos(),
        ];
        Self::look_at(eye, center, up, fov_y_deg, 0.01, 100.0, width, height)
    }

    /// Row-major flattened matrices for tensor upload.
    pub fn view_flat(&self) -> Vec<f32> {
        self.view.iter().flatten().copied().collect()
    }

    pub fn proj_flat(&self) -> Vec<f32> {
        self.proj.iter().flatten().copied().collect()
    }

    /// `[4, 4]` constant tensors for the constant-camera render path.
    pub fn matrix_constants(&self) -> (resin_dsl::Tensor, resin_dsl::Tensor) {
        (
            resin_dsl::Tensor::constant_f32(&[4, 4], &self.view_flat()),
            resin_dsl::Tensor::constant_f32(&[4, 4], &self.proj_flat()),
        )
    }

    pub fn focal_x(&self) -> f32 {
        self.proj[0][0]
    }

    pub fn focal_y(&self) -> f32 {
        self.proj[1][1]
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
