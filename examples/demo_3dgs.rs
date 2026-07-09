//! Render the gnomen gaussian cloud and write a PPM image.
//!
//! The whole renderer is composed from general tensor ops (no 3DGS kernels):
//! depth radix argsort, gathers, a transmittance scan, and reductions.
//!
//! ```sh
//! cargo run --example demo_3dgs --release            # auto backend
//! cargo run --example demo_3dgs --release -- --backend cpu --size 48
//! ```

use std::io::Write;

use resin::dsl::Tensor;
use resin::gaussians::{gnomen_cloud, render, Camera, CloudData, GaussianCloud};
use resin::jit::{DeviceValue, HostArray, Jit};

fn main() {
    let mut backend = String::from("auto");
    let mut size = 64usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => backend = args.next().expect("--backend <cpu|wgpu|auto>"),
            "--size" => {
                size = args
                    .next()
                    .expect("--size <pixels>")
                    .parse()
                    .expect("size must be an integer")
            }
            other => panic!("unknown argument {other}"),
        }
    }

    let camera = Camera::gnomen_default(size, size);
    let data = gnomen_cloud();

    let image = match backend.as_str() {
        "cpu" => render_with(resin::jit::CpuJit, &data, &camera),
        #[cfg(feature = "wgpu")]
        "wgpu" => render_with(resin::jit::WgpuJit::default(), &data, &camera),
        "auto" => render_auto(&data, &camera),
        other => panic!("backend {other} not available in this build"),
    };

    let path = "demo_3dgs.ppm";
    write_ppm(path, camera.width, camera.height, &image);
    println!(
        "wrote {path} ({}x{}, max channel {:.3})",
        camera.width,
        camera.height,
        image.iter().fold(0.0f32, |a, &b| a.max(b))
    );
}

fn render_auto(data: &CloudData, camera: &Camera) -> Vec<f32> {
    #[cfg(feature = "wgpu")]
    if resin::jit::wgpu::gpu_available() {
        println!("backend: wgpu");
        return render_with(resin::jit::WgpuJit::default(), data, camera);
    }
    println!("backend: cpu");
    render_with(resin::jit::CpuJit, data, camera)
}

fn render_with<J: Jit>(jit: J, data: &CloudData, camera: &Camera) -> Vec<f32> {
    let n = data.count();
    let cloud = GaussianCloud {
        means: jit
            .upload(&HostArray::from_f32(&[n, 3], &data.means_flat()))
            .expect("upload means"),
        scales: jit
            .upload(&HostArray::from_f32(&[n, 3], &data.scales_flat()))
            .expect("upload scales"),
        quats: jit
            .upload(&HostArray::from_f32(&[n, 4], &data.quats_flat()))
            .expect("upload quats"),
        colors: jit
            .upload(&HostArray::from_f32(&[n, 3], &data.colors_flat()))
            .expect("upload colors"),
        opacities: jit
            .upload(&HostArray::from_f32(&[n], &data.opacities))
            .expect("upload opacities"),
    };
    let camera = camera.clone();
    let f = jit.jit(move |cloud: &GaussianCloud<Tensor>| render(cloud, &camera));
    f.call(&cloud)
        .expect("render")
        .host()
        .expect("download")
        .to_f32()
}

fn write_ppm(path: &str, width: usize, height: usize, rgb: &[f32]) {
    assert_eq!(rgb.len(), width * height * 3);
    let mut out = Vec::with_capacity(rgb.len() + 32);
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    out.extend(
        rgb.iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8),
    );
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&out))
        .expect("write ppm");
}
