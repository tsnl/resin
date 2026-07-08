//! Rasterized Cornell box with deferred Lambert shading, written as PPM
//! (`docs/hw-nodes.md`).
//!
//! Usage:
//!   cargo run --example demo_raster
//!   cargo run --example demo_raster -- --backend vulkan --size 64x64 --out cornell.ppm
//!   cargo run --example demo_raster -- --backend auto

use std::env;
use std::path::PathBuf;

use resin::render::ppm::write_ppm;
use resin::render::raster;
use resin::render::{cornell_box, Camera};

/// Roughly from the ceiling panel into the room.
const LIGHT_DIR: [f32; 3] = [0.3, -1.0, 0.25];
const AMBIENT: f32 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendKind {
    Cpu,
    Wgpu,
    Vulkan,
    Auto,
}

fn parse_args() -> (BackendKind, usize, usize, PathBuf) {
    let mut backend = BackendKind::Auto;
    let mut width = 256usize;
    let mut height = 256usize;
    let mut out = PathBuf::from("cornell_raster.ppm");
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                let value = args
                    .next()
                    .unwrap_or_else(|| panic!("--backend requires cpu|wgpu|vulkan|auto"));
                backend = match value.as_str() {
                    "cpu" => BackendKind::Cpu,
                    "wgpu" => BackendKind::Wgpu,
                    "vulkan" => BackendKind::Vulkan,
                    "auto" => BackendKind::Auto,
                    other => panic!("unknown backend {other:?}; expected cpu|wgpu|vulkan|auto"),
                };
            }
            "--size" => {
                let value = args.next().unwrap_or_else(|| panic!("--size requires WxH"));
                let (w, h) = value
                    .split_once('x')
                    .unwrap_or_else(|| panic!("invalid size {value:?}; expected WxH"));
                width = w.parse().unwrap_or_else(|_| panic!("invalid width {w:?}"));
                height = h.parse().unwrap_or_else(|_| panic!("invalid height {h:?}"));
            }
            "--out" => {
                out = PathBuf::from(args.next().unwrap_or_else(|| panic!("--out requires a path")));
            }
            flag => panic!(
                "unknown flag {flag}; try --backend cpu|wgpu|vulkan|auto, --size WxH, --out path.ppm"
            ),
        }
    }
    (backend, width, height, out)
}

/// Prefer the backend closest to the hardware: vulkan > wgpu > cpu.
fn auto_backend() -> BackendKind {
    #[cfg(feature = "vulkan")]
    if resin::jit::backends::vulkan::shared_context_available() {
        return BackendKind::Vulkan;
    }
    #[cfg(feature = "wgpu")]
    if resin::jit::backends::wgpu::shared_context_available() {
        return BackendKind::Wgpu;
    }
    BackendKind::Cpu
}

fn main() {
    let (requested, width, height, out) = parse_args();
    let backend = match requested {
        BackendKind::Auto => auto_backend(),
        other => other,
    };

    let scene = cornell_box();
    let camera = Camera::cornell_default(width, height);
    eprintln!("rendering {width}x{height} Cornell box on {backend:?} …");
    let image = match backend {
        #[cfg(feature = "cpu")]
        BackendKind::Cpu => raster::render(
            resin::jit::backends::cpu::CpuJit,
            &scene,
            &camera,
            LIGHT_DIR,
            AMBIENT,
        ),
        #[cfg(not(feature = "cpu"))]
        BackendKind::Cpu => panic!("built without the `cpu` feature"),
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => raster::render(
            resin::jit::backends::wgpu::WgpuJit,
            &scene,
            &camera,
            LIGHT_DIR,
            AMBIENT,
        ),
        #[cfg(not(feature = "wgpu"))]
        BackendKind::Wgpu => panic!("built without the `wgpu` feature"),
        #[cfg(feature = "vulkan")]
        BackendKind::Vulkan => raster::render(
            resin::jit::backends::vulkan::VulkanJit,
            &scene,
            &camera,
            LIGHT_DIR,
            AMBIENT,
        ),
        #[cfg(not(feature = "vulkan"))]
        BackendKind::Vulkan => panic!("built without the `vulkan` feature"),
        BackendKind::Auto => unreachable!("auto resolved above"),
    };

    write_ppm(&out, width, height, &image).expect("write ppm");
    eprintln!("wrote {}", out.display());
}
