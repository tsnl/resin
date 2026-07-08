//! Path-traced Cornell box written as a PPM image.
//!
//! Usage:
//!   cargo run --example demo_pathtrace
//!   cargo run --example demo_pathtrace -- --backend vulkan --size 128x128 \
//!       --spp 16 --bounces 4 --out cornell.ppm
//!
//! `--backend auto` (the default) prefers hardware ray query on Vulkan, then
//! wgpu (which itself picks hardware or the compute fallback), then CPU.

use std::env;
use std::path::PathBuf;

use resin::render::pathtrace::{render, PathTracerConfig};
use resin::render::ppm::write_ppm;
use resin::render::{cornell_box, Camera};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendKind {
    Cpu,
    Wgpu,
    Vulkan,
    Auto,
}

struct Args {
    backend: BackendKind,
    width: usize,
    height: usize,
    spp: usize,
    bounces: usize,
    out: PathBuf,
}

fn parse_args() -> Args {
    let mut parsed = Args {
        backend: BackendKind::Auto,
        width: 128,
        height: 128,
        spp: 16,
        bounces: 3,
        out: PathBuf::from("pathtrace.ppm"),
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next()
                .unwrap_or_else(|| panic!("{flag} requires a value"))
        };
        match arg.as_str() {
            "--backend" => {
                parsed.backend = match value("--backend").as_str() {
                    "cpu" => BackendKind::Cpu,
                    "wgpu" => BackendKind::Wgpu,
                    "vulkan" => BackendKind::Vulkan,
                    "auto" => BackendKind::Auto,
                    other => panic!("unknown backend {other:?}; expected cpu|wgpu|vulkan|auto"),
                };
            }
            "--size" => {
                let size = value("--size");
                let (w, h) = size
                    .split_once('x')
                    .unwrap_or_else(|| panic!("--size expects WxH, got {size:?}"));
                parsed.width = w.parse().unwrap_or_else(|_| panic!("bad width {w:?}"));
                parsed.height = h.parse().unwrap_or_else(|_| panic!("bad height {h:?}"));
            }
            "--spp" => {
                let spp = value("--spp");
                parsed.spp = spp.parse().unwrap_or_else(|_| panic!("bad spp {spp:?}"));
            }
            "--bounces" => {
                let bounces = value("--bounces");
                parsed.bounces = bounces
                    .parse()
                    .unwrap_or_else(|_| panic!("bad bounces {bounces:?}"));
            }
            "--out" => parsed.out = PathBuf::from(value("--out")),
            flag => panic!(
                "unknown flag {flag}; try --backend cpu|wgpu|vulkan|auto, --size WxH, --spp N, \
                 --bounces B, --out path.ppm"
            ),
        }
    }
    parsed
}

/// Hardware-first backend choice among the compiled-in ones.
fn resolve_auto() -> BackendKind {
    #[cfg(feature = "vulkan")]
    if resin::jit::backends::vulkan::ray_query_available() {
        return BackendKind::Vulkan;
    }
    #[cfg(feature = "wgpu")]
    if resin::jit::backends::wgpu::shared_context_available() {
        return BackendKind::Wgpu;
    }
    #[cfg(feature = "cpu")]
    return BackendKind::Cpu;
    #[allow(unreachable_code)]
    {
        panic!("no backend available: build with cpu/wgpu/vulkan features")
    }
}

fn main() {
    let args = parse_args();
    let scene = cornell_box();
    let camera = Camera::cornell_default(args.width, args.height);
    let config = PathTracerConfig {
        bounces: args.bounces,
        seed: 0,
    };

    let backend = if args.backend == BackendKind::Auto {
        resolve_auto()
    } else {
        args.backend
    };
    eprintln!(
        "rendering {}x{} spp={} bounces={} on {backend:?}",
        args.width, args.height, args.spp, args.bounces
    );

    let image = match backend {
        #[cfg(feature = "cpu")]
        BackendKind::Cpu => render(
            resin::jit::backends::cpu::CpuJit,
            &scene,
            &camera,
            args.spp,
            &config,
        ),
        #[cfg(not(feature = "cpu"))]
        BackendKind::Cpu => panic!("built without the `cpu` feature"),
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => render(
            resin::jit::backends::wgpu::WgpuJit,
            &scene,
            &camera,
            args.spp,
            &config,
        ),
        #[cfg(not(feature = "wgpu"))]
        BackendKind::Wgpu => panic!("built without the `wgpu` feature"),
        #[cfg(feature = "vulkan")]
        BackendKind::Vulkan => render(
            resin::jit::backends::vulkan::VulkanJit,
            &scene,
            &camera,
            args.spp,
            &config,
        ),
        #[cfg(not(feature = "vulkan"))]
        BackendKind::Vulkan => panic!("built without the `vulkan` feature"),
        BackendKind::Auto => unreachable!("resolved above"),
    };

    write_ppm(&args.out, args.width, args.height, &image).expect("write PPM");
    eprintln!("wrote {}", args.out.display());
}
