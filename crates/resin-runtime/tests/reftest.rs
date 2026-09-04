//! GPU reftests. Set `RESIN_UPDATE_REFS=1` to rewrite the checked-in PNGs.

use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use resin_runtime::testing::{GpuLock, lock_gpu};
use resin_runtime::{ResinGpu, ResinMemory, ResinStatus, read_png, write_png};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

#[repr(C)]
struct Root {
    width: u32,
    height: u32,
    pixels: u64,
}

struct Gpu {
    inner: ResinGpu,
    _lock: GpuLock,
}

impl Deref for Gpu {
    type Target = ResinGpu;
    fn deref(&self) -> &ResinGpu {
        &self.inner
    }
}

impl DerefMut for Gpu {
    fn deref_mut(&mut self) -> &mut ResinGpu {
        &mut self.inner
    }
}

#[test]
fn hello_triangle() {
    let Some(mut gpu) = require_gpu() else {
        return;
    };
    let Some(vert) = compile_shader("hello_triangle.glsl", "vert", Some("VERTEX")) else {
        return;
    };
    let Some(frag) = compile_shader("hello_triangle.glsl", "frag", Some("FRAGMENT")) else {
        return;
    };

    let pipeline = gpu
        .create_graphics_pipeline(&vert, &frag)
        .expect("graphics pipeline");
    let mut image = gpu.create_image(WIDTH, HEIGHT).expect("image");
    let pixels = gpu
        .malloc(
            WIDTH as usize * HEIGHT as usize * 4,
            4,
            ResinMemory::Readback,
        )
        .expect("malloc");

    let mut commands = gpu.start_command_recording().expect("record");
    commands
        .begin_rendering(&mut image, [0.0, 0.0, 0.0, 1.0])
        .expect("begin rendering");
    commands.set_pipeline(&pipeline).expect("set pipeline");
    commands.draw(0, 3).expect("draw");
    commands.end_rendering().expect("end rendering");
    commands
        .copy_image_to_buffer(&mut image, &pixels)
        .expect("copy");
    gpu.submit(commands).expect("submit");

    let host = pixels.host_bytes().expect("mapped readback");
    assert_reftest("hello_triangle.png", host);
}

#[test]
fn compute_gradient() {
    let Some(mut gpu) = require_gpu() else {
        return;
    };
    let Some(spv) = compile_shader("compute.glsl", "comp", None) else {
        return;
    };

    let pipeline = gpu.create_compute_pipeline(&spv).expect("compute pipeline");
    let pixels = gpu
        .malloc(
            WIDTH as usize * HEIGHT as usize * 4,
            4,
            ResinMemory::Default,
        )
        .expect("malloc pixels");
    let root = gpu
        .malloc(size_of::<Root>(), 8, ResinMemory::Default)
        .expect("malloc root");

    let pixels_device = gpu
        .host_to_device(pixels.host_pointer())
        .expect("pixels device address");
    unsafe {
        root.host_pointer().cast::<Root>().write(Root {
            width: WIDTH,
            height: HEIGHT,
            pixels: pixels_device,
        });
    }
    let root_device = gpu
        .host_to_device(root.host_pointer())
        .expect("root device address");

    let mut commands = gpu.start_command_recording().expect("record");
    commands.set_pipeline(&pipeline).expect("set pipeline");
    commands
        .dispatch(root_device, WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1)
        .expect("dispatch");
    gpu.submit(commands).expect("submit");

    let host = pixels.host_bytes().expect("mapped default allocation");
    assert_reftest("compute.png", host);
}

fn require_gpu() -> Option<Gpu> {
    let lock = lock_gpu();
    match ResinGpu::create() {
        Ok(inner) => Some(Gpu { inner, _lock: lock }),
        Err(ResinStatus::VulkanUnavailable | ResinStatus::Unsupported) => {
            eprintln!("skipping: no suitable Vulkan device");
            None
        }
        Err(err) => panic!("ResinGpu::create failed: {err:?}"),
    }
}

fn compile_shader(name: &str, stage: &str, define: Option<&str>) -> Option<Vec<u8>> {
    let src = data_dir().join(name);
    let mut cmd = Command::new("glslc");
    cmd.arg(format!("-fshader-stage={stage}"))
        .arg("--target-env=vulkan1.3")
        .arg("-O");
    if let Some(define) = define {
        cmd.arg(format!("-D{define}"));
    }
    cmd.arg("-o").arg("-").arg(&src);
    let output = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).output() {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("skipping: glslc not found");
            return None;
        }
        Err(err) => panic!("failed to spawn glslc: {err}"),
    };
    assert!(
        output.status.success(),
        "glslc {name} {stage} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(output.stdout)
}

fn assert_reftest(name: &str, pixels: &[u8]) {
    let reference_path = data_dir().join(name);
    if update_refs() {
        write_png(&reference_path, WIDTH, HEIGHT, 4, pixels)
            .unwrap_or_else(|err| panic!("update {name}: {err:?}"));
        eprintln!("updated {}", reference_path.display());
        return;
    }
    if !reference_path.exists() {
        panic!(
            "missing reference {}; run with RESIN_UPDATE_REFS=1",
            reference_path.display()
        );
    }
    let reference =
        read_png(&reference_path, 4).unwrap_or_else(|err| panic!("read {name}: {err:?}"));
    assert_eq!(reference.width, WIDTH);
    assert_eq!(reference.height, HEIGHT);
    assert_eq!(reference.channels, 4);
    if reference.pixels == pixels {
        return;
    }

    let actual_path = std::env::temp_dir().join(format!("resin-actual-{name}"));
    write_png(&actual_path, WIDTH, HEIGHT, 4, pixels)
        .unwrap_or_else(|err| panic!("write actual {name}: {err:?}"));
    let (mismatched, max_delta) = pixel_diff(&reference.pixels, pixels);
    panic!(
        "{name} mismatch: {mismatched} pixels differ, max channel delta {max_delta}.\n\
         actual: {}\n\
         reference: {}\n\
         rewrite with RESIN_UPDATE_REFS=1",
        actual_path.display(),
        reference_path.display()
    );
}

fn pixel_diff(reference: &[u8], actual: &[u8]) -> (usize, u8) {
    let n = reference.len().min(actual.len());
    let mut mismatched = 0usize;
    let mut max_delta = 0u8;
    for pixel in (0..n).step_by(4) {
        let mut pixel_delta = 0u8;
        for channel in 0..4 {
            let i = pixel + channel;
            if i >= n {
                break;
            }
            pixel_delta = pixel_delta.max(reference[i].abs_diff(actual[i]));
        }
        if pixel_delta != 0 {
            mismatched += 1;
            max_delta = max_delta.max(pixel_delta);
        }
    }
    if reference.len() != actual.len() {
        mismatched += 1;
    }
    (mismatched, max_delta)
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
}

fn update_refs() -> bool {
    matches!(
        std::env::var("RESIN_UPDATE_REFS").as_deref(),
        Ok("1") | Ok("true")
    )
}
