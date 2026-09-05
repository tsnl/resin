//! Shared GPU reftest harness. Set `RESIN_UPDATE_REFS=1` to rewrite PNGs.

use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use resin_runtime::testing::{GpuLock, lock_gpu};
use resin_runtime::{ResinGpu, ResinStatus, image_read_png, image_write_png};

pub struct Gpu {
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

pub fn require_gpu() -> Option<Gpu> {
    let lock = lock_gpu();
    match ResinGpu::create() {
        Ok(inner) => Some(Gpu { inner, _lock: lock }),
        Err(ResinStatus::VulkanUnavailable | ResinStatus::Unsupported) => {
            assert!(
                !gpu_required(),
                "RESIN_REQUIRE_GPU is set but no suitable Vulkan device is available"
            );
            eprintln!("skipping: no suitable Vulkan device");
            None
        }
        Err(err) => panic!("ResinGpu::create failed: {err:?}"),
    }
}

pub fn compile_shader(src: &str, stage: &str, define: Option<&str>) -> Option<Vec<u8>> {
    let mut cmd = Command::new("glslc");
    cmd.arg(format!("-fshader-stage={stage}"))
        .arg("--target-env=vulkan1.3")
        .arg("-O");
    if let Some(define) = define {
        cmd.arg(format!("-D{define}"));
    }
    cmd.arg("-o").arg("-").arg("-");
    let mut child = match cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            assert!(
                !gpu_required(),
                "RESIN_REQUIRE_GPU is set but glslc is unavailable"
            );
            eprintln!("skipping: glslc not found");
            return None;
        }
        Err(err) => panic!("failed to spawn glslc: {err}"),
    };
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(src.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "glslc {stage} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(output.stdout)
}

/// Compare `pixels` to the PNG sitting next to `test_file` (`file!()`).
pub fn assert_reftest(test_file: &str, width: u32, height: u32, pixels: &[u8], rgb_tolerance: u8) {
    let reference_path = sibling(test_file, "png");
    let name = reference_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    if update_refs() {
        image_write_png(&reference_path, width, height, 4, pixels)
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
        image_read_png(&reference_path, 4).unwrap_or_else(|err| panic!("read {name}: {err:?}"));
    assert_eq!(reference.width, width);
    assert_eq!(reference.height, height);
    assert_eq!(reference.channels, 4);
    if reference.pixels.len() == pixels.len()
        && reference
            .pixels
            .iter()
            .zip(pixels)
            .enumerate()
            .all(|(i, (&a, &b))| a.abs_diff(b) <= if i % 4 == 3 { 0 } else { rgb_tolerance })
    {
        return;
    }

    let actual_path = std::env::temp_dir().join(format!("resin-actual-{name}"));
    image_write_png(&actual_path, width, height, 4, pixels)
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

fn sibling(test_file: &str, extension: &str) -> PathBuf {
    let stem = Path::new(test_file).file_stem().expect("test file name");
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(stem)
        .with_extension(extension)
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

fn update_refs() -> bool {
    matches!(
        std::env::var("RESIN_UPDATE_REFS").as_deref(),
        Ok("1") | Ok("true")
    )
}

fn gpu_required() -> bool {
    matches!(
        std::env::var("RESIN_REQUIRE_GPU").as_deref(),
        Ok("1") | Ok("true")
    )
}
