//! CIFAR-10 host loader: download, binary batch parse, and batch normalization.
//!
//! Official binary layout (channel-first planar RGB): each sample is 1 label
//! byte + 3072 pixel bytes (`R…` then `G…` then `B…`, each plane 32×32).

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const IMG_W: usize = 32;
pub const IMG_H: usize = 32;
pub const IMG_C: usize = 3;
/// Planar CHW size: `C * H * W`.
pub const IMG_CHW: usize = IMG_C * IMG_H * IMG_W;
pub const NUM_CLS: usize = 10;

const RECORD: usize = 1 + IMG_CHW;
const BATCH_N: usize = 10_000;
const ARCHIVE: &str = "cifar-10-binary.tar.gz";
const EXTRACTED_DIR: &str = "cifar-10-batches-bin";
const URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

/// CIFAR-10 images as planar `u8` pixels (0–255, CHW) and labels (0–9).
pub struct Cifar10Dataset {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
    pub n: usize,
}

impl Cifar10Dataset {
    pub fn load(split: &str, cache_dir: impl AsRef<Path>) -> Result<Self, String> {
        let batch_names: &[&str] = match split {
            "train" => &[
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin",
            ],
            "test" => &["test_batch.bin"],
            other => return Err(format!("unknown CIFAR-10 split: {other}")),
        };
        let cache_dir = cache_dir.as_ref();
        fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;
        let data_dir = ensure_cifar10_dir(cache_dir)?;

        let mut images = Vec::new();
        let mut labels = Vec::new();
        for name in batch_names {
            read_batch(&data_dir.join(name), &mut images, &mut labels)?;
        }
        let n = labels.len();
        if images.len() != n * IMG_CHW {
            return Err("CIFAR-10 image/label count mismatch".into());
        }
        Ok(Self { images, labels, n })
    }

    /// Normalized `f32` images (planar CHW) and one-hot `f32` labels.
    pub fn batch_f32(&self, indices: &[usize]) -> (Vec<f32>, Vec<f32>) {
        batch_f32_from_indices(&self.images, &self.labels, indices)
    }
}

fn batch_f32_from_indices(images: &[u8], labels: &[u8], indices: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let b = indices.len();
    let mut xs = vec![0f32; b * IMG_CHW];
    let mut ys = vec![0f32; b * NUM_CLS];
    for (row, &idx) in indices.iter().enumerate() {
        let base = idx * IMG_CHW;
        for i in 0..IMG_CHW {
            xs[row * IMG_CHW + i] = images[base + i] as f32 / 255.0;
        }
        let label = labels[idx] as usize;
        ys[row * NUM_CLS + label] = 1.0;
    }
    (xs, ys)
}

fn ensure_cifar10_dir(cache_dir: &Path) -> Result<PathBuf, String> {
    let data_dir = cache_dir.join(EXTRACTED_DIR);
    if data_dir.join("data_batch_1.bin").is_file() {
        return Ok(data_dir);
    }
    let tgz_path = cache_dir.join(ARCHIVE);
    if !tgz_path.is_file() {
        download(URL, &tgz_path)?;
    }
    extract_tgz(&tgz_path, cache_dir)?;
    if !data_dir.join("data_batch_1.bin").is_file() {
        return Err(format!(
            "CIFAR-10 extract missing {EXTRACTED_DIR}/data_batch_1.bin"
        ));
    }
    Ok(data_dir)
}

fn download(url: &str, dest: &Path) -> Result<(), String> {
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .map_err(|e| format!("failed to spawn curl: {e}"))?;
    if !status.success() {
        return Err(format!("curl failed for {url}: {status}"));
    }
    Ok(())
}

fn extract_tgz(tgz: &Path, dest_dir: &Path) -> Result<(), String> {
    let status = std::process::Command::new("tar")
        .args(["-xzf"])
        .arg(tgz)
        .arg("-C")
        .arg(dest_dir)
        .status()
        .map_err(|e| format!("failed to spawn tar: {e}"))?;
    if !status.success() {
        return Err(format!("tar -xzf failed: {status}"));
    }
    Ok(())
}

fn read_batch(path: &Path, images: &mut Vec<u8>, labels: &mut Vec<u8>) -> Result<(), String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; BATCH_N * RECORD];
    f.read_exact(&mut buf)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    images.reserve(BATCH_N * IMG_CHW);
    labels.reserve(BATCH_N);
    for rec in 0..BATCH_N {
        let off = rec * RECORD;
        labels.push(buf[off]);
        images.extend_from_slice(&buf[off + 1..off + RECORD]);
    }
    Ok(())
}
