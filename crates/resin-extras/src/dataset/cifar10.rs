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
/// One batch file is exactly `BATCH_N` records.
const BATCH_BYTES: u64 = (BATCH_N * RECORD) as u64;
const ARCHIVE: &str = "cifar-10-binary.tar.gz";
const EXTRACTED_DIR: &str = "cifar-10-batches-bin";
/// Fast mirror first; official host as fallback.
const URLS: &[&str] = &[
    "http://mirror.tensorflow.org/www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
];
/// MD5 of the official `cifar-10-binary.tar.gz` archive.
const ARCHIVE_MD5: &str = "c32a1d4ab5d03f1284b67883e8d87530";

const TRAIN_BATCHES: &[&str] = &[
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];
const TEST_BATCHES: &[&str] = &["test_batch.bin"];
const ALL_BATCHES: &[&str] = &[
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
    "test_batch.bin",
];

/// CIFAR-10 images as planar `u8` pixels (0–255, CHW) and labels (0–9).
pub struct Cifar10Dataset {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
    pub n: usize,
}

impl Cifar10Dataset {
    pub fn load(split: &str, cache_dir: impl AsRef<Path>) -> Result<Self, String> {
        let batch_names: &[&str] = match split {
            "train" => TRAIN_BATCHES,
            "test" => TEST_BATCHES,
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
    if batches_ok(&data_dir) {
        return Ok(data_dir);
    }
    // Incomplete extract from a prior truncated download — start clean.
    let _ = fs::remove_dir_all(&data_dir);

    let tgz_path = cache_dir.join(ARCHIVE);
    ensure_archive(&tgz_path)?;
    if let Err(e) = extract_tgz(&tgz_path, cache_dir) {
        // Corrupt/truncated .tar.gz is the usual cause; drop it and retry once.
        let _ = fs::remove_file(&tgz_path);
        let _ = fs::remove_dir_all(&data_dir);
        ensure_archive(&tgz_path)?;
        extract_tgz(&tgz_path, cache_dir).map_err(|e2| {
            format!("{e}; retry after re-download also failed: {e2}")
        })?;
    }
    if !batches_ok(&data_dir) {
        return Err(format!(
            "CIFAR-10 extract incomplete under {}",
            data_dir.display()
        ));
    }
    Ok(data_dir)
}

fn batches_ok(data_dir: &Path) -> bool {
    ALL_BATCHES.iter().all(|name| {
        let p = data_dir.join(name);
        p.is_file()
            && fs::metadata(&p)
                .map(|m| m.len() == BATCH_BYTES)
                .unwrap_or(false)
    })
}

fn ensure_archive(tgz_path: &Path) -> Result<(), String> {
    if tgz_path.is_file() && md5_hex(tgz_path)? == ARCHIVE_MD5 {
        return Ok(());
    }
    let _ = fs::remove_file(tgz_path);

    let mut last_err = String::from("no download URLs tried");
    for url in URLS {
        eprintln!("downloading CIFAR-10 → {} …\n  from {url}", tgz_path.display());
        match download(url, tgz_path) {
            Ok(()) => match md5_hex(tgz_path) {
                Ok(sum) if sum == ARCHIVE_MD5 => return Ok(()),
                Ok(sum) => {
                    let _ = fs::remove_file(tgz_path);
                    last_err = format!(
                        "MD5 mismatch for {url}: got {sum}, expected {ARCHIVE_MD5}"
                    );
                }
                Err(e) => {
                    let _ = fs::remove_file(tgz_path);
                    last_err = e;
                }
            },
            Err(e) => {
                let _ = fs::remove_file(tgz_path);
                last_err = e;
            }
        }
    }
    Err(format!("CIFAR-10 download failed: {last_err}"))
}

fn download(url: &str, dest: &Path) -> Result<(), String> {
    // Write to a temp path then rename so a killed curl never leaves a
    // "complete-looking" truncated archive.
    let tmp = dest.with_extension("tar.gz.partial");
    let _ = fs::remove_file(&tmp);
    let status = std::process::Command::new("curl")
        .args(["-fL", "--retry", "3", "--retry-delay", "2", "-o"])
        .arg(&tmp)
        .arg(url)
        .status()
        .map_err(|e| format!("failed to spawn curl: {e}"))?;
    if !status.success() {
        let _ = fs::remove_file(&tmp);
        return Err(format!("curl failed for {url}: {status}"));
    }
    fs::rename(&tmp, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!("failed to finalize download: {e}")
    })?;
    Ok(())
}

/// Lowercase hex MD5 via `md5sum` (Linux/coreutils) or `md5 -q` (macOS).
fn md5_hex(path: &Path) -> Result<String, String> {
    let try_md5sum = std::process::Command::new("md5sum")
        .arg(path)
        .output();
    if let Ok(out) = try_md5sum {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            let hex = s.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
            if hex.len() == 32 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
                return Ok(hex);
            }
            return Err(format!("unexpected md5sum output: {s:?}"));
        }
    }
    let out = std::process::Command::new("md5")
        .args(["-q"])
        .arg(path)
        .output()
        .map_err(|e| format!("failed to spawn md5/md5sum: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "md5 failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let hex = String::from_utf8_lossy(&out.stdout).trim().to_ascii_lowercase();
    if hex.len() == 32 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
        Ok(hex)
    } else {
        Err(format!("unexpected md5 output: {hex:?}"))
    }
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
        return Err(format!(
            "tar -xzf {} failed: {status} (often a truncated download)",
            tgz.display()
        ));
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
