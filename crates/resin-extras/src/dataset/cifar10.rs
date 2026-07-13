//! CIFAR-10 host dataset: download the binary tarball, parse batch files.
//!
//! Official binary layout (channel-first planar RGB): each sample is 1 label
//! byte + 3072 pixel bytes (`R…` then `G…` then `B…`, each plane 32×32).

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::{Dataset, Split, dataset_dir, download_file};

pub const IMG_W: usize = 32;
pub const IMG_H: usize = 32;
pub const IMG_C: usize = 3;
pub type Image = [u8; IMG_CHW];

/// Planar CHW size: `C * H * W`.
pub const IMG_CHW: usize = IMG_C * IMG_H * IMG_W;

pub const CLASSES: &[&str; 10] = &[
    "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
];

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

/// One CIFAR-10 sample: planar CHW pixels and class index.
pub struct Example<'a> {
    pub image: &'a Image,
    pub label: u8,
}

/// CIFAR-10 images as planar `u8` pixels (0–255, CHW) and labels (0–9).
pub struct Cifar10 {
    pub images: Vec<Image>,
    pub labels: Vec<u8>,
    pub n: usize,
}

impl Dataset for Cifar10 {
    /// Row index into the split. Richer keys (index + aug params) can replace
    /// this later; [`get`](Dataset::get) stays a pure function of the key.
    type Key = usize;
    type Val<'a> = Example<'a>;

    const NAME: &'static str = "cifar-10";

    fn load(split: Split) -> Result<Self, String> {
        let dir = Self::download()?;
        let batch_names: &[&str] = match split {
            Split::Train => TRAIN_BATCHES,
            Split::Test => TEST_BATCHES,
        };
        let data_dir = dir.join(EXTRACTED_DIR);
        let mut images = Vec::new();
        let mut labels = Vec::new();
        for name in batch_names {
            read_batch(&data_dir.join(name), &mut images, &mut labels)?;
        }
        let n = labels.len();
        if images.len() != n {
            return Err("CIFAR-10 image/label count mismatch".into());
        }
        Ok(Self { images, labels, n })
    }

    fn len(&self) -> usize {
        self.n
    }

    fn get<'a>(&'a self, &index: &usize) -> Example<'a> {
        Example {
            image: &self.images[index],
            label: self.labels[index],
        }
    }

    fn keys(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.len()
    }
}

// ── Download ─────────────────────────────────────────────────────────────────

impl Cifar10 {
    fn download() -> Result<PathBuf, String> {
        let dir = dataset_dir(Self::NAME);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        ensure_extracted(&dir)?;
        Ok(dir)
    }
}

fn ensure_extracted(cache_dir: &Path) -> Result<PathBuf, String> {
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
        extract_tgz(&tgz_path, cache_dir)
            .map_err(|e2| format!("{e}; retry after re-download also failed: {e2}"))?;
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
        eprintln!(
            "downloading CIFAR-10 → {} …\n  from {url}",
            tgz_path.display()
        );
        match download_file(url, tgz_path) {
            Ok(()) => match md5_hex(tgz_path) {
                Ok(sum) if sum == ARCHIVE_MD5 => return Ok(()),
                Ok(sum) => {
                    let _ = fs::remove_file(tgz_path);
                    last_err = format!("MD5 mismatch for {url}: got {sum}, expected {ARCHIVE_MD5}");
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

/// Lowercase hex MD5 via `md5sum` (Linux/coreutils) or `md5 -q` (macOS).
fn md5_hex(path: &Path) -> Result<String, String> {
    let try_md5sum = std::process::Command::new("md5sum").arg(path).output();
    if let Ok(out) = try_md5sum {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            let hex = s
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
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
    let hex = String::from_utf8_lossy(&out.stdout)
        .trim()
        .to_ascii_lowercase();
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

// ── Load / parse ─────────────────────────────────────────────────────────────

fn read_batch(path: &Path, images: &mut Vec<Image>, labels: &mut Vec<u8>) -> Result<(), String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; BATCH_N * RECORD];
    f.read_exact(&mut buf)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    images.reserve(BATCH_N);
    labels.reserve(BATCH_N);
    for rec in 0..BATCH_N {
        let off = rec * RECORD;
        labels.push(buf[off]);
        let pixels = &buf[off + 1..off + RECORD];
        images.push(Image::try_from(pixels).expect("record pixels are IMG_CHW"));
    }
    Ok(())
}
