//! MNIST host dataset: download IDX files, parse, batch-normalize.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::{Dataset, Split, dataset_dir, download_file};

pub const IMG_W: usize = 28;
pub const IMG_H: usize = 28;
pub const IMG_WH: usize = IMG_W * IMG_H;
pub type Image = [u8; IMG_WH];

pub const CLASSES: &'static [&'static str; 10] =
    &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// One MNIST sample: normalized pixels and class index.
pub struct Example<'a> {
    pub image: &'a Image,
    pub label: u8,
}

/// MNIST images as `u8` pixels (0–255) and labels as class indices (0–9).
pub struct Mnist {
    pub images: Vec<Image>,
    pub labels: Vec<u8>,
    pub n: usize,
}

impl Dataset for Mnist {
    /// Row index into the split. Richer keys (index + aug params) can replace
    /// this later; [`get`](Dataset::get) stays a pure function of the key.
    type Key = usize;
    type Val<'a> = Example<'a>;

    const NAME: &'static str = "mnist";

    fn load(split: Split) -> Result<Self, String> {
        let dir = Self::download()?;
        let (img_name, lbl_name) = match split {
            Split::Train => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            Split::Test => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        };
        let images = read_idx_images(&dir.join(img_name))?;
        let labels = read_idx_labels(&dir.join(lbl_name))?;
        if images.len() != labels.len() {
            return Err("MNIST image/label count mismatch".into());
        }
        let n = labels.len();
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

impl Mnist {
    fn download() -> Result<PathBuf, String> {
        let dir = dataset_dir(Self::NAME);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        for name in [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ] {
            ensure_file(&dir, name)?;
        }
        Ok(dir)
    }
}

fn ensure_file(dir: &Path, name: &str) -> Result<PathBuf, String> {
    let path = dir.join(name);
    if path.is_file() {
        return Ok(path);
    }
    let url = format!("https://ossci-datasets.s3.amazonaws.com/mnist/{name}.gz");
    let gz_path = dir.join(format!("{name}.gz"));
    eprintln!("downloading MNIST {name} …");
    download_file(&url, &gz_path)?;
    gunzip_file(&gz_path, &path)?;
    let _ = fs::remove_file(&gz_path);
    Ok(path)
}

fn gunzip_file(gz_path: &Path, out_path: &Path) -> Result<(), String> {
    let status = std::process::Command::new("gzip")
        .args(["-dc"])
        .arg(gz_path)
        .stdout(File::create(out_path).map_err(|e| e.to_string())?)
        .status()
        .map_err(|e| format!("failed to spawn gzip: {e}"))?;
    if !status.success() {
        return Err(format!("gzip -dc failed: {status}"));
    }
    Ok(())
}

// ── Load / parse ─────────────────────────────────────────────────────────────

fn read_u32_be(r: &mut impl Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u32::from_be_bytes(buf))
}

fn read_idx_images(path: &Path) -> Result<Vec<Image>, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let magic = read_u32_be(&mut f)?;
    if magic != 2051 {
        return Err(format!("bad MNIST image magic: {magic}"));
    }
    let n = read_u32_be(&mut f)? as usize;
    let rows = read_u32_be(&mut f)? as usize;
    let cols = read_u32_be(&mut f)? as usize;
    if rows != IMG_H || cols != IMG_W {
        return Err(format!("unexpected MNIST image size {rows}x{cols}"));
    }
    // One bulk read, then split into fixed-size rows.
    let mut flat = vec![0u8; n * IMG_WH];
    f.read_exact(&mut flat).map_err(|e| e.to_string())?;
    let images = flat
        .chunks_exact(IMG_WH)
        .map(|chunk| Image::try_from(chunk).expect("chunk is IMG_WH"))
        .collect();
    Ok(images)
}

fn read_idx_labels(path: &Path) -> Result<Vec<u8>, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let magic = read_u32_be(&mut f)?;
    if magic != 2049 {
        return Err(format!("bad MNIST label magic: {magic}"));
    }
    let n = read_u32_be(&mut f)? as usize;
    let mut data = vec![0u8; n];
    f.read_exact(&mut data).map_err(|e| e.to_string())?;
    Ok(data)
}
