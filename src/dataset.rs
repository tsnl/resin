//! Host dataset loaders (MNIST and friends).

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const IMG_W: usize = 28;
pub const IMG_H: usize = 28;
pub const IMG_WH: usize = IMG_W * IMG_H;
pub const NUM_CLS: usize = 10;

/// MNIST images as `u8` pixels (0–255) and labels as class indices (0–9).
pub struct MnistDataset {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
    pub n: usize,
}

impl MnistDataset {
    pub fn load(split: &str, cache_dir: impl AsRef<Path>) -> Result<Self, String> {
        let (img_name, lbl_name) = match split {
            "train" => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            "test" => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
            other => return Err(format!("unknown MNIST split: {other}")),
        };
        let cache_dir = cache_dir.as_ref();
        fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;

        let img_path = ensure_mnist_file(cache_dir, img_name)?;
        let lbl_path = ensure_mnist_file(cache_dir, lbl_name)?;

        let images = read_idx_images(&img_path)?;
        let labels = read_idx_labels(&lbl_path)?;
        if images.len() / IMG_WH != labels.len() {
            return Err("MNIST image/label count mismatch".into());
        }
        let n = labels.len();
        Ok(Self { images, labels, n })
    }

    /// Normalized `f32` images and one-hot `f32` labels for a batch of indices.
    pub fn batch_f32(&self, indices: &[usize]) -> (Vec<f32>, Vec<f32>) {
        batch_f32_from_indices(&self.images, &self.labels, indices)
    }
}

fn batch_f32_from_indices(images: &[u8], labels: &[u8], indices: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let b = indices.len();
    let mut xs = vec![0f32; b * IMG_WH];
    let mut ys = vec![0f32; b * NUM_CLS];
    for (row, &idx) in indices.iter().enumerate() {
        let base = idx * IMG_WH;
        for i in 0..IMG_WH {
            xs[row * IMG_WH + i] = images[base + i] as f32 / 255.0;
        }
        let label = labels[idx] as usize;
        ys[row * NUM_CLS + label] = 1.0;
    }
    (xs, ys)
}

fn ensure_mnist_file(cache_dir: &Path, name: &str) -> Result<PathBuf, String> {
    let path = cache_dir.join(name);
    if path.is_file() {
        return Ok(path);
    }
    let url = format!("https://ossci-datasets.s3.amazonaws.com/mnist/{name}.gz");
    let gz_path = cache_dir.join(format!("{name}.gz"));
    download(&url, &gz_path)?;
    gunzip_file(&gz_path, &path)?;
    let _ = fs::remove_file(&gz_path);
    Ok(path)
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

fn read_u32_be(r: &mut impl Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u32::from_be_bytes(buf))
}

fn read_idx_images(path: &Path) -> Result<Vec<u8>, String> {
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
    let mut data = vec![0u8; n * IMG_WH];
    f.read_exact(&mut data).map_err(|e| e.to_string())?;
    Ok(data)
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

/// Shuffled batch index iterator (Fisher–Yates with LCG).
pub struct BatchIndices {
    order: Vec<usize>,
    n: usize,
    batch_size: usize,
    pos: usize,
    drop_last: bool,
}

impl BatchIndices {
    pub fn new(n: usize, batch_size: usize, seed: u64, drop_last: bool) -> Self {
        let mut order: Vec<usize> = (0..n).collect();
        let mut state = seed;
        for i in (1..n).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (state as usize) % (i + 1);
            order.swap(i, j);
        }
        if drop_last {
            let usable = (n / batch_size) * batch_size;
            order.truncate(usable);
        }
        Self {
            order,
            n,
            batch_size,
            pos: 0,
            drop_last,
        }
    }

    pub fn rewind(&mut self, seed: u64) {
        *self = Self::new(self.n, self.batch_size, seed, self.drop_last);
    }

    pub fn next_batch(&mut self) -> Option<&[usize]> {
        if self.pos + self.batch_size > self.order.len() {
            return None;
        }
        let start = self.pos;
        self.pos += self.batch_size;
        Some(&self.order[start..self.pos])
    }
}