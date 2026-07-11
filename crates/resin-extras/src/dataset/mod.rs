//! Host datasets (map-style) and samplers.
//!
//! A [`Dataset`] is an indexable table: deterministic `get(key) → item`. Keys are
//! not limited to plain row indices — a [`Sampler`] may put augmentation
//! parameters (crop, flip, …) into the key so the dataset stays pure.
//!
//! [`IndexSampler`] is the basic shuffle-over-row-indices implementation.
//! Streaming sources (`IterDataset` / `IterSampler`) are intentionally out of
//! scope here; add them later as a parallel path when shard/token streams land.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub mod cifar10;
pub mod mnist;
pub mod sampler;

pub use cifar10::Cifar10;
pub use mnist::{IMG_H, IMG_W, IMG_WH, Mnist, NUM_CLS};
pub use sampler::{IndexSampler, Sampler};

// ── Shared root ──────────────────────────────────────────────────────────────

/// Root directory for all dataset downloads.
///
/// Order: `RESIN_DATA_DIR` env, else `~/.cache/resin`, else `./data`.
pub fn data_root() -> PathBuf {
    if let Some(dir) = env::var_os("RESIN_DATA_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home).join(".cache/resin");
    }
    PathBuf::from("data")
}

/// `data_root()/name` — the on-disk directory for one dataset.
pub fn dataset_dir(name: &str) -> PathBuf {
    data_root().join(name)
}

// ── Split ────────────────────────────────────────────────────────────────────

/// Train or test partition of a dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Test,
}

// ── Dataset (map-style) ──────────────────────────────────────────────────────

/// Map-style dataset: deterministic lookup by key.
///
/// # Keys
///
/// [`Key`] may be a plain row index (`usize`) or a richer value that also
/// carries sampler-drawn augmentation parameters. Implementors must treat
/// [`get`](Dataset::get) as a pure function of `(self, key)` — no hidden RNG.
/// That keeps multi-worker / cache / resume behavior boring and correct.
///
/// # Streaming
///
/// For shard streams and LM pretraining, prefer a future `IterDataset` rather
/// than stretching this trait.
pub trait Dataset: Sized {
    /// Lookup key. Often `usize` (row index); may include aug parameters.
    type Key;

    /// One sample returned by [`get`](Dataset::get).
    type Item;

    /// Subdirectory under [`data_root`], e.g. `"mnist"` or `"cifar-10"`.
    const NAME: &'static str;

    /// Ensure raw files exist under [`dataset_dir`]`(Self::NAME)`. Idempotent.
    /// Returns the dataset directory path.
    fn download() -> Result<PathBuf, String>;

    /// Load a train or test split. Downloads first if needed.
    fn load(split: Split) -> Result<Self, String>;

    /// Number of underlying rows (the domain of plain index keys).
    fn len(&self) -> usize;

    /// Deterministic sample for `key`. Must not draw random numbers internally.
    fn get(&self, key: &Self::Key) -> Self::Item;
}

// ── Shared download helper ───────────────────────────────────────────────────

/// Download `url` to `dest` via curl. Writes through a `.partial` sibling so a
/// killed curl never leaves a truncated file that looks complete.
pub(crate) fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = dest.with_extension("partial");
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
