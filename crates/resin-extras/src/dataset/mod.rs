//! Dataset = key-value map containing training data, resident in host memory.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub mod cifar10;
pub mod mnist;

pub trait Dataset: Sized {
    type Key;
    type Val<'a>
    where
        Self: 'a;

    const NAME: &'static str;

    /// Load a train or test split. Downloads first if needed.
    fn load(split: Split) -> Result<Self, String>;

    /// Number of underlying rows (the domain of plain index keys).
    fn len(&self) -> usize;

    /// Deterministic sample for `key`.
    fn get<'a>(&'a self, key: &Self::Key) -> Self::Val<'a>;
}

/// `data_root()/name` — the on-disk directory for one dataset.
pub fn dataset_dir(name: &str) -> PathBuf {
    data_root().join(name)
}

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

/// Train or test partition of a dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Test,
}

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
