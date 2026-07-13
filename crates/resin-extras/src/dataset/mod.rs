//! Dataset = key-value map containing training data, resident in host memory.
//!
//! Walk the map with [`Dataset::keys`] / [`Dataset::iter`] — not an `Iterator`
//! impl on the store itself (values may borrow from the dataset).

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

    /// Every key in the map (order is dataset-defined; e.g. `0..len()`).
    fn keys(&self) -> impl Iterator<Item = Self::Key> + '_;

    /// `(key, value)` pairs via [`keys`](Self::keys) + [`get`](Self::get).
    fn iter(&self) -> impl Iterator<Item = (Self::Key, Self::Val<'_>)> + '_ {
        self.keys().map(|k| {
            let v = self.get(&k);
            (k, v)
        })
    }

    /// Every value in the map
    fn values(&self) -> impl Iterator<Item = Self::Val<'_>> + '_ {
        self.keys().map(|k| self.get(&k))
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny host map: label is the value (owned so the test stays free of downloads).
    struct Tiny {
        labels: Vec<u8>,
    }

    impl Dataset for Tiny {
        type Key = usize;
        type Val<'a> = u8;

        const NAME: &'static str = "tiny";

        fn load(_split: Split) -> Result<Self, String> {
            unreachable!("test fixture")
        }

        fn len(&self) -> usize {
            self.labels.len()
        }

        fn get<'a>(&'a self, &index: &usize) -> u8 {
            self.labels[index]
        }

        fn keys(&self) -> impl Iterator<Item = usize> + '_ {
            0..self.len()
        }
    }

    #[test]
    fn keys_and_iter_walk_in_order() {
        let d = Tiny {
            labels: vec![3, 1, 4],
        };
        assert_eq!(d.keys().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(d.iter().collect::<Vec<_>>(), vec![(0, 3), (1, 1), (2, 4)]);
    }
}
