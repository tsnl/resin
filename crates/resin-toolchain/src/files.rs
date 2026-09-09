use crate::Error;
use resin_common::prelude::*;
use std::{fs, io, path::Path};

pub(super) fn write_output(bytes: &[u8], output: &Path) -> Result<(), Error> {
    let temp = TempDir::new(parent(output)).map_err(io_error)?;
    let path = temp.path().join("output");
    fs::write(&path, bytes).map_err(io_error)?;
    fs::rename(path, output).map_err(io_error)
}

pub(super) fn copy_output(source: &Path, output: &Path) -> Result<(), Error> {
    fs::create_dir_all(parent(output)).map_err(io_error)?;
    let temp = TempDir::new(parent(output)).map_err(io_error)?;
    let path = temp.path().join("output");
    fs::copy(source, &path).map_err(io_error)?;
    fs::rename(path, output).map_err(io_error)
}

pub(super) fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

pub(super) fn io_error(error: io::Error) -> Error {
    Error(error.to_string())
}

/// A directory lock covers cache inspection, replacement, and the caller's use.
pub(super) fn lock_directory(directory: &Path) -> Result<fs::File, Error> {
    fs::create_dir_all(directory)?;
    let lock = fs::File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("lock"))?;
    lock.lock()?;
    Ok(lock)
}
