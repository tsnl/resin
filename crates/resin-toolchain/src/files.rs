use crate::Error;
use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};
use tempfile::TempDir;

pub(super) fn write_output(bytes: &[u8], output: &Path) -> Result<(), Error> {
    let temp = TempDir::new_in(parent(output)).map_err(io_error)?;
    let path = temp.path().join("output");
    fs::write(&path, bytes).map_err(io_error)?;
    fs::rename(path, output).map_err(io_error)
}

pub(super) fn copy_output(source: &Path, output: &Path) -> Result<(), Error> {
    fs::create_dir_all(parent(output)).map_err(io_error)?;
    let temp = TempDir::new_in(parent(output)).map_err(io_error)?;
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
    ensure_local_path(directory, Path::new("lock"))?;
    let lock = fs::File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("lock"))?;
    lock.lock()?;
    Ok(lock)
}

pub(super) fn write_changed(bytes: &[u8], output: &Path) -> Result<(), Error> {
    if fs::read(output).is_ok_and(|previous| previous == bytes) {
        return Ok(());
    }
    fs::create_dir_all(parent(output))?;
    write_output(bytes, output)
}

/// Removed generator inputs must not remain available to later graph commands.
pub(super) fn stage(project: &Path, work: &Path, cache: &Path) -> Result<(), Error> {
    for entry in fs::read_dir(project)? {
        if reserved(&entry?.file_name()) {
            return Err(Error(
                "source project contains a reserved toolchain filename".into(),
            ));
        }
    }
    let mut inputs = BTreeSet::new();
    collect_files(
        &fs::canonicalize(project)?,
        Path::new(""),
        &mut inputs,
        Some(&fs::canonicalize(cache)?),
    )?;
    let inventory = work.join(".resin-inputs");
    let previous = fs::read_to_string(&inventory).unwrap_or_default();
    for path in previous.lines().map(PathBuf::from) {
        if !inputs.contains(&path) {
            remove_file(&work.join(path))?;
        }
    }
    let mut names = String::new();
    for input in &inputs {
        let name = input
            .to_str()
            .filter(|name| !name.contains(['\n', '\r']))
            .ok_or_else(|| {
                Error("Ninja project filenames must be UTF-8 without newlines".into())
            })?;
        names.push_str(name);
        names.push('\n');
        ensure_local_path(work, input)?;
        copy_changed(&project.join(input), &work.join(input))?;
    }
    write_changed(names.as_bytes(), &inventory)
}

/// Ninja may overwrite intermediate outputs on failure; only successes are published.
pub(super) fn publish(work: &Path, directory: &Path) -> Result<(), Error> {
    let outputs = relative_files(work)?;
    for previous in relative_files(directory)? {
        if !outputs.contains(&previous) {
            remove_file(&directory.join(previous))?;
        }
    }
    for output in &outputs {
        ensure_local_path(directory, output)?;
        publish_file(&work.join(output), &directory.join(output))?;
    }
    Ok(())
}

fn publish_file(source: &Path, output: &Path) -> Result<(), Error> {
    let metadata = fs::metadata(source)?;
    if executable(source, &metadata) && fs::read(output).ok() != Some(fs::read(source)?) {
        fs::create_dir_all(parent(output))?;
        // Publish the child-produced inode, never one opened for writing here:
        // another thread's fork can briefly inherit a writer before exec closes it.
        fs::rename(source, output)?;
        fs::copy(output, source)?;
        fs::File::open(source)?.set_modified(metadata.modified()?)?;
        Ok(())
    } else {
        copy_changed(source, output)
    }
}

#[cfg(unix)]
fn executable(_: &Path, metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn executable(path: &Path, _: &fs::Metadata) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("exe"))
}

pub(super) fn ensure_local_path(root: &Path, relative: &Path) -> Result<(), Error> {
    let mut path = root.to_path_buf();
    for component in relative.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            return Err(Error(
                "project paths must remain inside the build directory".into(),
            ));
        }
        path.push(component);
        if fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return Err(Error(format!(
                "project paths cannot follow symlinks: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn copy_changed(source: &Path, output: &Path) -> Result<(), Error> {
    if fs::read(output).ok() == Some(fs::read(source)?) {
        fs::set_permissions(output, fs::metadata(source)?.permissions())?;
        return Ok(());
    }
    copy_output(source, output)
}

fn remove_file(path: &Path) -> Result<(), Error> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn relative_files(directory: &Path) -> Result<BTreeSet<PathBuf>, Error> {
    let mut paths = BTreeSet::new();
    collect_files(directory, Path::new(""), &mut paths, None)?;
    Ok(paths)
}

fn collect_files(
    root: &Path,
    relative: &Path,
    paths: &mut BTreeSet<PathBuf>,
    excluded: Option<&Path>,
) -> Result<(), Error> {
    for entry in fs::read_dir(root.join(relative))? {
        let entry = entry?;
        if excluded == Some(entry.path().as_path()) {
            continue;
        }
        if matches!(
            entry.file_name().to_str(),
            Some(".ninja-work" | ".ninja_log" | ".ninja_deps" | ".resin-inputs")
        ) {
            continue;
        }
        let path = relative.join(entry.file_name());
        ensure_local_path(root, &path)?;
        if entry.file_type()?.is_dir() {
            collect_files(root, &path, paths, excluded)?;
        } else {
            paths.insert(path);
        }
    }
    Ok(())
}

fn reserved(name: &std::ffi::OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(
            "lock"
                | "toolchain.ninja"
                | "toolchain.state"
                | ".ninja-work"
                | ".ninja_log"
                | ".ninja_deps"
                | ".resin-inputs"
        )
    )
}
