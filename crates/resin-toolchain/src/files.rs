//! Async staging and immutable artifact publication. Only staging holds a lock.
use crate::{Error, NativeInputs};
use resin_executor::Cancellation;
use std::{
    collections::BTreeSet,
    io,
    path::{Path, PathBuf},
    time::Duration,
};
use tempfile::TempDir;
use tokio::{fs, io::AsyncReadExt};

pub(super) async fn native_inputs(directory: &Path) -> Result<NativeInputs, Error> {
    let bytes = match fs::read(directory.join("native-inputs.json")).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(NativeInputs::default()),
        Err(error) => return Err(error.into()),
    };
    let inputs: NativeInputs = serde_json::from_slice(&bytes)
        .map_err(|error| Error::new(format!("invalid native-inputs.json: {error}")))?;
    let sources: BTreeSet<_> = inputs
        .translation_units
        .iter()
        .map(|unit| &unit.source)
        .collect();
    let mut outputs = BTreeSet::new();
    for unit in &inputs.translation_units {
        ensure_local_path(directory, &unit.source).await?;
        ensure_local_path(directory, &unit.preprocessed).await?;
        if unit
            .preprocessed
            .extension()
            .is_none_or(|extension| extension != "i")
            || sources.contains(&unit.preprocessed)
            || !outputs.insert(&unit.preprocessed)
        {
            return Err(Error::new(
                "preprocessed C outputs must be distinct .i paths and cannot replace source inputs"
                    .into(),
            ));
        }
    }
    for path in &inputs.generated_prerequisites {
        ensure_local_path(directory, path).await?;
        if outputs.contains(path) {
            return Err(Error::new(
                "preprocessed C outputs cannot also be generated prerequisites".into(),
            ));
        }
    }
    Ok(inputs)
}

async fn previous_captures(directory: &Path) -> Result<Vec<PathBuf>, Error> {
    let bytes = match fs::read(directory.join("native-inputs.json")).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    // Cache metadata can predate the supplied project's schema. Earlier string
    // units did not create captures; only explicit output paths need retiring.
    let Ok(previous) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Ok(Vec::new());
    };
    let mut captures = Vec::new();
    for unit in previous
        .get("translation_units")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(path) = unit.get("preprocessed").and_then(serde_json::Value::as_str) {
            let path = PathBuf::from(path);
            ensure_local_path(directory, &path).await?;
            if path.extension().is_some_and(|extension| extension == "i") {
                captures.push(path);
            }
        }
    }
    Ok(captures)
}

pub(super) async fn temporary(directory: &Path) -> Result<TempDir, Error> {
    let directory = directory.to_path_buf();
    tokio::task::spawn_blocking(move || TempDir::new_in(directory))
        .await
        .map_err(|error| Error::new(error.to_string()))?
        .map_err(Into::into)
}

pub(super) async fn write_output(bytes: &[u8], output: &Path) -> Result<(), Error> {
    let temp = temporary(parent(output)).await?;
    let path = temp.path().join("output");
    fs::write(&path, bytes).await?;
    fs::rename(path, output).await.map_err(Into::into)
}

pub(super) async fn copy_output(source: &Path, output: &Path) -> Result<(), Error> {
    fs::create_dir_all(parent(output)).await?;
    let temp = temporary(parent(output)).await?;
    let path = temp.path().join("output");
    fs::copy(source, &path).await?;
    fs::rename(path, output).await.map_err(Into::into)
}

pub(super) async fn copy_artifact(
    source: &Path,
    output: &Path,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    cancellation.check()?;
    fs::create_dir_all(parent(output)).await?;
    let temp = temporary(parent(output)).await?;
    let path = temp.path().join("output");
    fs::copy(source, &path).await?;
    cancellation.check()?;
    fs::rename(path, output).await.map_err(Into::into)
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

pub(super) async fn lock_directory(
    directory: &Path,
    cancellation: &Cancellation,
) -> Result<std::fs::File, Error> {
    fs::create_dir_all(directory).await?;
    ensure_local_path(directory, Path::new("lock")).await?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("lock"))
        .await?
        .into_std()
        .await;
    loop {
        cancellation.check()?;
        match lock.try_lock() {
            Ok(()) => return Ok(lock),
            Err(std::fs::TryLockError::WouldBlock) => {}
            Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
        }
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::cancelled()),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
    }
}

pub(super) async fn install_changed(source: &Path, output: &Path) -> Result<(), Error> {
    if same_file_contents(source, output).await? {
        return Ok(());
    }
    fs::create_dir_all(parent(output)).await?;
    fs::rename(source, output).await.map_err(Into::into)
}

pub(super) async fn write_changed(bytes: &[u8], output: &Path) -> Result<(), Error> {
    if fs::read(output)
        .await
        .is_ok_and(|previous| previous == bytes)
    {
        return Ok(());
    }
    fs::create_dir_all(parent(output)).await?;
    write_output(bytes, output).await
}

/// Removed generator inputs must not remain available to later graph commands.
pub(super) async fn stage(
    project: &Path,
    work: &Path,
    cache: &Path,
    cancellation: &Cancellation,
) -> Result<NativeInputs, Error> {
    let mut entries = fs::read_dir(project).await?;
    while let Some(entry) = entries.next_entry().await? {
        if reserved(&entry.file_name()) {
            return Err(Error::new(
                "source project contains a reserved toolchain filename".into(),
            ));
        }
    }
    let inputs = relative_files(
        &fs::canonicalize(project).await?,
        Some(&fs::canonicalize(cache).await?),
        cancellation,
    )
    .await?;
    let native = native_inputs(project).await?;
    for unit in &native.translation_units {
        if inputs.iter().any(|input| {
            input.starts_with(&unit.preprocessed) || unit.preprocessed.starts_with(input)
        }) {
            return Err(Error::new(
                "preprocessed C outputs cannot replace supplied project files".into(),
            ));
        }
    }
    // Captures are Ninja inputs, so cleandead cannot remove them. Retire old
    // captures before staging: a replacement project may supply that same path.
    for captured in previous_captures(work).await? {
        cancellation.check()?;
        if !native
            .translation_units
            .iter()
            .any(|current| current.preprocessed == captured)
        {
            remove_file(&work.join(captured)).await?;
        }
    }
    if native.translation_units.is_empty() {
        remove_file(&work.join("native-inputs.state")).await?;
    }
    let inventory = work.join(".resin-inputs");
    let previous = fs::read_to_string(&inventory).await.unwrap_or_default();
    for path in previous.lines().map(PathBuf::from) {
        cancellation.check()?;
        if !inputs.contains(&path) {
            ensure_local_path(work, &path).await?;
            remove_file(&work.join(path)).await?;
        }
    }
    let mut names = String::new();
    for input in &inputs {
        cancellation.check()?;
        let name = input
            .to_str()
            .filter(|name| !name.contains(['\n', '\r']))
            .ok_or_else(|| {
                Error::new("Ninja project filenames must be UTF-8 without newlines".into())
            })?;
        names.push_str(name);
        names.push('\n');
        ensure_local_path(work, input).await?;
        copy_changed(&project.join(input), &work.join(input)).await?;
    }
    write_changed(names.as_bytes(), &inventory).await?;
    Ok(native)
}

/// Only completed builds replace the inspectable cache. Replacements use new inodes,
/// so artifact generations may retain links to any earlier published files.
pub(super) async fn publish(
    work: &Path,
    directory: &Path,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let outputs = relative_files(work, None, cancellation).await?;
    for previous in relative_files(directory, None, cancellation).await? {
        cancellation.check()?;
        if !outputs.contains(&previous) {
            remove_file(&directory.join(previous)).await?;
        }
    }
    for output in &outputs {
        cancellation.check()?;
        ensure_local_path(directory, output).await?;
        publish_file(&work.join(output), &directory.join(output)).await?;
    }
    Ok(())
}

/// Artifact files share only immutable published inodes, never mutable Ninja work.
/// Keep generations on the cache filesystem. Windows copies executable images so a
/// running artifact cannot deny replacement of the incremental cache file.
pub(super) async fn retain(
    directory: &Path,
    generations: &Path,
    cancellation: &Cancellation,
) -> Result<(TempDir, BTreeSet<PathBuf>), Error> {
    fs::create_dir_all(generations).await?;
    let artifact = temporary(generations).await?;
    let paths = relative_files(directory, None, cancellation).await?;
    for relative in &paths {
        cancellation.check()?;
        let target = artifact.path().join(relative);
        fs::create_dir_all(parent(&target)).await?;
        let source = directory.join(relative);
        #[cfg(not(windows))]
        fs::hard_link(source, target).await?;
        #[cfg(windows)]
        {
            fs::copy(&source, &target).await?;
            set_modified(&target, fs::metadata(source).await?.modified()?).await?;
        }
    }
    cancellation.check()?;
    Ok((artifact, paths))
}

async fn publish_file(source: &Path, output: &Path) -> Result<(), Error> {
    let metadata = fs::metadata(source).await?;
    if executable(source, &metadata) && !same_file_contents(source, output).await? {
        fs::create_dir_all(parent(output)).await?;
        // Publish the child-produced inode, never one opened for writing here:
        // another thread's fork can briefly inherit a writer before exec closes it.
        fs::rename(source, output).await?;
        fs::copy(output, source).await?;
        set_modified(source, metadata.modified()?).await?;
        Ok(())
    } else {
        copy_changed(source, output).await
    }
}

async fn set_modified(path: &Path, modified: std::time::SystemTime) -> Result<(), Error> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        std::fs::OpenOptions::new()
            .read(true)
            .write(cfg!(windows))
            .open(path)?
            .set_modified(modified)
    })
    .await
    .map_err(|error| Error::new(error.to_string()))??;
    Ok(())
}

#[cfg(unix)]
fn executable(_: &Path, metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn executable(path: &Path, _: &std::fs::Metadata) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("exe"))
}

pub(super) async fn ensure_local_path(root: &Path, relative: &Path) -> Result<(), Error> {
    let mut path = root.to_path_buf();
    for component in relative.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            return Err(Error::new(
                "project paths must remain inside the build directory".into(),
            ));
        }
        path.push(component);
        if fs::symlink_metadata(&path)
            .await
            .is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err(Error::new(format!(
                "project paths cannot follow symlinks: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

async fn same_file_contents(source: &Path, output: &Path) -> Result<bool, Error> {
    let mut previous = match fs::File::open(output).await {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let mut current = fs::File::open(source).await?;
    let mut remaining = current.metadata().await?.len();
    if previous.metadata().await?.len() != remaining {
        return Ok(false);
    }
    let (mut left, mut right) = (vec![0; 64 * 1024], vec![0; 64 * 1024]);
    while remaining != 0 {
        let count = remaining.min(left.len() as u64) as usize;
        current.read_exact(&mut left[..count]).await?;
        previous.read_exact(&mut right[..count]).await?;
        if left[..count] != right[..count] {
            return Ok(false);
        }
        remaining -= count as u64;
    }
    Ok(true)
}

async fn copy_changed(source: &Path, output: &Path) -> Result<(), Error> {
    if same_file_contents(source, output).await?
        && fs::metadata(output).await?.permissions() == fs::metadata(source).await?.permissions()
    {
        return Ok(());
    }
    copy_output(source, output).await
}

async fn remove_file(path: &Path) -> Result<(), Error> {
    match fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

async fn relative_files(
    root: &Path,
    excluded: Option<&Path>,
    cancellation: &Cancellation,
) -> Result<BTreeSet<PathBuf>, Error> {
    let mut paths = BTreeSet::new();
    let mut pending = vec![PathBuf::new()];
    while let Some(relative) = pending.pop() {
        cancellation.check()?;
        let mut entries = fs::read_dir(root.join(&relative)).await?;
        while let Some(entry) = entries.next_entry().await? {
            if excluded == Some(entry.path().as_path())
                || matches!(
                    entry.file_name().to_str(),
                    Some(".ninja-work" | ".ninja_log" | ".ninja_deps" | ".resin-inputs")
                )
            {
                continue;
            }
            let path = relative.join(entry.file_name());
            ensure_local_path(root, &path).await?;
            if entry.file_type().await?.is_dir() {
                pending.push(path);
            } else {
                paths.insert(path);
            }
        }
    }
    Ok(paths)
}

fn reserved(name: &std::ffi::OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(
            "lock"
                | "native-inputs.state"
                | "toolchain.ninja"
                | "toolchain.state"
                | ".ninja-work"
                | ".ninja_log"
                | ".ninja_deps"
                | ".resin-inputs"
        )
    )
}
