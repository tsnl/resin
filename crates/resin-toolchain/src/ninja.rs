//! Stage source files, let Ninja build their graph, then publish completed outputs.
use crate::{BuiltProject, CProfile, Error, NativeInputs, Settings, files};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub(super) async fn build(
    project: &Path,
    name: &str,
    entry: &str,
    profile: CProfile,
    settings: &Settings,
    cancellation: &resin_executor::Cancellation,
) -> Result<BuiltProject, Error> {
    let root = cache_directory(settings, name, entry);
    let _lock = files::lock_directory(&root, cancellation).await?;
    let directory = root.join(profile.directory());
    let work = directory.join(".ninja-work");
    files::ensure_local_path(&root, Path::new(profile.directory())).await?;
    files::ensure_local_path(&directory, Path::new(".ninja-work")).await?;
    let inputs = files::stage(project, &work, &settings.cache, cancellation).await?;
    settings
        .configure(profile, &work, &inputs, cancellation)
        .await?;
    prerequisites(&work, &inputs, settings, cancellation).await?;
    settings
        .preprocess(profile, &work, &inputs, cancellation)
        .await?;
    let mut command = settings.command(&settings.ninja)?;
    command.current_dir(&work).args(["-j", "1"]);
    super::process::run(command, "native build", cancellation).await?;
    let mut clean = settings.command(&settings.ninja)?;
    clean.current_dir(&work).args(["-t", "cleandead"]);
    super::process::run(clean, "removing obsolete build outputs", cancellation).await?;
    files::publish(&work, &directory, cancellation).await?;
    files::ensure_local_path(&settings.cache, Path::new(".artifacts")).await?;
    let (directory, files) =
        files::retain(&directory, &settings.cache.join(".artifacts"), cancellation).await?;
    Ok(BuiltProject {
        directory: Arc::new(directory),
        files: Arc::new(files),
    })
}

async fn prerequisites(
    directory: &Path,
    inputs: &NativeInputs,
    settings: &Settings,
    cancellation: &resin_executor::Cancellation,
) -> Result<(), Error> {
    if inputs.generated_prerequisites.is_empty() {
        return Ok(());
    }
    let mut command = settings.command(&settings.ninja)?;
    command
        .current_dir(directory)
        .args(["-j", "1", "--"])
        .args(&inputs.generated_prerequisites);
    super::process::run(command, "native preprocessing prerequisites", cancellation).await
}

impl CProfile {
    pub(super) fn directory(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }

    pub(super) fn optimization(self) -> &'static str {
        match self {
            Self::Debug => "-O0",
            Self::Release => "-O3",
        }
    }
}

fn cache_directory(settings: &Settings, name: &str, entry: &str) -> PathBuf {
    let mut hash = blake3::Hasher::new();
    hash.update(b"resin-native-slot-v1");
    for text in [name, entry] {
        hash.update(&(text.len() as u64).to_le_bytes());
        hash.update(text.as_bytes());
    }
    let digest = hash.finalize().to_hex();
    settings
        .cache
        .join(format!("{}-{}", cache_label(name), &digest[..16]))
}

// Cosmetic label only: the complete opaque source name remains in the hash.
fn cache_label(name: &str) -> String {
    let filename = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let stem = filename.rsplit_once('.').map_or(filename, |(stem, _)| stem);
    let label: String = stem
        .chars()
        .filter(|c| c.is_alphanumeric() || " _-".contains(*c))
        .take(64)
        .collect();
    if label.is_empty() {
        "program".into()
    } else {
        label
    }
}
