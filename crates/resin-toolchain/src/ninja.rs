//! Stage source files, let Ninja build their graph, then publish completed outputs.
use crate::{BuiltProject, CProfile, Error, Settings, files};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::Arc,
};

pub(super) fn build(
    project: &Path,
    name: &str,
    entry: &str,
    profile: CProfile,
    settings: &Settings,
) -> Result<BuiltProject, Error> {
    let root = cache_directory(settings, name, entry);
    let lock = Arc::new(files::lock_directory(&root)?);
    let directory = root.join(profile.directory());
    let work = directory.join(".ninja-work");
    files::ensure_local_path(&root, Path::new(profile.directory()))?;
    files::ensure_local_path(&directory, Path::new(".ninja-work"))?;
    files::stage(project, &work, &settings.cache)?;
    settings.configure(profile, &work)?;
    let mut command = settings.command(&settings.ninja)?;
    command.current_dir(&work);
    super::process::run(command, "native build")?;
    let mut clean = settings.command(&settings.ninja)?;
    clean.current_dir(&work).args(["-t", "cleandead"]);
    super::process::run(clean, "removing obsolete build outputs")?;
    files::publish(&work, &directory)?;
    Ok(BuiltProject { directory, lock })
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
    let mut hash = DefaultHasher::new();
    name.hash(&mut hash);
    entry.hash(&mut hash);
    settings
        .cache
        .join(format!("{}-{:016x}", cache_label(name), hash.finish()))
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
