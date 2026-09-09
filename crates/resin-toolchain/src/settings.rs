//! Explicit host settings supplied by the compiler's caller.
use crate::Error;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

/// Environment and paths resolved before compilation. Discovery failures are retained
/// so unused tools (in particular glslc for host-only code) remain optional.
pub(super) struct Settings {
    pub(super) cc: Result<PathBuf, String>,
    pub(super) glslc: Result<PathBuf, String>,
    pub(super) runtime_include: PathBuf,
    pub(super) runtime_library: Result<PathBuf, String>,
    pub(super) environment: BTreeMap<OsString, OsString>,
    pub(super) cache: PathBuf,
    pub(super) directory: PathBuf,
    pub(super) temporary: PathBuf,
    pub(super) executable: PathBuf,
}

impl Settings {
    pub(super) fn command(&self, executable: &Path) -> Command {
        let mut command = Command::new(executable);
        command
            .current_dir(&self.directory)
            .env_clear()
            .envs(&self.environment);
        command
    }

    pub(crate) fn cc(&self) -> Result<&Path, Error> {
        self.cc.as_deref().map_err(|error| Error(error.clone()))
    }

    pub(crate) fn glslc(&self) -> Result<&Path, Error> {
        self.glslc.as_deref().map_err(|error| Error(error.clone()))
    }

    pub(crate) fn runtime_library(&self) -> Result<&Path, Error> {
        self.runtime_library
            .as_deref()
            .map_err(|error| Error(error.clone()))
    }
}
