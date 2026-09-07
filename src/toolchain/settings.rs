//! Explicit host settings supplied by the compiler's caller.
use crate::backend::Error;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
};

/// Environment and paths resolved before compilation. Discovery failures are retained
/// so unused tools (in particular glslc for host-only code) remain optional.
pub struct Settings {
    pub cc: Result<PathBuf, String>,
    pub glslc: Result<PathBuf, String>,
    pub runtime_include: PathBuf,
    pub runtime_library: Result<PathBuf, String>,
    pub environment: BTreeMap<OsString, OsString>,
    pub cache: PathBuf,
    pub directory: PathBuf,
    pub temporary: PathBuf,
    pub executable: PathBuf,
}

impl Settings {
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
