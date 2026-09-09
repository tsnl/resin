//! Native tools resolved from explicit process inputs, with locked build caches.
//!
//! `Environment` is caller-owned input. `Toolchain` resolves it once and hides
//! process invocation, dependency tracking, and cache maintenance. Missing tools
//! are reported only when an operation needs them, so host builds need no glslc.

use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use resin_common::types::shader::Stage;

mod c;
mod dependencies;
mod environment;
mod files;
mod platform;
mod process;
mod settings;
mod shaders;

use files::{copy_output, io_error, parent, write_output};
pub use platform::DEFAULT_C_COMPILER;
use platform::RUNTIME_ARCHIVE;
use resin_common::TempDir;
use settings::Settings;

/// Captured process inputs; callers may supply these without changing OS state.
pub struct Environment {
    pub variables: BTreeMap<OsString, OsString>,
    pub directory: PathBuf,
    pub executable: PathBuf,
    pub temporary: PathBuf,
}

impl Environment {
    pub fn capture() -> io::Result<Self> {
        Ok(Self {
            variables: std::env::vars_os().collect(),
            directory: std::env::current_dir()?,
            executable: std::env::current_exe()?,
            temporary: std::env::temp_dir(),
        })
    }

    /// Resolve explicit compiler choices, then CC/GLSLC, then platform defaults.
    pub fn toolchain(&self, cc: Option<&OsStr>, glslc: Option<&OsStr>) -> Toolchain {
        self.resolve_tools(cc, glslc)
    }

    pub fn variable(&self, name: &str) -> Option<&OsStr> {
        #[cfg(windows)]
        return self
            .variables
            .iter()
            .find(|(key, _)| {
                key.to_str()
                    .is_some_and(|key| key.eq_ignore_ascii_case(name))
            })
            .map(|(_, value)| value.as_os_str());
        #[cfg(not(windows))]
        self.variables
            .get(OsStr::new(name))
            .map(OsString::as_os_str)
    }

    pub fn path(&self, variable: &str, fallback: impl AsRef<Path>) -> PathBuf {
        self.directory.join(
            self.variable(variable)
                .map(Path::new)
                .unwrap_or_else(|| fallback.as_ref()),
        )
    }
}

/// Resolved tools and their immutable execution environment.
pub struct Toolchain {
    settings: Settings,
}

impl Toolchain {
    /// Compile and retain a cached executable, locked until the result is dropped.
    pub fn build_c(
        &self,
        file: &Path,
        entry: &str,
        source: &str,
        profile: CProfile,
    ) -> Result<Executable, Error> {
        c::build_c(file, entry, source, &self.settings, profile)
    }

    /// Compile C directly to an output path using the release profile.
    pub fn compile_c(&self, source: &str, output: &Path) -> Result<(), Error> {
        c::compile_c(source, output, &self.settings)
    }

    /// Compile GLSL and reuse a valid cached SPIR-V result when available.
    pub fn build_glsl(&self, source: &str, stage: Stage) -> Result<Vec<u8>, Error> {
        shaders::build_glsl(source, stage, &self.settings)
    }

    /// Compile GLSL to validated SPIR-V bytes without using the build cache.
    pub fn compile_glsl(&self, source: &str, stage: Stage) -> Result<Vec<u8>, Error> {
        shaders::compile_glsl(source, stage, &self.settings)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CProfile {
    Debug,
    Release,
}

/// A native executable whose cache lock is retained during copying and execution.
pub struct Executable {
    executable: PathBuf,
    _lock: fs::File,
}

impl Executable {
    pub fn path(&self) -> &Path {
        &self.executable
    }

    pub fn copy_to(&self, output: &Path) -> Result<(), Error> {
        copy_output(self.path(), output)
    }

    pub fn run(&self) -> Result<i32, Error> {
        self.run_with_args(&[])
    }

    /// Pass literal OS arguments; inherit the caller's execution environment.
    pub fn run_with_args(&self, args: &[OsString]) -> Result<i32, Error> {
        Ok(Command::new(self.path())
            .args(args)
            .status()?
            .code()
            .unwrap_or(1))
    }
}

#[derive(Debug)]
pub struct Error(String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}
