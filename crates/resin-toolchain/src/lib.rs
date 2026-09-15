//! Build an on-disk Ninja project with captured tools and a locked native cache.
//! The included `toolchain.ninja` supplies `optimize_shader`, `embed_shader`, and
//! `compile_program` rules; generated `build.ninja` files describe their dependencies.
//! Ninja owns the graph and incremental work. Successful outputs remain available
//! through `BuiltProject`; executable handles retain its lock during copying or use.

use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

mod environment;
mod files;
mod ninja;
mod platform;
mod process;
mod settings;

use files::copy_output;
use platform::RUNTIME_ARCHIVE;
use settings::Settings;

#[cfg(not(windows))]
pub const DEFAULT_C_COMPILER: &str = "cc";
#[cfg(all(windows, target_env = "msvc"))]
pub const DEFAULT_C_COMPILER: &str = "clang";
#[cfg(all(windows, not(target_env = "msvc")))]
pub const DEFAULT_C_COMPILER: &str = "gcc";

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

    /// Resolve explicit tool choices, then CC/SPIRV_OPT, then platform defaults.
    pub fn toolchain(&self, cc: Option<&OsStr>, spirv_opt: Option<&OsStr>) -> Toolchain {
        self.resolve_tools(cc, spirv_opt)
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
    /// Stage a complete source directory, build its `build.ninja`, and retain outputs.
    /// `name` and `entry` identify a stable cache independently of temporary inputs.
    pub fn build(
        &self,
        project: &Path,
        name: &str,
        entry: &str,
        profile: CProfile,
    ) -> Result<BuiltProject, Error> {
        ninja::build(project, name, entry, profile, &self.settings)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CProfile {
    Debug,
    Release,
}

/// Successful project files, protected from another build until all handles drop.
#[derive(Debug)]
pub struct BuiltProject {
    directory: PathBuf,
    lock: Arc<fs::File>,
}

impl BuiltProject {
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn path(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.directory.join(relative)
    }

    pub fn executable(&self, relative: impl AsRef<Path>) -> Result<Executable, Error> {
        let executable = self.path(relative);
        if !executable.is_file() {
            return Err(Error(format!(
                "built executable not found: {}",
                executable.display()
            )));
        }
        Ok(Executable {
            executable,
            _lock: self.lock.clone(),
        })
    }
}

/// A native executable whose cache lock is retained during copying and execution.
#[derive(Debug)]
pub struct Executable {
    executable: PathBuf,
    _lock: Arc<fs::File>,
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
