//! Build native executables with captured tools and bounded asynchronous execution.
//! Completed host objects link directly; on-disk C/SPIR-V projects build through Ninja.
//! The included `toolchain.ninja` supplies `optimize_shader`, `embed_shader`, and
//! `compile_program` rules; generated `build.ninja` files describe their dependencies.
//! Ninja owns the graph and incremental work. Successful outputs remain available
//! through immutable `BuiltProject` generations, independently of later cache builds.

use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

mod environment;
mod files;
mod headers;
mod interop;
mod ninja;
mod object;
mod platform;
mod process;
mod settings;
mod shader;

use platform::RUNTIME_ARCHIVE;
use resin_executor::{Cancellation, Execution};
use settings::Settings;
use tempfile::TempDir;
use tokio::process::Command;

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

/// Optional `native-inputs.json` in a generated Ninja project. Every field defaults
/// to empty. Declared C projects compile captured `.i` files with the
/// `compile_preprocessed_program` rule and depend on `native-inputs.state`. Other native graphs can omit it.
/// The toolchain preprocesses these units with the exact configured C flags and
/// captured environment, covering transitive/default headers and conditional includes.
/// Compilation consumes those captured bytes; later header edits affect the next build.
/// The configured tool installation and linked libraries must stay stable during a build.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NativeInputs {
    /// Check compiler-reported dependencies before compiling captured C. Allowed files
    /// must belong to this staged project, the configured runtime, or compiler-discovered
    /// system include roots. This validates input ownership; it is not a process sandbox.
    pub restrict_header_paths: bool,
    /// Original C files and the captured `.i` files consumed by their Ninja edges.
    pub translation_units: Vec<CTranslationUnit>,
    /// Literal language/code-generation arguments used for scanning and compilation.
    pub c_flags: Vec<String>,
    /// Include/define arguments used only while capturing preprocessed C.
    pub preprocessing_flags: Vec<String>,
    /// Project-relative Ninja targets needed before preprocessing, such as shader headers.
    pub generated_prerequisites: Vec<PathBuf>,
}

/// One C input and its project-relative captured output. The paths must differ,
/// and `preprocessed` must end in `.i` so native compilers consume preprocessed C.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CTranslationUnit {
    pub source: PathBuf,
    pub preprocessed: PathBuf,
}

//
// Captured C interoperability
//

/// Scalar C boundary types. Integer widths are 8, 16, 32, or 64; floats are 32 or 64.
/// Void is permitted only as a function result. Pointee types do not affect pointer ABI.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ForeignScalar {
    Void,
    Bool,
    Integer { bits: u8, signed: bool },
    Float { bits: u8 },
    Pointer,
}

/// A requested C function declaration and its exact scalar ABI. `name` must be a C
/// identifier. Only externally linked, nonvariadic functions with the host C calling
/// convention are supported; macros and inline-only definitions are not link contracts.
/// Parameter attributes not exposed by CIndex are rejected because they can add
/// hidden arguments that are absent from the reported scalar function type.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ForeignFunction {
    pub name: String,
    pub params: Vec<ForeignScalar>,
    pub result: ForeignScalar,
}

/// Complete staged header bytes and ordered include context for C interoperability.
/// File/include-directory paths are portable relative paths. `includes` names staged
/// or system headers. Header bytes never come from client filesystem paths.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ForeignInputs {
    pub files: BTreeMap<Arc<str>, Arc<[u8]>>,
    pub includes: Vec<String>,
    pub include_directories: Vec<String>,
    pub functions: Vec<ForeignFunction>,
}

/// A validated external declaration. `symbol` is the linker import name before the
/// platform's global symbol prefix, suitable for a native object emitter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignDeclaration {
    pub name: String,
    pub symbol: String,
    pub c_signature: String,
}

/// Completed declaration metadata extracted from captured preprocessed C headers.
/// No C function bodies or objects are generated. The linker resolves these symbols
/// from its configured libraries; analysis cannot prove their implementations exist.
/// Cache identity belongs to the caller and must include the inputs and captured Clang
/// installation/settings. Keep the installed SDK and compiler stable while reusing it.
#[derive(Clone, Debug)]
pub struct ForeignAnalysis {
    declarations: Arc<[ForeignDeclaration]>,
    includes: Arc<[String]>,
    diagnostics: Arc<[String]>,
}

impl ForeignAnalysis {
    pub fn declarations(&self) -> &[ForeignDeclaration] {
        &self.declarations
    }
    pub fn includes(&self) -> &[String] {
        &self.includes
    }
    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }
}

/// Native operation whose configured tools and SDK inputs form a cache identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeOperation {
    Foreign,
    Shader,
    Link { runtime: bool },
}

/// Resolved tools and their immutable execution environment.
#[derive(Clone)]
pub struct Toolchain {
    settings: Arc<Settings>,
}

impl Toolchain {
    /// Hash the configured inputs used by this operation, including tool file contents.
    /// Unused tools are neither discovered nor read. Explicit header/library roots
    /// are captured where relevant; default SDK installations must stay stable for
    /// the service lifetime. Inputs must stay stable until the operation completes.
    pub async fn fingerprint(
        &self,
        operation: NativeOperation,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<String, Error> {
        let settings = self.settings.clone();
        let execution = execution.clone();
        process::supervise(cancellation, move |cancellation| async move {
            let _permit = execution.acquire(&cancellation).await?;
            settings
                .operation_fingerprint(operation, &cancellation)
                .await
        })
        .await
    }

    /// Preprocess captured headers and validate directly linkable C declarations.
    /// CLANG selects the preprocessor; LIBCLANG_PATH selects its CIndex library
    /// (file or directory). No C object is compiled. Cancellation owns and reaps
    /// native work before removing temporary inputs; returned metadata owns its data.
    pub async fn analyze_foreign(
        &self,
        inputs: Arc<ForeignInputs>,
        temporary: &Path,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<ForeignAnalysis, Error> {
        let (temporary, settings, execution) = (
            temporary.to_path_buf(),
            self.settings.clone(),
            execution.clone(),
        );
        process::supervise(cancellation, move |cancellation| async move {
            interop::analyze(inputs, &temporary, &settings, &execution, &cancellation).await
        })
        .await
    }

    /// Link a self-contained host object with the configured C compiler's linker driver.
    /// The object supplies `main` and may reference the host C/math libraries. This
    /// operation does not compile C, run Ninja, or link Resin's runtime archive.
    /// Each call reserves one execution slot and returns a separately owned temporary
    /// executable generation; callers decide whether to retain and reuse the result.
    /// Cancelling or dropping the future terminates its process tree.
    pub async fn link_object(
        &self,
        object: Arc<[u8]>,
        temporary: &Path,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<Executable, Error> {
        let temporary = temporary.to_path_buf();
        let settings = self.settings.clone();
        let execution = execution.clone();
        process::supervise(cancellation, move |cancellation| async move {
            let _permit = execution.acquire(&cancellation).await?;
            object::link(object, &temporary, &settings, &cancellation).await
        })
        .await
    }

    /// Stage a complete project, run Ninja, and retain an immutable output generation.
    /// `name` and `entry` select incremental work independently of temporary inputs.
    /// Each build reserves one execution slot and runs Ninja with one native job.
    /// Cancelling or dropping the future terminates its process tree; awaited cancellation
    /// finishes cleanup before returning. Retained artifacts never hold the staging lock.
    pub async fn build(
        &self,
        project: &Path,
        name: &str,
        entry: &str,
        profile: CProfile,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<BuiltProject, Error> {
        let (project, name, entry) = (project.to_path_buf(), name.to_owned(), entry.to_owned());
        let settings = self.settings.clone();
        let execution = execution.clone();
        process::supervise(cancellation, move |cancellation| async move {
            let _permit = execution.acquire(&cancellation).await?;
            ninja::build(&project, &name, &entry, profile, &settings, &cancellation).await
        })
        .await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CProfile {
    Debug,
    Release,
}

/// Immutable project files. The final project or executable owner removes its generation.
#[derive(Clone, Debug)]
pub struct BuiltProject {
    directory: Arc<TempDir>,
    files: Arc<std::collections::BTreeSet<PathBuf>>,
}

impl BuiltProject {
    pub fn directory(&self) -> &Path {
        self.directory.path()
    }

    pub fn path(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.directory.path().join(relative)
    }

    pub fn executable(&self, relative: impl AsRef<Path>) -> Result<Executable, Error> {
        if !self.files.contains(relative.as_ref()) {
            return Err(Error::new(format!(
                "built executable not found: {}",
                self.path(relative).display()
            )));
        }
        Ok(Executable {
            executable: self.path(relative),
            _directory: self.directory.clone(),
        })
    }
}

/// A native executable retaining its immutable artifact generation during use.
#[derive(Clone, Debug)]
pub struct Executable {
    executable: PathBuf,
    _directory: Arc<TempDir>,
}

impl Executable {
    pub fn path(&self) -> &Path {
        &self.executable
    }

    /// Copy atomically. Cancellation never leaves a partially copied destination.
    pub async fn copy_to(
        &self,
        output: &Path,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        let artifact = self.clone();
        let output = output.to_path_buf();
        let execution = execution.clone();
        process::supervise(cancellation, move |cancellation| async move {
            let _permit = execution.acquire(&cancellation).await?;
            files::copy_artifact(artifact.path(), &output, &cancellation).await
        })
        .await
    }

    pub async fn run(
        &self,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<i32, Error> {
        self.run_with_args(&[], execution, cancellation).await
    }

    /// Pass literal OS arguments; inherit the caller's execution environment.
    pub async fn run_with_args(
        &self,
        args: &[OsString],
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<i32, Error> {
        let artifact = self.clone();
        let args = args.to_vec();
        let execution = execution.clone();
        process::supervise(cancellation, move |cancellation| async move {
            let _permit = execution.acquire(&cancellation).await?;
            let mut command = Command::new(artifact.path());
            command.args(args);
            let status = process::status(command, &cancellation).await?;
            Ok(status.code().unwrap_or(1))
        })
        .await
    }
}

/// A native operation failed or was cancelled. Cancellation is distinguishable from diagnostics.
#[derive(Debug)]
pub struct Error {
    message: String,
    cancelled: bool,
}

impl Error {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    fn new(message: String) -> Self {
        Self {
            message,
            cancelled: false,
        }
    }

    fn cancelled() -> Self {
        Self {
            message: "native operation cancelled".into(),
            cancelled: true,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::new(error.to_string())
    }
}

impl From<resin_executor::Error> for Error {
    fn from(error: resin_executor::Error) -> Self {
        if matches!(error, resin_executor::Error::Cancelled) {
            Self::cancelled()
        } else {
            Self::new(error.to_string())
        }
    }
}

//
// Completed object and shader operations
//

/// Immutable object inputs and the native runtime required by their references.
/// Object order is preserved; platform libraries follow the captured runtime archive.
#[derive(Clone, Debug)]
pub struct NativeLink {
    pub objects: Vec<Arc<[u8]>>,
    pub runtime: bool,
}

impl Toolchain {
    /// Link captured objects into a separately owned executable generation.
    /// This invokes the configured linker driver without running C compilation or Ninja.
    pub async fn link_native(
        &self,
        inputs: NativeLink,
        temporary: &Path,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<Executable, Error> {
        let temporary = temporary.to_path_buf();
        let settings = self.settings.clone();
        let execution = execution.clone();
        process::supervise(cancellation, move |cancellation| async move {
            let _permit = execution.acquire(&cancellation).await?;
            object::link_native(inputs, &temporary, &settings, &cancellation).await
        })
        .await
    }

    /// Optimize captured SPIR-V for Vulkan 1.3 and return owned binary bytes.
    /// Cancellation or abandonment drains the optimizer before staging is removed.
    pub async fn optimize_shader(
        &self,
        bytes: Arc<[u8]>,
        temporary: &Path,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<Arc<[u8]>, Error> {
        let temporary = temporary.to_path_buf();
        let settings = self.settings.clone();
        let execution = execution.clone();
        process::supervise(cancellation, move |cancellation| async move {
            let _permit = execution.acquire(&cancellation).await?;
            shader::optimize(bytes, &temporary, &settings, &cancellation).await
        })
        .await
    }
}
