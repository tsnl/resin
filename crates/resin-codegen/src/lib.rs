//! Generate C, SPIR-V, and a Ninja dependency graph from verified LIR in one pass.
//! The returned project describes inputs for native tools; it owns no compiler state.
//! Target languages and lowering are private. Generation runs no external tools.
//!
//! ```compile_fail,E0603
//! use resin_codegen::c;
//! ```
//!
//! ```compile_fail,E0603
//! use resin_codegen::spirv;
//! ```
//!
//! ```compile_fail,E0432
//! use resin_codegen::{CModule, SpirvModule};
//! ```

use resin_executor::{Cancellation, Execution};
use resin_types::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tempfile::TempDir;

mod c;
mod error;
mod layout;
mod numeric;
mod spirv;

//
// Generated source files and the binary headers they will need
//

/// Complete native inputs supplied by the application. Files live under `native/`.
/// Missing bindings preserve literal includes for direct compiler clients; services
/// must validate complete bindings before generation. No header is read here.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeHeaders {
    pub files: BTreeMap<Arc<str>, Arc<[u8]>>,
    pub bindings: BTreeMap<resin_lir::ForeignHeader, NativeInclude>,
    pub include_directories: Vec<Arc<str>>,
    /// An explicit runtime binding prevents user include roots replacing the compiler ABI.
    /// It also enables native dependency validation against staged and toolchain include roots.
    pub runtime: Option<NativeInclude>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum NativeInclude {
    Staged { path: Arc<str> },
    System { spelling: Arc<str> },
}

/// Immutable source files ready for native staging. Clones retain the same owned directory.
/// Files outlive the verified input and are removed when the final project owner drops.
#[derive(Debug, Clone)]
pub struct GeneratedProject {
    directory: Arc<TempDir>,
    build_file: PathBuf,
    program: Option<PathBuf>,
    c_source: Option<PathBuf>,
    shaders: Arc<[GeneratedShader]>,
    name: String,
    entry: Option<String>,
}

impl GeneratedProject {
    pub fn directory(&self) -> &Path {
        self.directory.path()
    }

    /// Ninja dependency graph; `toolchain.ninja` supplies native rules and settings.
    pub fn build_file(&self) -> &Path {
        &self.build_file
    }

    /// Planned executable path, absent for a shader-only project.
    pub fn program(&self) -> Option<&Path> {
        self.program.as_deref()
    }

    /// Host translation unit, absent for a shader-only project.
    pub fn c_source(&self) -> Option<&Path> {
        self.c_source.as_deref()
    }

    pub fn shaders(&self) -> &[GeneratedShader] {
        &self.shaders
    }

    /// Opaque diagnostic name of the host entry's source; never interpreted as a path.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn entry(&self) -> Option<&str> {
        self.entry.as_deref()
    }
}

/// A generated SPIR-V module and its planned optimized binary and C header.
#[derive(Debug)]
pub struct GeneratedShader {
    function: FunctionId,
    stage: Stage,
    unoptimized_spirv: PathBuf,
    spirv: PathBuf,
    header: PathBuf,
    symbol: String,
}

impl GeneratedShader {
    pub fn function(&self) -> FunctionId {
        self.function
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// Unoptimized SPIR-V binary emitted directly from verified LIR.
    pub fn unoptimized_spirv(&self) -> &Path {
        &self.unoptimized_spirv
    }

    /// Planned optimized SPIR-V binary path, produced by the toolchain.
    pub fn spirv(&self) -> &Path {
        &self.spirv
    }

    /// Planned header path; source generation does not create this file.
    pub fn header(&self) -> &Path {
        &self.header
    }

    /// Four-byte-aligned byte array in the header, with a matching `<symbol>_length`.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// Generate an owned source project on a bounded worker before running native tools.
/// The existing temporary parent receives a unique child directory for this generation.
/// All target lowering succeeds before any files are written or the project is returned.
/// A requested host entry emits C and its embedded shaders; `None` emits the requested shaders.
/// The toolchain configures and runs the returned Ninja graph. Preprocessed C, optimized
/// SPIR-V, headers, and executables are planned outputs until then. Paths are absolute.
/// Failure or cancellation releases the unpublished directory. Native tools must stage
/// these inputs elsewhere rather than changing a retained generated project.
/// Host projects include `native-inputs.json`: original/captured translation-unit paths,
/// literal `c_flags` and `preprocessing_flags`, and generated header prerequisites.
/// The toolchain captures `main.c` as `main.i` and records its bytes in `native-inputs.state`.
/// Native compilation consumes that captured input without reading headers again.
/// Shader-only projects have no C input metadata.
///
/// ```compile_fail
/// let module = resin_lir::Module::default();
/// resin_codegen::generate(std::sync::Arc::new(module), Some("main".into()),
///     std::path::Path::new("build"), &resin_executor::Execution::default(),
///     &resin_executor::Cancellation::new());
/// ```
pub async fn generate(
    checked: Arc<resin_lir::VerifiedModule>,
    host_entry: Option<String>,
    headers: Arc<NativeHeaders>,
    temporary_parent: &Path,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<GeneratedProject, GenerationError> {
    let temporary_parent = temporary_parent.to_path_buf();
    execution
        .run(cancellation, move |cancellation| {
            let parent = std::path::absolute(temporary_parent)?;
            let directory = Arc::new(
                tempfile::Builder::new()
                    .prefix("resin-codegen-")
                    .tempdir_in(parent)?,
            );
            generate_project(
                checked.view(),
                host_entry.as_deref(),
                &headers,
                directory,
                cancellation,
            )
        })
        .await?
}

fn generate_project(
    checked: resin_lir::Verified<'_>,
    host_entry: Option<&str>,
    headers: &NativeHeaders,
    directory: Arc<TempDir>,
    cancellation: &Cancellation,
) -> Result<GeneratedProject, GenerationError> {
    validate_headers(headers)?;
    let project = describe_project(checked.module(), host_entry, directory)?;
    cancellation.check()?;
    let c = host_entry
        .map(|entry| generate_host(checked, entry, headers))
        .transpose()?;
    let mut shaders = Vec::with_capacity(project.shaders.len());
    for shader in project.shaders.iter() {
        cancellation.check()?;
        shaders.push(generate_shader(checked, shader)?);
    }
    cancellation.check()?;
    let build = build_graph(&project);
    write_sources(
        &project,
        c.as_deref(),
        &shaders,
        &build,
        headers,
        cancellation,
    )?;
    Ok(project)
}

#[derive(Debug)]
pub enum GenerationError {
    Execution { error: resin_executor::Error },
    Codegen { error: Error },
}

impl From<resin_executor::Error> for GenerationError {
    fn from(error: resin_executor::Error) -> Self {
        Self::Execution { error }
    }
}

impl From<Error> for GenerationError {
    fn from(error: Error) -> Self {
        Self::Codegen { error }
    }
}

impl From<std::io::Error> for GenerationError {
    fn from(error: std::io::Error) -> Self {
        Self::Codegen {
            error: error.into(),
        }
    }
}

impl std::fmt::Display for GenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution { error } => error.fmt(f),
            Self::Codegen { error } => error.fmt(f),
        }
    }
}

impl std::error::Error for GenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution { error } => Some(error),
            Self::Codegen { error } => Some(error),
        }
    }
}

/// A target failure, retaining source context independently of terminal rendering.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    location: Option<resin_source::SourceLocation>,
    context: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    UnsupportedTarget,
    InvalidProgram,
    InvalidLir,
    Io,
}

impl Error {
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
    pub fn message(&self) -> &str {
        &self.message
    }
    pub fn location(&self) -> Option<&resin_source::SourceLocation> {
        self.location.as_ref()
    }

    fn new(kind: ErrorKind, message: String) -> Self {
        Self {
            kind,
            message,
            location: None,
            context: None,
        }
    }
    fn unsupported(message: String) -> Self {
        Self::new(ErrorKind::UnsupportedTarget, message)
    }
    fn invalid(message: String) -> Self {
        Self::new(ErrorKind::InvalidLir, message)
    }
    fn program(message: String) -> Self {
        Self::new(ErrorKind::InvalidProgram, message)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(origin) = &self.location {
            write!(f, "{}: ", error::source_context(origin))?;
        }
        if let Some(context) = &self.context {
            write!(f, "{context}: ")?;
        }
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::new(ErrorKind::Io, error.to_string())
    }
}

//
// Plan stable filenames from verified function identities
//

fn describe_project(
    module: &resin_lir::Module,
    entry: Option<&str>,
    directory: Arc<TempDir>,
) -> Result<GeneratedProject, Error> {
    if entry.is_none() && module.shaders.is_empty() {
        return Err(Error::invalid("no shader entries were requested".into()));
    }
    let shaders = module
        .shaders
        .iter()
        .filter(|(_, shader)| entry.is_none() || shader.embedded)
        .map(|(&function, shader)| describe_shader(function, &shader.stage, directory.path()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GeneratedProject {
        build_file: directory.path().join("build.ninja"),
        program: entry.map(|_| directory.path().join(program_filename())),
        c_source: entry.map(|_| directory.path().join("main.c")),
        name: project_name(module, entry),
        entry: entry.map(str::to_owned),
        directory,
        shaders: shaders.into(),
    })
}

fn project_name(module: &resin_lir::Module, entry: Option<&str>) -> String {
    entry
        .and_then(|name| module.entries.get(name))
        .and_then(|function| module.origins.functions.get(function))
        .map(|origin| origin.source.name().to_owned())
        .unwrap_or_else(|| {
            if entry.is_some() {
                "program"
            } else {
                "shaders"
            }
            .into()
        })
}

fn describe_shader(
    function: FunctionId,
    stage: &str,
    directory: &Path,
) -> Result<GeneratedShader, Error> {
    Ok(GeneratedShader {
        function,
        stage: stage.parse().map_err(Error::unsupported)?,
        unoptimized_spirv: directory.join(format!("shader_{}.unoptimized.spv", function.index())),
        spirv: directory.join(format!("shader_{}.spv", function.index())),
        header: directory.join(shader_header(function)),
        symbol: shader_symbol(function),
    })
}

fn shader_header(function: FunctionId) -> String {
    format!("shader_{}.h", function.index())
}

fn shader_symbol(function: FunctionId) -> String {
    format!("r_spv{}", function.index())
}

//
// Lower every target before publishing any source files
//

fn generate_host(
    checked: resin_lir::Verified<'_>,
    entry: &str,
    headers: &NativeHeaders,
) -> Result<String, Error> {
    c::lower::generate(checked, entry, headers).map(|module| c::print::module(&module))
}

fn generate_shader(
    checked: resin_lir::Verified<'_>,
    shader: &GeneratedShader,
) -> Result<Vec<u8>, Error> {
    spirv::generate(checked, shader.function, shader.stage)
}

fn write_sources(
    project: &GeneratedProject,
    c: Option<&str>,
    shaders: &[Vec<u8>],
    build: &str,
    headers: &NativeHeaders,
    cancellation: &Cancellation,
) -> Result<(), GenerationError> {
    cancellation.check()?;
    if let (Some(path), Some(source)) = (&project.c_source, c) {
        write_source(path, source.as_bytes())?;
        write_source(
            &project.directory().join("native-inputs.json"),
            native_inputs(project, headers).as_bytes(),
        )?;
    }
    let mut directories = BTreeSet::new();
    for path in &headers.include_directories {
        stage_directory(project.directory(), path, &mut directories)?;
    }
    for (path, bytes) in &headers.files {
        cancellation.check()?;
        stage_directory(
            project.directory(),
            path.rsplit_once('/').expect("native parent").0,
            &mut directories,
        )?;
        write_source(&project.directory().join(path.as_ref()), bytes)?;
    }
    for (shader, source) in project.shaders.iter().zip(shaders) {
        cancellation.check()?;
        write_source(&shader.unoptimized_spirv, source)?;
    }
    cancellation.check()?;
    write_source(&project.build_file, build.as_bytes())?;
    Ok(())
}

fn write_source(path: &Path, source: &[u8]) -> Result<(), Error> {
    use std::io::Write;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .and_then(|mut file| file.write_all(source))
        .map_err(|error| Error::new(ErrorKind::Io, format!("{}: {error}", path.display())))
}

fn stage_directory(root: &Path, path: &str, created: &mut BTreeSet<String>) -> Result<(), Error> {
    for end in path
        .match_indices('/')
        .map(|(index, _)| index)
        .chain(std::iter::once(path.len()))
    {
        let prefix = &path[..end];
        if created.insert(prefix.to_owned()) {
            // A pre-existing differently-spelled directory is an actual filesystem
            // alias (including Unicode case folding); reject it rather than merge.
            fs::create_dir(root.join(prefix)).map_err(|error| {
                Error::new(ErrorKind::Io, format!("native directory {prefix}: {error}"))
            })?;
        }
    }
    Ok(())
}

//
// Build dependencies: SPIR-V → optimized SPIR-V → embedded headers → host executable
//

fn program_filename() -> String {
    format!("program{}", std::env::consts::EXE_SUFFIX)
}

fn native_inputs(project: &GeneratedProject, headers: &NativeHeaders) -> String {
    let mut preprocessing_flags = Vec::new();
    if let Some(NativeInclude::Staged { path }) = &headers.runtime {
        // Runtime headers include one another with quoted root-relative names.
        // Keep their pragma-once identities in the same frozen snapshot, before
        // the toolchain's ordinary runtime -I fallback or user include roots.
        let parent = path
            .rsplit_once('/')
            .expect("validated staged runtime path")
            .0;
        preprocessing_flags.extend(["-iquote".to_owned(), parent.to_owned()]);
    }
    preprocessing_flags.extend(
        headers
            .include_directories
            .iter()
            .map(|path| format!("-I{path}")),
    );
    serde_json::json!({
        "translation_units": [{"source": "main.c", "preprocessed": "main.i"}],
        "c_flags": [],
        "restrict_header_paths": headers.runtime.is_some(),
        "preprocessing_flags": preprocessing_flags,
        "generated_prerequisites": project.shaders.iter().map(|shader| shader_header(shader.function)).collect::<Vec<_>>(),
    }).to_string()
}

fn validate_headers(headers: &NativeHeaders) -> Result<(), Error> {
    fn staged(path: &str) -> Result<(), Error> {
        if !path.starts_with("native/")
            || path.split('/').any(|part| {
                part.is_empty()
                    || part == "."
                    || part == ".."
                    || part.ends_with([' ', '.'])
                    || device_name(part)
            })
            || path
                .chars()
                .any(|ch| ch.is_control() || "\\:\"<>|?*".contains(ch))
        {
            return Err(Error::invalid(format!(
                "invalid staged native path: {path}"
            )));
        }
        Ok(())
    }
    for path in headers
        .files
        .keys()
        .chain(headers.include_directories.iter())
    {
        staged(path)?;
    }
    let mut aliases = BTreeMap::new();
    for path in headers
        .files
        .keys()
        .chain(headers.include_directories.iter())
    {
        for end in path
            .match_indices('/')
            .map(|(index, _)| index)
            .chain(std::iter::once(path.len()))
        {
            let prefix = &path[..end];
            if aliases
                .insert(prefix.to_ascii_lowercase(), prefix.to_owned())
                .is_some_and(|old| old != prefix)
            {
                return Err(Error::invalid(
                    "native paths collide under ASCII case folding".into(),
                ));
            }
        }
    }
    for include in headers.bindings.values().chain(headers.runtime.iter()) {
        match include {
            NativeInclude::Staged { path } => {
                staged(path)?;
                if !headers.files.contains_key(path) {
                    return Err(Error::invalid(format!(
                        "missing staged native header: {path}"
                    )));
                }
            }
            NativeInclude::System { spelling } if !resin_types::Foreign::valid_header(spelling) => {
                return Err(Error::invalid(format!("invalid system header: {spelling}")));
            }
            NativeInclude::System { .. } => {}
        }
    }
    for path in headers.files.keys() {
        let mut prefix = path.as_ref();
        while let Some((parent, _)) = prefix.rsplit_once('/') {
            if headers.files.contains_key(parent) {
                return Err(Error::invalid(format!(
                    "native file/directory conflict: {parent}"
                )));
            }
            prefix = parent;
        }
    }
    Ok(())
}

fn build_graph(project: &GeneratedProject) -> String {
    let mut out = String::from(
        "# Native tool paths and platform flags are supplied by the toolchain.\ninclude toolchain.ninja\n\n",
    );
    for shader in project.shaders.iter() {
        shader_rules(&mut out, shader);
    }
    if project.program.is_some() {
        host_rule(&mut out, &project.shaders);
    }
    default_target(&mut out, project);
    out
}

fn shader_rules(out: &mut String, shader: &GeneratedShader) {
    use std::fmt::Write;
    let id = shader.function.index();
    writeln!(
        out,
        "build shader_{id}.spv: optimize_shader shader_{id}.unoptimized.spv | toolchain.state"
    )
    .unwrap();
    writeln!(
        out,
        "build shader_{id}.h: embed_shader shader_{id}.spv | toolchain.state"
    )
    .unwrap();
    writeln!(out, "  symbol = {}\n", shader.symbol).unwrap();
}

fn host_rule(out: &mut String, shaders: &[GeneratedShader]) {
    use std::fmt::Write;
    write!(
        out,
        "build {}: compile_preprocessed_program main.i | toolchain.state $runtime_library native-inputs.state",
        program_filename()
    )
    .unwrap();
    for shader in shaders {
        write!(out, " {}", shader_header(shader.function)).unwrap();
    }
    out.push_str("\n\n");
}

fn default_target(out: &mut String, project: &GeneratedProject) {
    use std::fmt::Write;
    out.push_str("build all: phony");
    if project.program.is_some() {
        write!(out, " {}", program_filename()).unwrap();
    } else {
        for shader in project.shaders.iter() {
            write!(out, " {}", shader_header(shader.function)).unwrap();
        }
    }
    out.push_str("\ndefault all\n");
}

fn device_name(part: &str) -> bool {
    let base = part
        .split('.')
        .next()
        .unwrap_or(part)
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    matches!(
        base.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || ["COM", "LPT"].iter().any(|prefix| {
        base.strip_prefix(prefix).is_some_and(|number| {
            matches!(
                number,
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    })
}
