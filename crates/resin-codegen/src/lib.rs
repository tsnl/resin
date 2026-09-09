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

use resin_types::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
};

mod c;
mod error;
mod layout;
mod numeric;
mod spirv;

//
// Generated source files and the binary headers they will need
//

/// Source files ready for native tools. The caller owns the output directory.
/// Files and metadata remain usable after the verified input is dropped.
#[derive(Debug)]
pub struct GeneratedProject {
    directory: PathBuf,
    build_file: PathBuf,
    program: Option<PathBuf>,
    c_source: Option<PathBuf>,
    shaders: Vec<GeneratedShader>,
    name: String,
    entry: Option<String>,
}

impl GeneratedProject {
    pub fn directory(&self) -> &Path {
        &self.directory
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

/// Generate a source project and its complete build graph before running native tools.
/// All target lowering succeeds before any files are written.
/// A host entry emits C and its embedded shaders; `None` emits all declared shaders.
/// The toolchain configures and runs the returned Ninja graph. Optimized SPIR-V, headers, and
/// executables are planned outputs until then. Paths in the result are absolute.
/// I/O failure may leave partially written files.
///
/// ```compile_fail,E0308
/// let module = resin_lir::Module::default();
/// resin_codegen::generate(&module, Some("main"), std::path::Path::new("build"));
/// ```
pub fn generate(
    checked: resin_lir::Verified<'_>,
    host_entry: Option<&str>,
    directory: &Path,
) -> Result<GeneratedProject, Error> {
    let project = describe_project(checked.module(), host_entry, directory)?;
    let c = host_entry
        .map(|entry| generate_host(checked, entry))
        .transpose()?;
    let shaders = project
        .shaders
        .iter()
        .map(|shader| generate_shader(checked, shader))
        .collect::<Result<Vec<_>, _>>()?;
    let build = build_graph(&project);
    write_sources(&project, c.as_deref(), &shaders, &build)?;
    Ok(project)
}

#[derive(Debug)]
pub struct Error(String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

//
// Plan stable filenames from verified function identities
//

fn describe_project(
    module: &resin_lir::Module,
    entry: Option<&str>,
    directory: &Path,
) -> Result<GeneratedProject, Error> {
    let directory = std::path::absolute(directory)?;
    let shaders = module
        .shaders
        .iter()
        .filter(|(_, shader)| entry.is_none() || shader.embedded)
        .map(|(&function, shader)| describe_shader(function, &shader.stage, &directory))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GeneratedProject {
        build_file: directory.join("build.ninja"),
        program: entry.map(|_| directory.join(program_filename())),
        c_source: entry.map(|_| directory.join("main.c")),
        name: project_name(module, entry),
        entry: entry.map(str::to_owned),
        directory,
        shaders,
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
        stage: stage.parse().map_err(Error)?,
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

fn generate_host(checked: resin_lir::Verified<'_>, entry: &str) -> Result<String, Error> {
    c::lower::generate(checked, entry).map(|module| c::print::module(&module))
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
) -> Result<(), Error> {
    fs::create_dir_all(&project.directory)?;
    if let (Some(path), Some(source)) = (&project.c_source, c) {
        write_source(path, source.as_bytes())?;
    }
    for (shader, source) in project.shaders.iter().zip(shaders) {
        write_source(&shader.unoptimized_spirv, source)?;
    }
    write_source(&project.build_file, build.as_bytes())
}

fn write_source(path: &Path, source: &[u8]) -> Result<(), Error> {
    if fs::read(path).is_ok_and(|current| current == source) {
        return Ok(());
    }
    fs::write(path, source).map_err(|error| Error(format!("{}: {error}", path.display())))
}

//
// Build dependencies: SPIR-V → optimized SPIR-V → embedded headers → host executable
//

fn program_filename() -> String {
    format!("program{}", std::env::consts::EXE_SUFFIX)
}

fn build_graph(project: &GeneratedProject) -> String {
    let mut out = String::from(
        "# Native tool paths and platform flags are supplied by the toolchain.\ninclude toolchain.ninja\n\n",
    );
    for shader in &project.shaders {
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
        "build {}: compile_program main.c | toolchain.state $runtime_library",
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
        for shader in &project.shaders {
            write!(out, " {}", shader_header(shader.function)).unwrap();
        }
    }
    out.push_str("\ndefault all\n");
}
