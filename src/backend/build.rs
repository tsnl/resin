//! Compilation orchestration shared by CLI modes and other compiler hosts.
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

use super::{Error, c, glsl};
use crate::{
    compiler::{Input, Session},
    ir, toolchain,
};

/// Input and tool choices for native compilation or source/SPIR-V generation.
pub struct Request {
    pub input: Input,
    pub destination: Option<PathBuf>,
    pub cc: Option<OsString>,
    pub glslc: Option<OsString>,
    pub stage: Option<glsl::Stage>,
}

#[derive(Clone, Copy)]
pub enum Target {
    C,
    Glsl,
    Spirv,
}

/// Retains the build-cache lock until the caller has finished using the executable.
pub struct Executable {
    build: toolchain::CBuild,
}

impl Executable {
    pub fn path(&self) -> &Path {
        self.build.executable()
    }

    pub fn run(&self) -> Result<i32, Error> {
        Ok(Command::new(self.path()).status()?.code().unwrap_or(1))
    }
}

/// Build native code and optionally copy a release executable to the destination.
/// Without a destination, use the debug cache. Execution is an explicit separate step.
pub fn compile(request: &Request) -> Result<Executable, Error> {
    request.validate()?;
    let snapshot = Session::default().analyze(&request.input.path)?;
    let source = host_source(request, snapshot.module()?)?;
    let output = request.host_destination()?;
    let profile = if output.is_some() {
        toolchain::CProfile::Release
    } else {
        toolchain::CProfile::Debug
    };
    let compiler = compiler(&request.cc, "CC", toolchain::DEFAULT_C_COMPILER);
    let build = toolchain::build_c(
        &request.input.path,
        &request.input.entry,
        &source,
        &compiler,
        profile,
    )?;
    if let Some(output) = output {
        toolchain::copy_output(build.executable(), &output)?;
    }
    Ok(Executable { build })
}

/// Generate target output. Text targets include a trailing newline; SPIR-V is binary.
pub fn generate(request: &Request, target: Target) -> Result<Vec<u8>, Error> {
    request.validate()?;
    let snapshot = Session::default().analyze(&request.input.path)?;
    let module = snapshot.module()?;
    let source = match target {
        Target::C => host_source(request, module)?,
        Target::Glsl | Target::Spirv => {
            let stage = shader_stage(request, module)?;
            let source = glsl::emit(module, &request.input.entry, stage)?;
            if matches!(target, Target::Spirv) {
                return toolchain::compile_glsl(
                    &source,
                    stage,
                    &compiler(&request.glslc, "GLSLC", "glslc"),
                );
            }
            source
        }
    };
    Ok(format!("{source}\n").into_bytes())
}

fn host_source(request: &Request, module: &ir::Module) -> Result<String, Error> {
    let shaders = super::build_shaders(module, &compiler(&request.glslc, "GLSLC", "glslc"))?;
    c::emit_with_shaders(module, &request.input.entry, &shaders)
}

fn shader_stage(request: &Request, module: &ir::Module) -> Result<glsl::Stage, Error> {
    let declared = module
        .entries
        .get(request.input.entry.as_str())
        .and_then(|id| module.shaders.get(id));
    match (request.stage, declared) {
        (Some(stage), Some(entry)) if stage.name() != entry.stage.as_ref() => Err(Error(
            "--stage conflicts with the function's shader decorator".into(),
        )),
        (Some(stage), _) => Ok(stage),
        (None, Some(entry)) => entry.stage.parse(),
        (None, None) => Ok(glsl::Stage::Compute),
    }
}

impl Request {
    fn validate(&self) -> Result<(), Error> {
        if let Some(output) = &self.destination {
            toolchain::protect_source(&self.input.path, output)?;
        }
        Ok(())
    }

    fn host_destination(&self) -> Result<Option<PathBuf>, Error> {
        let Some(path) = &self.destination else {
            return Ok(None);
        };
        let trailing_separator = path
            .as_os_str()
            .as_encoded_bytes()
            .last()
            .is_some_and(|&b| b == b'/' || b == std::path::MAIN_SEPARATOR as u8);
        let output = if path.is_dir() || trailing_separator {
            let mut name = self
                .input
                .path
                .file_stem()
                .ok_or_else(|| Error("source file needs a name".into()))?
                .to_os_string();
            if self.input.entry != "main" {
                name.push(format!("-{}", self.input.entry));
            }
            name.push(std::env::consts::EXE_SUFFIX);
            path.join(name)
        } else {
            path.clone()
        };
        toolchain::protect_source(&self.input.path, &output)?;
        Ok(Some(output))
    }
}

fn compiler(option: &Option<OsString>, env: &str, fallback: &str) -> OsString {
    option
        .clone()
        .or_else(|| std::env::var_os(env))
        .unwrap_or_else(|| fallback.into())
}
