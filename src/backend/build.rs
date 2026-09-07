//! Compilation orchestration shared by CLI modes and other compiler hosts.
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

use super::{Error, c, glsl};
use crate::{
    compiler::{Request, Session, Target},
    ir, toolchain,
};

/// A completed compilation, retaining executable locks through execution.
pub enum Artifact {
    Executable(Executable),
    Bytes(Vec<u8>),
}

impl Artifact {
    pub fn run(&self) -> Result<i32, Error> {
        match self {
            Self::Executable(executable) => executable.run(),
            Self::Bytes(_) => Err(Error("only a native executable can be run".into())),
        }
    }
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

fn native(request: &Request, module: &ir::Module) -> Result<Executable, Error> {
    let source = host_source(request, module)?;
    let output = host_destination(request)?;
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

/// Compile the requested artifact without executing it.
/// Native outputs are copied when a destination is supplied; otherwise they remain
/// in the debug cache. Text artifacts include a trailing newline; SPIR-V is binary.
pub fn compile(request: &Request) -> Result<Artifact, Error> {
    validate(request)?;
    let snapshot = Session::default().analyze(&request.input.path)?;
    let module = snapshot.module()?;
    let source = match request.target {
        Target::Executable => return native(request, module).map(Artifact::Executable),
        Target::C => host_source(request, module)?,
        Target::Glsl | Target::Spirv => {
            let stage = shader_stage(request, module)?;
            let source = glsl::emit(module, &request.input.entry, stage)?;
            if matches!(request.target, Target::Spirv) {
                return toolchain::compile_glsl(
                    &source,
                    stage,
                    &compiler(&request.glslc, "GLSLC", "glslc"),
                )
                .map(Artifact::Bytes);
            }
            source
        }
    };
    Ok(Artifact::Bytes(format!("{source}\n").into_bytes()))
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

fn validate(request: &Request) -> Result<(), Error> {
    if let Some(output) = &request.destination {
        toolchain::protect_source(&request.input.path, output)?;
    }
    Ok(())
}

fn host_destination(request: &Request) -> Result<Option<PathBuf>, Error> {
    let Some(path) = &request.destination else {
        return Ok(None);
    };
    let trailing_separator = path
        .as_os_str()
        .as_encoded_bytes()
        .last()
        .is_some_and(|&b| b == b'/' || b == std::path::MAIN_SEPARATOR as u8);
    let output = if path.is_dir() || trailing_separator {
        let mut name = request
            .input
            .path
            .file_stem()
            .ok_or_else(|| Error("source file needs a name".into()))?
            .to_os_string();
        if request.input.entry != "main" {
            name.push(format!("-{}", request.input.entry));
        }
        name.push(std::env::consts::EXE_SUFFIX);
        path.join(name)
    } else {
        path.clone()
    };
    toolchain::protect_source(&request.input.path, &output)?;
    Ok(Some(output))
}

fn compiler(option: &Option<OsString>, env: &str, fallback: &str) -> OsString {
    option
        .clone()
        .or_else(|| std::env::var_os(env))
        .unwrap_or_else(|| fallback.into())
}
