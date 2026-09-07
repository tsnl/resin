//! Artifact generation from checked IR and explicit build settings.
use std::{path::Path, process::Command};

use super::{Error, c, glsl};
use crate::{
    compiler::{Request, Target},
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
    let build = toolchain::build_c(
        &request.input().path,
        &request.input().entry,
        &source,
        &request.options().tools,
        request.options().profile,
    )?;
    if let Some(output) = request.destination() {
        toolchain::copy_output(build.executable(), output)?;
    }
    Ok(Executable { build })
}

/// Generate an artifact from an already analyzed module. Native outputs retain
/// their cache lock; textual artifacts include a trailing newline.
pub(crate) fn generate(request: &Request, module: &ir::Module) -> Result<Artifact, Error> {
    let source = match request.target() {
        Target::Executable => return native(request, module).map(Artifact::Executable),
        Target::C => host_source(request, module)?,
        Target::Glsl | Target::Spirv => {
            let stage = shader_stage(request, module)?;
            let source = glsl::emit(module, &request.input().entry, stage)?;
            if matches!(request.target(), Target::Spirv) {
                return toolchain::compile_glsl(&source, stage, &request.options().tools)
                    .map(Artifact::Bytes);
            }
            source
        }
    };
    Ok(Artifact::Bytes(format!("{source}\n").into_bytes()))
}

fn host_source(request: &Request, module: &ir::Module) -> Result<String, Error> {
    let shaders = super::build_shaders(module, &request.options().tools)?;
    c::emit_with_shaders(module, &request.input().entry, &shaders)
}

fn shader_stage(request: &Request, module: &ir::Module) -> Result<glsl::Stage, Error> {
    let declared = module
        .entries
        .get(request.input().entry.as_str())
        .and_then(|id| module.shaders.get(id));
    match (request.options().stage, declared) {
        (Some(stage), Some(entry)) if stage.name() != entry.stage.as_ref() => Err(Error(
            "--stage conflicts with the function's shader decorator".into(),
        )),
        (Some(stage), _) => Ok(stage),
        (None, Some(entry)) => entry.stage.parse(),
        (None, None) => Ok(glsl::Stage::Compute),
    }
}
