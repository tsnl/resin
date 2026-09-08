//! Artifact generation from checked IR and explicit build settings.
use std::{ffi::OsString, path::Path, process::Command};

use super::{Error, c};
use crate::{compiler::Request, ir, toolchain};

/// Retains the build-cache lock until the caller has finished using the executable.
pub struct Executable {
    build: toolchain::CBuild,
}

impl Executable {
    pub fn path(&self) -> &Path {
        self.build.executable()
    }

    pub fn run(&self) -> Result<i32, Error> {
        self.run_with_args(&[])
    }

    /// Execute with literal OS arguments, retaining the build-cache lock throughout.
    pub fn run_with_args(&self, args: &[OsString]) -> Result<i32, Error> {
        Ok(Command::new(self.path())
            .args(args)
            .status()?
            .code()
            .unwrap_or(1))
    }
}

pub(crate) fn generate(
    request: &Request,
    checked: ir::verify::Verified<'_>,
) -> Result<Executable, Error> {
    let shaders = super::shaders::build_verified(checked, &request.options().tools)?;
    let source = c::emit_verified(checked, &request.input().entry, &shaders)?;
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
