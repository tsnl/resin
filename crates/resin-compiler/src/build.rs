//! Artifact generation from checked IR and explicit build settings.
use crate::lir_verifier;

use crate::Error;
use crate::{Request, toolchain};

pub(crate) fn generate(
    request: &Request,
    checked: lir_verifier::Verified<'_>,
) -> Result<toolchain::Executable, Error> {
    let shaders = super::shaders::build_verified(checked, &request.options.tools)?;
    let source = emit_c(checked, &request.input.entry, &shaders)?;
    let build = request.options.tools.build_c(
        &request.input.path,
        &request.input.entry,
        &source,
        request.options.profile,
    )?;
    if let Some(output) = request.destination() {
        build.copy_to(output)?;
    }
    Ok(build)
}

pub(super) fn emit_c(
    checked: crate::lir_verifier::Verified<'_>,
    entry: &str,
    shaders: &[crate::codegen::Shader],
) -> Result<String, crate::Error> {
    crate::codegen::generate_c(checked, entry, shaders)
        .map(|module| crate::codegen::print_c(&module))
        .map_err(Into::into)
}
