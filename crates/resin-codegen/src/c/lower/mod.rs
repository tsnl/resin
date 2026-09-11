//! Verified LIR → C source tree. Runtime and ABI choices are made here.
use crate::Error;
use resin_lir::Module;
use resin_lir::Verified;
use resin_types::prelude::*;
use types::Types;

mod entry;
mod foreign;
mod formatting;
mod function;
mod gpu;
mod ops;
mod projection;
mod types;
mod value;

#[derive(Clone)]
struct Slot {
    ty: Ty,
    expr: String,
    live: Option<String>,
}

pub fn generate(checked: Verified<'_>, entry: &str) -> Result<crate::c::CModule, Error> {
    let module = checked.module();
    let analysis = checked.analysis();
    let types = Types::new(module, &analysis.types);
    let entry = entry_function(&types, entry)?;
    let functions = analysis
        .functions
        .iter()
        .enumerate()
        .map(|(i, flow)| function::lower(&types, i, flow))
        .collect::<Result<_, _>>()?;
    Ok(crate::c::CModule {
        includes: includes(module),
        assertions: host_assertions(),
        declarations: types.declarations(),
        local_includes: module
            .shaders
            .iter()
            .filter(|(_, shader)| shader.embedded)
            .map(|(&function, _)| crate::shader_header(function))
            .collect(),
        prototypes: (0..module.functions.len())
            .map(|i| function::signature(&types, i))
            .collect(),
        lifecycle: types.lifecycle(),
        functions,
        entry,
    })
}

fn includes(module: &Module) -> Vec<String> {
    let mut headers: Vec<String> = [
        "resin_runtime.h",
        "stddef.h",
        "stdio.h",
        "stdlib.h",
        "math.h",
        "float.h",
        "string.h",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let foreign: std::collections::BTreeSet<_> = module
        .functions
        .iter()
        .filter_map(|f| f.foreign.as_ref().map(|f| f.header.to_string()))
        .collect();
    headers.extend(foreign);
    headers
}

fn host_assertions() -> Vec<(String, String)> {
    vec![(
        "sizeof(void *) == 8 && sizeof(size_t) == 8".into(),
        "Resin currently requires a 64-bit host".into(),
    )]
}

fn entry_function(types: &Types<'_>, name: &str) -> Result<crate::c::CFunction, Error> {
    let mut body = "  (void)r_argc; (void)r_argv;\n  atexit(resin_cleanup);\n".to_string();
    body.push_str(&entry::emit(types, name)?);
    Ok(crate::c::CFunction {
        signature: "int main(int r_argc, char **r_argv)".into(),
        body: crate::c::CBody::Inline(body),
    })
}
