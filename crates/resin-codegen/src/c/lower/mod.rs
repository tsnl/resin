//! Verified LIR → C source tree. Runtime and ABI choices are made here.
use crate::Error;
use resin_lir::{Module, Ty};
use resin_lir_verifier::Verified;
use types::Types;

mod entry;
mod foreign;
mod formatting;
mod function;
mod ops;
mod types;
mod value;

#[derive(Clone)]
struct Slot {
    ty: Ty,
    expr: String,
    live: Option<String>,
}

use crate::Shader;

pub fn generate(
    checked: Verified<'_>,
    entry: &str,
    shaders: &[Shader],
) -> Result<crate::CModule, Error> {
    let module = checked.module();
    let analysis = checked.analysis();
    let types = Types::new(module, &analysis.types, shaders);
    let entry = entry_function(&types, entry)?;
    let data = shader_data(shaders)?;
    let functions = analysis
        .functions
        .iter()
        .enumerate()
        .map(|(i, flow)| function::lower(&types, i, flow))
        .collect::<Result<_, _>>()?;
    Ok(crate::CModule {
        includes: includes(module),
        assertions: host_assertions(),
        declarations: types.declarations(),
        data,
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

fn shader_data(shaders: &[Shader]) -> Result<Vec<Vec<u32>>, Error> {
    shaders
        .iter()
        .map(|shader| {
            if shader.words.len() < 5 || shader.words[0] != 0x07230203 {
                return Err(Error("invalid embedded SPIR-V".into()));
            }
            Ok(shader.words.clone())
        })
        .collect()
}

fn entry_function(types: &Types<'_>, name: &str) -> Result<crate::CFunction, Error> {
    let mut body = "  (void)r_argc; (void)r_argv;\n  atexit(resin_cleanup);\n".to_string();
    body.push_str(&entry::emit(types, name)?);
    Ok(crate::CFunction {
        signature: "int main(int r_argc, char **r_argv)".into(),
        body: crate::CBody::Inline(body),
    })
}
