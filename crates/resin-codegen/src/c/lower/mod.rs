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
mod ownership;
mod pipeline;
mod projection;
mod representation;
mod types;
mod value;

#[derive(Clone)]
struct Slot {
    ty: Ty,
    expr: String,
    live: Option<String>,
}

pub fn generate(
    checked: Verified<'_>,
    entry: &str,
    headers: &crate::NativeHeaders,
) -> Result<crate::c::CModule, Error> {
    let module = checked.module();
    let analysis = checked.analysis();
    let types = Types::new(module, &analysis.types);
    let entry = entry_function(&types, entry)?;
    let functions = analysis
        .functions
        .iter()
        .enumerate()
        .filter(|(i, _)| module.functions[*i].profile == resin_lir::Profile::Host)
        .map(|(i, flow)| function::lower(&types, i, flow))
        .collect::<Result<_, _>>()?;
    let (includes, native_includes) = includes(module, headers);
    Ok(crate::c::CModule {
        includes,
        assertions: host_assertions(),
        declarations: types.declarations(),
        local_includes: module
            .shaders
            .iter()
            .filter(|(_, shader)| shader.embedded)
            .map(|(&function, _)| crate::shader_header(function))
            .chain(native_includes)
            .collect(),
        prototypes: (0..module.functions.len())
            .filter(|&i| module.functions[i].profile == resin_lir::Profile::Host)
            .map(|i| function::signature(&types, i))
            .collect(),
        lifecycle: types.lifecycle(),
        functions,
        entry,
    })
}

fn includes(module: &Module, bindings: &crate::NativeHeaders) -> (Vec<String>, Vec<String>) {
    let mut system = [
        "stddef.h", "stdio.h", "stdlib.h", "math.h", "float.h", "string.h",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();
    let mut staged = Vec::new();
    let runtime = bindings
        .runtime
        .clone()
        .unwrap_or_else(|| crate::NativeInclude::System {
            spelling: "resin_runtime.h".into(),
        });
    let foreign = module.foreign_headers.iter().chain(
        module
            .functions
            .iter()
            .filter(|function| function.profile == resin_lir::Profile::Host)
            .filter_map(|function| function.foreign.as_ref().map(|foreign| &foreign.header)),
    );
    let includes = std::iter::once(runtime).chain(foreign.map(|header| {
        bindings
            .bindings
            .get(header)
            .cloned()
            .unwrap_or_else(|| crate::NativeInclude::System {
                spelling: header.spelling.clone(),
            })
    }));
    for include in includes {
        match include {
            crate::NativeInclude::System { spelling } => system.push(spelling.to_string()),
            crate::NativeInclude::Staged { path } => staged.push(path.to_string()),
        }
    }
    system.sort();
    system.dedup();
    staged.sort();
    staged.dedup();
    (system, staged)
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
