use std::fmt::Write;

use crate::ir::{self, Module, Ty};

use super::Error;
use types::Types;

mod entry;
mod foreign;
mod function;
mod ops;
mod print;
mod types;
mod value;

#[derive(Clone)]
struct Slot {
    ty: Ty,
    expr: String,
}

pub struct Shader {
    pub function: ir::FunctionId,
    pub stage: super::glsl::Stage,
    pub words: Vec<u32>,
}

/// Emit a C11 executable. Function zero initializes the module; optional main is () -> int or ().
pub fn emit(module: &Module) -> Result<String, Error> {
    emit_with_shaders(module, &[])
}

pub fn emit_with_shaders(module: &Module, shaders: &[Shader]) -> Result<String, Error> {
    let analysis = ir::verify::analyze(module)?;
    let init = module
        .functions
        .first()
        .ok_or_else(|| Error("missing module initializer".into()))?;
    if init.ty()
        != Some(Ty::Function {
            param: Box::new(Ty::Unit),
            result: Box::new(Ty::Unit),
        })
        || init.foreign.is_some()
    {
        return Err(Error(
            "module initializer must be a () -> () function".into(),
        ));
    }
    let types = Types::collect(module, shaders, &analysis);
    let mut out =
        "#include <resin_runtime.h>\n#include <stdlib.h>\n#include <math.h>\n".to_string();
    out.push_str("_Static_assert(sizeof(void *) == 8 && sizeof(size_t) == 8, \"Resin currently requires a 64-bit host\");\n");
    let headers: std::collections::BTreeSet<_> = module
        .functions
        .iter()
        .filter_map(|function| function.foreign.as_ref().map(|foreign| &foreign.header))
        .collect();
    for header in headers {
        writeln!(out, "#include <{header}>").unwrap();
    }
    out.push_str(&types.declarations());
    for (index, shader) in shaders.iter().enumerate() {
        if shader.words.len() < 5 || shader.words[0] != 0x07230203 {
            return Err(Error("invalid embedded SPIR-V".into()));
        }
        writeln!(out, "static uint32_t r_spv{index}[] = {{").unwrap();
        for word in &shader.words {
            writeln!(out, "  0x{word:08x},").unwrap();
        }
        out.push_str("};\n");
    }
    for (index, global) in module.globals.iter().enumerate() {
        writeln!(out, "static {} r_g{index};", types.name(&global.ty)).unwrap();
    }
    for (index, function) in module.functions.iter().enumerate() {
        writeln!(
            out,
            "{} r_fn{index}({} r_arg);",
            types.name(&function.result),
            types.name(&function.locals[function.param.index()].ty)
        )
        .unwrap();
    }
    for (index, flow) in analysis.iter().enumerate() {
        out.push_str(&function::emit(&types, index, flow)?);
    }
    out.push_str("int main(void) {\n  atexit(resin_cleanup);\n  r_fn0(0);\n");
    for index in 0..module.globals.len() {
        writeln!(out, "  (void)&r_g{index};").unwrap();
    }
    out.push_str(&entry::emit(&types)?);
    out.push_str("}\n");
    Ok(out)
}
