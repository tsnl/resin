use std::fmt::Write;

use crate::ir::{self, Instr, Module, Ty};

use super::Error;
use types::Types;

mod function;
mod ops;
mod types;
mod value;

#[derive(Clone)]
struct Slot {
    ty: Ty,
    expr: String,
}

/// Emit a C11 executable. Function zero initializes the module; optional main is () -> int or ().
pub fn emit(module: &Module) -> Result<String, Error> {
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
        || !init.nonlocals.is_empty()
    {
        return Err(Error(
            "module initializer must be an uncaptured () -> () function".into(),
        ));
    }
    let mut types = Types::new(module);
    for (index, _) in module.types.iter().enumerate() {
        types.intern(&Ty::Defined {
            definition: ir::TypeId::from_index(index),
        });
    }
    for global in &module.globals {
        types.intern(&global.ty);
    }
    for (function, flow) in module.functions.iter().zip(&analysis) {
        types.intern(&function.ty().unwrap());
        for local in &function.locals {
            types.intern(&local.ty);
        }
        for capture in &function.nonlocals {
            types.intern(&capture.ty);
        }
        for ty in flow
            .inputs
            .iter()
            .flatten()
            .chain(flow.results.iter().flatten().flatten())
        {
            types.intern(ty);
        }
        for block in &function.blocks {
            for instr in &block.instrs {
                if let Instr::Push { value } = instr {
                    types.value(value);
                }
            }
        }
    }
    let mut out = include_str!("runtime.h").to_string();
    out.push_str(&types.declarations());
    for (index, global) in module.globals.iter().enumerate() {
        writeln!(out, "static {} r_g{index};", types.name(&global.ty)).unwrap();
    }
    for (index, function) in module.functions.iter().enumerate() {
        if !function.nonlocals.is_empty() {
            writeln!(out, "typedef struct {{").unwrap();
            for (i, capture) in function.nonlocals.iter().enumerate() {
                writeln!(out, "  {} c{i};", types.name(&capture.ty)).unwrap();
            }
            writeln!(out, "}} r_env{index};").unwrap();
        }
        writeln!(
            out,
            "static {} r_fn{index}(void *r_env, {} r_arg);",
            types.name(&function.result),
            types.name(&function.locals[function.param.index()].ty)
        )
        .unwrap();
    }
    for (index, flow) in analysis.iter().enumerate() {
        out.push_str(&function::emit(&types, index, flow)?);
    }
    out.push_str("int main(void) {\n  atexit(r_cleanup);\n  r_fn0(NULL, 0);\n");
    for index in 0..module.globals.len() {
        writeln!(out, "  (void)&r_g{index};").unwrap();
    }
    let entries: Vec<_> = module
        .globals
        .iter()
        .enumerate()
        .filter(|(_, global)| global.name.as_ref() == "main")
        .collect();
    match entries.as_slice() {
        [] => out.push_str("  return 0;\n"),
        [(index, global)] => {
            let Ty::Function { param, result } = &global.ty else {
                return Err(Error("main must be a function".into()));
            };
            if param.as_ref() != &Ty::Unit || !matches!(result.as_ref(), Ty::Unit | Ty::Int32) {
                return Err(Error("main must have type () -> int or () -> ()".into()));
            }
            writeln!(
                out,
                "  if (!r_g{index}.call) r_fail(\"main is not initialized\");"
            )
            .unwrap();
            if result.as_ref() == &Ty::Unit {
                writeln!(out, "  r_g{index}.call(r_g{index}.env, 0);\n  return 0;").unwrap();
            } else {
                writeln!(out, "  return r_g{index}.call(r_g{index}.env, 0);").unwrap();
            }
        }
        _ => return Err(Error("multiple globals named main".into())),
    }
    out.push_str("}\n");
    Ok(out)
}
