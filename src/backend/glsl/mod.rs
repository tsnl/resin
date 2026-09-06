use crate::{
    backend::Error,
    ir::{self, Module, Ty},
};

mod entry;
mod function;
mod types;

use types::Types;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Compute,
    Vertex,
    Fragment,
}

impl std::str::FromStr for Stage {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        match name {
            "compute" => Ok(Self::Compute),
            "vertex" => Ok(Self::Vertex),
            "fragment" => Ok(Self::Fragment),
            _ => Err(Error("stage must be compute, vertex, or fragment".into())),
        }
    }
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
    pub fn entry(self) -> &'static str {
        match self {
            Self::Compute => "kernel",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}

pub fn emit(module: &Module, entry: &str, stage: Stage) -> Result<String, Error> {
    let Some(ir::Entry::Function(function)) = module.entries.get(entry) else {
        return Err(Error(format!(
            "expected a visible shader function named {entry:?}"
        )));
    };
    emit_function(module, *function, stage)
}

pub fn emit_function(
    module: &Module,
    entry: ir::FunctionId,
    stage: Stage,
) -> Result<String, Error> {
    let analysis = ir::verify::analyze(module)?;
    let mut reachable = Vec::new();
    visit(
        module,
        entry.index(),
        &mut vec![false; module.functions.len()],
        &mut reachable,
    )?;
    let function = &module.functions[entry.index()];
    let mut types = Types::new(module);
    for &index in &reachable {
        let function = &module.functions[index];
        for local in &function.locals {
            types.register(&local.ty)?;
        }
        types.register(&function.result)?;
        for ty in analysis[index].inputs.iter().flatten() {
            if matches!(ty, Ty::Function { .. }) {
                return Err(Error(
                    "shader profile cannot carry addresses or functions across block edges".into(),
                ));
            }
            if let Ty::Pointer { pointee } = ty {
                types.register(pointee)?;
            } else {
                types.register(ty)?;
            }
        }
        for ty in analysis[index].results.iter().flatten().flatten() {
            match ty {
                Ty::Pointer { pointee } => types.register(pointee)?,
                Ty::Function { .. } => {}
                _ => types.register(ty)?,
            }
        }
    }
    let wrapper = entry::emit(
        &types,
        &function.locals[function.param.index()].ty,
        &function.result,
        stage,
    )?
    .replace("r_entry", &format!("r_fn{}", entry.index()));
    let mut functions = String::new();
    for index in reachable {
        functions.push_str(&function::emit(
            &mut types,
            &module.functions[index],
            &analysis[index],
            &format!("r_fn{index}"),
        )?);
    }
    let mut out = "#version 460\n#extension GL_EXT_buffer_reference : require\n#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n".to_string();
    out.push_str(&types.declarations());
    out.push_str(&functions);
    out.push_str(&wrapper);
    Ok(out)
}

fn visit(
    module: &Module,
    index: usize,
    active: &mut [bool],
    result: &mut Vec<usize>,
) -> Result<(), Error> {
    let function = module
        .functions
        .get(index)
        .ok_or_else(|| Error("invalid shader function".into()))?;
    if result.contains(&index) {
        return Ok(());
    }
    let name = function.name.as_deref().unwrap_or("<unnamed>");
    if active[index] {
        return Err(Error(format!("recursive shader call graph at {name}")));
    }
    if function.foreign.is_some() {
        return Err(Error(format!("shader cannot call foreign function {name}")));
    }
    active[index] = true;
    for instr in function.blocks.iter().flat_map(|block| &block.instrs) {
        match instr {
            ir::Instr::Function { function } => visit(module, function.index(), active, result)?,
            ir::Instr::CallBuiltin { name, .. } if name.as_ref() == "print" => {
                return Err(Error("print is only supported in host programs".into()));
            }
            _ => {}
        }
    }
    active[index] = false;
    result.push(index);
    Ok(())
}
