use crate::Error;
use resin_types::prelude::*;

mod entry;
mod function;
mod ops;
mod reachable;
mod symbols;
mod types;

use types::Types;

pub fn generate(
    checked: resin_lir::Verified<'_>,
    entry: FunctionId,
    stage: Stage,
) -> Result<crate::glsl::GlslModule, Error> {
    let module = checked.module();
    let analysis = checked.analysis();
    let reachable = reachable::functions(module, entry.index())?;
    let function = &module.functions[entry.index()];
    let mut types = Types::new(module, &analysis.types);
    for &index in &reachable {
        register_function_types(&mut types, index, &analysis.functions[index])?;
    }
    let wrapper = shader_wrapper(&types, function, entry.index(), stage)?;
    let mut functions = Vec::new();
    for index in reachable {
        functions.push(function::lower(
            &mut types,
            &module.functions[index],
            &analysis.functions[index],
            &format!("r_fn{index}"),
            index,
        )?);
    }
    Ok(crate::glsl::GlslModule {
        extensions: types.extensions().into(),
        declarations: types.declarations(),
        globals: "bool r_failed = false;\n".into(),
        functions,
        entry: wrapper,
    })
}

#[derive(Clone, PartialEq)]
struct Slot {
    ty: Ty,
    expr: String,
    local: bool,
}

impl Slot {
    fn symbolic(&self) -> bool {
        self.local || matches!(self.ty, Ty::Function { .. })
    }
    fn local_result(instr: &resin_lir::Instr, args: &[Self]) -> bool {
        matches!(instr, resin_lir::Instr::LocalAddress { .. })
            || matches!(
                instr,
                resin_lir::Instr::AccessStatic { .. } | resin_lir::Instr::AccessDynamic
            ) && args[0].local
    }
}

fn register_function_types(
    types: &mut Types<'_>,
    index: usize,
    flow: &resin_lir::FunctionTypes,
) -> Result<(), Error> {
    let function = &types.module.functions[index];
    for ty in function
        .locals
        .iter()
        .map(|local| &local.ty)
        .chain([&function.result])
    {
        types
            .register(ty)
            .map_err(|e| Error::at(types.module, index, None, e))?;
    }
    for ty in flow
        .inputs
        .iter()
        .flatten()
        .chain(flow.results.iter().flatten().flatten())
    {
        register_operand(types, ty)?;
    }
    Ok(())
}

fn register_operand(types: &mut Types<'_>, ty: &Ty) -> Result<(), Error> {
    match ty {
        Ty::Function { .. } => Ok(()), // Direct functions have symbolic slots.
        Ty::Pointer { pointee } => types.register(pointee),
        _ => types.register(ty),
    }
}

fn shader_wrapper(
    types: &Types<'_>,
    function: &resin_lir::Function,
    index: usize,
    stage: Stage,
) -> Result<String, Error> {
    let typer = TyperContext::from_definitions(types.module.types.clone());
    let interface = resin_types::shader::validate(
        &typer,
        &function.locals[0].ty,
        &function.result,
        function.foreign.is_some(),
        stage.name(),
    )
    .map_err(Error)?;
    Ok(
        entry::emit(types, &function.locals[0].ty, &function.result, &interface)
            .replace("r_entry", &format!("r_fn{index}")),
    )
}
