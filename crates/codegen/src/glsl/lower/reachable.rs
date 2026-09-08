//! Find shader callees in dependency order, rejecting recursive and host-only calls.
use crate::{
    Error,
    lir::{self, Module},
};

pub(super) fn functions(module: &Module, entry: usize) -> Result<Vec<usize>, Error> {
    let mut result = vec![];
    visit(
        module,
        entry,
        &mut vec![false; module.functions.len()],
        &mut result,
    )?;
    Ok(result)
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
        return Err(Error::at(
            module,
            index,
            None,
            Error(format!("recursive shader call graph at {name}")),
        ));
    }
    if function.foreign.is_some() {
        return Err(Error::at(
            module,
            index,
            None,
            Error(format!("shader cannot call foreign function {name}")),
        ));
    }
    active[index] = true;
    for (b, block) in function.blocks.iter().enumerate() {
        for (i, instr) in block.instrs.iter().enumerate() {
            match instr {
                lir::Instr::Function { function } => {
                    visit(module, function.index(), active, result)
                        .map_err(|error| Error::at(module, index, Some((b, i)), error))?
                }
                lir::Instr::CallBuiltin { name, .. }
                    if matches!(name.as_ref(), "print" | "fmt") =>
                {
                    return Err(Error::at(
                        module,
                        index,
                        Some((b, i)),
                        Error(format!("{name} is only supported in host programs")),
                    ));
                }
                _ => {}
            }
        }
    }
    active[index] = false;
    result.push(index);
    Ok(())
}
