use crate::{
    backend::Error,
    ir::{Module, Ty},
};

pub(super) fn emit(module: &Module, entry: &str) -> Result<String, Error> {
    let id = module.entries.get(entry)
        .ok_or_else(|| Error(format!("entry function `{entry}` is not exported; add `export {{ {entry} }};` to the entry file")))?;
    let function = &module.functions[id.index()];
    if function.foreign.is_some()
        || function.locals[function.param.index()].ty != Ty::Unit
        || !matches!(function.result, Ty::Unit | Ty::Int32)
    {
        return Err(Error(format!(
            "entry function `{entry}` must be a Resin function of type () -> int or () -> ()"
        )));
    }
    Ok(if function.result == Ty::Unit {
        format!("  r_fn{}(0);\n  return 0;\n", id.index())
    } else {
        format!("  return r_fn{}(0);\n", id.index())
    })
}
