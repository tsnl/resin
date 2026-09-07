use super::types::Types;
use crate::{backend::Error, ir::Ty};

pub(super) fn emit(types: &Types<'_>, entry: &str) -> Result<String, Error> {
    let module = types.module;
    let id = module.entries.get(entry)
        .ok_or_else(|| Error(format!("entry function `{entry}` is not exported; add `export {{ {entry} }};` to the entry file")))?;
    let function = &module.functions[id.index()];
    if function.foreign.is_some()
        || function.locals[0].ty != Ty::Unit
        || !matches!(function.result, Ty::Unit | Ty::Int32 | Ty::Result { .. })
    {
        return Err(Error(format!(
            "entry function `{entry}` must be a Resin function, take (), and return int, (), or Result of either"
        )));
    }
    if let Ty::Result { value, error } = &function.result {
        if !matches!(value.as_ref(), Ty::Unit | Ty::Int32) {
            return Err(Error(
                "Result entry points must have an int or () success type".into(),
            ));
        }
        let mut error_name = "\"invalid error tag\"".to_string();
        for definition in error.variants().unwrap() {
            let name = quoted(&module.types[definition.index()].name);
            error_name = if matches!(error.as_ref(), Ty::Defined { .. }) {
                name
            } else {
                format!(
                    "(r_result.payload.v1.tag == {}u ? {name} : {error_name})",
                    definition.tag()
                )
            };
        }
        let success = if value.as_ref() == &Ty::Unit {
            "0"
        } else {
            "r_result.payload.v0"
        };
        return Ok(format!(
            "  {} r_result = r_fn{}(0);\n  if (r_result.tag == 1u) {{ fprintf(stderr, \"unhandled error: %s\\n\", {error_name}); return 1; }}\n  return {success};\n",
            types.name(&function.result),
            id.index()
        ));
    }
    Ok(if function.result == Ty::Unit {
        format!("  r_fn{}(0);\n  return 0;\n", id.index())
    } else {
        format!("  return r_fn{}(0);\n", id.index())
    })
}

fn quoted(text: &str) -> String {
    let escaped: String = text
        .bytes()
        .map(|byte| match byte {
            b' '..=b'~' if !matches!(byte, b'"' | b'\\') => char::from(byte).to_string(),
            _ => format!("\\{byte:03o}"),
        })
        .collect();
    format!("\"{escaped}\"")
}
