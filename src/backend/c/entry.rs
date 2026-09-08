use super::types::Types;
use crate::{backend::Error, ir::Ty};

pub(super) fn emit(types: &Types<'_>, entry: &str) -> Result<String, Error> {
    let module = types.module;
    let id = module.entries.get(entry)
        .ok_or_else(|| Error(format!("entry function `{entry}` is not exported; add `export {{ {entry} }};` to the entry file")))?;
    let function = &module.functions[id.index()];
    let parameter = &function.locals[0].ty;
    let process_inputs = matches!(parameter, Ty::Record { fields } if fields.len() == 3
        && fields[0].ty == Ty::Int32 && string_array(&fields[1].ty) && string_array(&fields[2].ty));
    if function.foreign.is_some()
        || !(parameter == &Ty::Unit || process_inputs)
        || !matches!(function.result, Ty::Unit | Ty::Int32 | Ty::Result { .. })
    {
        return Err(Error(format!(
            "entry function `{entry}` must be a Resin function, take () or (int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>), and return int, (), or Result of either"
        )));
    }
    let setup = if process_inputs {
        format!(
            "  char **r_arguments, **r_environment;\n  r_argc = resin_process_init(r_argc, (const char *const *)r_argv, &r_arguments, &r_environment);\n  {} r_entry_arg = {{r_argc, (void *)r_arguments, (void *)r_environment}};\n",
            types.name(parameter)
        )
    } else {
        String::new()
    };
    let argument = if process_inputs { "r_entry_arg" } else { "0" };
    if let Ty::Result { value, error } = &function.result {
        if !matches!(value.as_ref(), Ty::Unit | Ty::Int32) {
            return Err(Error(
                "Result entry points must have an int or () success type".into(),
            ));
        }
        let mut error_name = "\"invalid error tag\"".to_string();
        for definition in error.variants().unwrap() {
            let name = quoted(module.types[definition.index()].name().unwrap());
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
        let mut cleanup = String::new();
        types.drop_value(&function.result, "r_result", &mut cleanup);
        return Ok(format!(
            "{setup}  {} r_result = r_fn{}({argument});\n  if (r_result.tag == 1u) {{ fprintf(stderr, \"unhandled error: %s\\n\", {error_name}); {cleanup} return 1; }}\n  return {success};\n",
            types.name(&function.result),
            id.index()
        ));
    }
    Ok(if function.result == Ty::Unit {
        format!("{setup}  r_fn{}({argument});\n  return 0;\n", id.index())
    } else {
        format!("{setup}  return r_fn{}({argument});\n", id.index())
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

fn string_array(ty: &Ty) -> bool {
    matches!(ty, Ty::Pointer { pointee } if matches!(pointee.as_ref(), Ty::Pointer { pointee } if pointee.as_ref() == &Ty::UInt8))
}
