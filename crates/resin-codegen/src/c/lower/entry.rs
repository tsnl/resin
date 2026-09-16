use super::types::Types;
use crate::Error;
use resin_types::prelude::*;

pub(super) fn emit(types: &Types<'_>, entry: &str) -> Result<String, Error> {
    let module = types.module;
    let id = module.entries.get(entry)
        .ok_or_else(|| Error(format!("entry function `{entry}` is not exported; add `export {{ {entry} }};` to the entry file")))?;
    let function = &module.functions[id.index()];
    let parameters = &function.locals[..function.parameter_count];
    let process_inputs = matches!(parameters, [argc, argv, envp]
        if argc.ty == Ty::Int32 && string_array(&argv.ty) && string_array(&envp.ty));
    if function.foreign.is_some()
        || !(parameters.is_empty() || process_inputs)
        || !function
            .result
            .members()
            .iter()
            .all(|ty| matches!(ty, Ty::Unit | Ty::Int32 | Ty::Error { .. }))
    {
        return Err(Error(format!(
            "entry function `{entry}` must be a Resin function, take () or (int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>), and return int, (), or a union of those with Err<E>"
        )));
    }
    let setup = if process_inputs {
        "  char **r_arguments, **r_environment;\n  r_argc = resin_process_init(r_argc, (const char *const *)r_argv, &r_arguments, &r_environment);\n".to_owned()
    } else {
        String::new()
    };
    let argument = if process_inputs {
        "r_argc, (void *)r_arguments, (void *)r_environment"
    } else {
        ""
    };
    if matches!(function.result, Ty::Union { .. } | Ty::Error { .. }) {
        let mut out = format!(
            "{setup}  {} r_result = r_fn{}({argument});\n",
            types.name(&function.result),
            id.index()
        );
        let mut cleanup = String::new();
        types.drop_value(&function.result, "r_result", &mut cleanup);
        for member in function.result.members() {
            let value = if matches!(function.result, Ty::Union { .. }) {
                out.push_str(&format!(
                    "  if (r_result.tag == {}u) {{\n",
                    types.id(&member)
                ));
                format!("r_result.payload.v{}", types.id(&member))
            } else {
                out.push_str("  {\n");
                "r_result".into()
            };
            match member {
                Ty::Error { payload } => {
                    let descriptor = super::representation::descriptor(types, &payload);
                    out.push_str(&format!("    resin_report_error(&{descriptor}, &({value}).value);\n{cleanup}    return 1;\n"));
                }
                Ty::Unit => out.push_str(&format!("{cleanup}    return 0;\n")),
                _ => out.push_str(&format!(
                    "    int code = {value};\n{cleanup}    return code;\n"
                )),
            }
            out.push_str("  }\n");
        }
        out.push_str("  resin_fail(\"invalid entry result tag\"); return 1;\n");
        return Ok(out);
    }
    Ok(if function.result == Ty::Unit {
        format!("{setup}  r_fn{}({argument});\n  return 0;\n", id.index())
    } else {
        format!("{setup}  return r_fn{}({argument});\n", id.index())
    })
}

fn string_array(ty: &Ty) -> bool {
    matches!(ty, Ty::Pointer { pointee } if matches!(pointee.as_ref(), Ty::Pointer { pointee } if pointee.as_ref() == &Ty::UInt8))
}
