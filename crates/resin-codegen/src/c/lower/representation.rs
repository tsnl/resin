//! Compact, read-only descriptions for the host's value renderer. Pointers are leaves.
use super::types::Types;
use resin_types::prelude::*;
use std::fmt::Write;

pub(super) fn descriptor(types: &Types<'_>, ty: &Ty) -> String {
    let id = types.id(ty);
    if types.representations.borrow_mut().insert(id) && text_view(types, ty).is_none() {
        match ty {
            Ty::Defined { definition } => {
                descriptor(
                    types,
                    types.module.types[definition.index()].body().unwrap(),
                );
            }
            Ty::Error { payload } => {
                descriptor(types, payload);
            }
            Ty::Array { element, .. } => {
                descriptor(types, element);
            }
            Ty::Record { fields } => {
                for field in fields {
                    descriptor(types, &field.ty);
                }
            }
            Ty::Union { .. } | Ty::Result { .. } => {
                for (_, payload) in ty.payloads().unwrap() {
                    descriptor(types, &payload);
                }
            }
            _ => {}
        }
    }
    format!("r_repr_{id}")
}

pub(super) fn declarations(types: &Types<'_>) -> String {
    let ids: Vec<_> = types.representations.borrow().iter().copied().collect();
    let mut out = String::new();
    for id in &ids {
        writeln!(out, "static const ResinReprType r_repr_{id};").unwrap();
    }
    for id in ids {
        let ty = types.table.types().nth(id).unwrap();
        let name = types.name(&ty);
        let label = resin_types::format_type(&ty, &types.module.types);
        let mut fields = vec![];
        let mut length = 0;
        let hook = text_view(types, &ty);
        let kind = if hook.is_some() {
            "TEXT"
        } else {
            match &ty {
                Ty::Unit => "UNIT",
                Ty::None => "NONE",
                Ty::Bool => "BOOL",
                Ty::Int8 => "I8",
                Ty::Int16 => "I16",
                Ty::Int32 => "I32",
                Ty::Int64 => "I64",
                Ty::UInt8 => "U8",
                Ty::UInt16 => "U16",
                Ty::UInt32 => "U32",
                Ty::UInt64 => "U64",
                Ty::Float32 => "F32",
                Ty::Float64 => "F64",
                Ty::Str => "STR",
                Ty::Pointer { .. } => "POINTER",
                Ty::Defined { definition } => {
                    let body = types.module.types[definition.index()].body().unwrap();
                    fields.push(field(
                        types,
                        "",
                        body,
                        format!("offsetof({name}, value)"),
                        0,
                    ));
                    "NAMED"
                }
                Ty::Error { payload } => {
                    fields.push(field(
                        types,
                        "",
                        payload,
                        format!("offsetof({name}, value)"),
                        0,
                    ));
                    "ERROR"
                }
                Ty::Array {
                    element,
                    length: count,
                } => {
                    length = *count;
                    fields.push(field(
                        types,
                        "",
                        element,
                        format!("offsetof({name}, items)"),
                        0,
                    ));
                    "ARRAY"
                }
                Ty::Record { fields: members } => {
                    for (index, member) in members.iter().enumerate() {
                        fields.push(field(
                            types,
                            &member.name,
                            &member.ty,
                            format!("offsetof({name}, f{index})"),
                            0,
                        ));
                    }
                    if members
                        .iter()
                        .enumerate()
                        .all(|(i, f)| f.name.as_ref() == format!("_{i}"))
                    {
                        "TUPLE"
                    } else {
                        "RECORD"
                    }
                }
                Ty::Union { .. } | Ty::Result { .. } => {
                    for (case, payload) in ty.payloads().unwrap() {
                        let tag = types.tag(&case);
                        let label = match case {
                            Case::Ok => "ok",
                            Case::Err => "err",
                            _ => "",
                        };
                        fields.push(field(
                            types,
                            label,
                            &payload,
                            format!("offsetof({name}, payload.v{tag})"),
                            tag,
                        ));
                    }
                    "UNION"
                }
                _ => "OPAQUE",
            }
        };
        let callback = if let Some(hook) = hook {
            writeln!(out, "{};", super::function::signature(types, hook.index())).unwrap();
            let result = types.name(&types.module.functions[hook.index()].result);
            writeln!(out, "static ResinPrintBytes r_repr_text_{id}(const void *data) {{ {result} view = r_fn{}(({name} *)data); return (ResinPrintBytes){{ view.f0, view.f1 }}; }}", hook.index()).unwrap();
            format!("r_repr_text_{id}")
        } else {
            "NULL".into()
        };
        let field_count = fields.len();
        let pointer = if fields.is_empty() {
            "NULL".into()
        } else {
            writeln!(
                out,
                "static const ResinReprField r_repr_fields_{id}[] = {{ {} }};",
                fields.join(", ")
            )
            .unwrap();
            format!("r_repr_fields_{id}")
        };
        writeln!(out, "static const ResinReprType r_repr_{id} = {{ RESIN_REPR_{kind}, {}, sizeof({name}), {length}, {pointer}, {field_count}, {callback} }};", quoted(&label)).unwrap();
    }
    out
}

fn field(types: &Types<'_>, name: &str, ty: &Ty, offset: String, tag: u32) -> String {
    format!(
        "{{ {}, &r_repr_{}, {offset}, {tag}u }}",
        quoted(name),
        types.id(ty)
    )
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

fn text_view(types: &Types<'_>, ty: &Ty) -> Option<FunctionId> {
    let Ty::Defined { definition } = ty else {
        return None;
    };
    types.module.text_views.get(definition).copied()
}
