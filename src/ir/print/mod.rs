//! IR → S-expression formatting via `sexpfmt`.

use ::sexpfmt::{PrinterConfig, SExp, SExpBookendStyle, sexp_to_string};

use crate::ir::{Function, Instr, Module, Terminator, Ty, Value};

mod names;

use names::{FunctionNames, Names};

pub fn format_module(module: &Module) -> String {
    let sexp = sexp_module(module);
    let config = PrinterConfig {
        indent_width: 2,
        margin_width: 80,
    };
    sexp_to_string(&sexp, &config)
}

fn sexp_module(module: &Module) -> SExp {
    let names = Names::new(module);
    let mut items = Vec::new();
    for (name, entry) in &module.entries {
        let target = names.functions.get(entry.index());
        items.push(list(
            "entry",
            vec![
                symbol(name.as_ref()),
                symbol(target.map_or("invalid", AsRef::as_ref)),
            ],
        ));
    }
    for (index, def) in module.types.iter().enumerate() {
        items.push(list(
            "type",
            vec![
                symbol(names.types[index].as_ref()),
                def.body()
                    .map_or_else(|| symbol("incomplete"), |body| sexp_ty(&names, body)),
            ],
        ));
    }
    for (function, entry) in &module.shaders {
        items.push(list(
            "shader-entry",
            vec![
                symbol(names.functions[function.index()].as_ref()),
                symbol(entry.stage.as_ref()),
                symbol(if entry.embedded {
                    "embedded"
                } else {
                    "candidate"
                }),
            ],
        ));
    }
    for (index, function) in module.functions.iter().enumerate() {
        items.push(sexp_function(&names, index, function));
    }
    list("module", items)
}

fn sexp_function(names: &Names, index: usize, function: &Function) -> SExp {
    let fn_names = FunctionNames::new(function);
    let mut items = vec![
        symbol(names.functions[index].as_ref()),
        symbol(fn_names.locals[0].as_ref()),
        sexp_ty(names, &function.result),
    ];
    if let Some(foreign) = &function.foreign {
        items.push(list("extern", vec![symbol(foreign.header.as_ref())]));
    }
    for (i, local) in function.locals.iter().enumerate().skip(1) {
        items.push(list(
            "local",
            vec![
                symbol(fn_names.locals[i].as_ref()),
                sexp_ty(names, &local.ty),
            ],
        ));
    }
    for (i, block) in function.blocks.iter().enumerate() {
        let mut block_items = vec![symbol(fn_names.blocks[i].as_ref())];
        block_items.extend(
            block
                .instrs
                .iter()
                .map(|instr| sexp_instr(names, &fn_names, instr)),
        );
        block_items.push(sexp_terminator(&fn_names, &block.terminator));
        items.push(list("block", block_items));
    }
    list("function", items)
}

fn sexp_instr(names: &Names, fn_names: &FunctionNames, instr: &Instr) -> SExp {
    match instr {
        Instr::SetLocal { local } => list(
            "set-local",
            vec![symbol(fn_names.locals[local.index()].as_ref())],
        ),
        Instr::MakeVariant { ty, tag } => list(
            "make-variant",
            vec![sexp_ty(names, ty), symbol(tag.to_string())],
        ),
        Instr::VariantTag => symbol("variant-tag"),
        Instr::VariantPayload { tag } => list("variant-payload", vec![symbol(tag.to_string())]),
        Instr::Widen { ty } => list("widen", vec![sexp_ty(names, ty)]),
        Instr::Shader { function, stage } => list(
            "shader",
            vec![
                symbol(names.functions[function.index()].as_ref()),
                symbol(stage.as_ref()),
            ],
        ),
        Instr::NumericCast { ty } => list("numeric-cast", vec![sexp_ty(names, ty)]),
        Instr::PointerCast { ty } => list("pointer-cast", vec![sexp_ty(names, ty)]),
        Instr::Push { value } => list("push", vec![sexp_value(names, value)]),
        Instr::LocalAddress { local } => list(
            "local-addr",
            vec![symbol(fn_names.locals[local.index()].as_ref())],
        ),
        Instr::AccessStatic { index } => list("access-static", vec![symbol(index.to_string())]),
        Instr::AccessDynamic => symbol("access-dynamic"),
        Instr::Load => symbol("load"),
        Instr::Store => symbol("store"),
        Instr::Discard => symbol("discard"),
        Instr::Ascribe { ty } => list("ascribe", vec![sexp_ty(names, ty)]),
        Instr::MakeRecord { fields } => list(
            "make-record",
            fields.iter().map(|field| symbol(field.as_ref())).collect(),
        ),
        Instr::MakeArray { elements, element } => list(
            "make-array",
            vec![symbol(elements.to_string()), sexp_ty(names, element)],
        ),
        Instr::Function { function } => list(
            "function-ref",
            vec![symbol(names.functions[function.index()].as_ref())],
        ),
        Instr::Call => symbol("call"),
        Instr::CallBuiltin {
            name,
            params,
            result,
        } => list(
            "call-builtin",
            vec![
                symbol(name.as_ref()),
                group(params.iter().map(|ty| sexp_ty(names, ty)).collect()),
                sexp_ty(names, result),
            ],
        ),
    }
}

fn sexp_terminator(fn_names: &FunctionNames, terminator: &Terminator) -> SExp {
    match terminator {
        Terminator::Break { target } => list(
            "break",
            vec![symbol(fn_names.blocks[target.index()].as_ref())],
        ),
        Terminator::Branch { then, els } => list(
            "branch",
            vec![
                symbol(fn_names.blocks[then.index()].as_ref()),
                symbol(fn_names.blocks[els.index()].as_ref()),
            ],
        ),
        Terminator::Return => symbol("return"),
    }
}

fn sexp_value(names: &Names, value: &Value) -> SExp {
    match value {
        Value::Type { ty } => list("type", vec![sexp_ty(names, ty)]),
        Value::Unit => symbol("unit"),
        Value::Bool { value } => list("bool", vec![symbol(value.to_string())]),
        Value::Int8 { value } => list("sbyte", vec![symbol(value.to_string())]),
        Value::Int16 { value } => list("short", vec![symbol(value.to_string())]),
        Value::Int32 { value } => list("int", vec![symbol(value.to_string())]),
        Value::Int64 { value } => list("long", vec![symbol(value.to_string())]),
        Value::UInt8 { value } => list("ubyte", vec![symbol(value.to_string())]),
        Value::UInt16 { value } => list("ushort", vec![symbol(value.to_string())]),
        Value::UInt32 { value } => list("uint", vec![symbol(value.to_string())]),
        Value::UInt64 { value } => list("ulong", vec![symbol(value.to_string())]),
        Value::Float32 { value } => list("float32", vec![symbol(value.to_string())]),
        Value::Float64 { value } => list("float64", vec![symbol(value.to_string())]),
        Value::Array { value } => list(
            "array",
            value
                .elements
                .iter()
                .map(|element| sexp_value(names, element))
                .collect(),
        ),
        Value::Record { value } => list(
            "record",
            value
                .fields
                .iter()
                .map(|field| {
                    list(
                        "field",
                        vec![symbol(field.name.as_ref()), sexp_value(names, &field.value)],
                    )
                })
                .collect(),
        ),
        Value::StaticAddress { .. } => symbol("static-addr"),
        Value::DynamicAddress { address } => {
            list("dynamic-addr", vec![symbol(address.to_string())])
        }
    }
}

fn sexp_ty(names: &Names, ty: &Ty) -> SExp {
    match ty {
        Ty::Union { variants } => list(
            "union",
            variants
                .iter()
                .map(|definition| {
                    sexp_ty(
                        names,
                        &Ty::Defined {
                            definition: *definition,
                        },
                    )
                })
                .collect(),
        ),
        Ty::Result { value, error } => {
            list("result", vec![sexp_ty(names, value), sexp_ty(names, error)])
        }
        Ty::Type => symbol("type"),
        Ty::Unit => symbol("unit"),
        Ty::Bool => symbol("bool"),
        Ty::Int8 => symbol("sbyte"),
        Ty::Int16 => symbol("short"),
        Ty::Int32 => symbol("int"),
        Ty::Int64 => symbol("long"),
        Ty::UInt8 => symbol("ubyte"),
        Ty::UInt16 => symbol("ushort"),
        Ty::UInt32 => symbol("uint"),
        Ty::UInt64 => symbol("ulong"),
        Ty::Float32 => symbol("float32"),
        Ty::Float64 => symbol("float64"),
        Ty::Foreign { name } => list("foreign", vec![symbol(name.as_ref())]),
        Ty::Defined { definition } => names
            .types
            .get(definition.index())
            .map(|name| symbol(name.as_ref()))
            .unwrap_or_else(|| symbol(format!("type.{}", definition.index()))),
        Ty::Pointer { pointee } => list("ptr", vec![sexp_ty(names, pointee)]),
        Ty::Span { element } => list("span", vec![sexp_ty(names, element)]),
        Ty::Array { element, length } => list(
            "array",
            vec![sexp_ty(names, element), symbol(length.to_string())],
        ),
        Ty::Record { fields } => list(
            "record",
            fields
                .iter()
                .map(|field| {
                    list(
                        "field",
                        vec![symbol(field.name.as_ref()), sexp_ty(names, &field.ty)],
                    )
                })
                .collect(),
        ),
        Ty::Function { param, result } => {
            list("func", vec![sexp_ty(names, param), sexp_ty(names, result)])
        }
    }
}

fn list(head: &str, items: Vec<SExp>) -> SExp {
    let mut children = Vec::with_capacity(items.len() + 1);
    children.push(symbol(head));
    children.extend(items);
    SExp::List(children, SExpBookendStyle::Parentheses)
}

fn group(items: Vec<SExp>) -> SExp {
    if items.is_empty() {
        SExp::Null(SExpBookendStyle::Parentheses)
    } else {
        SExp::List(items, SExpBookendStyle::Parentheses)
    }
}

fn symbol(s: impl Into<String>) -> SExp {
    SExp::Atom(s.into())
}
