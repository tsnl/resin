//! IR → S-expression formatting via `sexpfmt`.

use std::{collections::HashSet, sync::Arc};

use ::sexpfmt::{PrinterConfig, SExp, SExpBookendStyle, sexp_to_string};

use super::{Function, Instr, Module, Terminator, Ty, Value};

pub fn format_module(module: &Module) -> String {
    let sexp = sexp_module(module);
    let config = PrinterConfig {
        indent_width: 2,
        margin_width: 80,
    };
    sexp_to_string(&sexp, &config)
}

struct Names {
    types: Vec<Arc<str>>,
    globals: Vec<Arc<str>>,
    functions: Vec<Arc<str>>,
}

struct FunctionNames {
    locals: Vec<Arc<str>>,
    nonlocals: Vec<Arc<str>>,
    blocks: Vec<Arc<str>>,
}

impl Names {
    fn new(module: &Module) -> Self {
        Self {
            types: uniquify(
                module
                    .types
                    .iter()
                    .map(|def| Some(def.name.clone()))
                    .collect(),
                "type",
            ),
            globals: uniquify(
                module
                    .globals
                    .iter()
                    .map(|global| Some(global.name.clone()))
                    .collect(),
                "g",
            ),
            functions: uniquify(
                module
                    .functions
                    .iter()
                    .map(|function| function.name.clone())
                    .collect(),
                "fn",
            ),
        }
    }

    fn function(&self, module: &Module, index: usize) -> FunctionNames {
        let function = &module.functions[index];
        FunctionNames {
            locals: uniquify(
                function
                    .locals
                    .iter()
                    .map(|local| local.name.clone())
                    .collect(),
                "l",
            ),
            nonlocals: uniquify(
                function
                    .nonlocals
                    .iter()
                    .map(|nonlocal| nonlocal.name.clone())
                    .collect(),
                "n",
            ),
            blocks: uniquify(
                function
                    .blocks
                    .iter()
                    .map(|block| block.name.clone())
                    .collect(),
                "b",
            ),
        }
    }
}

fn uniquify(preferred: Vec<Option<Arc<str>>>, fallback_prefix: &str) -> Vec<Arc<str>> {
    let mut used = HashSet::new();
    let mut names = Vec::with_capacity(preferred.len());
    for (index, name) in preferred.into_iter().enumerate() {
        let mut candidate: Arc<str> =
            name.unwrap_or_else(|| format!("{fallback_prefix}.{index}").into());
        if used.contains(&candidate) {
            let base = candidate.clone();
            let mut suffix = 1;
            loop {
                let next: Arc<str> = format!("{base}.{suffix}").into();
                if !used.contains(&next) {
                    candidate = next;
                    break;
                }
                suffix += 1;
            }
        }
        used.insert(candidate.clone());
        names.push(candidate);
    }
    names
}

fn sexp_module(module: &Module) -> SExp {
    let names = Names::new(module);
    let mut items = Vec::new();
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
    for (index, global) in module.globals.iter().enumerate() {
        items.push(list(
            "global",
            vec![
                symbol(names.globals[index].as_ref()),
                sexp_ty(&names, &global.ty),
            ],
        ));
    }
    for (index, function) in module.functions.iter().enumerate() {
        items.push(sexp_function(module, &names, index, function));
    }
    list("module", items)
}

fn sexp_function(module: &Module, names: &Names, index: usize, function: &Function) -> SExp {
    let fn_names = names.function(module, index);
    let mut items = vec![
        symbol(names.functions[index].as_ref()),
        symbol(fn_names.locals[function.param.index()].as_ref()),
        sexp_ty(names, &function.result),
    ];
    for (i, nonlocal) in function.nonlocals.iter().enumerate() {
        items.push(list(
            "nonlocal",
            vec![
                symbol(fn_names.nonlocals[i].as_ref()),
                sexp_ty(names, &nonlocal.ty),
            ],
        ));
    }
    for (i, local) in function.locals.iter().enumerate() {
        if function.param.index() == i {
            continue;
        }
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
        Instr::Push { value } => list("push", vec![sexp_value(names, value)]),
        Instr::LocalAddress { local } => list(
            "local-addr",
            vec![symbol(fn_names.locals[local.index()].as_ref())],
        ),
        Instr::GlobalAddress { global } => list(
            "global-addr",
            vec![symbol(names.globals[global.index()].as_ref())],
        ),
        Instr::NonLocalAddress { nonlocal } => list(
            "nonlocal-addr",
            vec![symbol(fn_names.nonlocals[nonlocal.index()].as_ref())],
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
        Instr::MakeClosure { function, captures } => list(
            "make-closure",
            vec![
                symbol(names.functions[function.index()].as_ref()),
                symbol(captures.to_string()),
            ],
        ),
        Instr::Call => symbol("call"),
        Instr::CurrentClosure => symbol("current-closure"),
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
        Value::Closure { value } => list(
            "closure",
            vec![symbol(names.functions[value.function.index()].as_ref())],
        ),
    }
}

fn sexp_ty(names: &Names, ty: &Ty) -> SExp {
    match ty {
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
