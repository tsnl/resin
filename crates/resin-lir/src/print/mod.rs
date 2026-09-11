//! IR → S-expression formatting via `sexpfmt`.

use ::sexpfmt::{PrinterConfig, SExp, SExpBookendStyle, sexp_to_string};
use resin_types::prelude::*;

use crate::{BlockId, Function, Instr, Module, Terminator};

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
            if def.name().is_some() {
                "type"
            } else {
                "structural-type"
            },
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
    if function.foreign.is_none() {
        let mut printer = Blocks {
            names,
            fn_names: &fn_names,
            function,
            printed: vec![false; function.blocks.len()],
        };
        items.extend(printer.region(function.entry));
        for i in 0..function.blocks.len() {
            if !printer.printed[i] {
                items.push(list("unreachable", printer.region(BlockId::from_index(i))));
            }
        }
    }
    list("function", items)
}

fn sexp_instr(names: &Names, fn_names: &FunctionNames, instr: &Instr) -> SExp {
    match instr {
        Instr::GpuNew { allocator, element } => list(
            "gpu-new",
            vec![
                symbol(names.functions[allocator.index()].as_ref()),
                sexp_ty(names, element),
            ],
        ),
        Instr::GpuAllocate { allocator, element } => list(
            "gpu-allocate",
            vec![
                symbol(names.functions[allocator.index()].as_ref()),
                sexp_ty(names, element),
            ],
        ),
        Instr::GpuAllocateNative => symbol("gpu-allocate-native"),
        Instr::GpuSlice => symbol("gpu-slice"),
        Instr::GpuReadOnly => symbol("gpu-read-only"),
        Instr::GpuWriteOnly => symbol("gpu-write-only"),
        Instr::GpuCopyTo => symbol("gpu-copy-to"),
        Instr::GpuProject { allocator, shader } => list(
            "gpu-project",
            vec![
                symbol(names.functions[allocator.index()].as_ref()),
                symbol(names.functions[shader.index()].as_ref()),
            ],
        ),
        Instr::GpuArgumentsDispatch => symbol("gpu-dispatch"),
        Instr::GpuArgumentsDraw => symbol("gpu-draw"),
        Instr::GpuCopyImage => symbol("gpu-copy-image"),
        Instr::TransferLoad => symbol("transfer-load"),
        Instr::ForgetLocal { local } => list(
            "forget-local",
            vec![symbol(fn_names.locals[local.index()].as_ref())],
        ),
        Instr::ArcNew => symbol("arc-new"),
        Instr::ArcData => symbol("arc-data"),
        Instr::Downgrade => symbol("downgrade"),
        Instr::Upgrade => symbol("upgrade"),
        Instr::WeakEmpty { pointee } => list("weak-empty", vec![sexp_ty(names, pointee)]),
        Instr::TakeLocal { local } => list(
            "take-local",
            vec![symbol(fn_names.locals[local.index()].as_ref())],
        ),
        Instr::DropLocal { local } => list(
            "drop-local",
            vec![symbol(fn_names.locals[local.index()].as_ref())],
        ),
        Instr::SetLocal { local } => list(
            "set-local",
            vec![symbol(fn_names.locals[local.index()].as_ref())],
        ),
        Instr::MakeVariant { ty, tag } => list(
            "make-variant",
            vec![sexp_ty(names, ty), sexp_case(names, tag)],
        ),
        Instr::ExcludeNone => symbol("exclude-none"),
        Instr::IsVariant { tag } => list("is-variant", vec![sexp_case(names, tag)]),
        Instr::VariantPayload { tag } => list("variant-payload", vec![sexp_case(names, tag)]),
        Instr::Widen { ty } => list("widen", vec![sexp_ty(names, ty)]),
        Instr::Shader { function, stage } => list(
            "shader",
            vec![
                symbol(names.functions[function.index()].as_ref()),
                symbol(stage.as_ref()),
            ],
        ),
        Instr::Eliminate { result } => list("eliminate-never", vec![sexp_ty(names, result)]),
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
        Instr::Replace => symbol("replace"),
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

/// Print ownership nesting while keeping invalid trees inspectable.
struct Blocks<'a> {
    names: &'a Names,
    fn_names: &'a FunctionNames,
    function: &'a Function,
    printed: Vec<bool>,
}

impl Blocks<'_> {
    fn region(&mut self, mut id: BlockId) -> Vec<SExp> {
        let mut blocks = Vec::new();
        loop {
            let Some(block) = self.function.blocks.get(id.index()) else {
                blocks.push(list("invalid-block", vec![symbol(id.index().to_string())]));
                return blocks;
            };
            let name = symbol(self.fn_names.blocks[id.index()].as_ref());
            if std::mem::replace(&mut self.printed[id.index()], true) {
                blocks.push(list("reused-block", vec![name]));
                return blocks;
            }
            let mut items = vec![name];
            items.extend(
                block
                    .instrs
                    .iter()
                    .map(|instr| sexp_instr(self.names, self.fn_names, instr)),
            );
            let (terminator, next) = self.terminator(&block.terminator);
            items.push(terminator);
            blocks.push(list("block", items));
            let Some(next) = next else { return blocks };
            id = next;
        }
    }

    fn terminator(&mut self, terminator: &Terminator) -> (SExp, Option<BlockId>) {
        match *terminator {
            Terminator::Merge => (symbol("merge"), None),
            Terminator::LoopTest => (symbol("loop-test"), None),
            Terminator::Continue => (symbol("continue"), None),
            Terminator::Return => (symbol("return"), None),
            Terminator::If { then, els, next } => (
                list(
                    "if",
                    vec![
                        list("then", self.region(then)),
                        list("else", self.region(els)),
                    ],
                ),
                next,
            ),
            Terminator::Loop {
                condition,
                body,
                next,
            } => (
                list(
                    "loop",
                    vec![
                        list("condition", self.region(condition)),
                        list("body", self.region(body)),
                    ],
                ),
                next,
            ),
        }
    }
}

fn sexp_value(names: &Names, value: &Value) -> SExp {
    match value {
        Value::Type { ty } => list("type", vec![sexp_ty(names, ty)]),
        Value::None => symbol("None"),
        Value::Str { value } => list("str", value.iter().map(|v| symbol(v.to_string())).collect()),
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
                .map(|member| sexp_ty(names, member))
                .collect(),
        ),
        Ty::Result { value, error } => {
            list("result", vec![sexp_ty(names, value), sexp_ty(names, error)])
        }
        Ty::Type => symbol("type"),
        Ty::Unit => symbol("unit"),
        Ty::None => symbol("None"),
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
        Ty::Str => symbol("str"),
        Ty::Foreign { name } => list("foreign", vec![symbol(name.as_ref())]),
        Ty::Defined { definition } => names
            .types
            .get(definition.index())
            .map(|name| symbol(name.as_ref()))
            .unwrap_or_else(|| symbol(format!("type.{}", definition.index()))),
        Ty::Pointer { pointee } => list("ptr", vec![sexp_ty(names, pointee)]),
        Ty::GpuPointer { pointee } => list("gpu-ptr", vec![sexp_ty(names, pointee)]),
        Ty::GpuSpan { element } => list("gpu-span", vec![sexp_ty(names, element)]),
        Ty::GpuArguments => symbol("GpuArguments"),
        Ty::Arc { pointee } => list("arc", vec![sexp_ty(names, pointee)]),
        Ty::Weak { pointee } => list("weak", vec![sexp_ty(names, pointee)]),
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

fn sexp_case(names: &Names, case: &Case) -> SExp {
    match case {
        Case::Ok => symbol("ok"),
        Case::Err => symbol("err"),
        Case::Type(ty) => sexp_ty(names, ty),
    }
}
