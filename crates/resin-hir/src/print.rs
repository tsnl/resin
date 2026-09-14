//! Inspect the resolved tree without the AST, scopes, or inference state.
use crate::{Arguments, Function, MatchArm, Module, Parameter, Statement, Term, TermKind};
use crate::{Case, Type, TypeDefinition};
use resin_types::TypeId;
use std::collections::BTreeMap;
use std::sync::Arc;

use sexpfmt::{PrinterConfig, SExp, SExpBookendStyle, sexp_to_string};

pub(super) fn format_type(ty: &Type, definitions: &[TypeDefinition]) -> String {
    TypeNames {
        definitions: definitions
            .iter()
            .map(|definition| definition.name.clone())
            .collect(),
        parameters: definitions
            .iter()
            .flat_map(|definition| &definition.type_params)
            .map(|parameter| (parameter.id, parameter.name.val.clone()))
            .collect(),
    }
    .format(ty)
}

pub fn format_module(module: &Module) -> String {
    let printer = Printer {
        types: TypeNames {
            definitions: module
                .types
                .iter()
                .map(|definition| definition.name.clone())
                .collect(),
            parameters: module
                .functions
                .iter()
                .flat_map(|function| &function.signature.type_params)
                .chain(
                    module
                        .types
                        .iter()
                        .flat_map(|definition| &definition.type_params),
                )
                .map(|parameter| (parameter.id, parameter.name.val.clone()))
                .collect(),
        },
    };
    sexp_to_string(
        &printer.module(module),
        &PrinterConfig {
            indent_width: 2,
            margin_width: 100,
        },
    )
}

struct Printer {
    types: TypeNames,
}

impl Printer {
    fn module(&self, module: &Module) -> SExp {
        let types = module
            .types
            .iter()
            .enumerate()
            .map(|(i, def)| self.definition(i, def));
        let entries = module
            .entries
            .iter()
            .map(|(name, id)| list("export", vec![atom(name), function_id(id.index())]));
        let shaders = module.shaders.iter().map(|(id, entry)| {
            list(
                "shader",
                vec![
                    function_id(id.index()),
                    atom(&entry.stage),
                    atom(entry.embedded),
                ],
            )
        });
        let functions = module
            .functions
            .iter()
            .enumerate()
            .map(|(i, f)| self.function(i, f));
        list(
            "hir",
            types
                .chain(entries)
                .chain(shaders)
                .chain(functions)
                .collect(),
        )
    }

    fn definition(&self, index: usize, definition: &TypeDefinition) -> SExp {
        let mut fields = vec![
            atom(format!("type{index}")),
            self.ty(&Type::Defined {
                arguments: definition
                    .type_params
                    .iter()
                    .map(|parameter| Type::Parameter {
                        parameter: parameter.id,
                    })
                    .collect(),
                definition: TypeId::from_index(index),
            }),
        ];
        fields.push(self.ty(&definition.body));
        fields.extend(definition.methods.iter().map(|(name, function)| {
            list("method", vec![atom(name), function_id(function.index())])
        }));
        if let Some(drop) = definition.drop {
            fields.push(list("drop", vec![function_id(drop.index())]));
        }
        list("type", fields)
    }

    fn function(&self, index: usize, function: &Function) -> SExp {
        let mut fields = vec![
            function_id(index),
            atom(&function.name),
            self.ty(&function.signature.result.ty),
        ];
        fields.extend(function.signature.type_params.iter().map(|parameter| {
            list(
                "type-param",
                vec![atom(parameter.id.index()), atom(&parameter.name.val)],
            )
        }));
        fields.extend(function.signature.params.iter().map(|p| self.parameter(p)));
        if let Some(foreign) = &function.foreign_header {
            fields.push(list("extern", vec![quoted(foreign)]));
        }
        if let Some(body) = &function.body {
            fields.push(self.term(body));
        }
        list("def", fields)
    }

    fn parameter(&self, parameter: &Parameter) -> SExp {
        let binding = parameter.binding.map_or_else(|| atom("_"), binding);
        list(
            "param",
            vec![
                binding,
                atom(&parameter.name.val),
                self.ty(&parameter.annotation.ty),
            ],
        )
    }

    fn ty(&self, ty: &Type) -> SExp {
        quoted(self.types.format(ty))
    }

    fn term(&self, term: &Term) -> SExp {
        list("typed", vec![self.ty(&term.ty), self.kind(&term.kind)])
    }

    fn kind(&self, kind: &TermKind) -> SExp {
        match kind {
            TermKind::Constant { value } => list("constant", vec![quoted(format!("{value:?}"))]),
            TermKind::Numeric { text } => list("numeric", vec![quoted(text)]),
            TermKind::Layout { of, size } => {
                list(if *size { "sizeof" } else { "alignof" }, vec![self.ty(of)])
            }
            TermKind::Local { binding: id, name } => {
                list("local", vec![binding(*id), atom(&name.val)])
            }
            TermKind::Function {
                function,
                type_args,
            } => list(
                "apply",
                std::iter::once(function_id(function.index()))
                    .chain(type_args.iter().map(|ty| self.ty(ty)))
                    .collect(),
            ),
            TermKind::DependentMethod { lookup } => {
                list("dependent-method", vec![quoted(self.types.method(lookup))])
            }
            TermKind::DependentMethodCall {
                lookup,
                receiver,
                args,
            } => {
                let mut fields = vec![quoted(self.types.method(lookup))];
                fields.extend(receiver.iter().map(|receiver| self.term(receiver)));
                fields.extend(args.iter().map(|arg| self.term(arg)));
                list("dependent-method-call", fields)
            }
            TermKind::Shader { function, stage } => {
                list("spirv", vec![function_id(function.index()), atom(stage)])
            }
            TermKind::Unwrap { value } => list("unwrap", vec![self.term(value)]),
            TermKind::Try { value } => list("try", vec![self.term(value)]),
            TermKind::Match { value, arms } => list(
                "match",
                std::iter::once(self.term(value))
                    .chain(arms.iter().map(|a| self.arm(a)))
                    .collect(),
            ),
            TermKind::If { cond, then, els } => {
                list("if", vec![self.term(cond), self.term(then), self.term(els)])
            }
            TermKind::While { cond, body } => list("while", vec![self.term(cond), self.term(body)]),
            TermKind::Block { stmts, tail } => list(
                "block",
                stmts
                    .iter()
                    .map(|s| self.statement(s))
                    .chain(std::iter::once(self.term(tail)))
                    .collect(),
            ),
            TermKind::Record { fields } => list(
                "record",
                fields
                    .iter()
                    .map(|(name, t)| list("field", vec![atom(&name.val), self.term(t)]))
                    .collect(),
            ),
            TermKind::Array { elems } => {
                list("array", elems.iter().map(|t| self.term(t)).collect())
            }
            TermKind::Builtin { name, args } => {
                list(name, args.iter().map(|t| self.term(t)).collect())
            }
            TermKind::Call { func, args } => list(
                "call",
                vec![
                    self.term(func),
                    list("args", args.iter().map(|arg| self.term(arg)).collect()),
                ],
            ),
            TermKind::Intrinsic { op, args } => list(
                "intrinsic",
                vec![atom(format!("{op:?}")), self.arguments(args)],
            ),
            TermKind::Adapt { conversion, arg } => list(
                "adapt",
                vec![quoted(format!("{conversion:?}")), self.term(arg)],
            ),
            TermKind::Convert { arg } => list("convert", vec![self.term(arg)]),
            TermKind::ArcNew { value } => list("arc", vec![self.term(value)]),
            TermKind::GpuNew { allocator, args } => list(
                "gpu-new",
                vec![function_id(allocator.index()), self.arguments(args)],
            ),
            TermKind::GpuAllocate { allocator, args } => list(
                "gpu-allocate",
                vec![function_id(allocator.index()), self.arguments(args)],
            ),
            TermKind::GpuPipelineCreate {
                factory,
                shaders,
                args,
            } => list(
                "gpu-pipeline-create",
                vec![
                    function_id(factory.index()),
                    list(
                        "shaders",
                        shaders
                            .iter()
                            .map(|shader| function_id(shader.index()))
                            .collect(),
                    ),
                    self.arguments(args),
                ],
            ),
            TermKind::GpuPipelineDispatch {
                context,
                allocator,
                record,
                args,
            } => list(
                "gpu-pipeline-dispatch",
                vec![
                    function_id(context.index()),
                    atom(
                        allocator
                            .map(|id| id.index().to_string())
                            .unwrap_or_else(|| "none".into()),
                    ),
                    function_id(record.index()),
                    self.arguments(args),
                ],
            ),
            TermKind::HostAllocate { error, args } => list(
                "host-allocate",
                vec![function_id(error.index()), self.arguments(args)],
            ),
            TermKind::WeakEmpty { ty } => list("weak-empty", vec![self.ty(ty)]),
            TermKind::Result { failure, arg } => {
                list(if *failure { "err" } else { "ok" }, vec![self.term(arg)])
            }
            TermKind::Absurd { arg } => list("absurd", vec![self.term(arg)]),
            TermKind::Assign { place, value } => {
                list("assign", vec![self.term(place), self.term(value)])
            }
            TermKind::Address { place } => list("address", vec![self.term(place)]),
            TermKind::Deref { pointer } => list("deref", vec![self.term(pointer)]),
            TermKind::Field { base, name } => list("field", vec![self.term(base), atom(name)]),
        }
    }

    fn arguments(&self, args: &Arguments) -> SExp {
        let mut fields = vec![list(
            "params",
            args.params.iter().map(|ty| self.ty(ty)).collect(),
        )];
        fields.extend(args.values.iter().map(|arg| self.term(arg)));
        list("arguments", fields)
    }

    fn arm(&self, arm: &MatchArm) -> SExp {
        let tag = match &arm.tag {
            Case::Ok => atom("ok"),
            Case::Err => atom("err"),
            Case::Type { ty } => self.ty(ty),
        };
        list(
            "arm",
            vec![
                tag,
                arm.binding.map_or_else(|| atom("_"), binding),
                self.term(&arm.body),
            ],
        )
    }

    fn statement(&self, statement: &Statement) -> SExp {
        match statement {
            Statement::Define {
                binding: id,
                name,
                init,
            } => list(
                "define",
                vec![binding(*id), atom(&name.val), self.term(init)],
            ),
            Statement::Declare {
                binding: id,
                name,
                ty,
            } => list(
                "declare",
                vec![binding(*id), atom(&name.val), self.ty(&ty.ty)],
            ),
            Statement::Expr { term } => self.term(term),
        }
    }
}

fn function_id(index: usize) -> SExp {
    atom(format!("fn{index}"))
}
fn binding(index: usize) -> SExp {
    atom(format!("%{index}"))
}
fn atom(value: impl ToString) -> SExp {
    SExp::Atom(value.to_string())
}
fn quoted(value: impl AsRef<str>) -> SExp {
    atom(format!("{:?}", value.as_ref()))
}
fn list(head: &str, fields: Vec<SExp>) -> SExp {
    SExp::List(
        std::iter::once(atom(head)).chain(fields).collect(),
        SExpBookendStyle::Parentheses,
    )
}

/// Names needed to render HIR types, independent of concrete layout or inference.
pub(super) struct TypeNames {
    pub definitions: Vec<Arc<str>>,
    pub parameters: BTreeMap<crate::TypeParameterId, Arc<str>>,
}

impl TypeNames {
    pub(super) fn format(&self, ty: &Type) -> String {
        match ty {
            Type::Union { variants } => {
                if variants.is_empty() {
                    return "Never".into();
                }
                variants
                    .iter()
                    .map(|member| self.format(member))
                    .collect::<Vec<_>>()
                    .join(" | ")
            }
            Type::Result { value, error } => {
                format!("Result<{}, {}>", self.format(value), self.format(error))
            }
            Type::Type => "type".into(),
            Type::Unit => "()".into(),
            Type::None => "None".into(),
            Type::Bool => "bool".into(),
            Type::Int8 => "sbyte".into(),
            Type::Int16 => "short".into(),
            Type::Int32 => "int".into(),
            Type::Int64 => "long".into(),
            Type::UInt8 => "ubyte".into(),
            Type::UInt16 => "ushort".into(),
            Type::UInt32 => "uint".into(),
            Type::UInt64 => "ulong".into(),
            Type::Float32 => "float32".into(),
            Type::Float64 => "float64".into(),
            Type::Str => "str".into(),
            Type::Foreign { name } => name.to_string(),
            Type::Parameter { parameter } => self
                .parameters
                .get(parameter)
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("T{}", parameter.index())),
            Type::Member { base, name } => format!("{}.{}", self.format(base), name),
            Type::Method { lookup } => self.method(lookup),
            Type::FunctionParameter { function, index } => {
                format!("parameter({}, {index})", self.format(function))
            }
            Type::FunctionResult { function } => {
                format!("result({})", self.format(function))
            }
            Type::Defined {
                definition,
                arguments,
            } => {
                let name = self
                    .definitions
                    .get(definition.index())
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "?".into());
                if arguments.is_empty() {
                    name
                } else {
                    format!(
                        "{name}<{}>",
                        arguments
                            .iter()
                            .map(|argument| self.format(argument))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Type::Pointer { pointee } => format!("Ptr<{}>", self.format(pointee)),
            Type::GpuPointer { pointee } => format!("GpuPtr<{}>", self.format(pointee)),
            Type::GpuSpan { element } => format!("GpuSpan<{}>", self.format(element)),
            Type::GpuArguments => "GpuArguments".into(),
            Type::GpuComputePipeline { root, owner } => format!(
                "GpuComputePipeline<{}, {}>",
                self.format(root),
                self.format(owner)
            ),
            Type::GpuGraphicsPipeline { root, owner } => format!(
                "GpuGraphicsPipeline<{}, {}>",
                self.format(root),
                self.format(owner)
            ),
            Type::ArcPtr { pointee } => format!("ArcPtr<{}>", self.format(pointee)),
            Type::ArcSpan { element } => format!("ArcSpan<{}>", self.format(element)),
            Type::WeakSpan { element } => format!("WeakSpan<{}>", self.format(element)),
            Type::WeakPtr { pointee } => format!("WeakPtr<{}>", self.format(pointee)),
            Type::Span { element } => format!("Span<{}>", self.format(element)),
            Type::Array { element, length } => {
                format!("[{}; {length}]", self.format(element))
            }
            Type::Record { fields }
                if !fields.is_empty()
                    && fields
                        .iter()
                        .enumerate()
                        .all(|(i, field)| field.name.as_ref() == format!("_{i}")) =>
            {
                let values = fields
                    .iter()
                    .map(|field| self.format(&field.ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({values}{})", if fields.len() == 1 { "," } else { "" })
            }
            Type::Record { fields } => format!(
                "{{ {} }}",
                fields
                    .iter()
                    .map(|f| format!("{}: {}", f.name, self.format(&f.ty)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Type::Function { params, result } => {
                format!(
                    "({}) -> {}",
                    params
                        .iter()
                        .map(|ty| self.format(ty))
                        .collect::<Vec<_>>()
                        .join(", "),
                    self.format(result)
                )
            }
        }
    }

    fn method(&self, lookup: &crate::MethodLookup) -> String {
        let receiver = self.format(&lookup.receiver);
        let arguments = lookup
            .type_args
            .iter()
            .map(|ty| self.format(ty))
            .collect::<Vec<_>>()
            .join(", ");
        let access = if lookup.associated { "::" } else { "." };
        format!("method({receiver}{access}{}::<{arguments}>)", lookup.name)
    }
}
