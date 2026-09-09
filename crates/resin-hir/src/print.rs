//! Inspect the resolved tree without the AST, scopes, or inference state.
use crate::types::{Case, Ty, TypeDef, TypeId, print::format_type};
use crate::{Arguments, Function, MatchArm, Module, Parameter, Statement, Term, TermKind};
use sexpfmt::{PrinterConfig, SExp, SExpBookendStyle, sexp_to_string};

pub fn format_module(module: &Module) -> String {
    let printer = Printer {
        types: &module.types,
    };
    sexp_to_string(
        &printer.module(module),
        &PrinterConfig {
            indent_width: 2,
            margin_width: 100,
        },
    )
}

struct Printer<'a> {
    types: &'a [TypeDef],
}

impl Printer<'_> {
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

    fn definition(&self, index: usize, definition: &TypeDef) -> SExp {
        let mut fields = vec![
            atom(format!("type{index}")),
            self.ty(&definition.ty(TypeId::from_index(index))),
        ];
        if let Some(body) = definition.body() {
            fields.push(self.ty(body));
        }
        if let Some(drop) = definition.drop_hook() {
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
        fields.extend(function.signature.params.iter().map(|p| self.parameter(p)));
        if let Some(foreign) = &function.foreign {
            fields.push(list("extern", vec![quoted(&foreign.header)]));
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

    fn ty(&self, ty: &Ty) -> SExp {
        quoted(format_type(ty, self.types))
    }

    fn term(&self, term: &Term) -> SExp {
        list("typed", vec![self.ty(&term.ty), self.kind(&term.kind)])
    }

    fn kind(&self, kind: &TermKind) -> SExp {
        match kind {
            TermKind::Constant(value) => list("constant", vec![quoted(format!("{value:?}"))]),
            TermKind::Local { binding: id, name } => {
                list("local", vec![binding(*id), atom(&name.val)])
            }
            TermKind::Function { function } => function_id(function.index()),
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
            TermKind::Call { func, arg } => list("call", vec![self.term(func), self.term(arg)]),
            TermKind::Pack(args) => self.arguments(args),
            TermKind::Intrinsic { op, args } => list(
                "intrinsic",
                vec![atom(format!("{op:?}")), self.arguments(args)],
            ),
            TermKind::Adapt { conversion, arg } => list(
                "adapt",
                vec![quoted(format!("{conversion:?}")), self.term(arg)],
            ),
            TermKind::Convert { conversion, arg } => list(
                "convert",
                vec![quoted(format!("{conversion:?}")), self.term(arg)],
            ),
            TermKind::ArcNew { value } => list("arc", vec![self.term(value)]),
            TermKind::WeakEmpty { pointee } => list("weak-empty", vec![self.ty(pointee)]),
            TermKind::Result { failure, arg } => {
                list(if *failure { "err" } else { "ok" }, vec![self.term(arg)])
            }
            TermKind::Absurd { arg } => list("absurd", vec![self.term(arg)]),
            TermKind::Assign { place, value } => {
                list("assign", vec![self.term(place), self.term(value)])
            }
            TermKind::Address { place } => list("address", vec![self.term(place)]),
            TermKind::Deref { pointer } => list("deref", vec![self.term(pointer)]),
            TermKind::Field { base, access } => list(
                "field",
                vec![self.term(base), quoted(format!("{access:?}"))],
            ),
        }
    }

    fn arguments(&self, args: &Arguments) -> SExp {
        let mut fields = vec![list(
            "params",
            args.params.iter().map(|ty| self.ty(ty)).collect(),
        )];
        if let Some(receiver) = &args.receiver {
            fields.push(list("receiver", vec![self.term(receiver)]));
        }
        fields.push(self.term(&args.argument));
        list("pack", fields)
    }

    fn arm(&self, arm: &MatchArm) -> SExp {
        let tag = match &arm.tag {
            Case::Ok => atom("ok"),
            Case::Err => atom("err"),
            Case::Type(ty) => self.ty(ty),
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
