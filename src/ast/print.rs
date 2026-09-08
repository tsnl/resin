//! AST → S-expression formatting via `sexpfmt`.

use ::sexpfmt::{PrinterConfig, SExp, SExpBookendStyle, sexp_to_string};

use super::*;

pub fn format_source(file: &SourceFile) -> String {
    format(sexp_source(file))
}

pub fn format_program(program: &Program) -> String {
    format(list(
        "program",
        program
            .modules
            .iter()
            .map(|module| {
                list(
                    "file",
                    vec![
                        string(module.path.to_string_lossy()),
                        sexp_source(&module.file),
                    ],
                )
            })
            .collect(),
    ))
}

fn format(sexp: SExp) -> String {
    let config = PrinterConfig {
        indent_width: 2,
        margin_width: 80,
    };
    sexp_to_string(&sexp, &config)
}

fn sexp_source(file: &SourceFile) -> SExp {
    let mut items = Vec::new();
    if !file.exports.is_empty() {
        items.push(list(
            "export",
            file.exports
                .iter()
                .map(|name| symbol(name.val.as_ref()))
                .collect(),
        ));
    }
    if !file.imports.is_empty() {
        items.push(list(
            "import",
            file.imports
                .iter()
                .map(|path| string(path.val.as_ref()))
                .collect(),
        ));
    }
    items.extend(file.stmts.iter().map(sexp_stmt));
    list("source", items)
}

fn sexp_stmt(stmt: &Stmt) -> SExp {
    match &stmt.val {
        StmtKind::Struct { name, body } => list_sp(
            "struct",
            stmt.span,
            vec![symbol(name.val.as_ref()), sexp_typespec(body)],
        ),
        StmtKind::ForeignType { name } => {
            list_sp("extern-type", stmt.span, vec![symbol(name.val.as_ref())])
        }
        StmtKind::ForeignFunction {
            header,
            name,
            params,
            result,
        } => list_sp(
            "extern",
            stmt.span,
            vec![
                string(header.as_ref()),
                symbol(name.val.as_ref()),
                group(
                    params
                        .iter()
                        .map(|(name, ann)| {
                            list("param", vec![symbol(name.val.as_ref()), sexp_typespec(ann)])
                        })
                        .collect(),
                ),
                sexp_typespec(result),
            ],
        ),
        StmtKind::Impl { owner, methods } => list_sp(
            "impl",
            stmt.span,
            std::iter::once(symbol(owner.val.as_ref()))
                .chain(methods.iter().map(sexp_stmt))
                .collect(),
        ),
        StmtKind::Function {
            name,
            params,
            result,
            body,
            decorators,
            ..
        } => list_sp(
            "def",
            stmt.span,
            vec![
                symbol(name.val.as_ref()),
                group(
                    params
                        .iter()
                        .map(|(name, ann)| {
                            list("param", vec![symbol(name.val.as_ref()), sexp_typespec(ann)])
                        })
                        .collect(),
                ),
                sexp_typespec(result),
                sexp_term(body),
                list(
                    "decorators",
                    decorators
                        .iter()
                        .map(|name| symbol(name.val.as_ref()))
                        .collect(),
                ),
            ],
        ),
        StmtKind::Define { name, init } => list_sp(
            "define",
            stmt.span,
            vec![symbol(name.val.as_ref()), sexp_term(init)],
        ),
        StmtKind::DefineType { name, init } => list_sp(
            "define-type",
            stmt.span,
            vec![symbol(name.val.as_ref()), sexp_typespec(init)],
        ),
        StmtKind::Declare { name, ann } => list_sp(
            "declare",
            stmt.span,
            vec![symbol(name.val.as_ref()), sexp_typespec(ann)],
        ),
        StmtKind::Expr { term } => list_sp("expr", stmt.span, vec![sexp_term(term)]),
        StmtKind::Defer { body } => list_sp("defer", stmt.span, vec![sexp_term(body)]),
    }
}

fn sexp_term(term: &Term) -> SExp {
    match &term.val {
        TermKind::Unwrap { value } => list_sp("unwrap", term.span, vec![sexp_term(value)]),
        TermKind::Try { value } => list_sp("try", term.span, vec![sexp_term(value)]),
        TermKind::Match { value, arms } => {
            let mut items = vec![sexp_term(value)];
            items.extend(arms.iter().map(|arm| {
                let variant = match &arm.variant {
                    MatchVariant::Ok => symbol("ok"),
                    MatchVariant::Err => symbol("err"),
                    MatchVariant::Type(ty) => sexp_typespec(ty),
                };
                list(
                    "arm",
                    vec![
                        variant,
                        symbol(arm.name.as_ref().map_or("_", |name| name.val.as_ref())),
                        sexp_term(&arm.body),
                    ],
                )
            }));
            list_sp("match", term.span, items)
        }
        TermKind::Hole { children } => {
            list_sp("hole", term.span, children.iter().map(sexp_term).collect())
        }
        TermKind::FieldHole { base } => list_sp("field-hole", term.span, vec![sexp_term(base)]),
        TermKind::Var { name } => symbol(name.val.as_ref()),
        TermKind::Num { value } => symbol(value.as_ref()),
        TermKind::String { value } => string(value.as_ref()),
        TermKind::If { cond, then, els } => list_sp(
            "if",
            term.span,
            vec![sexp_term(cond), sexp_term(then), sexp_term(els)],
        ),
        TermKind::While { cond, body } => {
            list_sp("while", term.span, vec![sexp_term(cond), sexp_term(body)])
        }
        TermKind::Array { elems } => {
            list_sp("array", term.span, elems.iter().map(sexp_term).collect())
        }
        TermKind::Record { fields } => list_sp(
            "record",
            term.span,
            fields
                .iter()
                .map(|(name, val)| list("field", vec![symbol(name.val.as_ref()), sexp_term(val)]))
                .collect(),
        ),
        TermKind::Block { stmts, tail } => {
            let mut items: Vec<_> = stmts.iter().map(sexp_stmt).collect();
            items.push(sexp_term(tail));
            list_sp("block", term.span, items)
        }
        TermKind::None => symbol("None"),
        TermKind::Unit => list_sp("unit", term.span, vec![]),
        TermKind::Call { func, arg } => {
            list_sp("call", term.span, vec![sexp_term(func), sexp_term(arg)])
        }
        TermKind::Builtin { name, args } => list_sp(
            "builtin",
            term.span,
            vec![
                symbol(name.as_ref()),
                group(args.iter().map(sexp_term).collect()),
            ],
        ),
        TermKind::Assign { place, value } => list_sp(
            "assign",
            term.span,
            vec![sexp_term(place), sexp_term(value)],
        ),
        TermKind::Deref { pointer } => list_sp("deref", term.span, vec![sexp_term(pointer)]),
        TermKind::Address { place } => list_sp("address", term.span, vec![sexp_term(place)]),
        TermKind::Field { base, name } => list_sp(
            "field",
            term.span,
            vec![sexp_term(base), symbol(name.val.as_ref())],
        ),
        TermKind::Type { ty } => sexp_typespec(ty),
    }
}

fn sexp_typespec(ts: &Type) -> SExp {
    match &ts.val {
        TypeKind::Union { left, right } => list_sp(
            "union-type",
            ts.span,
            vec![sexp_typespec(left), sexp_typespec(right)],
        ),
        TypeKind::Result { value, error } => list_sp(
            "result-type",
            ts.span,
            vec![sexp_typespec(value), sexp_typespec(error)],
        ),
        TypeKind::Hole => list_sp("type-hole", ts.span, vec![]),
        TypeKind::Infer => list_sp("infer-type", ts.span, vec![]),
        TypeKind::Unit => list_sp("unit-type", ts.span, vec![]),
        TypeKind::Atom { name } => symbol(name.val.as_ref()),
        TypeKind::App { head, arg } => list_sp(
            "type-app",
            ts.span,
            vec![symbol(head.val.as_ref()), sexp_typespec(arg)],
        ),
        TypeKind::Func { from, to } => list_sp(
            "func-type",
            ts.span,
            vec![sexp_typespec(from), sexp_typespec(to)],
        ),
        TypeKind::Record { fields } => list_sp(
            "record-type",
            ts.span,
            fields
                .iter()
                .map(|(name, ann)| {
                    list("field", vec![symbol(name.val.as_ref()), sexp_typespec(ann)])
                })
                .collect(),
        ),
    }
}

fn list(head: &str, items: Vec<SExp>) -> SExp {
    let mut children = Vec::with_capacity(items.len() + 1);
    children.push(symbol(head));
    children.extend(items);
    SExp::List(children, SExpBookendStyle::Parentheses)
}

fn list_sp(head: &str, span: Span, items: Vec<SExp>) -> SExp {
    let mut children = Vec::with_capacity(items.len() + 2);
    children.push(symbol(head));
    children.push(span_str(span));
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

fn string(s: impl Into<String>) -> SExp {
    SExp::Atom(format!("{:?}", s.into()))
}

fn span_str(span: Span) -> SExp {
    string(format!("{}:{}", span.start, span.end))
}
