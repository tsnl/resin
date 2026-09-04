//! AST → S-expression formatting via `sexpfmt`.

use sexpfmt::{PrinterConfig, SExp, SExpBookendStyle, sexp_to_string};

use crate::ast::*;

//
// API
//

pub fn format_source(file: &SourceFile) -> String {
    let sexp = sexp_source(file);
    let config = PrinterConfig {
        indent_width: 2,
        margin_width: 80,
    };
    sexp_to_string(&sexp, &config)
}

//
// SExp builders
//

fn sexp_source(file: &SourceFile) -> SExp {
    list("source", file.stmts.iter().map(sexp_stmt).collect())
}

fn sexp_stmt(stmt: &Stmt) -> SExp {
    match &stmt.val {
        StmtKind::Define { name, init } => list_sp(
            "define",
            stmt.span,
            vec![symbol(name.val.as_ref()), sexp_term(init)],
        ),
        StmtKind::Declare { name, ann } => list_sp(
            "declare",
            stmt.span,
            vec![symbol(name.val.as_ref()), sexp_typespec(ann)],
        ),
    }
}

fn sexp_term(term: &Term) -> SExp {
    match &term.val {
        TermKind::Var(v) => symbol(v.val.as_ref()),
        TermKind::Num(n) => symbol(n.as_ref()),
        TermKind::Lambda { params, body } => {
            let param_sexps: Vec<SExp> = params
                .iter()
                .map(|(name, ann)| {
                    list("param", vec![symbol(name.val.as_ref()), sexp_typespec(ann)])
                })
                .collect();
            list_sp(
                "lambda",
                term.span,
                vec![group(param_sexps), sexp_term(body)],
            )
        }
        TermKind::If { cond, then, els } => list_sp(
            "if",
            term.span,
            vec![sexp_term(cond), sexp_term(then), sexp_term(els)],
        ),
        TermKind::Array(elems) => {
            list_sp("array", term.span, elems.iter().map(sexp_term).collect())
        }
        TermKind::Record(fields) => list_sp(
            "record",
            term.span,
            fields
                .iter()
                .map(|(name, val)| list("field", vec![symbol(name.val.as_ref()), sexp_term(val)]))
                .collect(),
        ),
        TermKind::Block { stmts, tail } => list_sp(
            "block",
            term.span,
            vec![
                group(stmts.iter().map(sexp_stmt).collect()),
                sexp_term(tail),
            ],
        ),
        TermKind::Unit => list_sp("unit", term.span, vec![]),
        TermKind::Call { func, args } => list_sp(
            "call",
            term.span,
            vec![sexp_term(func), group(args.iter().map(sexp_term).collect())],
        ),
        TermKind::Type(ty) => sexp_typespec(ty),
    }
}

fn sexp_typespec(ts: &Type) -> SExp {
    match &ts.val {
        TypeKind::Atom(name) => symbol(name.val.as_ref()),
        TypeKind::App { head, arg } => list_sp(
            "type-app",
            ts.span,
            vec![symbol(head.val.as_ref()), sexp_term(arg)],
        ),
        TypeKind::Func { from, to } => list_sp(
            "func-type",
            ts.span,
            vec![sexp_typespec(from), sexp_typespec(to)],
        ),
        TypeKind::Record(fields) => list_sp(
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

//
// SExp builder helpers
//

fn list(head: &str, items: Vec<SExp>) -> SExp {
    let mut children = Vec::with_capacity(items.len() + 1);
    children.push(symbol(head));
    children.extend(items);
    SExp::List(children, SExpBookendStyle::Parentheses)
}

fn list_sp(head: &str, span: Span, items: Vec<SExp>) -> SExp {
    let mut children = Vec::with_capacity(items.len() + 2);
    children.push(symbol(head));
    children.extend(items);
    children.push(span_str(span));
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
