use std::fmt::{self, Write};

use num::ToPrimitive;

use crate::ast;
use crate::vocab;
use crate::Symbol;
pub fn print(file: &ast::File) -> String {
    let mut buf = String::new();
    write_file(&mut buf, file).unwrap();
    buf
}

fn write_each<T>(
    f: &mut String,
    items: &[T],
    write_item: impl Fn(&mut String, &T) -> fmt::Result,
) -> fmt::Result {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            f.write_char(' ')?;
        }
        write_item(f, item)?;
    }
    Ok(())
}

fn write_list<T>(
    f: &mut String,
    items: &[T],
    write_item: impl Fn(&mut String, &T) -> fmt::Result,
) -> fmt::Result {
    f.write_char('(')?;
    write_each(f, items, write_item)?;
    f.write_char(')')
}

// --- File ---

fn write_file(f: &mut String, file: &ast::File) -> fmt::Result {
    f.write_str("(file ")?;
    write_each(f, &file.stmts, write_stmt)?;
    f.write_char(')')
}

// --- Stmt ---

fn write_stmt(f: &mut String, stmt: &ast::Stmt) -> fmt::Result {
    match stmt {
        ast::Stmt::Def(inner) => write_stmt_def(f, inner),
        ast::Stmt::Let(inner) => write_stmt_let(f, inner),
        ast::Stmt::Discard(inner) => write_stmt_discard(f, inner),
        ast::Stmt::TypeSig(inner) => write_stmt_type_sig(f, inner),
    }
}

fn write_stmt_def(f: &mut String, inner: &ast::stmt::Def) -> fmt::Result {
    let ast::stmt::Def { name, args, body, .. } = inner;
    f.write_str("(def ")?;
    write!(f, "{name}")?;
    if !args.is_empty() {
        f.write_char(' ')?;
        write_list(f, args, |f, (sym, _)| write!(f, "{sym}"))?;
    }
    f.write_char(' ')?;
    write_expr(f, body)?;
    f.write_char(')')
}

fn write_stmt_let(f: &mut String, inner: &ast::stmt::Let) -> fmt::Result {
    let ast::stmt::Let { pattern, init, .. } = inner;
    f.write_str("(let ")?;
    write_pattern(f, pattern)?;
    f.write_char(' ')?;
    write_expr(f, init)?;
    f.write_char(')')
}

fn write_stmt_discard(f: &mut String, inner: &ast::stmt::Discard) -> fmt::Result {
    let ast::stmt::Discard { val, .. } = inner;
    f.write_str("(discard ")?;
    write_expr(f, val)?;
    f.write_char(')')
}

fn write_stmt_type_sig(f: &mut String, inner: &ast::stmt::TypeSig) -> fmt::Result {
    let ast::stmt::TypeSig { name, sig, .. } = inner;
    f.write_str("(type-sig ")?;
    write!(f, "{name} ")?;
    write_expr(f, sig)?;
    f.write_char(')')
}

// --- Expr ---

fn write_expr(f: &mut String, expr: &ast::Expr) -> fmt::Result {
    match expr {
        ast::Expr::Literal(inner) => write_expr_literal(f, inner),
        ast::Expr::Name(inner) => write_expr_name(f, inner),
        ast::Expr::Dot(inner) => write_expr_dot(f, inner),
        ast::Expr::Apply(inner) => write_expr_apply(f, inner),
        ast::Expr::If(inner) => write_expr_if(f, inner),
        ast::Expr::Chain(inner) => write_expr_chain(f, inner),
        ast::Expr::Tuple(inner) => write_expr_tuple(f, inner),
        ast::Expr::Match(inner) => write_expr_match(f, inner),
        ast::Expr::Ctor(inner) => write_expr_ctor(f, inner),
        ast::Expr::Grad(inner) => {
            f.write_str("(grad ")?;
            write_expr(f, &inner.func)?;
            f.write_char(')')
        }
        ast::Expr::As(inner) => {
            f.write_str("(as ")?;
            write_expr(f, &inner.expr)?;
            f.write_char(' ')?;
            write_expr(f, &inner.target)?;
            f.write_char(')')
        }
    }
}

fn write_expr_literal(f: &mut String, inner: &ast::expr::Literal) -> fmt::Result {
    write_literal(f, &inner.val)
}

fn write_expr_name(f: &mut String, inner: &ast::expr::Name) -> fmt::Result {
    write!(f, "{}", inner.name)
}

fn write_expr_dot(f: &mut String, inner: &ast::expr::Dot) -> fmt::Result {
    f.write_str("(. ")?;
    write_expr(f, &inner.base)?;
    write!(f, " {})", inner.field)
}

fn write_expr_apply(f: &mut String, inner: &ast::expr::Apply) -> fmt::Result {
    f.write_str("(apply ")?;
    write_expr(f, &inner.callee)?;
    f.write_char(' ')?;
    write_each(f, &inner.args, write_expr)?;
    f.write_char(')')
}

fn write_expr_if(f: &mut String, inner: &ast::expr::If) -> fmt::Result {
    f.write_str("(if ")?;
    write_each(f, &inner.cond_branch_vec, |f, (cond, body)| {
        f.write_char('(')?;
        write_expr(f, cond)?;
        f.write_char(' ')?;
        write_expr(f, body)?;
        f.write_char(')')
    })?;
    f.write_str(" (else ")?;
    write_expr(f, &inner.else_branch)?;
    f.write_str("))")
}

fn write_expr_chain(f: &mut String, inner: &ast::expr::Chain) -> fmt::Result {
    f.write_str("(chain ")?;
    write_each(f, &inner.stmt_vec, write_stmt)?;
    f.write_char(')')
}

fn write_expr_tuple(f: &mut String, inner: &ast::expr::Tuple) -> fmt::Result {
    f.write_str("(tuple ")?;
    write_each(f, &inner.elements, write_expr)?;
    f.write_char(')')
}

fn write_expr_match(f: &mut String, inner: &ast::expr::Match) -> fmt::Result {
    f.write_str("(match ")?;
    write_expr(f, &inner.scrutinee)?;
    f.write_char(' ')?;
    write_each(f, &inner.arms, |f, (pat, body)| {
        f.write_char('(')?;
        write_pattern(f, pat)?;
        f.write_char(' ')?;
        write_expr(f, body)?;
        f.write_char(')')
    })?;
    f.write_char(')')
}

fn write_expr_ctor(f: &mut String, inner: &ast::expr::Ctor) -> fmt::Result {
    f.write_str("(ctor ")?;
    write_type(f, &inner.ty)?;
    f.write_char(')')
}

// --- Pattern ---

fn write_pattern(f: &mut String, pat: &ast::Pattern) -> fmt::Result {
    match pat {
        ast::Pattern::Hole(inner) => write_pattern_hole(f, inner),
        ast::Pattern::Literal(inner) => write_pattern_literal(f, inner),
        ast::Pattern::Name(inner) => write_pattern_name(f, inner),
        ast::Pattern::Constructor(inner) => write_pattern_constructor(f, inner),
    }
}

fn write_pattern_hole(f: &mut String, _inner: &ast::pattern::Hole) -> fmt::Result {
    f.write_char('_')
}

fn write_pattern_literal(f: &mut String, inner: &ast::pattern::Literal) -> fmt::Result {
    write_literal(f, &inner.val)
}

fn write_pattern_name(f: &mut String, inner: &ast::pattern::Name) -> fmt::Result {
    write!(f, "{}", inner.name)
}

fn write_pattern_constructor(f: &mut String, inner: &ast::pattern::Constructor) -> fmt::Result {
    match &inner.arg {
        Some(sub) => {
            f.write_str("(ctor ")?;
            write!(f, "{} ", inner.name)?;
            write_pattern(f, sub)?;
            f.write_char(')')
        }
        None => write!(f, "{}", inner.name),
    }
}

// --- Type ---

fn write_type(f: &mut String, ty: &ast::Type) -> fmt::Result {
    match ty {
        ast::Type::Name { name } => write_type_name(f, name),
        ast::Type::Apply { name, args } => write_type_apply(f, name, args),
        ast::Type::Record { fields } => write_type_record(f, fields),
        ast::Type::Enum { variants } => write_type_enum(f, variants),
    }
}

fn write_type_name(f: &mut String, name: &Symbol) -> fmt::Result {
    write!(f, "{name}")
}

fn write_type_apply(f: &mut String, name: &Symbol, args: &[ast::Expr]) -> fmt::Result {
    f.write_str("(type-apply ")?;
    write!(f, "{name} ")?;
    write_each(f, args, write_expr)?;
    f.write_char(')')
}

fn write_type_record(f: &mut String, fields: &[(Symbol, ast::Expr)]) -> fmt::Result {
    f.write_str("(record ")?;
    write_each(f, fields, |f: &mut String, (name, expr)| {
        f.write_char('(')?;
        write!(f, "{name} ")?;
        write_expr(f, expr)?;
        f.write_char(')')
    })?;
    f.write_char(')')
}

fn write_type_enum(f: &mut String, variants: &[(Symbol, Option<ast::Expr>)]) -> fmt::Result {
    f.write_str("(enum ")?;
    write_each(f, variants, |f: &mut String, (name, payload)| {
        match payload {
            Some(expr) => {
                write!(f, "({name} ")?;
                write_expr(f, expr)?;
                f.write_char(')')
            }
            None => write!(f, "{name}"),
        }
    })?;
    f.write_char(')')
}

// --- Literal ---

fn write_literal(f: &mut String, lit: &vocab::Literal) -> fmt::Result {
    match lit {
        vocab::Literal::Number(n) => {
            if n.is_integer() {
                write!(f, "{}", n.value.numer())
            } else if n.force_float {
                write!(f, "{}", n.value.to_f64().unwrap())
            } else {
                write!(f, "{}/{}", n.value.numer(), n.value.denom())
            }
        }
        vocab::Literal::String(s) => write!(f, "\"{}\"", s.content),
        vocab::Literal::Bool(b) => f.write_str(if b.value { "true" } else { "false" }),
    }
}
