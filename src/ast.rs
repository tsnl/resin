use crate::{Literal, Operator};

use super::{Span, Symbol};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Def {
    name: Symbol,
    template_args: Vec<Bind>,
    args: Vec<Bind>,
    ret_ty: Option<Expr>,
    body: Expr,
    span: Span,
}
impl Def {
    pub fn new(
        name: Symbol,
        template_args: Vec<Bind>,
        args: Vec<Bind>,
        ret_ty: Option<Expr>,
        body: Expr,
        span: Span,
    ) -> Self {
        Self {
            name,
            template_args,
            args,
            ret_ty,
            body,
            span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bind {
    name: Symbol,
    expr: Expr,
    span: Span,
}
impl Bind {
    pub fn new(name: Symbol, expr: Expr, span: Span) -> Self {
        Self { name, expr, span }
    }
    pub fn name(&self) -> &Symbol {
        &self.name
    }
    pub fn expr(&self) -> &Expr {
        &self.expr
    }
    pub fn span(&self) -> &Span {
        &self.span
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Literal(Box<LiteralExpr>),
    Name(Box<NameExpr>),
    Chain(Box<ChainExpr>),
    Record(Box<RecordExpr>),
    Array(Box<ArrayExpr>),
    Call(Box<CallExpr>),
    Dot(Box<DotExpr>),
    If(Box<IfExpr>),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralExpr {
    pub literal: Literal,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameExpr {
    pub name: Symbol,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainExpr {
    pub binds: Vec<Bind>,
    pub tail: Option<Expr>,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordExpr {
    pub fields: Vec<Bind>,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayExpr {
    pub elements: Vec<Expr>,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallExpr {
    pub func: Expr,
    pub args: Vec<Expr>,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotExpr {
    pub expr: Expr,
    pub field: Symbol,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfExpr {
    pub cond: Expr,
    pub then_branch: Expr,
    pub else_branch: Expr,
    pub span: Span,
}
impl Expr {
    pub fn new_literal(literal: Literal, span: Span) -> Self {
        Self::Literal(Box::new(LiteralExpr { literal, span }))
    }
    pub fn new_name(name: Symbol, span: Span) -> Self {
        Self::Name(Box::new(NameExpr { name, span }))
    }
    pub fn new_chain(binds: Vec<Bind>, tail: Option<Expr>, span: Span) -> Self {
        Self::Chain(Box::new(ChainExpr { binds, tail, span }))
    }
    pub fn new_record(fields: Vec<Bind>, span: Span) -> Self {
        Self::Record(Box::new(RecordExpr { fields, span }))
    }
    pub fn new_array(elements: Vec<Expr>, span: Span) -> Self {
        Self::Array(Box::new(ArrayExpr { elements, span }))
    }
    pub fn new_call(func: Expr, args: Vec<Expr>, span: Span) -> Self {
        Self::Call(Box::new(CallExpr { func, args, span }))
    }
    pub fn new_dot(expr: Expr, field: Symbol, span: Span) -> Self {
        Self::Dot(Box::new(DotExpr { expr, field, span }))
    }
    pub fn new_if(cond: Expr, then_branch: Expr, else_branch: Expr, span: Span) -> Self {
        Self::If(Box::new(IfExpr {
            cond,
            then_branch,
            else_branch,
            span,
        }))
    }
    pub fn new_op(op: Operator, args: Vec<Expr>, span: Span) -> Self {
        let name = Symbol::from(Into::<&str>::into(op));
        let func = Expr::new_name(name, span.clone());
        Self::new_call(func, args, span)
    }
}
