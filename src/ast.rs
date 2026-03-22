use crate::{Span, Symbol, vocab};

#[derive(Debug)]
pub struct File {
    pub stmts: Vec<Stmt>,
}

#[derive(Debug)]
pub enum Type {
    Name {
        name: Symbol,
    },
    Apply {
        name: Symbol,
        args: Vec<Expr>,
    },
    Record {
        fields: Vec<(Symbol, Expr)>,
    },
    Enum {
        variants: Vec<(Symbol, Option<Expr>)>,
    },
}

crate::define_tree! {
  pub enum Stmt {
    Def {
      name: Symbol,
      args: Vec<(Symbol, Span)>,
      body: Expr,
      span: Span,
    },
    Let {
      pattern: Pattern,
      init: Expr,
      span: Span,
    },
    Discard {
      val: Expr,
      span: Span,
    },
    TypeSig {
      name: Symbol,
      sig: Expr,
      span: Span,
    },
  }
  pub enum Expr {
    Literal {
      val: vocab::Literal,
      span: Span,
    },
    Name {
      name: Symbol,
      span: Span,
    },
    Dot {
      base: Expr,
      field: Symbol,
      span: Span,
    },
    Apply {
      callee: Expr,
      args: Vec<Expr>,
      span: Span,
    },
    If {
      cond_branch_vec: Vec<(Expr, Expr)>,
      else_branch: Expr,
      span: Span,
    },
    Chain {
      stmt_vec: Vec<Stmt>,
      span: Span,
    },
    Tuple {
      elements: Vec<Expr>,
      span: Span,
    },
    Match {
      scrutinee: Expr,
      arms: Vec<(Pattern, Expr)>,
      span: Span,
    },
    Ctor {
      ty: Type,
      span: Span,
    },
    Grad {
      func: Expr,
      span: Span,
    },
    As {
      expr: Expr,
      target: Expr,
      span: Span,
    }
  }
  pub enum Pattern {
    Hole {
      span: Span,
    },
    Literal {
      val: vocab::Literal,
      span: Span,
    },
    Name {
      name: Symbol,
      span: Span,
    },
    Constructor {
      name: Symbol,
      arg: Option<Pattern>,
      span: Span,
    }
  }
}

impl Stmt {
    pub fn span(&self) -> &Span {
        match self {
            Stmt::Def(inner) => &inner.span,
            Stmt::Let(inner) => &inner.span,
            Stmt::Discard(inner) => &inner.span,
            Stmt::TypeSig(inner) => &inner.span,
        }
    }
}

impl Expr {
    pub fn span(&self) -> &Span {
        match self {
            Expr::Literal(inner) => &inner.span,
            Expr::Name(inner) => &inner.span,
            Expr::Dot(inner) => &inner.span,
            Expr::Apply(inner) => &inner.span,
            Expr::If(inner) => &inner.span,
            Expr::Chain(inner) => &inner.span,
            Expr::Tuple(inner) => &inner.span,
            Expr::Match(inner) => &inner.span,
            Expr::Ctor(inner) => &inner.span,
            Expr::Grad(inner) => &inner.span,
            Expr::As(inner) => &inner.span,
        }
    }
}

impl Pattern {
    pub fn span(&self) -> &Span {
        match self {
            Pattern::Hole(inner) => &inner.span,
            Pattern::Literal(inner) => &inner.span,
            Pattern::Name(inner) => &inner.span,
            Pattern::Constructor(inner) => &inner.span,
        }
    }
}
