use crate::{
    Symbol,
    fb::Loc,
    vocab::{self, BuiltinType},
};

#[derive(Debug)]
pub struct File {
    pub stmts: Vec<Stmt>,
}

crate::define_tree! {
  pub enum Stmt {
    DefFun {
      name: Symbol,
      template_args: Vec<AnnId>,
      args: Vec<AnnId>,
      ret_ty: Option<TySpec>,
      body: Term,
      loc: Loc,
    },
    DefNewType {
      name: Symbol,
      template_args: Vec<AnnId>,
      old_ty: TySpec,
      loc: Loc,
    },
    DefEnumType {
      name: Symbol,
      template_args: Vec<AnnId>,
      variants: Vec<(Symbol, Option<TySpec>)>,
      loc: Loc,
    },
    Let {
      pattern: Pattern,
      init: Term,
      loc: Loc,
    },
    Discard {
      val: Term,
      loc: Loc,
    },
  }
  pub enum Term {
    Hole {
      loc: Loc,
    },
    Literal {
      val: vocab::Literal,
      loc: Loc,
    },
    Name {
      name: Symbol,
      loc: Loc,
    },
    Dot {
      base: Term,
      field: vocab::Literal,
      loc: Loc,
    },
    Apply {
      callee: Term,
      args: Vec<Term>,
      loc: Loc,
    },
    If {
      cond_branch_vec: Vec<(Term, Term)>,
      else_branch: Option<Term>,
      loc: Loc,
    },
    Chain {
      stmt_vec: Vec<Stmt>,
      loc: Loc,
    },
    Tuple {
      elements: Vec<Term>,
      loc: Loc,
    },
    NamedTuple {
      fields: Vec<(Symbol, Term)>,
      loc: Loc,
    },
    Constructor {
      name: Symbol,
      loc: Loc,
    },
    Match {
      scrutinee: Term,
      arms: Vec<(Pattern, Term)>,
      loc: Loc,
    }
  }
  pub enum Pattern {
    Hole {
      loc: Loc,
    },
    Literal {
      val: vocab::Literal,
      loc: Loc,
    },
    Name {
      name: Symbol,
      ann: Option<TySpec>,
      loc: Loc,
    },
    Constructor {
      name: Symbol,
      arg: Option<Pattern>,
      loc: Loc,
    },
    Tuple {
      elements: Vec<Pattern>,
      loc: Loc,
    },
    NamedTuple {
      fields: Vec<(Symbol, Option<Pattern>)>,
      loc: Loc,
    },
    Pointer {
      pointee: Pattern,
      loc: Loc,
    }
  }
  pub enum TySpec {
    Hole {
      loc: Loc,
    },
    Builtin {
      kind: BuiltinType,
      loc: Loc,
    },
    Name {
      name: Symbol,
      loc: Loc,
    },
    TemplateInstance {
      name: Symbol,
      template_args: Vec<TySpec>,
      loc: Loc,
    },
    NamedTuple {
      fields: Vec<AnnId>,
      loc: Loc,
    }
  }
  pub enum AnnId {
    Lid {
      name: Symbol,
      ty: TySpec,
      loc: Loc,
    },
    Uid {
      name: Symbol,
      ty: Option<TySpec>,
      loc: Loc,
    }
  }
}

impl Stmt {
    pub fn loc(&self) -> &Loc {
        match self {
            Stmt::DefFun(inner) => {
                let stmt::DefFun { loc, .. } = &**inner;
                loc
            }
            Stmt::Let(inner) => {
                let stmt::Let { loc, .. } = &**inner;
                loc
            }
            Stmt::Discard(inner) => {
                let stmt::Discard { loc, .. } = &**inner;
                loc
            }
            Stmt::DefNewType(def_new_type) => {
                let stmt::DefNewType { loc, .. } = &**def_new_type;
                loc
            }
            Stmt::DefEnumType(def_enum_type) => {
                let stmt::DefEnumType { loc, .. } = &**def_enum_type;
                loc
            }
        }
    }
}
impl Term {
    pub fn loc(&self) -> &Loc {
        match self {
            Term::Hole(inner) => {
                let term::Hole { loc, .. } = &**inner;
                loc
            }
            Term::Literal(inner) => {
                let term::Literal { loc, .. } = &**inner;
                loc
            }
            Term::Name(inner) => {
                let term::Name { loc, .. } = &**inner;
                loc
            }
            Term::Apply(inner) => {
                let term::Apply { loc, .. } = &**inner;
                loc
            }
            Term::Dot(dot) => {
                let term::Dot { loc, .. } = &**dot;
                loc
            }
            Term::If(inner) => {
                let term::If { loc, .. } = &**inner;
                loc
            }
            Term::Chain(inner) => {
                let term::Chain { loc, .. } = &**inner;
                loc
            }
            Term::Tuple(inner) => {
                let term::Tuple { loc, .. } = &**inner;
                loc
            }
            Term::NamedTuple(inner) => {
                let term::NamedTuple { loc, .. } = &**inner;
                loc
            }
            Term::Constructor(inner) => {
                let term::Constructor { loc, .. } = &**inner;
                loc
            }
            Term::Match(inner) => {
                let term::Match { loc, .. } = &**inner;
                loc
            }
        }
    }
}

impl Pattern {
    pub fn loc(&self) -> &Loc {
        match self {
            Pattern::Hole(inner) => {
                let pattern::Hole { loc, .. } = &**inner;
                loc
            }
            Pattern::Literal(inner) => {
                let pattern::Literal { loc, .. } = &**inner;
                loc
            }
            Pattern::Name(inner) => {
                let pattern::Name { loc, .. } = &**inner;
                loc
            }
            Pattern::Constructor(inner) => {
                let pattern::Constructor { loc, .. } = &**inner;
                loc
            }
            Pattern::Tuple(inner) => {
                let pattern::Tuple { loc, .. } = &**inner;
                loc
            }
            Pattern::NamedTuple(anonymous_struct) => {
                let pattern::NamedTuple { loc, .. } = &**anonymous_struct;
                loc
            }
            Pattern::Pointer(pointer) => {
                let pattern::Pointer { loc, .. } = &**pointer;
                loc
            }
        }
    }
}
