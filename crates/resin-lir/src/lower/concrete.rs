//! Concrete expressions consumed by storage lowering, one function at a time.
use crate::Foreign;
use resin_hir::BindingId;
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(super) struct Function {
    pub(super) location: Option<SourceLocation>,
    pub(super) name: Arc<str>,
    pub(super) signature: Signature,
    /// External declarations have no function body.
    pub(super) foreign: Option<Foreign>,
    pub(super) body: Option<Term>,
}

#[derive(Debug, Clone)]
pub(super) struct Signature {
    pub(super) params: Vec<Parameter>,
    pub(super) result: Ty,
}

#[derive(Debug, Clone)]
pub(super) struct Parameter {
    /// Required for every parameter of a function with a body, including unused ones.
    pub(super) binding: Option<BindingId>,
    pub(super) name: Ident,
    pub(super) ty: Ty,
}

#[derive(Debug, Clone)]
pub(super) struct Term {
    pub(super) span: Span,
    pub(super) ty: Ty,
    pub(super) kind: TermKind,
}

#[derive(Debug, Clone)]
pub(super) enum TermKind {
    Break,
    Continue,
    Return {
        value: Box<Term>,
    },
    Move {
        place: Box<Term>,
    },
    Constant {
        value: Value,
    },
    Local {
        binding: BindingId,
        name: Ident,
        /// Whether the source binding permits a writable borrow.
        mutable: bool,
    },
    Function {
        function: FunctionId,
    },
    Unwrap {
        value: Box<Term>,
    },
    Try {
        value: Box<Term>,
    },
    Match {
        value: Box<Term>,
        arms: Vec<MatchArm>,
    },
    If {
        cond: Box<Term>,
        then: Box<Term>,
        els: Box<Term>,
    },
    ParallelMap {
        input: Box<Term>,
        element: ParallelParameter,
        body: Box<Term>,
    },
    ParallelReduce {
        input: Box<Term>,
        identity: Box<Term>,
        left: ParallelParameter,
        right: ParallelParameter,
        body: Box<Term>,
    },
    While {
        cond: Box<Term>,
        body: Box<Term>,
    },
    Block {
        stmts: Vec<Statement>,
        tail: Box<Term>,
    },
    Record {
        /// Initializers remain in evaluation order; their indices select the completed layout.
        fields: Vec<RecordInitializer>,
    },
    Array {
        elems: Vec<Term>,
    },
    Builtin {
        name: Arc<str>,
        args: Vec<Term>,
    },
    Call {
        func: Box<Term>,
        args: Vec<Term>,
    },
    Intrinsic {
        op: Intrinsic,
        type_args: Vec<Ty>,
        args: Arguments,
    },
    Adapt {
        conversion: ReceiverConversion,
        arg: Box<Term>,
    },
    Convert {
        conversion: resin_types::ExplicitConversion,
        arg: Box<Term>,
    },
    /// Create an owning pipeline whose root type comes from its shader declarations.
    GpuPipelineCreate {
        factory: FunctionId,
        shaders: Vec<FunctionId>,
        args: Arguments,
    },
    /// Project the checked host arguments and record a dispatch or draw.
    GpuPipelineDispatch {
        projection: Option<resin_types::GpuProjectionPlan>,
        context: FunctionId,
        allocator: Option<FunctionId>,
        record: FunctionId,
        args: Arguments,
    },
    Absurd {
        arg: Box<Term>,
    },
    Assign {
        place: Box<Term>,
        value: Box<Term>,
    },
    Address {
        place: Box<Term>,
    },
    Borrow {
        place: Box<Term>,
    },
    Deref {
        pointer: Box<Term>,
    },
    /// The projection is resolved during HIR construction; names are not looked up again.
    Field {
        base: Box<Term>,
        access: FieldAccess,
    },
}

#[derive(Debug, Clone)]
pub(super) struct RecordInitializer {
    /// Specialization assigns every declaration index exactly once.
    pub(super) index: usize,
    pub(super) value: Term,
}

#[derive(Debug, Clone)]
pub(super) struct Arguments {
    pub(super) values: Vec<Term>,
    pub(super) params: Vec<Ty>,
}

#[derive(Debug, Clone)]
pub(super) struct ParallelParameter {
    pub binding: BindingId,
    pub name: Ident,
    pub ty: Ty,
}

#[derive(Debug, Clone)]
pub(super) struct MatchArm {
    /// Destructure an Err member and widen its payload to this binding type.
    pub(super) error_payload: Option<Ty>,
    pub(super) tag: Case,
    pub(super) binding: Option<BindingId>,
    pub(super) body: Term,
}

#[derive(Debug, Clone)]
pub(super) enum Statement {
    Define {
        binding: BindingId,
        name: Ident,
        init: Term,
    },
    Declare {
        binding: BindingId,
        name: Ident,
        ty: Ty,
    },
    Expr {
        term: Term,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReceiverConversion {
    Value,
    ReadOnly,
    Borrow,
    Load,
}

impl Term {
    /// Whether evaluating this tree always leaves the current source path.
    pub(super) fn exits(&self) -> bool {
        match &self.kind {
            TermKind::Return { .. } | TermKind::Break | TermKind::Continue => true,
            TermKind::Block { stmts, tail } => {
                stmts.iter().any(|stmt| match stmt {
                    Statement::Define { init, .. } => init.exits(),
                    Statement::Expr { term } => term.exits(),
                    Statement::Declare { .. } => false,
                }) || tail.exits()
            }
            TermKind::If { cond, then, els } => cond.exits() || (then.exits() && els.exits()),
            TermKind::Match { value, arms } => {
                value.exits() || arms.iter().all(|arm| arm.body.exits())
            }
            TermKind::While { cond, .. } => cond.exits(),
            TermKind::ParallelMap { input, .. } => input.exits(),
            TermKind::ParallelReduce {
                input, identity, ..
            } => input.exits() || identity.exits(),
            TermKind::Call { func, args } => func.exits() || args.iter().any(Term::exits),
            TermKind::Builtin { args, .. } | TermKind::Array { elems: args } => {
                args.iter().any(Term::exits)
            }
            TermKind::Intrinsic { args, .. } => args.values.iter().any(Term::exits),
            TermKind::Record { fields } => fields.iter().any(|field| field.value.exits()),
            TermKind::Assign { place, value } => place.exits() || value.exits(),
            TermKind::Unwrap { value } | TermKind::Try { value } => value.exits(),
            TermKind::Adapt { arg, .. }
            | TermKind::Convert { arg, .. }
            | TermKind::Absurd { arg } => arg.exits(),
            TermKind::Address { place } | TermKind::Borrow { place } => place.exits(),
            TermKind::Deref { pointer } => pointer.exits(),
            TermKind::Field { base, .. } => base.exits(),
            _ => false,
        }
    }
}
