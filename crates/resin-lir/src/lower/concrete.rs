//! Concrete expressions consumed by storage lowering, one function at a time.
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

impl Signature {
    pub(super) fn parameter_type(&self) -> Ty {
        Ty::parameter(
            &self
                .params
                .iter()
                .map(|parameter| parameter.ty.clone())
                .collect::<Vec<_>>(),
        )
    }
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
    Constant {
        value: Value,
    },
    Local {
        binding: BindingId,
        name: Ident,
    },
    Function {
        function: FunctionId,
    },
    Shader {
        function: FunctionId,
        stage: Arc<str>,
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
    While {
        cond: Box<Term>,
        body: Box<Term>,
    },
    Block {
        stmts: Vec<Statement>,
        tail: Box<Term>,
    },
    Record {
        fields: Vec<(Ident, Term)>,
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
        arg: Box<Term>,
    },
    /// Evaluate a receiver, then unpack the remaining arguments in source order.
    Pack {
        args: Arguments,
    },
    Intrinsic {
        op: Intrinsic,
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
    ArcNew {
        value: Box<Term>,
    },
    /// Allocate and initialize a GPU element through the registered allocator.
    GpuNew {
        allocator: FunctionId,
        args: Arguments,
    },
    /// Allocate uninitialized GPU elements through the registered allocator.
    GpuAllocate {
        allocator: FunctionId,
        args: Arguments,
    },
    /// Create an owning pipeline whose root type comes from its shader declarations.
    GpuPipelineCreate {
        factory: FunctionId,
        shaders: Vec<FunctionId>,
        args: Arguments,
    },
    /// Project the checked host arguments and record a dispatch or draw.
    GpuPipelineDispatch {
        context: FunctionId,
        allocator: Option<FunctionId>,
        record: FunctionId,
        args: Arguments,
    },
    WeakEmpty {
        pointee: Ty,
    },
    Result {
        failure: bool,
        arg: Box<Term>,
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
pub(super) struct Arguments {
    pub(super) receiver: Option<Box<Term>>,
    pub(super) argument: Box<Term>,
    pub(super) params: Vec<Ty>,
}

#[derive(Debug, Clone)]
pub(super) struct MatchArm {
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
    Address,
    Load,
    ArcAddress,
    ArcLoad,
}
