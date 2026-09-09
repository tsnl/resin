//! Low-level language: typed stack instructions and explicit control flow.
//! [`generate`] translates HIR into this representation.
//! Construction and printing internals are private.
//!
//! ```compile_fail,E0603
//! use resin_lir::lower;
//! ```

use resin_common::prelude::*;
mod lower;
mod print;

use std::{collections::BTreeMap, sync::Arc};

define_id! {
    pub struct BlockId(usize);
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Option<Arc<str>>,
    pub foreign: Option<Foreign>,
    pub result: Ty,
    /// Local zero is always the parameter (unit, a single value, or a tuple).
    /// Every function, including a foreign declaration, must have this slot.
    pub locals: Vec<Local>,
    pub entry: BlockId,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    pub name: Option<Arc<str>>,
    pub instrs: Vec<Instr>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub name: Option<Arc<str>>,
    pub ty: Ty,
}

impl Function {
    pub fn ty(&self) -> Option<Ty> {
        Some(Ty::Function {
            param: Box::new(self.locals.first()?.ty.clone()),
            result: Box::new(self.result.clone()),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    /// Transfer a prepared payload directly into a new shared allocation.
    ArcNew,
    /// Internal field transfer during parameter destructuring.
    TransferLoad,
    ForgetLocal {
        local: LocalId,
    },
    /// Borrow a payload address; the generator retains the owner for the access.
    ArcData,
    Downgrade,
    Upgrade,
    WeakEmpty {
        pointee: Ty,
    },
    /// Transfer a compiler temporary without introducing a source-level move.
    TakeLocal {
        local: LocalId,
    },
    /// Destroy an initialized local and clear its initialization flag.
    DropLocal {
        local: LocalId,
    },
    SetLocal {
        local: LocalId,
    },
    MakeVariant {
        ty: Ty,
        tag: Case,
    },
    IsVariant {
        tag: Case,
    },
    /// Exclude None from a value; trap if that is the active case.
    ExcludeNone,
    VariantPayload {
        tag: Case,
    },
    Widen {
        ty: Ty,
    },
    Shader {
        function: FunctionId,
        stage: Arc<str>,
    },
    PointerCast {
        ty: Ty,
    },
    Push {
        value: Value,
    },

    LocalAddress {
        local: LocalId,
    },

    /// Project a type-directed child from an aggregate value or address.
    AccessStatic {
        index: usize,
    },

    /// `[aggregate or address, index] -> [element or address]`.
    AccessDynamic,

    Load,

    /// `[address, value] -> [value]`: preserves the assigned value.
    Store,

    /// Exchange a live pointee with an owned replacement, returning the old value.
    Replace,

    /// `ty` must match the top value's type or differ by exactly one nominal layer.
    Ascribe {
        ty: Ty,
    },

    /// Explicit numeric conversion, trapping if an integer result is out of range.
    NumericCast {
        ty: Ty,
    },

    /// Consume Never and never return. `result` checks the unreachable continuation;
    /// it is not a runtime value. Backends terminate the block at this instruction.
    Eliminate {
        result: Ty,
    },

    Discard,

    /// Consume one value per field and construct a record in declaration order.
    MakeRecord {
        fields: Vec<Arc<str>>,
    },

    /// Consume elements in order; the explicit element type permits empty arrays.
    MakeArray {
        elements: usize,
        element: Ty,
    },

    Function {
        function: FunctionId,
    },

    /// Indirectly call the function preceding one argument value (possibly unit or a tuple).
    Call,

    /// Invoke a privileged builtin with its checked monomorphic signature.
    CallBuiltin {
        name: Arc<str>,
        params: Vec<Ty>,
        result: Ty,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// Transfer the current operand stack to another block unchanged.
    Break { target: BlockId },

    /// Consume a boolean and transfer the remaining stack to one target.
    Branch { then: BlockId, els: BlockId },

    /// Return the sole value on the operand stack from the function.
    Return,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackEffect {
    pub pops: usize,
    pub pushes: usize,
}

impl Instr {
    pub fn stack_effect(&self) -> StackEffect {
        match self {
            Self::WeakEmpty { .. }
            | Self::TakeLocal { .. }
            | Self::Shader { .. }
            | Self::Push { .. }
            | Self::Function { .. }
            | Self::LocalAddress { .. } => StackEffect { pops: 0, pushes: 1 },
            Self::TransferLoad
            | Self::ArcNew
            | Self::ArcData
            | Self::Downgrade
            | Self::Upgrade
            | Self::AccessStatic { .. }
            | Self::MakeVariant { .. }
            | Self::ExcludeNone
            | Self::IsVariant { .. }
            | Self::VariantPayload { .. }
            | Self::Widen { .. }
            | Self::Load
            | Self::Ascribe { .. }
            | Self::Eliminate { .. }
            | Self::NumericCast { .. }
            | Self::PointerCast { .. } => StackEffect { pops: 1, pushes: 1 },
            Self::AccessDynamic | Self::Store | Self::Replace => StackEffect { pops: 2, pushes: 1 },
            Self::ForgetLocal { .. } | Self::DropLocal { .. } => StackEffect { pops: 0, pushes: 0 },
            Self::Discard | Self::SetLocal { .. } => StackEffect { pops: 1, pushes: 0 },
            Self::MakeRecord { fields } => StackEffect {
                pops: fields.len(),
                pushes: 1,
            },
            Self::MakeArray { elements, .. } => StackEffect {
                pops: *elements,
                pushes: 1,
            },
            Self::Call => StackEffect { pops: 2, pushes: 1 },
            Self::CallBuiltin { params, .. } => StackEffect {
                pops: params.len(),
                pushes: 1,
            },
        }
    }
}

impl Terminator {
    pub const fn stack_effect(&self) -> StackEffect {
        match self {
            Self::Break { .. } => StackEffect { pops: 0, pushes: 0 },
            Self::Branch { .. } | Self::Return => StackEffect { pops: 1, pushes: 0 },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Module {
    /// Functions exported by the entry source file.
    pub entries: BTreeMap<Arc<str>, FunctionId>,
    /// Canonical nominal and structural definitions, indexed by [`TypeId`].
    pub types: TypeTable,
    pub functions: Vec<Function>,
    /// Decorated shader candidates and whether their static artifact is requested.
    pub shaders: BTreeMap<FunctionId, ShaderEntry>,
    /// Optional source origins; direct IR clients may leave this empty.
    pub origins: SourceMap,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SourceMap {
    pub sources: BTreeMap<std::path::PathBuf, Arc<str>>,
    pub functions: BTreeMap<FunctionId, SourceLocation>,
    pub instructions: BTreeMap<(FunctionId, BlockId, usize), SourceLocation>,
}

/// Lower a self-contained HIR module. Verification is a separate pass.
#[derive(Debug)]
pub struct Error {
    pub function: FunctionId,
    pub error: GenerateError,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for Error {}

/// Lower a typed tree into storage and control flow; verification is a separate pass.
pub fn generate(source: &resin_hir::Module) -> Result<Module, Error> {
    lower::generate(source)
}

/// Collect independent lowering errors across functions.
pub fn analyze(source: &resin_hir::Module) -> Result<Module, Vec<Error>> {
    lower::analyze(source)
}

pub fn format_module(module: &Module) -> String {
    print::format_module(module)
}
