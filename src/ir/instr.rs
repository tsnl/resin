//! Instructions for the typed stack IR.

use std::sync::Arc;

use crate::util::define_id;

use super::{FunctionId, LocalId, Ty, Value};

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
pub struct Foreign {
    pub header: Arc<str>,
    pub params: Vec<Ty>,
}

impl Foreign {
    pub(crate) fn valid(&self, result: &Ty) -> bool {
        self.params.iter().all(Ty::foreign_value)
            && (*result == Ty::Unit || result.foreign_value())
            && !self.header.is_empty()
            && self
                .header
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_./- :~()".contains(&c))
    }
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
    SetLocal {
        local: LocalId,
    },
    MakeVariant {
        ty: Ty,
        tag: u32,
    },
    VariantTag,
    VariantPayload {
        tag: u32,
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
            Self::Shader { .. }
            | Self::Push { .. }
            | Self::Function { .. }
            | Self::LocalAddress { .. } => StackEffect { pops: 0, pushes: 1 },
            Self::AccessStatic { .. }
            | Self::MakeVariant { .. }
            | Self::VariantTag
            | Self::VariantPayload { .. }
            | Self::Widen { .. }
            | Self::Load
            | Self::Ascribe { .. }
            | Self::Eliminate { .. }
            | Self::NumericCast { .. }
            | Self::PointerCast { .. } => StackEffect { pops: 1, pushes: 1 },
            Self::AccessDynamic | Self::Store => StackEffect { pops: 2, pushes: 1 },
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
