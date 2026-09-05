//! Instructions for the typed stack IR.

use std::sync::Arc;

use super::{BlockId, FunctionId, GlobalId, LocalId, NonLocalId, Ty, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Option<Arc<str>>,
    pub nonlocals: Vec<NonLocal>,
    pub param: LocalId,
    pub result: Ty,
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

/// An entry in a function's closure display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonLocal {
    pub name: Option<Arc<str>>,
    pub ty: Ty,
}

impl Function {
    pub fn ty(&self) -> Option<Ty> {
        Some(Ty::Function {
            param: Box::new(self.locals.get(self.param.index())?.ty.clone()),
            result: Box::new(self.result.clone()),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    Push {
        value: Value,
    },

    /// Push the currently executing closure, including its captured display.
    /// A local function's recursive name refers to this value, not its uninitialized destination.
    CurrentClosure,

    LocalAddress {
        local: LocalId,
    },

    GlobalAddress {
        global: GlobalId,
    },

    NonLocalAddress {
        nonlocal: NonLocalId,
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

    /// Consume `captures` values and construct a closure.
    MakeClosure {
        function: FunctionId,
        captures: usize,
    },

    /// Indirectly call the closure preceding one argument value (possibly unit or a tuple).
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
