//! Instructions for the typed stack IR.

use std::sync::Arc;

use super::{BlockId, FunctionId, GlobalId, LocalId, NonLocalId, Ty, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Option<Arc<str>>,

    /// Captured values in closure-display order.
    pub nonlocals: Vec<NonLocal>,

    /// Parameters in call order, installed into these locals before entry.
    pub params: Vec<LocalId>,

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
            params: self
                .params
                .iter()
                .map(|param| self.locals.get(param.index()).map(|local| local.ty.clone()))
                .collect::<Option<_>>()?,
            result: Box::new(self.result.clone()),
        })
    }
}

/// A typed stack-machine instruction.
///
/// Instructions are deliberately anonymous. Their results are pushed onto the
/// operand stack and are consumed by later instructions or stored in locals.
#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    /// Push a literal or compile-time-evaluated value.
    Push { value: Value },

    /// Push the address of a function-local allocation.
    LocalAddress { local: LocalId },

    /// Push the address of a module-level allocation.
    GlobalAddress { global: GlobalId },

    /// Push the address of an entry in the current closure display.
    NonLocalAddress { nonlocal: NonLocalId },

    /// Select one compile-time-known child from an aggregate value or address.
    /// The index is interpreted using the selected aggregate's type.
    AccessStatic { index: usize },

    /// Select an array element using a run-time index.
    ///
    /// This consumes the aggregate value or address followed by the index and
    /// pushes the selected value or address.
    AccessDynamic,

    /// Load a value through the address on top of the stack.
    Load,

    /// Store a value through an address, preserving the assigned value.
    ///
    /// Stack effect: `[address, value] -> [value]`.
    Store,

    /// Reinterpret the top of the stack as `ty`.
    ///
    /// Wraps or unwraps one nominal layer: `ty` is a nominal type whose
    /// defining body is the popped type, or the popped type is nominal and
    /// `ty` is its defining body. Identity ascriptions are omitted.
    Ascribe { ty: Ty },

    /// Discard the value on top of the stack.
    Discard,

    /// Consume one value per field and construct a record in declaration order.
    MakeRecord { fields: Vec<Arc<str>> },

    /// Consume `elements` values and construct an array in their original
    /// order. The element type is explicit so empty arrays remain typed.
    MakeArray { elements: usize, element: Ty },

    /// Consume `captures` values and construct a closure.
    MakeClosure {
        function: FunctionId,
        captures: usize,
    },

    /// Indirectly call the closure preceding `args` argument values.
    Call { args: usize },

    /// Invoke a privileged polymorphic builtin with its checked signature.
    /// Backend implementation selection remains deliberately deferred.
    CallBuiltin {
        name: Arc<str>,
        params: Vec<Ty>,
        result: Ty,
    },
}

/// The control-flow operation ending a basic block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// Transfer the current operand stack to another block unchanged.
    Break { target: BlockId },

    /// Consume a boolean and transfer the remaining stack to one target.
    Branch { then: BlockId, els: BlockId },

    /// Return the sole value on the operand stack from the function.
    Return,
}
