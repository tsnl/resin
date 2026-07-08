//! Element types and operators (runtime values; not type parameters).

use serde::{Deserialize, Serialize};

/// Errors from element-type operations.
#[derive(Debug, thiserror::Error)]
pub enum ElementTypeError {
    #[error("unsupported element type with kind={kind:?} and nbytes={nbytes}")]
    Unsupported { kind: ElementKind, nbytes: u32 },

    #[error("cannot join element types of different kinds: {lhs:?} and {rhs:?}")]
    KindMismatch { lhs: ElementKind, rhs: ElementKind },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementType {
    F4,
    F2,
    U4,
}

pub const F4: ElementType = ElementType::F4;
pub const F2: ElementType = ElementType::F2;
pub const U4: ElementType = ElementType::U4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementKind {
    Float,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryElementOperator {
    Neg,
    Exp,
    Log,
    Relu,
    Abs,
    Sqrt,
    Sin,
    Cos,
    Not,
    Floor,
    Ceil,
    /// Reinterpret the operand's bits as the kernel output element type.
    Bitcast,
    /// Numeric value conversion to the kernel output element type
    /// (f32 → u32 truncates toward zero; u32 → f32 rounds to nearest).
    Convert,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryAssocElementOperator {
    Mul,
    Add,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryElementOperator {
    Pow,
    Div,
    Sub,
    Assoc(BinaryAssocElementOperator),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryCompareOperator {
    Eq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryBitwiseOperator {
    Band,
    Bor,
    Bxor,
    Shl,
    Shr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementOperator {
    Unary(UnaryElementOperator),
    Binary(BinaryElementOperator),
    Compare(BinaryCompareOperator),
    Bitwise(BinaryBitwiseOperator),
}

impl ElementType {
    pub fn nbytes(self) -> u32 {
        match self {
            ElementType::F4 | ElementType::U4 => 4,
            ElementType::F2 => 2,
        }
    }

    pub fn kind(self) -> ElementKind {
        match self {
            ElementType::F4 | ElementType::F2 => ElementKind::Float,
            ElementType::U4 => ElementKind::Uint,
        }
    }
}

pub fn element_type_nbytes(element_type: ElementType) -> u32 {
    element_type.nbytes()
}

pub fn element_type_kind(element_type: ElementType) -> ElementKind {
    element_type.kind()
}

pub fn element_type(
    kind: ElementKind,
    nbytes: u32,
) -> Result<ElementType, ElementTypeError> {
    match (kind, nbytes) {
        (ElementKind::Float, 4) => Ok(ElementType::F4),
        (ElementKind::Float, 2) => Ok(ElementType::F2),
        (ElementKind::Uint, 4) => Ok(ElementType::U4),
        _ => Err(ElementTypeError::Unsupported { kind, nbytes }),
    }
}

pub fn element_type_join_kind(
    a: ElementKind,
    b: ElementKind,
) -> Result<ElementKind, ElementTypeError> {
    if a != b {
        return Err(ElementTypeError::KindMismatch { lhs: a, rhs: b });
    }
    Ok(a)
}

pub fn element_type_join(
    a: ElementType,
    b: ElementType,
) -> Result<ElementType, ElementTypeError> {
    let kind = element_type_join_kind(a.kind(), b.kind())?;
    let nbytes = a.nbytes().max(b.nbytes());
    element_type(kind, nbytes)
}