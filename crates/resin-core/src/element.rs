//! Element types and operators (runtime values; not type parameters).

use serde::{Deserialize, Serialize};

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

/// Flat set of elementwise operators. Grouped by arity / role in comments only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementOperator {
    // unary
    Relu,
    Neg,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Not,
    Floor,
    Ceil,
    /// Reinterpret bits. **Target type is the owning [`Node`]'s `element_type`**
    /// (output buffer); source type is `args[0].element_type()`. Same byte width
    /// is required (enforced when building the node / lowering).
    Bitcast,

    // binary arithmetic (incl. associative)
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,

    // binary comparison (result is 0/1 in the output element type)
    Eq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,

    // binary bitwise
    Band,
    Bor,
    Bxor,
    Shl,
    Shr,
}

impl ElementOperator {
    pub fn arity(self) -> usize {
        match self {
            Self::Relu
            | Self::Neg
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Sin
            | Self::Cos
            | Self::Not
            | Self::Floor
            | Self::Ceil
            | Self::Bitcast => 1,
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Pow
            | Self::Max
            | Self::Min
            | Self::Eq
            | Self::Ne
            | Self::Gt
            | Self::Lt
            | Self::Ge
            | Self::Le
            | Self::Band
            | Self::Bor
            | Self::Bxor
            | Self::Shl
            | Self::Shr => 2,
        }
    }

    /// Operators usable as reduction / scatter-accumulate combiners.
    pub fn is_binary_assoc(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Max | Self::Min)
    }
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

    pub fn from_kind_and_nbytes(kind: ElementKind, nbytes: u32) -> Result<Self, String> {
        match (kind, nbytes) {
            (ElementKind::Float, 4) => Ok(ElementType::F4),
            (ElementKind::Float, 2) => Ok(ElementType::F2),
            (ElementKind::Uint, 4) => Ok(ElementType::U4),
            _ => Err(format!(
                "unsupported element type with kind={kind:?} and nbytes={nbytes}"
            )),
        }
    }

    pub fn join(self, other: Self) -> Result<Self, String> {
        let kind = self.kind().join(other.kind())?;
        let nbytes = self.nbytes().max(other.nbytes());
        Self::from_kind_and_nbytes(kind, nbytes)
    }

    pub fn wgsl_name(self) -> &'static str {
        match self {
            ElementType::F4 => "f32",
            ElementType::F2 => "f16",
            ElementType::U4 => "u32",
        }
    }

    pub fn needs_enable_f16(self) -> bool {
        matches!(self, ElementType::F2)
    }
}

impl ElementKind {
    pub fn join(self, other: Self) -> Result<Self, String> {
        if self != other {
            return Err(format!(
                "cannot join different kinds' element types: {self:?} and {other:?}"
            ));
        }
        Ok(self)
    }
}
