//! Element types and operators (runtime values; not type parameters).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    F4,
    F2,
    U4,
}

pub const F4: ElementType = ElementType::F4;
pub const F2: ElementType = ElementType::F2;
pub const U4: ElementType = ElementType::U4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementKind {
    Float,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryElementOperator {
    Neg,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Not,
    Floor,
    Ceil,
    Bitcast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryAssocElementOperator {
    Mul,
    Add,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryElementOperator {
    Pow,
    Div,
    Sub,
    Assoc(BinaryAssocElementOperator),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryCompareOperator {
    Eq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryBitwiseOperator {
    Band,
    Bor,
    Bxor,
    Shl,
    Shr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

pub fn etype_nbytes(etype: ElementType) -> u32 {
    etype.nbytes()
}

pub fn etype_kind(etype: ElementType) -> ElementKind {
    etype.kind()
}

pub fn etype(kind: ElementKind, nbytes: u32) -> Result<ElementType, String> {
    match (kind, nbytes) {
        (ElementKind::Float, 4) => Ok(ElementType::F4),
        (ElementKind::Float, 2) => Ok(ElementType::F2),
        (ElementKind::Uint, 4) => Ok(ElementType::U4),
        _ => Err(format!(
            "unsupported etype with kind={kind:?} and nbytes={nbytes}"
        )),
    }
}

pub fn etype_join_kind(a: ElementKind, b: ElementKind) -> Result<ElementKind, String> {
    if a != b {
        return Err(format!("cannot join different kinds' etypes: {a:?} and {b:?}"));
    }
    Ok(a)
}

pub fn etype_join(a: ElementType, b: ElementType) -> Result<ElementType, String> {
    let kind = etype_join_kind(a.kind(), b.kind())?;
    let nbytes = a.nbytes().max(b.nbytes());
    etype(kind, nbytes)
}
