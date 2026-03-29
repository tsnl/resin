//
// ScalarType
//

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
}

//
// Operator
//

#[derive(Debug)]
pub enum Operator {
    // Unary operators:
    Pos,
    Neg,
    Not,

    // Binary operators
    Matmul,
    Mul,
    Div,
    Mod,
    Add,
    Sub,
    Shl,
    Shr,
    And,
    Xor,
    Or,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}
impl From<Operator> for &'static str {
    fn from(op: Operator) -> Self {
        match op {
            Operator::Pos => "__pos__",
            Operator::Neg => "__neg__",
            Operator::Not => "__not__",
            Operator::Matmul => "__matmul__",
            Operator::Mul => "__mul__",
            Operator::Div => "__div__",
            Operator::Mod => "__mod__",
            Operator::Add => "__add__",
            Operator::Sub => "__sub__",
            Operator::Shl => "__shl__",
            Operator::Shr => "__shr__",
            Operator::Xor => "__xor__",
            Operator::And => "__and__",
            Operator::Or => "__or__",
            Operator::Lt => "__lt__",
            Operator::Le => "__le__",
            Operator::Gt => "__gt__",
            Operator::Ge => "__ge__",
            Operator::Eq => "__eq__",
            Operator::Ne => "__ne__",
        }
    }
}
