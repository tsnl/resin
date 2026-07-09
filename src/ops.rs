//! Scalar operators shared by the DSL graph and the IR.
//!
//! Everything operates on f32. The nesting exists so reductions can demand an
//! associative operator ([`AssocOp`]) at the type level.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Relu,
    Abs,
    Sqrt,
    Sin,
    Cos,
}

/// Associative binary operators (usable as reduction folds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssocOp {
    Add,
    Mul,
    Max,
    Min,
}

impl AssocOp {
    /// Fold identity: `apply(identity, x) == x`.
    pub fn identity(self) -> f32 {
        match self {
            AssocOp::Add => 0.0,
            AssocOp::Mul => 1.0,
            AssocOp::Max => f32::NEG_INFINITY,
            AssocOp::Min => f32::INFINITY,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Assoc(AssocOp),
    Sub,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Unary(UnaryOp),
    Binary(BinaryOp),
}

/// Shorthands for the deeply nested variants.
impl Op {
    pub const NEG: Op = Op::Unary(UnaryOp::Neg);
    pub const EXP: Op = Op::Unary(UnaryOp::Exp);
    pub const LOG: Op = Op::Unary(UnaryOp::Log);
    pub const RELU: Op = Op::Unary(UnaryOp::Relu);
    pub const ABS: Op = Op::Unary(UnaryOp::Abs);
    pub const SQRT: Op = Op::Unary(UnaryOp::Sqrt);
    pub const SIN: Op = Op::Unary(UnaryOp::Sin);
    pub const COS: Op = Op::Unary(UnaryOp::Cos);
    pub const ADD: Op = Op::Binary(BinaryOp::Assoc(AssocOp::Add));
    pub const MUL: Op = Op::Binary(BinaryOp::Assoc(AssocOp::Mul));
    pub const MAX: Op = Op::Binary(BinaryOp::Assoc(AssocOp::Max));
    pub const MIN: Op = Op::Binary(BinaryOp::Assoc(AssocOp::Min));
    pub const SUB: Op = Op::Binary(BinaryOp::Sub);
    pub const DIV: Op = Op::Binary(BinaryOp::Div);
    pub const POW: Op = Op::Binary(BinaryOp::Pow);

    /// Number of operands the operator pops.
    pub fn arity(self) -> usize {
        match self {
            Op::Unary(_) => 1,
            Op::Binary(_) => 2,
        }
    }

    /// Apply on host f32 values (the reference semantics for all backends).
    pub fn apply(self, args: &[f32]) -> f32 {
        match self {
            Op::Unary(op) => {
                let x = args[0];
                match op {
                    UnaryOp::Neg => -x,
                    UnaryOp::Exp => x.exp(),
                    UnaryOp::Log => x.ln(),
                    UnaryOp::Relu => x.max(0.0),
                    UnaryOp::Abs => x.abs(),
                    UnaryOp::Sqrt => x.sqrt(),
                    UnaryOp::Sin => x.sin(),
                    UnaryOp::Cos => x.cos(),
                }
            }
            Op::Binary(op) => {
                let (a, b) = (args[0], args[1]);
                match op {
                    BinaryOp::Assoc(op) => op.apply(a, b),
                    BinaryOp::Sub => a - b,
                    BinaryOp::Div => a / b,
                    BinaryOp::Pow => a.powf(b),
                }
            }
        }
    }
}

impl AssocOp {
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            AssocOp::Add => a + b,
            AssocOp::Mul => a * b,
            AssocOp::Max => a.max(b),
            AssocOp::Min => a.min(b),
        }
    }
}
