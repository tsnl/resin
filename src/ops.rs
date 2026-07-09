//! Scalar operators and element types shared by the DSL graph and the IR.
//!
//! f32 is the default story; u32 exists so integer indexing and masks can be
//! first-class (gather indices, bit packing, compare→mask).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    F32,
    U32,
}

impl ElementType {
    pub fn nbytes(self) -> usize {
        4
    }

    pub fn name(self) -> &'static str {
        match self {
            ElementType::F32 => "f32",
            ElementType::U32 => "u32",
        }
    }
}

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
    Floor,
    Ceil,
    /// Reinterpret bits as `to` (same width).
    Bitcast { to: ElementType },
    /// Numeric conversion to `to` (f32→u32 truncates toward zero).
    Cast { to: ElementType },
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
    pub fn identity_f32(self) -> f32 {
        match self {
            AssocOp::Add => 0.0,
            AssocOp::Mul => 1.0,
            AssocOp::Max => f32::NEG_INFINITY,
            AssocOp::Min => f32::INFINITY,
        }
    }

    pub fn identity_u32(self) -> u32 {
        match self {
            AssocOp::Add => 0,
            AssocOp::Mul => 1,
            AssocOp::Max => u32::MIN,
            AssocOp::Min => u32::MAX,
        }
    }

    /// Backward-compat alias used by f32-only call sites.
    pub fn identity(self) -> f32 {
        self.identity_f32()
    }

    pub fn apply_f32(self, a: f32, b: f32) -> f32 {
        match self {
            AssocOp::Add => a + b,
            AssocOp::Mul => a * b,
            AssocOp::Max => a.max(b),
            AssocOp::Min => a.min(b),
        }
    }

    pub fn apply_u32(self, a: u32, b: u32) -> u32 {
        match self {
            AssocOp::Add => a.wrapping_add(b),
            AssocOp::Mul => a.wrapping_mul(b),
            AssocOp::Max => a.max(b),
            AssocOp::Min => a.min(b),
        }
    }

    pub fn apply(self, a: f32, b: f32) -> f32 {
        self.apply_f32(a, b)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Assoc(AssocOp),
    Sub,
    Div,
    Pow,
    /// Comparisons yield a 0/1 mask in the operand element type.
    CmpEq,
    CmpNe,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,
    /// Bitwise (u32 only).
    Band,
    Bor,
    Bxor,
    Shl,
    Shr,
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
    pub const FLOOR: Op = Op::Unary(UnaryOp::Floor);
    pub const CEIL: Op = Op::Unary(UnaryOp::Ceil);
    pub const ADD: Op = Op::Binary(BinaryOp::Assoc(AssocOp::Add));
    pub const MUL: Op = Op::Binary(BinaryOp::Assoc(AssocOp::Mul));
    pub const MAX: Op = Op::Binary(BinaryOp::Assoc(AssocOp::Max));
    pub const MIN: Op = Op::Binary(BinaryOp::Assoc(AssocOp::Min));
    pub const SUB: Op = Op::Binary(BinaryOp::Sub);
    pub const DIV: Op = Op::Binary(BinaryOp::Div);
    pub const POW: Op = Op::Binary(BinaryOp::Pow);
    pub const CMP_EQ: Op = Op::Binary(BinaryOp::CmpEq);
    pub const CMP_NE: Op = Op::Binary(BinaryOp::CmpNe);
    pub const CMP_LT: Op = Op::Binary(BinaryOp::CmpLt);
    pub const CMP_LE: Op = Op::Binary(BinaryOp::CmpLe);
    pub const CMP_GT: Op = Op::Binary(BinaryOp::CmpGt);
    pub const CMP_GE: Op = Op::Binary(BinaryOp::CmpGe);
    pub const BAND: Op = Op::Binary(BinaryOp::Band);
    pub const BOR: Op = Op::Binary(BinaryOp::Bor);
    pub const BXOR: Op = Op::Binary(BinaryOp::Bxor);
    pub const SHL: Op = Op::Binary(BinaryOp::Shl);
    pub const SHR: Op = Op::Binary(BinaryOp::Shr);

    pub fn cast(to: ElementType) -> Op {
        Op::Unary(UnaryOp::Cast { to })
    }

    pub fn bitcast(to: ElementType) -> Op {
        Op::Unary(UnaryOp::Bitcast { to })
    }

    /// Number of operands the operator pops.
    pub fn arity(self) -> usize {
        match self {
            Op::Unary(_) => 1,
            Op::Binary(_) => 2,
        }
    }

    /// Whether this op changes the element type of its operand (fusion barrier).
    pub fn changes_element_type(self) -> bool {
        matches!(
            self,
            Op::Unary(UnaryOp::Cast { .. } | UnaryOp::Bitcast { .. })
        )
    }

    /// Apply on host f32 values (the reference semantics for f32 kernels).
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
                    UnaryOp::Floor => x.floor(),
                    UnaryOp::Ceil => x.ceil(),
                    // Same-type cast/bitcast is identity at the f32 layer;
                    // cross-type kernels go through the typed interpreter.
                    UnaryOp::Bitcast { to: ElementType::F32 }
                    | UnaryOp::Cast { to: ElementType::F32 } => x,
                    UnaryOp::Bitcast { to: ElementType::U32 }
                    | UnaryOp::Cast { to: ElementType::U32 } => {
                        panic!("f32 apply cannot produce u32; use typed eval")
                    }
                }
            }
            Op::Binary(op) => {
                let (a, b) = (args[0], args[1]);
                match op {
                    BinaryOp::Assoc(op) => op.apply_f32(a, b),
                    BinaryOp::Sub => a - b,
                    BinaryOp::Div => a / b,
                    BinaryOp::Pow => a.powf(b),
                    BinaryOp::CmpEq => (a == b) as u32 as f32,
                    BinaryOp::CmpNe => (a != b) as u32 as f32,
                    BinaryOp::CmpLt => (a < b) as u32 as f32,
                    BinaryOp::CmpLe => (a <= b) as u32 as f32,
                    BinaryOp::CmpGt => (a > b) as u32 as f32,
                    BinaryOp::CmpGe => (a >= b) as u32 as f32,
                    BinaryOp::Band
                    | BinaryOp::Bor
                    | BinaryOp::Bxor
                    | BinaryOp::Shl
                    | BinaryOp::Shr => panic!("bitwise ops are u32-only"),
                }
            }
        }
    }
}
