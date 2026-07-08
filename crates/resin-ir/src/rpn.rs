use resin_core::ElementOperator;
use serde::{Deserialize, Serialize};

/// Reverse-polish expression over elementwise operand indices and operators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementRpnExpr {
    pub atoms: Vec<RpnAtom>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RpnAtom {
    /// Index into the kernel argument list.
    Arg(u32),
    Op(ElementOperator),
}