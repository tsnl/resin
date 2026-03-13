use crate::Symbol;
use std::fmt::Display;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Literal {
    Number(LiteralNumber),
    String(LiteralString),
    Symbol(LiteralSymbol),
    Bool(LiteralBool),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralNumber {
    pub value: num::BigRational,
    pub force_float: bool,
}
impl LiteralNumber {
    pub fn is_integer(&self) -> bool {
        self.value.is_integer() && !self.force_float
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralString {
    pub content: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralSymbol {
    pub content: Symbol,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralBool {
    pub value: bool,
}
impl Literal {
    pub fn new_symbol(content: Symbol) -> Self {
        Self::Symbol(LiteralSymbol { content })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOperator {
    Neg,
    Not,
    Ref,
    Deref,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOperator {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Shl,
    Shr,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitXor,
    BitOr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Bool,
    String,
    Symbol,
    Unit,
}
impl Display for BuiltinType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuiltinType::U8 => f.write_str("u8"),
            BuiltinType::U16 => f.write_str("u16"),
            BuiltinType::U32 => f.write_str("u32"),
            BuiltinType::U64 => f.write_str("u64"),
            BuiltinType::I8 => f.write_str("i8"),
            BuiltinType::I16 => f.write_str("i16"),
            BuiltinType::I32 => f.write_str("i32"),
            BuiltinType::I64 => f.write_str("i64"),
            BuiltinType::F32 => f.write_str("f32"),
            BuiltinType::F64 => f.write_str("f64"),
            BuiltinType::Bool => f.write_str("bool"),
            BuiltinType::String => f.write_str("string"),
            BuiltinType::Symbol => f.write_str("symbol"),
            BuiltinType::Unit => f.write_str("unit"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BuiltinOperatorNames {
    // Unary operators
    pub uop_neg: Symbol,   // "-(_)"
    pub uop_not: Symbol,   // "!(_)"
    pub uop_ref: Symbol,   // "&(_)"
    pub uop_deref: Symbol, // "*(_)"

    // Binary operators
    pub bop_add: Symbol,     // "+(_,_)"
    pub bop_sub: Symbol,     // "-(_,_)"
    pub bop_mul: Symbol,     // "*(_,_)"
    pub bop_div: Symbol,     // "/(_,_)"
    pub bop_rem: Symbol,     // "%(_,_)"
    pub bop_eq: Symbol,      // "==(_,_)"
    pub bop_ne: Symbol,      // "!=(_,_)"
    pub bop_lt: Symbol,      // "<(_,_)"
    pub bop_le: Symbol,      // "<=(_,_)"
    pub bop_gt: Symbol,      // ">(_,_)"
    pub bop_ge: Symbol,      // ">=(_,_)"
    pub bop_shl: Symbol,     // "<<(_,_)"
    pub bop_shr: Symbol,     // ">>(_,_)"
    pub bop_bit_and: Symbol, // "&(_,_)"
    pub bop_bit_xor: Symbol, // "^(_,_)"
    pub bop_bit_or: Symbol,  // "|(_,_)"

    // Special type names
    pub ty_pointer: Symbol,  // "Ptr[_]"
    pub ty_tuple: Symbol,    // "Tuple[...]"
    pub ty_function: Symbol, // "Fn[_,...]"
    pub ty_slice: Symbol,    // "Slice[_]"
    pub ty_array: Symbol,    // "Array[_,_]"
}

impl BuiltinOperatorNames {
    pub fn new() -> Self {
        Self {
            // Unary operators
            uop_neg: Symbol::from("-(_)"),
            uop_not: Symbol::from("!(_)"),
            uop_ref: Symbol::from("&(_)"),
            uop_deref: Symbol::from("*(_)"),

            // Binary operators
            bop_add: Symbol::from("+(_,_)"),
            bop_sub: Symbol::from("-(_,_)"),
            bop_mul: Symbol::from("*(_,_)"),
            bop_div: Symbol::from("/(_,_)"),
            bop_rem: Symbol::from("%(_,_)"),
            bop_eq: Symbol::from("==(_,_)"),
            bop_ne: Symbol::from("!=(_,_)"),
            bop_lt: Symbol::from("<(_,_)"),
            bop_le: Symbol::from("<=(_,_)"),
            bop_gt: Symbol::from(">(_,_)"),
            bop_ge: Symbol::from(">=(_,_)"),
            bop_shl: Symbol::from("<<(_,_)"),
            bop_shr: Symbol::from(">>(_,_)"),
            bop_bit_and: Symbol::from("&(_,_)"),
            bop_bit_xor: Symbol::from("^(_,_)"),
            bop_bit_or: Symbol::from("|(_,_)"),

            // Special type names
            ty_pointer: Symbol::from("Ptr[_]"),
            ty_tuple: Symbol::from("Tuple[...]"),
            ty_function: Symbol::from("Fn[_,...]"),
            ty_slice: Symbol::from("Slice[_]"),
            ty_array: Symbol::from("Array[_,_]"),
        }
    }
    pub fn for_unary_operator(&self, op: UnaryOperator) -> Symbol {
        match op {
            UnaryOperator::Neg => self.uop_neg.clone(),
            UnaryOperator::Not => self.uop_not.clone(),
            UnaryOperator::Ref => self.uop_ref.clone(),
            UnaryOperator::Deref => self.uop_deref.clone(),
        }
    }
    pub fn for_binary_operator(&self, op: BinaryOperator) -> Symbol {
        match op {
            BinaryOperator::Add => self.bop_add.clone(),
            BinaryOperator::Sub => self.bop_sub.clone(),
            BinaryOperator::Mul => self.bop_mul.clone(),
            BinaryOperator::Div => self.bop_div.clone(),
            BinaryOperator::Rem => self.bop_rem.clone(),
            BinaryOperator::Eq => self.bop_eq.clone(),
            BinaryOperator::Ne => self.bop_ne.clone(),
            BinaryOperator::Lt => self.bop_lt.clone(),
            BinaryOperator::Le => self.bop_le.clone(),
            BinaryOperator::Gt => self.bop_gt.clone(),
            BinaryOperator::Ge => self.bop_ge.clone(),
            BinaryOperator::Shl => self.bop_shl.clone(),
            BinaryOperator::Shr => self.bop_shr.clone(),
            BinaryOperator::BitAnd => self.bop_bit_and.clone(),
            BinaryOperator::BitXor => self.bop_bit_xor.clone(),
            BinaryOperator::BitOr => self.bop_bit_or.clone(),
        }
    }
}
