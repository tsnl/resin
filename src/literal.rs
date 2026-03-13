use num::BigRational;

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash, Clone, Debug)]
pub enum Literal {
    Unit,
    Bool(Box<BoolLiteral>),
    Number(Box<NumberLiteral>),
    String(Box<StringLiteral>),
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash, Clone, Debug)]
pub struct BoolLiteral {
    pub value: bool,
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash, Clone, Debug)]
pub struct NumberLiteral {
    pub value: BigRational,
    pub force_float: bool,
}
impl NumberLiteral {
    pub fn is_integral(&self) -> bool {
        self.value.is_integer() && !self.force_float
    }
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash, Clone, Debug)]
pub struct StringLiteral {
    pub value: String,
}
