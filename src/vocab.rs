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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Literal {
    Number(LiteralNumber),
    String(LiteralString),
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
pub struct LiteralBool {
    pub value: bool,
}
