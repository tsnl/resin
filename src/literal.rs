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
