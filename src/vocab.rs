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
