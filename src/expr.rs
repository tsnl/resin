pub struct Expr {
    dtype: DType,
    shape: Vec<usize>,
}

pub enum Detail {}

pub enum DType {
    Fp16,
    Fp32,
}
