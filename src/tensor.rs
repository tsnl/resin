use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, RangeBounds, Rem, Shl, Shr, Sub};

pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub detail: TensorDetail,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum DType {
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
}
pub enum TensorDetail {
    Constant(Box<[u8]>),
    Operator(Box<Operator>),
}
pub enum Operator {
    // Unary
    Neg(Tensor),
    Abs(Tensor),
    Exp(Tensor),
    Log(Tensor),
    BitNot(Tensor),

    // Matmul
    BatchMatmul(Tensor, Tensor),

    // Binary arithmetic
    Pow(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),
    Rem(Tensor, Tensor),
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),

    // Binary shift
    Shl(Tensor, Tensor),
    Shr(Tensor, Tensor),

    // Binary bitwise
    BitAnd(Tensor, Tensor),
    BitOr(Tensor, Tensor),
    BitXor(Tensor, Tensor),

    // Binary comparison:
    LessThan(Tensor, Tensor),
    LessOrEq(Tensor, Tensor),
    GreaterThan(Tensor, Tensor),
    GreaterOrEq(Tensor, Tensor),
    Eq(Tensor, Tensor),
    NotEq(Tensor, Tensor),

    // Shape manipulation
    Reshape(Tensor),
    Broadcast(Tensor),

    // Cast
    Into(Tensor, DType),
    View(Tensor, DType),
}

pub trait Scalar: bytemuck::Pod {
    fn dtype() -> DType;
}
impl Scalar for i32 {
    fn dtype() -> DType {
        DType::Int32
    }
}
impl Scalar for i64 {
    fn dtype() -> DType {
        DType::Int64
    }
}
impl Scalar for f32 {
    fn dtype() -> DType {
        DType::Float32
    }
}
impl Scalar for f64 {
    fn dtype() -> DType {
        DType::Float64
    }
}

impl<T: Scalar> From<&[T]> for Tensor {
    fn from(value: &[T]) -> Self {
        Tensor {
            dtype: T::dtype(),
            shape: vec![value.len() as u64],
            detail: TensorDetail::Constant(bytemuck::cast_slice(value).into()),
        }
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor {
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::Neg(self))),
        }
    }
}
impl Not for Tensor {
    type Output = Tensor;
    fn not(self) -> Self::Output {
        Tensor {
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::BitNot(self))),
        }
    }
}
impl Tensor {
    pub fn abs(self) -> Self {
        Tensor {
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::Abs(self))),
        }
    }
    pub fn exp(self) -> Self {
        Tensor {
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::Exp(self))),
        }
    }
    pub fn log(self) -> Self {
        Tensor {
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::Log(self))),
        }
    }
}

impl Tensor {
    pub fn bmm(self, other: Tensor) -> Self {
        assert_eq!(self.dtype, other.dtype);
        assert_eq!(self.shape.len(), 3);
        assert_eq!(other.shape.len(), 3);
        assert_eq!(self.shape[0], other.shape[0]);
        assert_eq!(self.shape[2], other.shape[1]);
        Tensor {
            dtype: self.dtype.clone(),
            shape: vec![self.shape[0], self.shape[1], other.shape[2]],
            detail: TensorDetail::Operator(Box::new(Operator::BatchMatmul(self, other))),
        }
    }
    pub fn matmul(self, other: Tensor) -> Self {
        let a_shape = self.shape.clone();
        let b_shape = other.shape.clone();
        assert_eq!(a_shape.len(), 2);
        assert_eq!(b_shape.len(), 2);
        let a = self.unsqueeze(0);
        let b = other.unsqueeze(0);
        let c = a.bmm(b);
        c.squeeze()
    }
}

impl Tensor {
    fn binary_op<F>(self, other: Tensor, op: F) -> Tensor
    where
        F: Fn(Tensor, Tensor) -> Operator,
    {
        assert_eq!(self.dtype, other.dtype);
        assert_eq!(self.shape, other.shape);
        Tensor {
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(op(self, other))),
        }
    }
}
impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::Mul)
    }
}
impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::Div)
    }
}
impl Rem for Tensor {
    type Output = Tensor;
    fn rem(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::Rem)
    }
}
impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::Add)
    }
}
impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::Sub)
    }
}
impl Shl for Tensor {
    type Output = Tensor;
    fn shl(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::Shl)
    }
}
impl Shr for Tensor {
    type Output = Tensor;
    fn shr(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::Shr)
    }
}
impl BitAnd for Tensor {
    type Output = Tensor;
    fn bitand(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::BitAnd)
    }
}
impl BitOr for Tensor {
    type Output = Tensor;
    fn bitor(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::BitOr)
    }
}
impl BitXor for Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: Tensor) -> Self::Output {
        self.binary_op(rhs, Operator::BitXor)
    }
}

impl Tensor {
    pub fn pow(self, other: Tensor) -> Self {
        self.binary_op(other, Operator::Pow)
    }
    pub fn lt(self, other: Tensor) -> Self {
        self.binary_op(other, Operator::LessThan)
    }
    pub fn le(self, other: Tensor) -> Self {
        self.binary_op(other, Operator::LessOrEq)
    }
    pub fn gt(self, other: Tensor) -> Self {
        self.binary_op(other, Operator::GreaterThan)
    }
    pub fn ge(self, other: Tensor) -> Self {
        self.binary_op(other, Operator::GreaterOrEq)
    }
    pub fn eq(self, other: Tensor) -> Self {
        self.binary_op(other, Operator::Eq)
    }
    pub fn ne(self, other: Tensor) -> Self {
        self.binary_op(other, Operator::NotEq)
    }
}

impl Tensor {
    pub fn reshape(self, shape: &[u64]) -> Self {
        let old_size: u64 = self.shape.iter().product();
        let new_size: u64 = shape.iter().product();
        assert_eq!(old_size, new_size, "Reshape size mismatch");
        Tensor {
            dtype: self.dtype.clone(),
            shape: shape.to_vec(),
            detail: TensorDetail::Operator(Box::new(Operator::Reshape(self))),
        }
    }
    pub fn broadcast(self, shape: &[u64]) -> Self {
        assert_eq!(self.shape.len(), shape.len());
        for (i, &dim) in self.shape.iter().enumerate() {
            assert!(dim == 1 || dim == shape[i]);
        }
        Tensor {
            dtype: self.dtype.clone(),
            shape: shape.to_vec(),
            detail: TensorDetail::Operator(Box::new(Operator::Broadcast(self))),
        }
    }
    pub fn into(self, dtype: DType) -> Self {
        Tensor {
            dtype,
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::Into(self, dtype))),
        }
    }
    pub fn view(self, dtype: DType) -> Self {
        Tensor {
            dtype,
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::View(self, dtype))),
        }
    }
    pub fn squeeze(self) -> Self {
        let new_shape: Vec<_> = self.shape.iter().cloned().filter(|&d| d != 1).collect();
        self.reshape(&new_shape)
    }
    pub fn unsqueeze(self, dim: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape)
    }
}
