use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

#[derive(Debug)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub detail: TensorDetail,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum DType {
    Float16,
    Float32,
}

#[derive(Debug)]
pub enum TensorDetail {
    Constant(Box<[f32]>),
    Operator(Box<Operator>),
}

#[derive(Debug)]
pub enum Operator {
    // Unary
    Neg(Tensor),
    Abs(Tensor),
    Exp(Tensor),
    Log(Tensor),

    // Matmul
    Bmm(Tensor, Tensor),

    // Binary arithmetic
    Pow(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),
    Rem(Tensor, Tensor),
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),

    // Shape manipulation
    Reshape(Tensor),
}

pub trait Scalar: bytemuck::Pod {
    fn dtype() -> DType;
}
impl Scalar for half::f16 {
    fn dtype() -> DType {
        DType::Float16
    }
}
impl Scalar for f32 {
    fn dtype() -> DType {
        DType::Float32
    }
}

pub fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Float16 => 2,
        DType::Float32 => 4,
    }
}

impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(value: [f32; N]) -> Self {
        let dtype = DType::Float32;
        let shape = vec![N];
        Tensor {
            dtype,
            shape,
            detail: TensorDetail::Constant(Box::from(value)),
        }
    }
}
impl<const M: usize, const N: usize> From<[[f32; N]; M]> for Tensor {
    fn from(value: [[f32; N]; M]) -> Self {
        let dtype = DType::Float32;
        let shape = vec![M, N];
        Tensor {
            dtype,
            shape,
            detail: TensorDetail::Constant(Box::from(value.as_flattened())),
        }
    }
}
impl<const L: usize, const M: usize, const N: usize> From<[[[f32; N]; M]; L]> for Tensor {
    fn from(value: [[[f32; N]; M]; L]) -> Self {
        let dtype = DType::Float32;
        let shape = vec![L, M, N];
        Tensor {
            dtype,
            shape,
            detail: TensorDetail::Constant(Box::from({
                let mut res = Vec::with_capacity(L * M * N);
                for i in 0..L {
                    for j in 0..M {
                        for k in 0..N {
                            res.push(value[i][j][k]);
                        }
                    }
                }
                res
            })),
        }
    }
}

impl Tensor {
    pub fn cast<S: Scalar>(self) -> Self {
        Self {
            dtype: S::dtype(),
            ..self
        }
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::Neg(self))),
        }
    }
}
impl Tensor {
    pub fn abs(self) -> Self {
        Tensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::Abs(self))),
        }
    }
    pub fn exp(self) -> Self {
        Tensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            detail: TensorDetail::Operator(Box::new(Operator::Exp(self))),
        }
    }
    pub fn log(self) -> Self {
        Tensor {
            dtype: self.dtype,
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
        let shape = vec![self.shape[0], self.shape[1], other.shape[2]];
        Tensor {
            dtype: self.dtype,
            shape,
            detail: TensorDetail::Operator(Box::new(Operator::Bmm(self, other))),
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
            dtype: self.dtype,
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
impl Tensor {
    pub fn pow(self, other: Tensor) -> Self {
        self.binary_op(other, Operator::Pow)
    }
}

impl Tensor {
    pub fn reshape(self, shape: impl Into<Vec<usize>>) -> Self {
        let new_shape: Vec<_> = shape.into();

        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(old_size, new_size, "Reshape size mismatch");

        Tensor {
            dtype: self.dtype,
            shape: new_shape,
            detail: TensorDetail::Operator(Box::new(Operator::Reshape(self))),
        }
    }
    pub fn squeeze(self) -> Self {
        let new_shape: Vec<_> = self.shape.iter().cloned().filter(|&d| d != 1).collect();
        self.reshape(new_shape)
    }
    pub fn unsqueeze(self, dim: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(new_shape)
    }
}
