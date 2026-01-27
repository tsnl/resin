use std::fmt;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub strides: Vec<u64>,
    pub detail: TensorDetail,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum DType {
    Int32,
    Float16,
    Float32,
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
impl Scalar for f32 {
    fn dtype() -> DType {
        DType::Float32
    }
}

pub fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Int32 => 4,
        DType::Float16 => 2,
        DType::Float32 => 4,
    }
}

pub fn contiguous_strides(shape: &[u64], dtype: DType) -> Vec<u64> {
    let elem_size = dtype_size(dtype) as u64;
    let mut strides = vec![0u64; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = elem_size;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

impl<T: Scalar> From<&[T]> for Tensor {
    fn from(value: &[T]) -> Self {
        let dtype = T::dtype();
        let shape = vec![value.len() as u64];
        let strides = contiguous_strides(&shape, dtype);
        Tensor {
            dtype,
            shape,
            strides,
            detail: TensorDetail::Constant(bytemuck::cast_slice(value).into()),
        }
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let strides = contiguous_strides(&self.shape, self.dtype);
        Tensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides,
            detail: TensorDetail::Operator(Box::new(Operator::Neg(self))),
        }
    }
}
impl Not for Tensor {
    type Output = Tensor;
    fn not(self) -> Self::Output {
        let strides = contiguous_strides(&self.shape, self.dtype);
        Tensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides,
            detail: TensorDetail::Operator(Box::new(Operator::BitNot(self))),
        }
    }
}
impl Tensor {
    pub fn abs(self) -> Self {
        let strides = contiguous_strides(&self.shape, self.dtype);
        Tensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides,
            detail: TensorDetail::Operator(Box::new(Operator::Abs(self))),
        }
    }
    pub fn exp(self) -> Self {
        let strides = contiguous_strides(&self.shape, self.dtype);
        Tensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides,
            detail: TensorDetail::Operator(Box::new(Operator::Exp(self))),
        }
    }
    pub fn log(self) -> Self {
        let strides = contiguous_strides(&self.shape, self.dtype);
        Tensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides,
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
        let strides = contiguous_strides(&shape, self.dtype);
        Tensor {
            dtype: self.dtype,
            shape,
            strides,
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
        let strides = contiguous_strides(&self.shape, self.dtype);
        Tensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides,
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
        let new_shape = shape.to_vec();
        let strides = contiguous_strides(&new_shape, self.dtype);
        Tensor {
            dtype: self.dtype,
            shape: new_shape,
            strides,
            detail: TensorDetail::Operator(Box::new(Operator::Reshape(self))),
        }
    }
    /// Broadcast tensor to new shape. This is a metadata-only operation.
    /// Sets stride to 0 for dimensions that are broadcast (size 1 -> size N).
    pub fn broadcast(self, shape: &[u64]) -> Self {
        assert_eq!(self.shape.len(), shape.len());
        let mut new_strides = Vec::with_capacity(shape.len());
        for (i, (&old_dim, &new_dim)) in self.shape.iter().zip(shape.iter()).enumerate() {
            assert!(old_dim == 1 || old_dim == new_dim, "Broadcast shape mismatch");
            if old_dim == 1 && new_dim > 1 {
                // Broadcast dimension: stride = 0
                new_strides.push(0);
            } else {
                new_strides.push(self.strides[i]);
            }
        }
        Tensor {
            dtype: self.dtype,
            shape: shape.to_vec(),
            strides: new_strides,
            detail: self.detail,
        }
    }
    pub fn into(self, dtype: DType) -> Self {
        let strides = contiguous_strides(&self.shape, dtype);
        Tensor {
            dtype,
            shape: self.shape.clone(),
            strides,
            detail: TensorDetail::Operator(Box::new(Operator::Into(self, dtype))),
        }
    }
    pub fn view(self, dtype: DType) -> Self {
        let strides = contiguous_strides(&self.shape, dtype);
        Tensor {
            dtype,
            shape: self.shape.clone(),
            strides,
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
    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape, self.dtype)
    }
}

// Custom Debug implementations for NumPy-style output

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.detail {
            TensorDetail::Constant(data) => {
                write!(f, "array(")?;
                format_array(f, data, &self.shape, self.dtype, 0)?;
                write!(f, ", dtype={:?})", self.dtype)
            }
            TensorDetail::Operator(op) => {
                if f.alternate() {
                    f.debug_struct("Tensor")
                        .field("dtype", &self.dtype)
                        .field("shape", &self.shape)
                        .field("op", op.as_ref())
                        .finish()
                } else {
                    write!(f, "Tensor {{ {:?}, {:?}, {:?} }}", self.dtype, self.shape, op.as_ref())
                }
            }
        }
    }
}

impl fmt::Debug for TensorDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorDetail::Constant(_) => write!(f, "Constant(...)"),
            TensorDetail::Operator(op) => {
                if f.alternate() {
                    write!(f, "Operator({:#?})", op)
                } else {
                    write!(f, "Operator({:?})", op)
                }
            }
        }
    }
}

impl fmt::Debug for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            self.fmt_pretty(f)
        } else {
            self.fmt_compact(f)
        }
    }
}

impl Operator {
    fn fmt_compact(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Neg(a) => write!(f, "Neg({:?})", a),
            Operator::Abs(a) => write!(f, "Abs({:?})", a),
            Operator::Exp(a) => write!(f, "Exp({:?})", a),
            Operator::Log(a) => write!(f, "Log({:?})", a),
            Operator::BitNot(a) => write!(f, "BitNot({:?})", a),
            Operator::BatchMatmul(a, b) => write!(f, "BatchMatmul({:?}, {:?})", a, b),
            Operator::Pow(a, b) => write!(f, "Pow({:?}, {:?})", a, b),
            Operator::Mul(a, b) => write!(f, "Mul({:?}, {:?})", a, b),
            Operator::Div(a, b) => write!(f, "Div({:?}, {:?})", a, b),
            Operator::Rem(a, b) => write!(f, "Rem({:?}, {:?})", a, b),
            Operator::Add(a, b) => write!(f, "Add({:?}, {:?})", a, b),
            Operator::Sub(a, b) => write!(f, "Sub({:?}, {:?})", a, b),
            Operator::Shl(a, b) => write!(f, "Shl({:?}, {:?})", a, b),
            Operator::Shr(a, b) => write!(f, "Shr({:?}, {:?})", a, b),
            Operator::BitAnd(a, b) => write!(f, "BitAnd({:?}, {:?})", a, b),
            Operator::BitOr(a, b) => write!(f, "BitOr({:?}, {:?})", a, b),
            Operator::BitXor(a, b) => write!(f, "BitXor({:?}, {:?})", a, b),
            Operator::LessThan(a, b) => write!(f, "LessThan({:?}, {:?})", a, b),
            Operator::LessOrEq(a, b) => write!(f, "LessOrEq({:?}, {:?})", a, b),
            Operator::GreaterThan(a, b) => write!(f, "GreaterThan({:?}, {:?})", a, b),
            Operator::GreaterOrEq(a, b) => write!(f, "GreaterOrEq({:?}, {:?})", a, b),
            Operator::Eq(a, b) => write!(f, "Eq({:?}, {:?})", a, b),
            Operator::NotEq(a, b) => write!(f, "NotEq({:?}, {:?})", a, b),
            Operator::Reshape(a) => write!(f, "Reshape({:?})", a),
            Operator::Into(a, dt) => write!(f, "Into({:?}, {:?})", a, dt),
            Operator::View(a, dt) => write!(f, "View({:?}, {:?})", a, dt),
        }
    }

    fn fmt_pretty(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Neg(a) => write!(f, "Neg(\n    {:#?}\n)", a),
            Operator::Abs(a) => write!(f, "Abs(\n    {:#?}\n)", a),
            Operator::Exp(a) => write!(f, "Exp(\n    {:#?}\n)", a),
            Operator::Log(a) => write!(f, "Log(\n    {:#?}\n)", a),
            Operator::BitNot(a) => write!(f, "BitNot(\n    {:#?}\n)", a),
            Operator::BatchMatmul(a, b) => write!(f, "BatchMatmul(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Pow(a, b) => write!(f, "Pow(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Mul(a, b) => write!(f, "Mul(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Div(a, b) => write!(f, "Div(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Rem(a, b) => write!(f, "Rem(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Add(a, b) => write!(f, "Add(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Sub(a, b) => write!(f, "Sub(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Shl(a, b) => write!(f, "Shl(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Shr(a, b) => write!(f, "Shr(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::BitAnd(a, b) => write!(f, "BitAnd(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::BitOr(a, b) => write!(f, "BitOr(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::BitXor(a, b) => write!(f, "BitXor(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::LessThan(a, b) => write!(f, "LessThan(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::LessOrEq(a, b) => write!(f, "LessOrEq(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::GreaterThan(a, b) => write!(f, "GreaterThan(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::GreaterOrEq(a, b) => write!(f, "GreaterOrEq(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Eq(a, b) => write!(f, "Eq(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::NotEq(a, b) => write!(f, "NotEq(\n    {:#?},\n    {:#?}\n)", a, b),
            Operator::Reshape(a) => write!(f, "Reshape(\n    {:#?}\n)", a),
            Operator::Into(a, dt) => write!(f, "Into(\n    {:#?},\n    {:?}\n)", a, dt),
            Operator::View(a, dt) => write!(f, "View(\n    {:#?},\n    {:?}\n)", a, dt),
        }
    }
}


fn format_scalar(f: &mut fmt::Formatter<'_>, data: &[u8], dtype: DType) -> fmt::Result {
    match dtype {
        DType::Int32 => {
            let v = i32::from_ne_bytes(data.try_into().unwrap());
            write!(f, "{}", v)
        }
        DType::Float16 => {
            let v = half::f16::from_ne_bytes(data.try_into().unwrap());
            let fv = v.to_f32();
            if fv.fract() == 0.0 && fv.abs() < 1e10 {
                write!(f, "{}.", fv)
            } else {
                write!(f, "{}", fv)
            }
        }
        DType::Float32 => {
            let v = f32::from_ne_bytes(data.try_into().unwrap());
            if v.fract() == 0.0 && v.abs() < 1e10 {
                write!(f, "{}.", v)
            } else {
                write!(f, "{}", v)
            }
        }
    }
}

fn format_array(
    f: &mut fmt::Formatter<'_>,
    data: &[u8],
    shape: &[u64],
    dtype: DType,
    depth: usize,
) -> fmt::Result {
    let elem_size = dtype_size(dtype);
    let pretty = f.alternate();

    if shape.is_empty() {
        // Scalar
        return format_scalar(f, data, dtype);
    }

    if shape.len() == 1 {
        // 1D array
        let n = shape[0] as usize;
        write!(f, "[")?;
        for i in 0..n {
            if i > 0 {
                write!(f, ", ")?;
            }
            let start = i * elem_size;
            let end = start + elem_size;
            format_scalar(f, &data[start..end], dtype)?;
        }
        write!(f, "]")
    } else {
        // Multi-dimensional array
        let outer_dim = shape[0] as usize;
        let inner_shape = &shape[1..];
        let inner_size: usize = inner_shape.iter().product::<u64>() as usize * elem_size;

        write!(f, "[")?;
        for i in 0..outer_dim {
            if i > 0 {
                if pretty {
                    // Pretty print: newlines and indentation
                    write!(f, ",")?;
                    writeln!(f)?;
                    for _ in 0..=depth {
                        write!(f, " ")?;
                    }
                } else {
                    // Compact: just comma and space
                    write!(f, ", ")?;
                }
            }
            let start = i * inner_size;
            let end = start + inner_size;
            format_array_inner(f, &data[start..end], inner_shape, dtype, depth + 1, pretty)?;
        }
        write!(f, "]")
    }
}

fn format_array_inner(
    f: &mut fmt::Formatter<'_>,
    data: &[u8],
    shape: &[u64],
    dtype: DType,
    depth: usize,
    pretty: bool,
) -> fmt::Result {
    let elem_size = dtype_size(dtype);

    if shape.is_empty() {
        return format_scalar(f, data, dtype);
    }

    if shape.len() == 1 {
        let n = shape[0] as usize;
        write!(f, "[")?;
        for i in 0..n {
            if i > 0 {
                write!(f, ", ")?;
            }
            let start = i * elem_size;
            let end = start + elem_size;
            format_scalar(f, &data[start..end], dtype)?;
        }
        write!(f, "]")
    } else {
        let outer_dim = shape[0] as usize;
        let inner_shape = &shape[1..];
        let inner_size: usize = inner_shape.iter().product::<u64>() as usize * elem_size;

        write!(f, "[")?;
        for i in 0..outer_dim {
            if i > 0 {
                if pretty {
                    write!(f, ",")?;
                    writeln!(f)?;
                    for _ in 0..=depth {
                        write!(f, " ")?;
                    }
                } else {
                    write!(f, ", ")?;
                }
            }
            let start = i * inner_size;
            let end = start + inner_size;
            format_array_inner(f, &data[start..end], inner_shape, dtype, depth + 1, pretty)?;
        }
        write!(f, "]")
    }
}
