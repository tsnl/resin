use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct Expr {
    pub shape: Vec<usize>,
    pub detail: Detail,
}

impl Expr {
    /// Create a 1D vector expression from a slice of f32 values.
    pub fn new_vector(device: &wgpu::Device, data: &[f32]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vector_constant"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        Expr {
            shape: vec![data.len()],
            detail: Detail::Constant(buffer),
        }
    }

    /// Create a 2D matrix expression from a 2D array of f32 values.
    pub fn new_matrix<const R: usize, const C: usize>(
        device: &wgpu::Device,
        data: &[[f32; C]; R],
    ) -> Self {
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matrix_constant"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        Expr {
            shape: vec![R, C],
            detail: Detail::Constant(buffer),
        }
    }

    /// Create a 3D tensor expression from a 3D array of f32 values.
    pub fn new_tensor<const B: usize, const R: usize, const C: usize>(
        device: &wgpu::Device,
        data: &[[[f32; C]; R]; B],
    ) -> Self {
        let flat: Vec<f32> = data.iter().flatten().flatten().copied().collect();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tensor_constant"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        Expr {
            shape: vec![B, R, C],
            detail: Detail::Constant(buffer),
        }
    }
}

pub enum Detail {
    Constant(wgpu::Buffer),
    Operator(Box<Operator>),
}

impl std::fmt::Debug for Detail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Detail::Constant(_) => f.debug_tuple("Constant").finish(),
            Detail::Operator(op) => f.debug_tuple("Operator").field(op).finish(),
        }
    }
}

#[derive(Debug)]
pub enum Operator {
    // Unary
    Neg(Expr),
    Abs(Expr),
    Exp(Expr),
    Log(Expr),

    // Matmul
    Bmm(Expr, Expr),

    // Binary arithmetic
    Pow(Expr, Expr),
    Mul(Expr, Expr),
    Div(Expr, Expr),
    Rem(Expr, Expr),
    Add(Expr, Expr),
    Sub(Expr, Expr),

    // Shape manipulation
    Reshape(Expr),
}

impl Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Self::Output {
        Expr {
            shape: self.shape.clone(),
            detail: Detail::Operator(Box::new(Operator::Neg(self))),
        }
    }
}
impl Expr {
    pub fn abs(self) -> Self {
        Expr {
            shape: self.shape.clone(),
            detail: Detail::Operator(Box::new(Operator::Abs(self))),
        }
    }
    pub fn exp(self) -> Self {
        Expr {
            shape: self.shape.clone(),
            detail: Detail::Operator(Box::new(Operator::Exp(self))),
        }
    }
    pub fn log(self) -> Self {
        Expr {
            shape: self.shape.clone(),
            detail: Detail::Operator(Box::new(Operator::Log(self))),
        }
    }
}

impl Expr {
    pub fn bmm(self, other: Expr) -> Self {
        assert_eq!(self.shape.len(), 3);
        assert_eq!(other.shape.len(), 3);
        assert_eq!(self.shape[0], other.shape[0]);
        assert_eq!(self.shape[2], other.shape[1]);
        let shape = vec![self.shape[0], self.shape[1], other.shape[2]];
        Expr {
            shape,
            detail: Detail::Operator(Box::new(Operator::Bmm(self, other))),
        }
    }
    pub fn matmul(self, other: Expr) -> Self {
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

impl Expr {
    fn binary_op<F>(self, other: Expr, op: F) -> Expr
    where
        F: Fn(Expr, Expr) -> Operator,
    {
        assert_eq!(self.shape, other.shape);
        Expr {
            shape: self.shape.clone(),
            detail: Detail::Operator(Box::new(op(self, other))),
        }
    }
}
impl Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Self::Output {
        self.binary_op(rhs, Operator::Mul)
    }
}
impl Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Self::Output {
        self.binary_op(rhs, Operator::Div)
    }
}
impl Rem for Expr {
    type Output = Expr;
    fn rem(self, rhs: Expr) -> Self::Output {
        self.binary_op(rhs, Operator::Rem)
    }
}
impl Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Self::Output {
        self.binary_op(rhs, Operator::Add)
    }
}
impl Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Self::Output {
        self.binary_op(rhs, Operator::Sub)
    }
}
impl Expr {
    pub fn pow(self, other: Expr) -> Self {
        self.binary_op(other, Operator::Pow)
    }
}

impl Expr {
    pub fn reshape(self, shape: impl Into<Vec<usize>>) -> Self {
        let new_shape: Vec<_> = shape.into();

        let old_size: usize = self.numel();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(old_size, new_size, "Reshape size mismatch");

        Expr {
            shape: new_shape,
            detail: Detail::Operator(Box::new(Operator::Reshape(self))),
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

impl Expr {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}
