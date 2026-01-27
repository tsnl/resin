use crate::tensor::{DType, Operator, Tensor, TensorDetail};

pub struct Buffer {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data: Box<[u8]>,
}

impl Buffer {
    pub fn to_tensor(self) -> Tensor {
        Tensor {
            dtype: self.dtype,
            shape: self.shape,
            detail: TensorDetail::Constant(self.data),
        }
    }
    fn numel(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }
}

pub fn eval(tensor: &Tensor) -> Tensor {
    eval_to_buffer(tensor).to_tensor()
}

fn eval_to_buffer(tensor: &Tensor) -> Buffer {
    match &tensor.detail {
        TensorDetail::Constant(data) => Buffer {
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
            data: data.clone(),
        },
        TensorDetail::Operator(op) => eval_op(op, tensor.dtype, &tensor.shape),
    }
}

fn eval_op(op: &Operator, dtype: DType, shape: &[u64]) -> Buffer {
    match op {
        // Unary operations
        Operator::Neg(a) => {
            let a = eval_to_buffer(a);
            unary_op(&a, dtype, shape, |dtype, a| match dtype {
                DType::Int32 => box_scalar(-i32::from_ne_bytes(a.try_into().unwrap())),
                DType::Int64 => box_scalar(-i64::from_ne_bytes(a.try_into().unwrap())),
                DType::Float16 => {
                    let v = half::f16::from_ne_bytes(a.try_into().unwrap());
                    box_scalar((-v).to_ne_bytes())
                }
                DType::Float32 => box_scalar(-f32::from_ne_bytes(a.try_into().unwrap())),
                DType::Float64 => box_scalar(-f64::from_ne_bytes(a.try_into().unwrap())),
            })
        }
        Operator::Abs(a) => {
            let a = eval_to_buffer(a);
            unary_op(&a, dtype, shape, |dtype, a| match dtype {
                DType::Int32 => box_scalar(i32::from_ne_bytes(a.try_into().unwrap()).abs()),
                DType::Int64 => box_scalar(i64::from_ne_bytes(a.try_into().unwrap()).abs()),
                DType::Float16 => {
                    let v = half::f16::from_ne_bytes(a.try_into().unwrap());
                    box_scalar(half::f16::from_f32(v.to_f32().abs()).to_ne_bytes())
                }
                DType::Float32 => box_scalar(f32::from_ne_bytes(a.try_into().unwrap()).abs()),
                DType::Float64 => box_scalar(f64::from_ne_bytes(a.try_into().unwrap()).abs()),
            })
        }
        Operator::Exp(a) => {
            let a = eval_to_buffer(a);
            unary_op(&a, dtype, shape, |dtype, a| match dtype {
                DType::Float16 => {
                    let v = half::f16::from_ne_bytes(a.try_into().unwrap());
                    box_scalar(half::f16::from_f32(v.to_f32().exp()).to_ne_bytes())
                }
                DType::Float32 => box_scalar(f32::from_ne_bytes(a.try_into().unwrap()).exp()),
                DType::Float64 => box_scalar(f64::from_ne_bytes(a.try_into().unwrap()).exp()),
                _ => panic!("exp not supported for {:?}", dtype),
            })
        }
        Operator::Log(a) => {
            let a = eval_to_buffer(a);
            unary_op(&a, dtype, shape, |dtype, a| match dtype {
                DType::Float16 => {
                    let v = half::f16::from_ne_bytes(a.try_into().unwrap());
                    box_scalar(half::f16::from_f32(v.to_f32().ln()).to_ne_bytes())
                }
                DType::Float32 => box_scalar(f32::from_ne_bytes(a.try_into().unwrap()).ln()),
                DType::Float64 => box_scalar(f64::from_ne_bytes(a.try_into().unwrap()).ln()),
                _ => panic!("log not supported for {:?}", dtype),
            })
        }
        Operator::BitNot(a) => {
            let a = eval_to_buffer(a);
            unary_op(&a, dtype, shape, |dtype, a| match dtype {
                DType::Int32 => box_scalar(!i32::from_ne_bytes(a.try_into().unwrap())),
                DType::Int64 => box_scalar(!i64::from_ne_bytes(a.try_into().unwrap())),
                _ => panic!("bitnot not supported for {:?}", dtype),
            })
        }

        // Binary arithmetic
        Operator::Add(a, b) => binary_arith(a, b, dtype, shape, |l, r| l + r, |l, r| l + r),
        Operator::Sub(a, b) => binary_arith(a, b, dtype, shape, |l, r| l - r, |l, r| l - r),
        Operator::Mul(a, b) => binary_arith(a, b, dtype, shape, |l, r| l * r, |l, r| l * r),
        Operator::Div(a, b) => binary_arith(a, b, dtype, shape, |l, r| l / r, |l, r| l / r),
        Operator::Rem(a, b) => binary_arith(a, b, dtype, shape, |l, r| l % r, |l, r| l % r),
        Operator::Pow(a, b) => {
            let a = eval_to_buffer(a);
            let b = eval_to_buffer(b);
            binary_op(&a, &b, dtype, shape, |dtype, av, bv| match dtype {
                DType::Int32 => {
                    let base = i32::from_ne_bytes(av.try_into().unwrap());
                    let exp = i32::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(base.pow(exp as u32))
                }
                DType::Int64 => {
                    let base = i64::from_ne_bytes(av.try_into().unwrap());
                    let exp = i64::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(base.pow(exp as u32))
                }
                DType::Float16 => {
                    let base = half::f16::from_ne_bytes(av.try_into().unwrap()).to_f32();
                    let exp = half::f16::from_ne_bytes(bv.try_into().unwrap()).to_f32();
                    box_scalar(half::f16::from_f32(base.powf(exp)).to_ne_bytes())
                }
                DType::Float32 => {
                    let base = f32::from_ne_bytes(av.try_into().unwrap());
                    let exp = f32::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(base.powf(exp))
                }
                DType::Float64 => {
                    let base = f64::from_ne_bytes(av.try_into().unwrap());
                    let exp = f64::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(base.powf(exp))
                }
            })
        }

        // Shift operations
        Operator::Shl(a, b) => {
            let a = eval_to_buffer(a);
            let b = eval_to_buffer(b);
            binary_op(&a, &b, dtype, shape, |dtype, av, bv| match dtype {
                DType::Int32 => {
                    let l = i32::from_ne_bytes(av.try_into().unwrap());
                    let r = i32::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l << r)
                }
                DType::Int64 => {
                    let l = i64::from_ne_bytes(av.try_into().unwrap());
                    let r = i64::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l << r)
                }
                _ => panic!("shl not supported for {:?}", dtype),
            })
        }
        Operator::Shr(a, b) => {
            let a = eval_to_buffer(a);
            let b = eval_to_buffer(b);
            binary_op(&a, &b, dtype, shape, |dtype, av, bv| match dtype {
                DType::Int32 => {
                    let l = i32::from_ne_bytes(av.try_into().unwrap());
                    let r = i32::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l >> r)
                }
                DType::Int64 => {
                    let l = i64::from_ne_bytes(av.try_into().unwrap());
                    let r = i64::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l >> r)
                }
                _ => panic!("shr not supported for {:?}", dtype),
            })
        }

        // Bitwise operations
        Operator::BitAnd(a, b) => {
            let a = eval_to_buffer(a);
            let b = eval_to_buffer(b);
            binary_op(&a, &b, dtype, shape, |dtype, av, bv| match dtype {
                DType::Int32 => {
                    let l = i32::from_ne_bytes(av.try_into().unwrap());
                    let r = i32::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l & r)
                }
                DType::Int64 => {
                    let l = i64::from_ne_bytes(av.try_into().unwrap());
                    let r = i64::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l & r)
                }
                _ => panic!("bitand not supported for {:?}", dtype),
            })
        }
        Operator::BitOr(a, b) => {
            let a = eval_to_buffer(a);
            let b = eval_to_buffer(b);
            binary_op(&a, &b, dtype, shape, |dtype, av, bv| match dtype {
                DType::Int32 => {
                    let l = i32::from_ne_bytes(av.try_into().unwrap());
                    let r = i32::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l | r)
                }
                DType::Int64 => {
                    let l = i64::from_ne_bytes(av.try_into().unwrap());
                    let r = i64::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l | r)
                }
                _ => panic!("bitor not supported for {:?}", dtype),
            })
        }
        Operator::BitXor(a, b) => {
            let a = eval_to_buffer(a);
            let b = eval_to_buffer(b);
            binary_op(&a, &b, dtype, shape, |dtype, av, bv| match dtype {
                DType::Int32 => {
                    let l = i32::from_ne_bytes(av.try_into().unwrap());
                    let r = i32::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l ^ r)
                }
                DType::Int64 => {
                    let l = i64::from_ne_bytes(av.try_into().unwrap());
                    let r = i64::from_ne_bytes(bv.try_into().unwrap());
                    box_scalar(l ^ r)
                }
                _ => panic!("bitxor not supported for {:?}", dtype),
            })
        }

        // Comparison operations (return same dtype, 1 for true, 0 for false)
        Operator::LessThan(a, b) => binary_cmp(a, b, dtype, shape, |o| o.is_lt()),
        Operator::LessOrEq(a, b) => binary_cmp(a, b, dtype, shape, |o| o.is_le()),
        Operator::GreaterThan(a, b) => binary_cmp(a, b, dtype, shape, |o| o.is_gt()),
        Operator::GreaterOrEq(a, b) => binary_cmp(a, b, dtype, shape, |o| o.is_ge()),
        Operator::Eq(a, b) => binary_cmp(a, b, dtype, shape, |o| o.is_eq()),
        Operator::NotEq(a, b) => binary_cmp(a, b, dtype, shape, |o| o.is_ne()),

        // Shape manipulation
        Operator::Reshape(a) => {
            let a = eval_to_buffer(a);
            Buffer {
                dtype,
                shape: shape.to_vec(),
                data: a.data,
            }
        }
        Operator::Broadcast(a) => {
            let a = eval_to_buffer(a);
            broadcast_impl(&a, shape)
        }

        // Matmul
        Operator::BatchMatmul(a, b) => {
            let a = eval_to_buffer(a);
            let b = eval_to_buffer(b);
            batch_matmul(&a, &b, dtype, shape)
        }

        // Cast operations
        Operator::Into(a, target_dtype) => {
            let a = eval_to_buffer(a);
            cast_into(&a, *target_dtype, shape)
        }
        Operator::View(a, _target_dtype) => {
            let a = eval_to_buffer(a);
            Buffer {
                dtype,
                shape: shape.to_vec(),
                data: a.data,
            }
        }
    }
}

fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Int32 => 4,
        DType::Int64 => 8,
        DType::Float16 => 2,
        DType::Float32 => 4,
        DType::Float64 => 8,
    }
}

fn box_scalar<T: bytemuck::Pod>(v: T) -> Box<[u8]> {
    bytemuck::bytes_of(&v).into()
}

fn unary_op<F>(a: &Buffer, dtype: DType, shape: &[u64], op: F) -> Buffer
where
    F: Fn(DType, &[u8]) -> Box<[u8]>,
{
    let elem_size = dtype_size(a.dtype);
    let numel = a.numel();
    let mut result = Vec::with_capacity(numel * elem_size);

    for i in 0..numel {
        let start = i * elem_size;
        let end = start + elem_size;
        let val = &a.data[start..end];
        let out = op(dtype, val);
        result.extend_from_slice(&out);
    }

    Buffer {
        dtype,
        shape: shape.to_vec(),
        data: result.into_boxed_slice(),
    }
}

fn binary_op<F>(a: &Buffer, b: &Buffer, dtype: DType, shape: &[u64], op: F) -> Buffer
where
    F: Fn(DType, &[u8], &[u8]) -> Box<[u8]>,
{
    let elem_size = dtype_size(a.dtype);
    let numel: usize = shape.iter().product::<u64>() as usize;
    let mut result = Vec::with_capacity(numel * elem_size);

    for i in 0..numel {
        let start = i * elem_size;
        let end = start + elem_size;
        let av = &a.data[start..end];
        let bv = &b.data[start..end];
        let out = op(dtype, av, bv);
        result.extend_from_slice(&out);
    }

    Buffer {
        dtype,
        shape: shape.to_vec(),
        data: result.into_boxed_slice(),
    }
}

fn binary_arith<IF, FF>(
    a: &Tensor,
    b: &Tensor,
    dtype: DType,
    shape: &[u64],
    int_op: IF,
    float_op: FF,
) -> Buffer
where
    IF: Fn(i64, i64) -> i64,
    FF: Fn(f64, f64) -> f64,
{
    let a = eval_to_buffer(a);
    let b = eval_to_buffer(b);
    binary_op(&a, &b, dtype, shape, |dtype, av, bv| match dtype {
        DType::Int32 => {
            let l = i32::from_ne_bytes(av.try_into().unwrap()) as i64;
            let r = i32::from_ne_bytes(bv.try_into().unwrap()) as i64;
            box_scalar(int_op(l, r) as i32)
        }
        DType::Int64 => {
            let l = i64::from_ne_bytes(av.try_into().unwrap());
            let r = i64::from_ne_bytes(bv.try_into().unwrap());
            box_scalar(int_op(l, r))
        }
        DType::Float16 => {
            let l = half::f16::from_ne_bytes(av.try_into().unwrap()).to_f64();
            let r = half::f16::from_ne_bytes(bv.try_into().unwrap()).to_f64();
            box_scalar(half::f16::from_f64(float_op(l, r)).to_ne_bytes())
        }
        DType::Float32 => {
            let l = f32::from_ne_bytes(av.try_into().unwrap()) as f64;
            let r = f32::from_ne_bytes(bv.try_into().unwrap()) as f64;
            box_scalar(float_op(l, r) as f32)
        }
        DType::Float64 => {
            let l = f64::from_ne_bytes(av.try_into().unwrap());
            let r = f64::from_ne_bytes(bv.try_into().unwrap());
            box_scalar(float_op(l, r))
        }
    })
}

fn binary_cmp<F>(a: &Tensor, b: &Tensor, dtype: DType, shape: &[u64], cmp: F) -> Buffer
where
    F: Fn(std::cmp::Ordering) -> bool,
{
    let a = eval_to_buffer(a);
    let b = eval_to_buffer(b);
    binary_op(&a, &b, dtype, shape, |dt, av, bv| {
        let ord = match dt {
            DType::Int32 => {
                let l = i32::from_ne_bytes(av.try_into().unwrap());
                let r = i32::from_ne_bytes(bv.try_into().unwrap());
                l.cmp(&r)
            }
            DType::Int64 => {
                let l = i64::from_ne_bytes(av.try_into().unwrap());
                let r = i64::from_ne_bytes(bv.try_into().unwrap());
                l.cmp(&r)
            }
            DType::Float16 => {
                let l = half::f16::from_ne_bytes(av.try_into().unwrap());
                let r = half::f16::from_ne_bytes(bv.try_into().unwrap());
                l.total_cmp(&r)
            }
            DType::Float32 => {
                let l = f32::from_ne_bytes(av.try_into().unwrap());
                let r = f32::from_ne_bytes(bv.try_into().unwrap());
                l.total_cmp(&r)
            }
            DType::Float64 => {
                let l = f64::from_ne_bytes(av.try_into().unwrap());
                let r = f64::from_ne_bytes(bv.try_into().unwrap());
                l.total_cmp(&r)
            }
        };
        let result = if cmp(ord) { 1 } else { 0 };
        match dt {
            DType::Int32 => box_scalar(result as i32),
            DType::Int64 => box_scalar(result as i64),
            DType::Float16 => box_scalar(half::f16::from_f32(result as f32).to_ne_bytes()),
            DType::Float32 => box_scalar(result as f32),
            DType::Float64 => box_scalar(result as f64),
        }
    })
}

fn broadcast_impl(a: &Buffer, target_shape: &[u64]) -> Buffer {
    let elem_size = dtype_size(a.dtype);
    let target_numel: usize = target_shape.iter().product::<u64>() as usize;
    let mut result = Vec::with_capacity(target_numel * elem_size);

    let ndim = target_shape.len();
    let src_shape = &a.shape;

    // Compute strides for source (0 stride for broadcast dimensions)
    let mut src_strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        if src_shape[i] == 1 {
            src_strides[i] = 0;
        } else {
            src_strides[i] = stride;
        }
        stride *= src_shape[i] as usize;
    }

    // Iterate over target shape
    for flat_idx in 0..target_numel {
        let mut src_idx = 0;
        let mut remaining = flat_idx;

        // Compute multi-dimensional index and map to source
        let mut divisor = target_numel;
        for i in 0..ndim {
            divisor /= target_shape[i] as usize;
            let coord = remaining / divisor;
            remaining %= divisor;
            src_idx += coord * src_strides[i];
        }

        let start = src_idx * elem_size;
        let end = start + elem_size;
        result.extend_from_slice(&a.data[start..end]);
    }

    Buffer {
        dtype: a.dtype,
        shape: target_shape.to_vec(),
        data: result.into_boxed_slice(),
    }
}

fn batch_matmul(a: &Buffer, b: &Buffer, dtype: DType, shape: &[u64]) -> Buffer {
    let batch = shape[0] as usize;
    let m = shape[1] as usize;
    let n = shape[2] as usize;
    let k = a.shape[2] as usize;

    let elem_size = dtype_size(dtype);
    let a_batch_size = m * k;
    let b_batch_size = k * n;
    let c_batch_size = m * n;

    let mut result = vec![0u8; batch * c_batch_size * elem_size];

    for batch_idx in 0..batch {
        let a_offset = batch_idx * a_batch_size * elem_size;
        let b_offset = batch_idx * b_batch_size * elem_size;
        let c_offset = batch_idx * c_batch_size * elem_size;

        for i in 0..m {
            for j in 0..n {
                let c_idx = c_offset + (i * n + j) * elem_size;

                match dtype {
                    DType::Int32 => {
                        let mut sum: i32 = 0;
                        for kk in 0..k {
                            let a_idx = a_offset + (i * k + kk) * elem_size;
                            let b_idx = b_offset + (kk * n + j) * elem_size;
                            let av =
                                i32::from_ne_bytes(a.data[a_idx..a_idx + 4].try_into().unwrap());
                            let bv =
                                i32::from_ne_bytes(b.data[b_idx..b_idx + 4].try_into().unwrap());
                            sum += av * bv;
                        }
                        result[c_idx..c_idx + 4].copy_from_slice(&sum.to_ne_bytes());
                    }
                    DType::Int64 => {
                        let mut sum: i64 = 0;
                        for kk in 0..k {
                            let a_idx = a_offset + (i * k + kk) * elem_size;
                            let b_idx = b_offset + (kk * n + j) * elem_size;
                            let av =
                                i64::from_ne_bytes(a.data[a_idx..a_idx + 8].try_into().unwrap());
                            let bv =
                                i64::from_ne_bytes(b.data[b_idx..b_idx + 8].try_into().unwrap());
                            sum += av * bv;
                        }
                        result[c_idx..c_idx + 8].copy_from_slice(&sum.to_ne_bytes());
                    }
                    DType::Float16 => {
                        let mut sum: f32 = 0.0;
                        for kk in 0..k {
                            let a_idx = a_offset + (i * k + kk) * elem_size;
                            let b_idx = b_offset + (kk * n + j) * elem_size;
                            let av = half::f16::from_ne_bytes(
                                a.data[a_idx..a_idx + 2].try_into().unwrap(),
                            );
                            let bv = half::f16::from_ne_bytes(
                                b.data[b_idx..b_idx + 2].try_into().unwrap(),
                            );
                            sum += av.to_f32() * bv.to_f32();
                        }
                        result[c_idx..c_idx + 2]
                            .copy_from_slice(&half::f16::from_f32(sum).to_ne_bytes());
                    }
                    DType::Float32 => {
                        let mut sum: f32 = 0.0;
                        for kk in 0..k {
                            let a_idx = a_offset + (i * k + kk) * elem_size;
                            let b_idx = b_offset + (kk * n + j) * elem_size;
                            let av =
                                f32::from_ne_bytes(a.data[a_idx..a_idx + 4].try_into().unwrap());
                            let bv =
                                f32::from_ne_bytes(b.data[b_idx..b_idx + 4].try_into().unwrap());
                            sum += av * bv;
                        }
                        result[c_idx..c_idx + 4].copy_from_slice(&sum.to_ne_bytes());
                    }
                    DType::Float64 => {
                        let mut sum: f64 = 0.0;
                        for kk in 0..k {
                            let a_idx = a_offset + (i * k + kk) * elem_size;
                            let b_idx = b_offset + (kk * n + j) * elem_size;
                            let av =
                                f64::from_ne_bytes(a.data[a_idx..a_idx + 8].try_into().unwrap());
                            let bv =
                                f64::from_ne_bytes(b.data[b_idx..b_idx + 8].try_into().unwrap());
                            sum += av * bv;
                        }
                        result[c_idx..c_idx + 8].copy_from_slice(&sum.to_ne_bytes());
                    }
                }
            }
        }
    }

    Buffer {
        dtype,
        shape: shape.to_vec(),
        data: result.into_boxed_slice(),
    }
}

fn cast_into(a: &Buffer, target_dtype: DType, shape: &[u64]) -> Buffer {
    let src_elem_size = dtype_size(a.dtype);
    let dst_elem_size = dtype_size(target_dtype);
    let numel = a.numel();
    let mut result = Vec::with_capacity(numel * dst_elem_size);

    for i in 0..numel {
        let start = i * src_elem_size;
        let end = start + src_elem_size;
        let src = &a.data[start..end];

        // Convert source to f64 as intermediate
        let val: f64 = match a.dtype {
            DType::Int32 => i32::from_ne_bytes(src.try_into().unwrap()) as f64,
            DType::Int64 => i64::from_ne_bytes(src.try_into().unwrap()) as f64,
            DType::Float16 => half::f16::from_ne_bytes(src.try_into().unwrap()).to_f64(),
            DType::Float32 => f32::from_ne_bytes(src.try_into().unwrap()) as f64,
            DType::Float64 => f64::from_ne_bytes(src.try_into().unwrap()),
        };

        // Convert f64 to target dtype
        let bytes: Box<[u8]> = match target_dtype {
            DType::Int32 => box_scalar(val as i32),
            DType::Int64 => box_scalar(val as i64),
            DType::Float16 => box_scalar(half::f16::from_f64(val).to_ne_bytes()),
            DType::Float32 => box_scalar(val as f32),
            DType::Float64 => box_scalar(val),
        };
        result.extend_from_slice(&bytes);
    }

    Buffer {
        dtype: target_dtype,
        shape: shape.to_vec(),
        data: result.into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn get_data(tensor: &Tensor) -> &[u8] {
        match &tensor.detail {
            TensorDetail::Constant(data) => data,
            _ => panic!("expected constant tensor"),
        }
    }

    #[test]
    fn test_add() {
        let a: Tensor = (&[1.0f32, 2.0, 3.0][..]).into();
        let b: Tensor = (&[4.0f32, 5.0, 6.0][..]).into();
        let c = a + b;
        let result = eval(&c);
        let data: &[f32] = bytemuck::cast_slice(get_data(&result));
        assert_eq!(data, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul() {
        let a: Tensor = (&[2.0f32, 3.0, 4.0][..]).into();
        let b: Tensor = (&[5.0f32, 6.0, 7.0][..]).into();
        let c = a * b;
        let result = eval(&c);
        let data: &[f32] = bytemuck::cast_slice(get_data(&result));
        assert_eq!(data, &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_neg() {
        let a: Tensor = (&[1.0f32, -2.0, 3.0][..]).into();
        let b = -a;
        let result = eval(&b);
        let data: &[f32] = bytemuck::cast_slice(get_data(&result));
        assert_eq!(data, &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_matmul() {
        // 2x3 @ 3x2 = 2x2
        let a: Tensor = (&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0][..]).into();
        let a = a.reshape(&[2, 3]);
        let b: Tensor = (&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0][..]).into();
        let b = b.reshape(&[3, 2]);
        let c = a.matmul(b);
        let result = eval(&c);
        let data: &[f32] = bytemuck::cast_slice(get_data(&result));
        // [1,2,3] @ [7,9,11; 8,10,12]^T = [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4,5,6] @ same = [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(data, &[58.0, 64.0, 139.0, 154.0]);
    }
}
