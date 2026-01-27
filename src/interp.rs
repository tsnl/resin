use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::tensor::{DType, Operator, Tensor, TensorDetail, contiguous_strides, dtype_size};

/// GPU buffer with metadata
struct GpuBuffer {
    buffer: wgpu::Buffer,
    dtype: DType,
    shape: Vec<u64>,
    strides: Vec<u64>,
}

pub struct CpuBuffer {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data: Box<[u8]>,
}

impl CpuBuffer {
    pub fn to_tensor(self) -> Tensor {
        let strides = contiguous_strides(&self.shape, self.dtype);
        Tensor {
            dtype: self.dtype,
            shape: self.shape,
            strides,
            detail: TensorDetail::Constant(self.data),
        }
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct PipelineKey {
    op: &'static str,
    dtype: DType,
}

pub struct Interpreter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_cache: HashMap<PipelineKey, wgpu::ComputePipeline>,
}

impl Interpreter {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            pipeline_cache: HashMap::new(),
        }
    }

    pub fn eval(&mut self, tensor: &Tensor) -> Tensor {
        let gpu_buf = self.eval_to_gpu(tensor);
        self.read_buffer(&gpu_buf).to_tensor()
    }

    fn eval_to_gpu(&mut self, tensor: &Tensor) -> GpuBuffer {
        match &tensor.detail {
            TensorDetail::Constant(data) => {
                let buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("constant"),
                        contents: data,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    });
                GpuBuffer {
                    buffer,
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                    strides: tensor.strides.clone(),
                }
            }
            TensorDetail::Operator(op) => {
                self.eval_op(op, tensor.dtype, &tensor.shape, &tensor.strides)
            }
        }
    }

    fn eval_op(
        &mut self,
        op: &Operator,
        dtype: DType,
        shape: &[u64],
        strides: &[u64],
    ) -> GpuBuffer {
        match op {
            // Unary operations
            Operator::Neg(a) => self.dispatch_unary(a, dtype, shape, "neg"),
            Operator::Abs(a) => self.dispatch_unary(a, dtype, shape, "abs"),
            Operator::Exp(a) => self.dispatch_unary(a, dtype, shape, "exp"),
            Operator::Log(a) => self.dispatch_unary(a, dtype, shape, "log"),
            Operator::BitNot(a) => self.dispatch_unary(a, dtype, shape, "bitnot"),

            // Binary arithmetic
            Operator::Add(a, b) => self.dispatch_binary(a, b, dtype, shape, "add"),
            Operator::Sub(a, b) => self.dispatch_binary(a, b, dtype, shape, "sub"),
            Operator::Mul(a, b) => self.dispatch_binary(a, b, dtype, shape, "mul"),
            Operator::Div(a, b) => self.dispatch_binary(a, b, dtype, shape, "div"),
            Operator::Rem(a, b) => self.dispatch_binary(a, b, dtype, shape, "rem"),
            Operator::Pow(a, b) => self.dispatch_binary(a, b, dtype, shape, "pow"),

            // Shift operations
            Operator::Shl(a, b) => self.dispatch_binary(a, b, dtype, shape, "shl"),
            Operator::Shr(a, b) => self.dispatch_binary(a, b, dtype, shape, "shr"),

            // Bitwise operations
            Operator::BitAnd(a, b) => self.dispatch_binary(a, b, dtype, shape, "bitand"),
            Operator::BitOr(a, b) => self.dispatch_binary(a, b, dtype, shape, "bitor"),
            Operator::BitXor(a, b) => self.dispatch_binary(a, b, dtype, shape, "bitxor"),

            // Comparison operations
            Operator::LessThan(a, b) => self.dispatch_binary(a, b, dtype, shape, "lt"),
            Operator::LessOrEq(a, b) => self.dispatch_binary(a, b, dtype, shape, "le"),
            Operator::GreaterThan(a, b) => self.dispatch_binary(a, b, dtype, shape, "gt"),
            Operator::GreaterOrEq(a, b) => self.dispatch_binary(a, b, dtype, shape, "ge"),
            Operator::Eq(a, b) => self.dispatch_binary(a, b, dtype, shape, "eq"),
            Operator::NotEq(a, b) => self.dispatch_binary(a, b, dtype, shape, "ne"),

            // Reshape - just pass through with new shape
            Operator::Reshape(a) => {
                let input = self.eval_to_gpu(a);
                GpuBuffer {
                    buffer: input.buffer,
                    dtype,
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                }
            }

            // Matmul
            Operator::BatchMatmul(a, b) => self.dispatch_matmul(a, b, dtype, shape),

            // Cast operations
            Operator::Into(a, target_dtype) => self.dispatch_cast(a, *target_dtype, shape),
            Operator::View(a, _target_dtype) => {
                let input = self.eval_to_gpu(a);
                GpuBuffer {
                    buffer: input.buffer,
                    dtype,
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                }
            }
        }
    }

    fn dispatch_unary(
        &mut self,
        input: &Tensor,
        dtype: DType,
        shape: &[u64],
        op_name: &'static str,
    ) -> GpuBuffer {
        let input_buf = self.eval_to_gpu(input);
        let numel: u64 = shape.iter().product();
        let elem_size = dtype_size(dtype);
        let output_size = numel as usize * elem_size;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unary_output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [numel as u32, 0, 0, 0]; // numel + padding
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let key = PipelineKey { op: op_name, dtype };
        self.ensure_pipeline(&key, generate_unary_shader);
        let pipeline = self.pipeline_cache.get(&key).unwrap();

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("unary_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unary_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((numel as u32 + 255) / 256, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        GpuBuffer {
            buffer: output_buffer,
            dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape, dtype),
        }
    }

    fn dispatch_binary(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        dtype: DType,
        shape: &[u64],
        op_name: &'static str,
    ) -> GpuBuffer {
        let a_buf = self.eval_to_gpu(a);
        let b_buf = self.eval_to_gpu(b);
        let numel: u64 = shape.iter().product();
        let elem_size = dtype_size(dtype);
        let output_size = numel as usize * elem_size;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [numel as u32, 0, 0, 0];
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let key = PipelineKey { op: op_name, dtype };
        self.ensure_pipeline(&key, generate_binary_shader);
        let pipeline = self.pipeline_cache.get(&key).unwrap();

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("binary_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((numel as u32 + 255) / 256, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        GpuBuffer {
            buffer: output_buffer,
            dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape, dtype),
        }
    }

    fn dispatch_matmul(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        dtype: DType,
        shape: &[u64],
    ) -> GpuBuffer {
        let a_buf = self.eval_to_gpu(a);
        let b_buf = self.eval_to_gpu(b);

        let batch = shape[0] as u32;
        let m = shape[1] as u32;
        let n = shape[2] as u32;
        let k = a.shape[2] as u32;

        let elem_size = dtype_size(dtype);
        let output_size = (batch as usize * m as usize * n as usize) * elem_size;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [batch, m, n, k];
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let key = PipelineKey {
            op: "matmul",
            dtype,
        };
        self.ensure_matmul_pipeline(&key);
        let pipeline = self.pipeline_cache.get(&key).unwrap();

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch one workgroup per output element
            pass.dispatch_workgroups(n, m, batch);
        }

        self.queue.submit(Some(encoder.finish()));

        GpuBuffer {
            buffer: output_buffer,
            dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape, dtype),
        }
    }

    fn dispatch_cast(&mut self, input: &Tensor, target_dtype: DType, shape: &[u64]) -> GpuBuffer {
        let input_buf = self.eval_to_gpu(input);
        let numel: u64 = shape.iter().product();
        let output_elem_size = dtype_size(target_dtype);
        let output_size = numel as usize * output_elem_size;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cast_output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [numel as u32, 0, 0, 0];
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cast_params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = self.create_cast_pipeline(input.dtype, target_dtype);

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cast_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cast_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cast_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((numel as u32 + 255) / 256, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        GpuBuffer {
            buffer: output_buffer,
            dtype: target_dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape, target_dtype),
        }
    }

    fn read_buffer(&self, gpu_buf: &GpuBuffer) -> CpuBuffer {
        let numel: u64 = gpu_buf.shape.iter().product();
        let elem_size = dtype_size(gpu_buf.dtype);
        let size = numel as usize * elem_size;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, &staging_buffer, 0, size as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range().to_vec().into_boxed_slice();
        staging_buffer.unmap();

        CpuBuffer {
            dtype: gpu_buf.dtype,
            shape: gpu_buf.shape.clone(),
            data,
        }
    }

    fn ensure_pipeline(&mut self, key: &PipelineKey, generator: fn(&str, DType) -> String) {
        if !self.pipeline_cache.contains_key(key) {
            let shader_source = generator(key.op, key.dtype);
            let shader_module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(key.op),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(key.op),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            self.pipeline_cache.insert(key.clone(), pipeline);
        }
    }

    fn ensure_matmul_pipeline(&mut self, key: &PipelineKey) {
        if !self.pipeline_cache.contains_key(key) {
            let shader_source = generate_matmul_shader(key.dtype);
            let shader_module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("matmul"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("matmul"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            self.pipeline_cache.insert(key.clone(), pipeline);
        }
    }

    fn create_cast_pipeline(&self, from_dtype: DType, to_dtype: DType) -> wgpu::ComputePipeline {
        let shader_source = generate_cast_shader(from_dtype, to_dtype);
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cast"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cast"),
                layout: None,
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
    }
}

fn wgsl_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Int32 => "i32",
        DType::Float16 => "f32", // Use f32 internally for f16
        DType::Float32 => "f32",
    }
}

fn generate_unary_shader(op: &str, dtype: DType) -> String {
    let wgsl_t = wgsl_type(dtype);
    let op_expr = match op {
        "neg" => "-input[idx]".to_string(),
        "abs" => format!("abs(input[idx])"),
        "exp" => format!("exp(input[idx])"),
        "log" => format!("log(input[idx])"),
        "bitnot" => "~input[idx]".to_string(),
        _ => panic!("Unknown unary op: {}", op),
    };

    format!(
        r#"
@group(0) @binding(0) var<storage, read> input: array<{dtype}>;
@group(0) @binding(1) var<storage, read_write> output: array<{dtype}>;

struct Params {{
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.numel) {{ return; }}
    output[idx] = {op_expr};
}}
"#,
        dtype = wgsl_t,
        op_expr = op_expr
    )
}

fn generate_binary_shader(op: &str, dtype: DType) -> String {
    let wgsl_t = wgsl_type(dtype);
    let op_expr = match op {
        "add" => "a[idx] + b[idx]".to_string(),
        "sub" => "a[idx] - b[idx]".to_string(),
        "mul" => "a[idx] * b[idx]".to_string(),
        "div" => "a[idx] / b[idx]".to_string(),
        "rem" => "a[idx] % b[idx]".to_string(),
        "pow" => "pow(a[idx], b[idx])".to_string(),
        "shl" => "a[idx] << u32(b[idx])".to_string(),
        "shr" => "a[idx] >> u32(b[idx])".to_string(),
        "bitand" => "a[idx] & b[idx]".to_string(),
        "bitor" => "a[idx] | b[idx]".to_string(),
        "bitxor" => "a[idx] ^ b[idx]".to_string(),
        "lt" => format!(
            "select({zero}, {one}, a[idx] < b[idx])",
            zero = zero_val(dtype),
            one = one_val(dtype)
        ),
        "le" => format!(
            "select({zero}, {one}, a[idx] <= b[idx])",
            zero = zero_val(dtype),
            one = one_val(dtype)
        ),
        "gt" => format!(
            "select({zero}, {one}, a[idx] > b[idx])",
            zero = zero_val(dtype),
            one = one_val(dtype)
        ),
        "ge" => format!(
            "select({zero}, {one}, a[idx] >= b[idx])",
            zero = zero_val(dtype),
            one = one_val(dtype)
        ),
        "eq" => format!(
            "select({zero}, {one}, a[idx] == b[idx])",
            zero = zero_val(dtype),
            one = one_val(dtype)
        ),
        "ne" => format!(
            "select({zero}, {one}, a[idx] != b[idx])",
            zero = zero_val(dtype),
            one = one_val(dtype)
        ),
        _ => panic!("Unknown binary op: {}", op),
    };

    format!(
        r#"
@group(0) @binding(0) var<storage, read> a: array<{dtype}>;
@group(0) @binding(1) var<storage, read> b: array<{dtype}>;
@group(0) @binding(2) var<storage, read_write> output: array<{dtype}>;

struct Params {{
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.numel) {{ return; }}
    output[idx] = {op_expr};
}}
"#,
        dtype = wgsl_t,
        op_expr = op_expr
    )
}

fn zero_val(dtype: DType) -> &'static str {
    match dtype {
        DType::Int32 => "0i",
        DType::Float16 | DType::Float32 => "0.0",
    }
}

fn one_val(dtype: DType) -> &'static str {
    match dtype {
        DType::Int32 => "1i",
        DType::Float16 | DType::Float32 => "1.0",
    }
}

fn generate_matmul_shader(dtype: DType) -> String {
    let wgsl_t = wgsl_type(dtype);
    let zero = zero_val(dtype);

    format!(
        r#"
@group(0) @binding(0) var<storage, read> a: array<{dtype}>;
@group(0) @binding(1) var<storage, read> b: array<{dtype}>;
@group(0) @binding(2) var<storage, read_write> output: array<{dtype}>;

struct Params {{
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
}}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let j = gid.x;  // column
    let i = gid.y;  // row
    let batch_idx = gid.z;

    if (i >= params.m || j >= params.n || batch_idx >= params.batch) {{ return; }}

    let a_batch_offset = batch_idx * params.m * params.k;
    let b_batch_offset = batch_idx * params.k * params.n;
    let c_batch_offset = batch_idx * params.m * params.n;

    var sum: {dtype} = {zero};
    for (var kk: u32 = 0u; kk < params.k; kk++) {{
        let a_val = a[a_batch_offset + i * params.k + kk];
        let b_val = b[b_batch_offset + kk * params.n + j];
        sum = sum + a_val * b_val;
    }}

    output[c_batch_offset + i * params.n + j] = sum;
}}
"#,
        dtype = wgsl_t,
        zero = zero
    )
}

fn generate_cast_shader(from_dtype: DType, to_dtype: DType) -> String {
    let from_t = wgsl_type(from_dtype);
    let to_t = wgsl_type(to_dtype);

    let convert_expr = if from_t == to_t {
        "input[idx]".to_string()
    } else {
        format!("{}(input[idx])", to_t)
    };

    format!(
        r#"
@group(0) @binding(0) var<storage, read> input: array<{from_t}>;
@group(0) @binding(1) var<storage, read_write> output: array<{to_t}>;

struct Params {{
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.numel) {{ return; }}
    output[idx] = {convert_expr};
}}
"#,
        from_t = from_t,
        to_t = to_t,
        convert_expr = convert_expr
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn create_interpreter() -> Interpreter {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .unwrap();
        Interpreter::new(device, queue)
    }

    fn get_data(tensor: &Tensor) -> &[u8] {
        match &tensor.detail {
            TensorDetail::Constant(data) => data,
            _ => panic!("expected constant tensor"),
        }
    }

    #[test]
    fn test_add() {
        let mut interp = create_interpreter();
        let a: Tensor = (&[1.0f32, 2.0, 3.0][..]).into();
        let b: Tensor = (&[4.0f32, 5.0, 6.0][..]).into();
        let c = a + b;
        let result = interp.eval(&c);
        let data: &[f32] = bytemuck::cast_slice(get_data(&result));
        assert_eq!(data, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul() {
        let mut interp = create_interpreter();
        let a: Tensor = (&[2.0f32, 3.0, 4.0][..]).into();
        let b: Tensor = (&[5.0f32, 6.0, 7.0][..]).into();
        let c = a * b;
        let result = interp.eval(&c);
        let data: &[f32] = bytemuck::cast_slice(get_data(&result));
        assert_eq!(data, &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_neg() {
        let mut interp = create_interpreter();
        let a: Tensor = (&[1.0f32, -2.0, 3.0][..]).into();
        let b = -a;
        let result = interp.eval(&b);
        let data: &[f32] = bytemuck::cast_slice(get_data(&result));
        assert_eq!(data, &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_matmul() {
        let mut interp = create_interpreter();
        // 2x3 @ 3x2 = 2x2
        let a: Tensor = (&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0][..]).into();
        let a = a.reshape(&[2, 3]);
        let b: Tensor = (&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0][..]).into();
        let b = b.reshape(&[3, 2]);
        let c = a.matmul(b);
        let result = interp.eval(&c);
        let data: &[f32] = bytemuck::cast_slice(get_data(&result));
        assert_eq!(data, &[58.0, 64.0, 139.0, 154.0]);
    }
}
