use std::alloc::Layout;

use crate::expr::{Detail, Expr, Operator};
use wgpu::include_wgsl;

pub struct Interp {
    device: wgpu::Device,
    queue: wgpu::Queue,
    binary_op_bind_group_layout: wgpu::BindGroupLayout,
    unary_op_bind_group_layout: wgpu::BindGroupLayout,
    gemm_bind_group_layout: wgpu::BindGroupLayout,
    binary_op_pow_pipeline: wgpu::ComputePipeline,
    binary_op_mul_pipeline: wgpu::ComputePipeline,
    binary_op_div_pipeline: wgpu::ComputePipeline,
    binary_op_rem_pipeline: wgpu::ComputePipeline,
    binary_op_add_pipeline: wgpu::ComputePipeline,
    binary_op_sub_pipeline: wgpu::ComputePipeline,
    unary_op_neg_pipeline: wgpu::ComputePipeline,
    unary_op_abs_pipeline: wgpu::ComputePipeline,
    unary_op_exp_pipeline: wgpu::ComputePipeline,
    unary_op_log_pipeline: wgpu::ComputePipeline,
    gemm_pipeline: wgpu::ComputePipeline,
}

impl Interp {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let binary_op_shader_module =
            device.create_shader_module(include_wgsl!("shaders/binary_op.wgsl"));
        let unary_op_shader_module =
            device.create_shader_module(include_wgsl!("shaders/unary_op.wgsl"));
        let gemm_shader_module = device.create_shader_module(include_wgsl!("shaders/gemm.wgsl"));

        let (binary_op_bind_group_layout, binary_op_pipeline_layout) =
            Self::create_binary_op_pipeline_layout(device);
        let (unary_op_bind_group_layout, unary_op_pipeline_layout) =
            Self::create_unary_op_pipeline_layout(device);
        let (gemm_bind_group_layout, gemm_pipeline_layout) =
            Self::create_gemm_pipeline_layout(device);

        Self {
            device: device.clone(),
            queue: queue.clone(),
            binary_op_bind_group_layout,
            unary_op_bind_group_layout,
            gemm_bind_group_layout,
            binary_op_pow_pipeline: Self::create_pipeline(
                device,
                &binary_op_pipeline_layout,
                &binary_op_shader_module,
                "binary_op_pow",
            ),
            binary_op_mul_pipeline: Self::create_pipeline(
                device,
                &binary_op_pipeline_layout,
                &binary_op_shader_module,
                "binary_op_mul",
            ),
            binary_op_div_pipeline: Self::create_pipeline(
                device,
                &binary_op_pipeline_layout,
                &binary_op_shader_module,
                "binary_op_div",
            ),
            binary_op_rem_pipeline: Self::create_pipeline(
                device,
                &binary_op_pipeline_layout,
                &binary_op_shader_module,
                "binary_op_mod",
            ),
            binary_op_add_pipeline: Self::create_pipeline(
                device,
                &binary_op_pipeline_layout,
                &binary_op_shader_module,
                "binary_op_add",
            ),
            binary_op_sub_pipeline: Self::create_pipeline(
                device,
                &binary_op_pipeline_layout,
                &binary_op_shader_module,
                "binary_op_sub",
            ),
            unary_op_neg_pipeline: Self::create_pipeline(
                device,
                &unary_op_pipeline_layout,
                &unary_op_shader_module,
                "unary_op_neg",
            ),
            unary_op_abs_pipeline: Self::create_pipeline(
                device,
                &unary_op_pipeline_layout,
                &unary_op_shader_module,
                "unary_op_abs",
            ),
            unary_op_exp_pipeline: Self::create_pipeline(
                device,
                &unary_op_pipeline_layout,
                &unary_op_shader_module,
                "unary_op_exp",
            ),
            unary_op_log_pipeline: Self::create_pipeline(
                device,
                &unary_op_pipeline_layout,
                &unary_op_shader_module,
                "unary_op_log",
            ),
            gemm_pipeline: Self::create_pipeline(
                device,
                &gemm_pipeline_layout,
                &gemm_shader_module,
                "gemm",
            ),
        }
    }
    fn create_binary_op_pipeline_layout(
        device: &wgpu::Device,
    ) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("binary_op_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("binary_op_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        (bind_group_layout, pipeline_layout)
    }
    fn create_unary_op_pipeline_layout(
        device: &wgpu::Device,
    ) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unary_op_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unary_op_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        (bind_group_layout, pipeline_layout)
    }
    fn create_gemm_pipeline_layout(
        device: &wgpu::Device,
    ) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        (bind_group_layout, pipeline_layout)
    }
    fn create_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(layout),
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }
}

impl Interp {
    pub fn eval(&self, expr: &Expr) -> wgpu::Buffer {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let result_buf = match &expr.detail {
            Detail::Constant(constant) => self.eval_constant_impl(constant),
            Detail::Reshape(inner) => self.eval_impl(inner, &mut enc),
            Detail::Operator(operator) => self.eval_operator_impl(operator, &mut enc),
        };

        self.queue.submit(Some(enc.finish()));

        result_buf
    }

    pub fn readback(&self, src: &wgpu::Buffer, numel: usize) -> Vec<f32> {
        let size = Layout::array::<f32>(numel).unwrap().size();
        let readback_buf = self.create_wgpu_buffer(
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size,
        );

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &readback_buf, 0, size as u64);
        self.queue.submit(Some(enc.finish()));

        let slice = readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let data: Vec<f32> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        readback_buf.unmap();

        data
    }

    fn eval_impl(&self, expr: &Expr, enc: &mut wgpu::CommandEncoder) -> wgpu::Buffer {
        match &expr.detail {
            Detail::Constant(constant) => self.eval_constant_impl(constant),
            Detail::Reshape(inner) => self.eval_impl(inner, enc),
            Detail::Operator(operator) => self.eval_operator_impl(operator, enc),
        }
    }

    fn eval_constant_impl(&self, buffer: &wgpu::Buffer) -> wgpu::Buffer {
        // Copy the buffer to avoid lifetime issues
        let size = buffer.size();
        let new_buf = self.create_wgpu_buffer(
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            size as usize,
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buffer, 0, &new_buf, 0, size);
        self.queue.submit(Some(enc.finish()));
        new_buf
    }

    fn eval_operator_impl(
        &self,
        operator: &Operator,
        enc: &mut wgpu::CommandEncoder,
    ) -> wgpu::Buffer {
        let pipeline = self.get_operator_pipeline(operator);
        let uniform_buffer = self.get_operator_uniform_buffer(operator);
        let (n, res, bg) = self.get_operator_dispatch_args(operator, &uniform_buffer, enc);

        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n, 1, 1);
        }
        res
    }
    fn get_operator_pipeline(&self, operator: &Operator) -> &wgpu::ComputePipeline {
        match operator {
            Operator::Neg(_) => &self.unary_op_neg_pipeline,
            Operator::Abs(_) => &self.unary_op_abs_pipeline,
            Operator::Exp(_) => &self.unary_op_exp_pipeline,
            Operator::Log(_) => &self.unary_op_log_pipeline,
            Operator::Pow(_, _) => &self.binary_op_pow_pipeline,
            Operator::Mul(_, _) => &self.binary_op_mul_pipeline,
            Operator::Div(_, _) => &self.binary_op_div_pipeline,
            Operator::Rem(_, _) => &self.binary_op_rem_pipeline,
            Operator::Add(_, _) => &self.binary_op_add_pipeline,
            Operator::Sub(_, _) => &self.binary_op_sub_pipeline,
            Operator::Bmm(_, _) => &self.gemm_pipeline,
        }
    }
    fn get_operator_uniform_buffer(&self, operator: &Operator) -> wgpu::Buffer {
        match operator {
            // Unary operators:
            Operator::Neg(expr)
            | Operator::Abs(expr)
            | Operator::Exp(expr)
            | Operator::Log(expr) => self.emplace_wgpu_buffer(
                wgpu::BufferUsages::UNIFORM,
                &[UnaryOpParams {
                    a_dim: UVec3::from_shape(&expr.shape),
                }],
            ),

            // Binary operators:
            Operator::Pow(lt, rt)
            | Operator::Mul(lt, rt)
            | Operator::Div(lt, rt)
            | Operator::Rem(lt, rt)
            | Operator::Add(lt, rt)
            | Operator::Sub(lt, rt) => self.emplace_wgpu_buffer(
                wgpu::BufferUsages::UNIFORM,
                &[BinaryOpParams {
                    a_dim: UVec3::from_shape(&lt.shape),
                    b_dim: UVec3::from_shape(&rt.shape),
                }],
            ),

            // GEMM operator:
            Operator::Bmm(lt, rt) => {
                let c_shape = vec![lt.shape[0], lt.shape[1], rt.shape[2]];
                self.emplace_wgpu_buffer(
                    wgpu::BufferUsages::UNIFORM,
                    &[GemmParams {
                        a_dim: UVec3::from_shape(&lt.shape),
                        b_dim: UVec3::from_shape(&rt.shape),
                        c_dim: UVec3::from_shape(&c_shape),
                        a_coeff: 1.0,
                        b_coeff: 1.0,
                        c_coeff: 0.0,
                        _pad: 0.0,
                    }],
                )
            }
        }
    }
    fn get_operator_dispatch_args(
        &self,
        operator: &Operator,
        uniform_buffer: &wgpu::Buffer,
        enc: &mut wgpu::CommandEncoder,
    ) -> (u32, wgpu::Buffer, wgpu::BindGroup) {
        match operator {
            Operator::Neg(expr)
            | Operator::Abs(expr)
            | Operator::Exp(expr)
            | Operator::Log(expr) => {
                self.get_unary_operator_dispatch_args(expr, uniform_buffer, enc)
            }
            Operator::Pow(lt, rt)
            | Operator::Mul(lt, rt)
            | Operator::Div(lt, rt)
            | Operator::Rem(lt, rt)
            | Operator::Add(lt, rt)
            | Operator::Sub(lt, rt) => {
                self.get_binary_operator_dispatch_args(lt, rt, uniform_buffer, enc)
            }
            Operator::Bmm(lt, rt) => {
                self.get_gemm_operator_dispatch_args(lt, rt, uniform_buffer, enc)
            }
        }
    }
    fn get_unary_operator_dispatch_args(
        &self,
        expr: &Expr,
        uniform_buffer: &wgpu::Buffer,
        enc: &mut wgpu::CommandEncoder,
    ) -> (u32, wgpu::Buffer, wgpu::BindGroup) {
        const WORKGROUP_SIZE: u32 = 32;
        let buf = self.eval_impl(expr, enc);
        let numel = expr.numel() as u32;
        let workgroups = (numel + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.unary_op_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf.as_entire_binding(),
                },
            ],
        });
        (workgroups, buf, bind_group)
    }
    fn get_binary_operator_dispatch_args(
        &self,
        lt: &Expr,
        rt: &Expr,
        uniform_buffer: &wgpu::Buffer,
        enc: &mut wgpu::CommandEncoder,
    ) -> (u32, wgpu::Buffer, wgpu::BindGroup) {
        const WORKGROUP_SIZE: u32 = 32;
        let lt_buf = self.eval_impl(lt, enc);
        let rt_buf = self.eval_impl(rt, enc);
        let numel = lt.numel() as u32;
        let workgroups = (numel + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.binary_op_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lt_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rt_buf.as_entire_binding(),
                },
            ],
        });
        (workgroups, lt_buf, bind_group)
    }
    fn get_gemm_operator_dispatch_args(
        &self,
        lt: &Expr,
        rt: &Expr,
        uniform_buffer: &wgpu::Buffer,
        enc: &mut wgpu::CommandEncoder,
    ) -> (u32, wgpu::Buffer, wgpu::BindGroup) {
        const WORKGROUP_SIZE: u32 = 32;
        let lt_buf = self.eval_impl(lt, enc);
        let rt_buf = self.eval_impl(rt, enc);
        let c_numel = lt.shape[0] * lt.shape[1] * rt.shape[2];
        let c_buf = self.create_wgpu_buffer(
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            Layout::array::<f32>(c_numel).unwrap().size(),
        );
        let workgroups = ((c_numel as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.gemm_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lt_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rt_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c_buf.as_entire_binding(),
                },
            ],
        });
        (workgroups, c_buf, bind_group)
    }
    fn emplace_wgpu_buffer<T: bytemuck::Pod>(
        &self,
        usages: wgpu::BufferUsages,
        data: &[T],
    ) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: true,
            size: Layout::array::<T>(data.len()).unwrap().size() as u64,
            usage: usages,
        });
        buffer
            .get_mapped_range_mut(..)
            .copy_from_slice(bytemuck::cast_slice(data));
        buffer.unmap();
        buffer
    }
    fn create_wgpu_buffer(&self, usages: wgpu::BufferUsages, nbytes: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            size: nbytes as u64,
            usage: usages,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UnaryOpParams {
    a_dim: UVec3,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryOpParams {
    a_dim: UVec3,
    b_dim: UVec3,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    a_dim: UVec3,
    b_dim: UVec3,
    c_dim: UVec3,
    a_coeff: f32,
    b_coeff: f32,
    c_coeff: f32,
    _pad: f32,
}

/// GPU-aligned vec3<u32> (padded to 16 bytes for WGSL alignment)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UVec3 {
    x: u32,
    y: u32,
    z: u32,
    _pad: u32,
}
impl UVec3 {
    fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z, _pad: 0 }
    }
    fn from_shape(shape: &[usize]) -> UVec3 {
        match shape.len() {
            1 => UVec3::new(1, 1, shape[0] as u32),
            2 => UVec3::new(1, shape[0] as u32, shape[1] as u32),
            3 => UVec3::new(shape[0] as u32, shape[1] as u32, shape[2] as u32),
            _ => panic!("Unsupported shape dimension: {}", shape.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    fn setup() -> (wgpu::Device, wgpu::Queue, Interp) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .expect("Failed to find an adapter");
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default()),
        )
        .expect("Failed to create device");
        let interp = Interp::new(&device, &queue);
        (device, queue, interp)
    }

    #[test]
    fn test_constant_readback() {
        let (device, _, interp) = setup();
        let constant = Expr::new_matrix(&device, &[[1.0, 2.0], [3.0, 4.0]]);
        let buf = interp.eval(&constant);
        let result = interp.readback(&buf, constant.numel());
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_binary_add() {
        let (device, _, interp) = setup();
        let a = Expr::new_vector(&device, &[1.0, 2.0, 3.0, 4.0]);
        let b = Expr::new_vector(&device, &[10.0, 20.0, 30.0, 40.0]);
        let expr = a + b;
        let buf = interp.eval(&expr);
        let result = interp.readback(&buf, expr.numel());
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_binary_mul() {
        let (device, _, interp) = setup();
        let a = Expr::new_vector(&device, &[1.0, 2.0, 3.0, 4.0]);
        let b = Expr::new_vector(&device, &[2.0, 3.0, 4.0, 5.0]);
        let expr = a * b;
        let buf = interp.eval(&expr);
        let result = interp.readback(&buf, expr.numel());
        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_unary_neg() {
        let (device, _, interp) = setup();
        let a = Expr::new_vector(&device, &[1.0, -2.0, 3.0, -4.0]);
        let expr = -a;
        let buf = interp.eval(&expr);
        let result = interp.readback(&buf, expr.numel());
        assert_eq!(result, vec![-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn test_bmm() {
        let (device, _, interp) = setup();
        let lt = Expr::new_tensor(&device, &[[[1.0, 0.0], [0.0, 1.0]]]);
        let rt = Expr::new_tensor(&device, &[[[2.0, 1.0], [0.0, 2.0]]]);
        let expr = lt.bmm(rt);
        let buf = interp.eval(&expr);
        let result = interp.readback(&buf, expr.numel());
        assert_eq!(result, vec![2.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_matmul() {
        let (device, _, interp) = setup();
        let lt = Expr::new_matrix(&device, &[[1.0, 0.0], [0.0, 1.0]]);
        let rt = Expr::new_matrix(&device, &[[2.0, 1.0], [0.0, 2.0]]);
        let expr = lt.matmul(rt);
        let buf = interp.eval(&expr);
        let result = interp.readback(&buf, expr.numel());
        assert_eq!(result, vec![2.0, 1.0, 0.0, 2.0]);
    }
}
