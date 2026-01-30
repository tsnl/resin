use std::alloc::Layout;

use crate::expr::{Detail, Expr, Operator};
use wgpu::include_wgsl;

pub struct Interp {
    device: wgpu::Device,
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
    pub fn new(device: &wgpu::Device) -> Self {
        let binary_op_shader_module =
            device.create_shader_module(include_wgsl!("shaders/binary_op.wgsl"));
        let unary_op_shader_module =
            device.create_shader_module(include_wgsl!("shaders/unary_op.wgsl"));
        let gemm_shader_module = device.create_shader_module(include_wgsl!("shaders/gemm.wgsl"));

        let binary_op_pipeline_layout = Self::create_binary_op_pipeline_layout(device);
        let unary_op_pipeline_layout = Self::create_unary_op_pipeline_layout(device);
        let gemm_pipeline_layout = Self::create_gemm_pipeline_layout(device);

        Self {
            device: device.clone(),
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
    fn create_binary_op_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
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
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("binary_op_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        })
    }
    fn create_unary_op_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
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
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unary_op_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        })
    }
    fn create_gemm_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
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
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        })
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
    pub fn eval(&self, expr: &Expr) -> Expr {
        todo!()
    }
    fn eval_impl(&self, expr: &Expr, enc: &mut wgpu::CommandEncoder) -> wgpu::Buffer {
        match &expr.detail {
            Detail::Constant(constant) => self.eval_constant_impl(constant, enc),
            Detail::Operator(operator) => self.eval_operator_impl(operator, enc),
        }
    }
    fn eval_constant_impl(&self, constant: &[f32], enc: &wgpu::CommandEncoder) -> wgpu::Buffer {
        self.emplace_wgpu_buffer(wgpu::BufferUsages::STORAGE, constant)
    }
    fn eval_operator_impl(
        &self,
        operator: &Operator,
        uniform_buffer: &wgpu::Buffer,
        enc: &mut wgpu::CommandEncoder,
    ) -> wgpu::Buffer {
        let pipeline = self.get_operator_pipeline(operator);
        let uniform_buffer = self.get_operator_uniform_buffer(operator);
        let (n, res, bg) = self.get_operator_dispatch_args(operator, &uniform_buffer, enc);

        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n, 1, 1);
        res
    }
    fn get_operator_pipeline(&self, operator: &Operator) -> &wgpu::ComputePipeline {
        todo!()
    }
    fn get_operator_uniform_buffer(&self, operator: &Operator) -> wgpu::Buffer {
        match operator {
            // Unary operators:
            // TODO

            // Binary operators:
            Operator::Pow(lt, rt)
            | Operator::Mul(lt, rt)
            | Operator::Div(lt, rt)
            | Operator::Rem(lt, rt)
            | Operator::Add(lt, rt)
            | Operator::Sub(lt, rt) => self.emplace_wgpu_buffer(
                wgpu::BufferUsages::UNIFORM,
                &[BinaryOpParams {
                    a_dim: SimdUVec3::from([lt.shape[0], lt.shape[1], lt.shape[2]]),
                    b_dim: SimdUVec3::from([rt.shape[0], rt.shape[1], rt.shape[2]]),
                }],
            ),

            // GEMM operator:
            // TODO

            // Placeholder until above todos are done
            _ => todo!(),
        }
    }
    fn get_operator_dispatch_args(
        &self,
        operator: &Operator,
        uniform_buffer: &wgpu::Buffer,
        enc: &mut wgpu::CommandEncoder,
    ) -> (u32, wgpu::Buffer, wgpu::BindGroup) {
        match operator {
            Operator::Pow(lt, rt) => {
                let lt_buf = self.eval_impl(lt, enc);
                let rt_buf = self.eval_impl(rt, enc);
                self.device
                    .create_bind_group(&wgpu::BindGroupDescriptor { entries: &[] })
            }
            _ => todo!(),
        }
    }
    fn emplace_wgpu_buffer<T: bytemuck::Pod>(
        &self,
        usages: wgpu::BufferUsages,
        data: &[T],
    ) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: true,
            size: std::mem::size_of::<T>() as u64 * data.len() as u64,
            usage: usages,
        });
        buffer
            .get_mapped_range_mut(..)
            .copy_from_slice(bytemuck::cast_slice(data));
        buffer.unmap();
        buffer
    }
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryOpParams {
    a_dim: simd_math::SimdUVec3<u32>,
    b_dim: simd_math::SimdUVec3<u32>,
}
