//! IR → [`VulkanProgram`] lowering: the shared WGSL emission translated to
//! SPIR-V with naga (`wgsl-in` → `spv-out`).

use std::collections::BTreeMap;

use resin_core::Tree;
use resin_ir::{BufferRef, BufferViewRef, IrKernel, IrProgram};

use super::error::VulkanLowerError;
use super::program::{
    VkBufferSpec, VkBufferViewSpec, VkDispatch, VkPipelineSpec, VkRasterStep, VkStep,
    VkTraceStep, VulkanProgram,
};
use crate::backends::wgsl::{dispatch_size_for_kernel, emit_wgsl_for_kernel, WgslKernelConfig};

/// How naga is driven for one SPIR-V translation. One place knows the
/// lang version, validation flags, and writer flags for the whole backend.
pub(super) struct SpirvOptions {
    pub stage: naga::ShaderStage,
    pub entry_point: &'static str,
    /// Enables naga's `RAY_QUERY` capability (emits `SPV_KHR_ray_query`).
    pub ray_query: bool,
    /// Off for graphics stages: y is flipped by the negative-height viewport
    /// instead of naga's coordinate-space adjustment.
    pub adjust_coordinate_space: bool,
}

pub(super) fn wgsl_to_spirv_with(
    wgsl: &str,
    options: SpirvOptions,
) -> Result<Vec<u32>, VulkanLowerError> {
    let module = naga::front::wgsl::parse_str(wgsl)
        .map_err(|e| VulkanLowerError::WgslParse(e.emit_to_string(wgsl)))?;

    let mut capabilities = naga::valid::Capabilities::default();
    if options.ray_query {
        capabilities |= naga::valid::Capabilities::RAY_QUERY;
    }
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .validate(&module)
        .map_err(|e| VulkanLowerError::WgslValidate(format!("{e:?}")))?;

    let mut spv_options = naga::back::spv::Options {
        // SPIR-V 1.4 needs a Vulkan 1.2 device (which the runtime requires)
        // and covers SPV_KHR_ray_query's entry-point interface rules.
        lang_version: (1, 4),
        ..Default::default()
    };
    if !options.adjust_coordinate_space {
        spv_options
            .flags
            .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    }
    naga::back::spv::write_vec(
        &module,
        &info,
        &spv_options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: options.stage,
            entry_point: options.entry_point.into(),
        }),
    )
    .map_err(|e| VulkanLowerError::SpvEmit(e.to_string()))
}

/// Translate WGSL to SPIR-V for a compute entry point named `main`.
pub(super) fn wgsl_to_spirv(wgsl: &str, ray_query: bool) -> Result<Vec<u32>, VulkanLowerError> {
    wgsl_to_spirv_with(
        wgsl,
        SpirvOptions {
            stage: naga::ShaderStage::Compute,
            entry_point: "main",
            ray_query,
            adjust_coordinate_space: true,
        },
    )
}

pub fn lower_ir_to_vulkan<P, S>(
    program: &IrProgram<P, S>,
    config: Option<WgslKernelConfig>,
) -> Result<VulkanProgram, VulkanLowerError>
where
    P: Tree<BufferRef>,
    S: Tree<BufferViewRef>,
{
    program.validate()?;
    let config = config.unwrap_or_default();

    let buffers: Vec<VkBufferSpec> = program
        .buffers
        .iter()
        .map(|b| {
            let count: u64 = if b.shape.is_empty() {
                1
            } else {
                b.shape.iter().map(|&d| u64::from(d)).product()
            };
            let nbytes = count * b.element_type.nbytes() as u64;
            VkBufferSpec {
                shape: b.shape.clone(),
                nbytes,
                init: b.init.clone(),
                readonly: b.readonly,
            }
        })
        .collect();

    let buffer_views: Vec<VkBufferViewSpec> = program
        .buffer_views
        .iter()
        .map(|v| VkBufferViewSpec {
            buffer_index: v.buffer_index.index(),
            offset: v.accessor.offset,
            shape: v.accessor.shape.clone(),
            pitch: v.accessor.pitch.clone(),
        })
        .collect();

    // Dedup pipelines by WGSL text before the (expensive) SPIR-V translation.
    let mut pipelines: Vec<VkPipelineSpec> = Vec::new();
    let mut pipeline_key_to_index: BTreeMap<(String, [u32; 3], u32, bool), usize> = BTreeMap::new();

    let mut queue = Vec::with_capacity(program.queue.len());
    for dispatch in &program.queue {
        let arg_buffer = |arg: usize| -> usize {
            buffer_views[dispatch.arg_view_indices[arg].index()].buffer_index
        };
        match &dispatch.kernel {
            IrKernel::TraceRays(k) => {
                queue.push(VkStep::TraceRays(VkTraceStep {
                    arg_buffer_indices: std::array::from_fn(arg_buffer),
                    output_buffer_index: dispatch.output_buffer_index.index(),
                    ray_count: k.ray_count(),
                    vertex_count: k.vertex_count(),
                    triangle_count: k.triangle_count(),
                }));
            }
            IrKernel::Rasterize(k) => {
                queue.push(VkStep::Rasterize(VkRasterStep {
                    positions_buffer_index: arg_buffer(0),
                    triangles_buffer_index: arg_buffer(1),
                    output_buffer_index: dispatch.output_buffer_index.index(),
                    height: k.height(),
                    width: k.width(),
                    triangle_count: k.triangle_count(),
                }));
            }
            kernel => {
                let wgsl = emit_wgsl_for_kernel(kernel, &config)?;
                let dispatch_size = dispatch_size_for_kernel(kernel, &config);
                let num_arg_bindings = kernel.arg_accessors().len() as u32;
                let clear = kernel.clear_output_before_dispatch();
                let key = (wgsl.clone(), dispatch_size, num_arg_bindings, clear);
                let pipeline_index = match pipeline_key_to_index.entry(key) {
                    std::collections::btree_map::Entry::Occupied(entry) => *entry.get(),
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        let idx = pipelines.len();
                        pipelines.push(VkPipelineSpec {
                            spirv: wgsl_to_spirv(&wgsl, false)?,
                            entry_point: "main".into(),
                            dispatch_size,
                            num_arg_bindings,
                            clear_output_before_dispatch: clear,
                        });
                        entry.insert(idx);
                        idx
                    }
                };

                queue.push(VkStep::Compute(VkDispatch {
                    pipeline_index,
                    arg_view_indices: dispatch
                        .arg_view_indices
                        .iter()
                        .map(|v| v.index())
                        .collect(),
                    output_buffer_index: dispatch.output_buffer_index.index(),
                }));
            }
        }
    }

    let mut param_buffer_indices = Vec::new();
    program.params.for_each_leaf(|buffer| {
        param_buffer_indices.push(buffer.index());
    });
    let mut sink_view_indices = Vec::new();
    program.sinks.for_each_leaf(|view| {
        sink_view_indices.push(view.index());
    });

    Ok(VulkanProgram {
        buffers,
        buffer_views,
        pipelines,
        queue,
        param_buffer_indices,
        sink_view_indices,
    })
}
