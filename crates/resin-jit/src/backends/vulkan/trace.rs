//! `trace_rays` executor for the Vulkan backend.
//!
//! Two modes (selected at dispatch time):
//! - **hardware**: `vkCmdBuildAccelerationStructuresKHR` + a WGSL ray-query
//!   kernel translated to `SPV_KHR_ray_query`, when the device enabled
//!   `VK_KHR_acceleration_structure` / `VK_KHR_ray_query`;
//! - **compute fallback**: host BVH build + the shared WGSL traversal kernel.

use ash::vk;

use super::error::{VkResultExt, VulkanRuntimeError};
use super::program::{VkPipelineSpec, VkTraceStep};
use super::runtime::{self, RunState, VulkanContext};
use crate::backends::hwtrace;

pub(super) fn execute(
    state: &mut RunState<'_>,
    step: &VkTraceStep,
) -> Result<(), VulkanRuntimeError> {
    if step.ray_count == 0 {
        return Ok(());
    }
    // An acceleration structure cannot be built over zero triangles; the
    // fallback's empty BVH handles that case on any device.
    if state.ctx.ray_query && step.triangle_count > 0 && !hwtrace::force_fallback() {
        execute_hardware(state, step)
    } else {
        execute_fallback(state, step)
    }
}

//
// Compute fallback
//

fn execute_fallback(
    state: &mut RunState<'_>,
    step: &VkTraceStep,
) -> Result<(), VulkanRuntimeError> {
    // Geometry may have been produced on-device by earlier steps: finish
    // pending work, read it back, and build the BVH on the host.
    state.flush()?;
    let vertex_bytes = runtime::read_buffer(state.ctx, &state.buffers[step.vertices_buffer_index()])?;
    let triangle_bytes =
        runtime::read_buffer(state.ctx, &state.buffers[step.triangles_buffer_index()])?;
    let vertices = hwtrace::bytes_to_f32(&vertex_bytes[..step.vertex_count as usize * 12]);
    let triangles = hwtrace::bytes_to_u32(&triangle_bytes[..step.triangle_count as usize * 12]);
    let bvh = hwtrace::build_bvh(&vertices, &triangles);

    // Uploaded BVH buffers append to `state.buffers`, which destroys them
    // with the run; program buffers keep their indices (append-only).
    let upload = |state: &mut RunState<'_>, words: &[u32]| -> Result<usize, VulkanRuntimeError> {
        let bytes = hwtrace::u32s_to_bytes(words);
        let buffer = runtime::create_buffer(
            state.ctx,
            bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
        )?;
        runtime::write_buffer(state.ctx, &buffer, &bytes)?;
        state.buffers.push(buffer);
        Ok(state.buffers.len() - 1)
    };
    let tris_index = upload(state, &bvh.packed_tris)?;
    let nodes_index = upload(state, &bvh.nodes)?;

    let groups_x = hwtrace::dispatch_x(step.ray_count).map_err(VulkanRuntimeError::Message)?;
    let spec = VkPipelineSpec {
        spirv: super::lower::wgsl_to_spirv(&hwtrace::fallback_kernel(step.ray_count), false)
            .map_err(|e| VulkanRuntimeError::Message(format!("fallback kernel: {e}")))?,
        entry_point: "main".into(),
        dispatch_size: [groups_x, 1, 1],
        num_arg_bindings: 7,
        clear_output_before_dispatch: false,
    };
    let pipeline = runtime::create_compute_pipeline(state, &spec)?;

    // Descriptor pool + set, all storage buffers (fallback kernel binding
    // order: output, origins, directions, t_min, t_max, vertices, tris,
    // nodes).
    let device = &state.ctx.device;
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(8)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .ctx("vkCreateDescriptorPool")?;
    state.track_descriptor_pool(pool);

    let set_layouts = [pipeline.set_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&set_layouts);
    let set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .ctx("vkAllocateDescriptorSets")?[0];

    let buffer_indices = [
        step.output_buffer_index,
        step.arg_buffer_indices[0],
        step.arg_buffer_indices[1],
        step.arg_buffer_indices[2],
        step.arg_buffer_indices[3],
        step.vertices_buffer_index(),
        tris_index,
        nodes_index,
    ];
    let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffer_indices
        .iter()
        .map(|&index| {
            vk::DescriptorBufferInfo::default()
                .buffer(state.buffers[index].buffer)
                .range(vk::WHOLE_SIZE)
        })
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
        .iter()
        .enumerate()
        .map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    state.full_barrier()?;
    let cmd = state.cmd()?;
    unsafe {
        let device = &state.ctx.device;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[set],
            &[],
        );
        device.cmd_dispatch(cmd, groups_x, 1, 1);
    }
    state.full_barrier()?;
    Ok(())
}

//
// Hardware ray query
//

/// Transient Vulkan handles for one hardware trace, destroyed after the
/// fence wait in [`RunState::flush`] guarantees the GPU is done with them.
/// Backing/scratch/instance *buffers* are pushed into `state.buffers`
/// instead (appending keeps program buffer indices valid and RunState's
/// drop destroys them after its own final wait). Handles a step failed to
/// reach stay null, which `vkDestroy*` ignores, so the same cleanup covers
/// partially recorded steps.
struct HwTrace {
    as_device: ash::khr::acceleration_structure::Device,
    blas: vk::AccelerationStructureKHR,
    tlas: vk::AccelerationStructureKHR,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
}

impl HwTrace {
    fn destroy(self, ctx: &VulkanContext) {
        let device = &ctx.device;
        unsafe {
            self.as_device.destroy_acceleration_structure(self.tlas, None);
            self.as_device.destroy_acceleration_structure(self.blas, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_shader_module(self.shader_module, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

fn execute_hardware(
    state: &mut RunState<'_>,
    step: &VkTraceStep,
) -> Result<(), VulkanRuntimeError> {
    let as_device =
        ash::khr::acceleration_structure::Device::new(&state.ctx.instance, &state.ctx.device);
    let mut hw = HwTrace {
        as_device,
        blas: vk::AccelerationStructureKHR::null(),
        tlas: vk::AccelerationStructureKHR::null(),
        set_layout: vk::DescriptorSetLayout::null(),
        pipeline_layout: vk::PipelineLayout::null(),
        shader_module: vk::ShaderModule::null(),
        pipeline: vk::Pipeline::null(),
        descriptor_pool: vk::DescriptorPool::null(),
    };
    let recorded = record_hardware(state, step, &mut hw);
    // Submit and wait even when recording failed halfway: the command buffer
    // may already reference these objects, and destroying resources a
    // recorded-but-pending command buffer uses would invalidate it.
    let flushed = state.flush();
    hw.destroy(state.ctx);
    recorded.and(flushed)
}

fn buffer_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    unsafe { device.get_buffer_device_address(&info) }
}

fn record_hardware(
    state: &mut RunState<'_>,
    step: &VkTraceStep,
    hw: &mut HwTrace,
) -> Result<(), VulkanRuntimeError> {
    let ctx = state.ctx;
    let device = &ctx.device;
    let as_device = hw.as_device.clone();

    // Make prior compute/transfer writes to the geometry buffers visible to
    // the acceleration-structure build.
    let cmd = state.cmd()?;
    barrier(
        ctx,
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::SHADER_READ,
    );

    // BLAS over the step's (device-address-capable) geometry buffers.
    let vertex_address = buffer_address(device, state.buffers[step.vertices_buffer_index()].buffer);
    let index_address = buffer_address(device, state.buffers[step.triangles_buffer_index()].buffer);
    let blas_triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertex_address,
        })
        .vertex_stride(12)
        .max_vertex(step.vertex_count.saturating_sub(1))
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_address,
        });
    let blas_geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: blas_triangles,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);
    hw.blas = build_acceleration_structure(
        state,
        &as_device,
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        blas_geometry,
        step.triangle_count,
    )?;

    // One identity-transform TLAS instance referencing the BLAS.
    let instance = vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: unsafe {
                as_device.get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(hw.blas),
                )
            },
        },
    };
    let instance_bytes = unsafe {
        std::slice::from_raw_parts(
            (&raw const instance).cast::<u8>(),
            size_of::<vk::AccelerationStructureInstanceKHR>(),
        )
    };
    let instance_buffer = runtime::create_buffer(
        ctx,
        instance_bytes.len() as u64,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        true,
    )?;
    runtime::write_buffer(ctx, &instance_buffer, instance_bytes)?;
    let instance_address = buffer_address(device, instance_buffer.buffer);
    state.buffers.push(instance_buffer);

    // The TLAS build reads the BLAS just built in this command buffer.
    let cmd = state.cmd()?;
    barrier(
        ctx,
        cmd,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
    );

    let tlas_instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: instance_address,
        });
    let tlas_geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: tlas_instances,
        });
    hw.tlas = build_acceleration_structure(
        state,
        &as_device,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        tlas_geometry,
        1,
    )?;

    // Ray queries run in the compute stage (there is no RT pipeline here).
    let cmd = state.cmd()?;
    barrier(
        ctx,
        cmd,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_READ,
    );

    // Pipeline with a custom descriptor layout: the shared all-storage
    // helper cannot express the TLAS binding.
    let spirv = super::lower::wgsl_to_spirv(&hwtrace::ray_query_kernel(step.ray_count), true)
        .map_err(|e| VulkanRuntimeError::Message(format!("ray query kernel: {e}")))?;
    let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
    hw.shader_module = unsafe { device.create_shader_module(&module_info, None) }
        .ctx("vkCreateShaderModule")?;

    let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..5)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    bindings.push(
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    );
    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    hw.set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
        .ctx("vkCreateDescriptorSetLayout")?;

    let set_layouts = [hw.set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    hw.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .ctx("vkCreatePipelineLayout")?;

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(hw.shader_module)
        .name(c"main");
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(hw.pipeline_layout);
    hw.pipeline = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .map_err(|(_, result)| VulkanRuntimeError::call("vkCreateComputePipelines", result))?[0];

    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(5),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    hw.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .ctx("vkCreateDescriptorPool")?;

    let set_layouts = [hw.set_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(hw.descriptor_pool)
        .set_layouts(&set_layouts);
    let set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .ctx("vkAllocateDescriptorSets")?[0];

    let buffer_indices = [
        step.output_buffer_index,
        step.arg_buffer_indices[0],
        step.arg_buffer_indices[1],
        step.arg_buffer_indices[2],
        step.arg_buffer_indices[3],
    ];
    let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffer_indices
        .iter()
        .map(|&index| {
            vk::DescriptorBufferInfo::default()
                .buffer(state.buffers[index].buffer)
                .range(vk::WHOLE_SIZE)
        })
        .collect();
    let mut writes: Vec<vk::WriteDescriptorSet> = buffer_infos
        .iter()
        .enumerate()
        .map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
    let structures = [hw.tlas];
    let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
        .acceleration_structures(&structures);
    let mut tlas_write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(5)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .push_next(&mut as_write);
    // push_next chains the AS payload but cannot infer the count from it.
    tlas_write.descriptor_count = 1;
    writes.push(tlas_write);
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    let groups_x = hwtrace::dispatch_x(step.ray_count).map_err(VulkanRuntimeError::Message)?;
    let cmd = state.cmd()?;
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, hw.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            hw.pipeline_layout,
            0,
            &[set],
            &[],
        );
        device.cmd_dispatch(cmd, groups_x, 1, 1);
    }
    state.full_barrier()?;
    Ok(())
}

/// Implementation minimum for acceleration-structure scratch addresses
/// (VkPhysicalDeviceAccelerationStructurePropertiesKHR).
fn scratch_offset_alignment(state: &RunState<'_>) -> u64 {
    let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut as_props);
    unsafe {
        state
            .ctx
            .instance
            .get_physical_device_properties2(state.ctx.physical_device, &mut props);
    }
    u64::from(as_props.min_acceleration_structure_scratch_offset_alignment.max(1))
}

/// Query build sizes, create backing + scratch buffers (pushed into
/// `state.buffers` for end-of-run destruction) and the AS handle, and record
/// the build into the current command buffer. The caller owns the returned
/// AS handle's destruction post-fence.
fn build_acceleration_structure(
    state: &mut RunState<'_>,
    as_device: &ash::khr::acceleration_structure::Device,
    ty: vk::AccelerationStructureTypeKHR,
    geometry: vk::AccelerationStructureGeometryKHR<'_>,
    primitive_count: u32,
) -> Result<vk::AccelerationStructureKHR, VulkanRuntimeError> {
    let ctx = state.ctx;
    let geometries = [geometry];
    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        as_device.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[primitive_count],
            &mut sizes,
        );
    }

    let backing = runtime::create_buffer(
        ctx,
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        true,
    )?;
    let backing_buffer = backing.buffer;
    state.buffers.push(backing);
    // The scratch device address must honor the implementation's
    // minAccelerationStructureScratchOffsetAlignment; the allocation's own
    // alignment is not guaranteed to, so over-allocate and round up.
    let scratch_alignment = scratch_offset_alignment(state);
    let scratch = runtime::create_buffer(
        ctx,
        sizes.build_scratch_size + scratch_alignment,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        true,
    )?;
    let scratch_address =
        buffer_address(&ctx.device, scratch.buffer).next_multiple_of(scratch_alignment);
    state.buffers.push(scratch);

    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(backing_buffer)
        .size(sizes.acceleration_structure_size)
        .ty(ty);
    let acceleration_structure =
        unsafe { as_device.create_acceleration_structure(&create_info, None) }
            .ctx("vkCreateAccelerationStructureKHR")?;

    build_info = build_info
        .dst_acceleration_structure(acceleration_structure)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_address,
        });
    let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
        .primitive_count(primitive_count);
    let cmd = match state.cmd() {
        Ok(cmd) => cmd,
        Err(err) => {
            unsafe { as_device.destroy_acceleration_structure(acceleration_structure, None) };
            return Err(err);
        }
    };
    unsafe {
        as_device.cmd_build_acceleration_structures(cmd, &[build_info], &[&[range]]);
    }

    Ok(acceleration_structure)
}

fn barrier(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_stage: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}
