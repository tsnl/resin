//! Naive Vulkan runtime driven with `ash`.
//!
//! Correctness-first, mirroring the wgpu runtime's philosophy: per-invoke
//! resources (buffers, pipelines, descriptor sets, command buffers) are
//! created from the lowered [`VulkanProgram`], executed with conservative
//! full barriers between dispatches, and destroyed afterwards. All memory is
//! host-visible + coherent (upload and readback are plain memcpys around a
//! fence wait); a staging/device-local split is a later optimization.
//!
//! Device requirements: Vulkan 1.2 with a compute queue. When
//! `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` (and their
//! dependencies) are present they are enabled, so `trace_rays` can take the
//! hardware path.

use std::ffi::CStr;
use std::sync::OnceLock;

use ash::vk;

use super::error::{VkResultExt, VulkanRuntimeError};
use super::program::{VkDispatch, VkPipelineSpec, VkStep, VulkanProgram};

pub struct VulkanContext {
    /// Kept alive for the device (the loader must outlive the instance; the
    /// context itself is never dropped once created in the static).
    #[allow(dead_code)]
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family_index: u32,
    pub queue: vk::Queue,
    pub memory_props: vk::PhysicalDeviceMemoryProperties,
    /// Hardware ray-query support (extensions enabled at device creation).
    pub ray_query: bool,
}

// The raw handles are externally synchronized by the queue-level locking the
// naive runtime performs (one submission at a time behind a mutex-free
// single-threaded invoke; wgpu's shared-context pattern applies here too).
unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

pub fn shared_context() -> Result<&'static VulkanContext, VulkanRuntimeError> {
    static CTX: OnceLock<Result<VulkanContext, String>> = OnceLock::new();
    match CTX.get_or_init(|| create_context().map_err(|e| e.to_string())) {
        Ok(ctx) => Ok(ctx),
        Err(msg) => Err(VulkanRuntimeError::Message(msg.clone())),
    }
}

const RT_EXTENSIONS: [&CStr; 3] = [
    ash::khr::acceleration_structure::NAME,
    ash::khr::ray_query::NAME,
    ash::khr::deferred_host_operations::NAME,
];

fn create_context() -> Result<VulkanContext, VulkanRuntimeError> {
    let entry = unsafe { ash::Entry::load() }
        .map_err(|e| VulkanRuntimeError::NoVulkan(e.to_string()))?;

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"resin")
        .api_version(vk::API_VERSION_1_2);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None) }
        .ctx("vkCreateInstance")?;

    let physical_devices =
        unsafe { instance.enumerate_physical_devices() }.ctx("vkEnumeratePhysicalDevices")?;

    // Prefer discrete > integrated > anything else with a compute queue.
    let mut best: Option<(vk::PhysicalDevice, u32, i32)> = None;
    for pd in physical_devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if vk::api_version_major(props.api_version) == 1
            && vk::api_version_minor(props.api_version) < 2
        {
            continue;
        }
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let Some(qfi) = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };
        if best.map(|(_, _, s)| score > s).unwrap_or(true) {
            best = Some((pd, qfi as u32, score));
        }
    }
    let (physical_device, queue_family_index, _) = best.ok_or(VulkanRuntimeError::NoDevice)?;

    // Enable ray-query extensions opportunistically.
    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .ctx("vkEnumerateDeviceExtensionProperties")?;
    let has_ext = |name: &CStr| {
        available
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(name))
    };
    let ray_query = RT_EXTENSIONS.iter().all(|name| has_ext(name));

    let queue_priorities = [1.0f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities)];

    let extension_ptrs: Vec<*const i8> = if ray_query {
        RT_EXTENSIONS.iter().map(|name| name.as_ptr()).collect()
    } else {
        Vec::new()
    };

    let mut features12 =
        vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(ray_query);
    let mut as_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
        .acceleration_structure(true);
    let mut rq_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);

    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&extension_ptrs)
        .push_next(&mut features12);
    if ray_query {
        device_info = device_info.push_next(&mut as_features).push_next(&mut rq_features);
    }

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .ctx("vkCreateDevice")?;
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
    let memory_props =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    Ok(VulkanContext {
        entry,
        instance,
        physical_device,
        device,
        queue_family_index,
        queue,
        memory_props,
        ray_query,
    })
}

/// One device buffer with its (host-visible, coherent) allocation.
pub(super) struct VkBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

impl VkBuffer {
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, VulkanRuntimeError> {
    (0..props.memory_type_count)
        .find(|&i| {
            (type_bits & (1 << i)) != 0
                && props.memory_types[i as usize].property_flags.contains(flags)
        })
        .ok_or_else(|| VulkanRuntimeError::Message("no host-visible memory type".into()))
}

pub(super) fn create_buffer(
    ctx: &VulkanContext,
    size: u64,
    usage: vk::BufferUsageFlags,
    device_address: bool,
) -> Result<VkBuffer, VulkanRuntimeError> {
    let device = &ctx.device;
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }.ctx("vkCreateBuffer")?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type = find_memory_type(
        &ctx.memory_props,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let mut flags_info =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let mut alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    if device_address {
        alloc = alloc.push_next(&mut flags_info);
    }
    let memory = match unsafe { device.allocate_memory(&alloc, None) }.ctx("vkAllocateMemory") {
        Ok(memory) => memory,
        Err(err) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(err);
        }
    };
    if let Err(err) =
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }.ctx("vkBindBufferMemory")
    {
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        return Err(err);
    }
    Ok(VkBuffer {
        buffer,
        memory,
        size,
    })
}

pub(super) fn write_buffer(
    ctx: &VulkanContext,
    buffer: &VkBuffer,
    bytes: &[u8],
) -> Result<(), VulkanRuntimeError> {
    debug_assert!(bytes.len() as u64 <= buffer.size);
    unsafe {
        let ptr = ctx
            .device
            .map_memory(buffer.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .ctx("vkMapMemory")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
        ctx.device.unmap_memory(buffer.memory);
    }
    Ok(())
}

pub(super) fn read_buffer(
    ctx: &VulkanContext,
    buffer: &VkBuffer,
) -> Result<Vec<u8>, VulkanRuntimeError> {
    let mut out = vec![0u8; buffer.size as usize];
    unsafe {
        let ptr = ctx
            .device
            .map_memory(buffer.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .ctx("vkMapMemory")?;
        std::ptr::copy_nonoverlapping(ptr.cast(), out.as_mut_ptr(), out.len());
        ctx.device.unmap_memory(buffer.memory);
    }
    Ok(out)
}

/// Per-invoke execution state: owns every transient Vulkan object and
/// destroys them on drop. Step executors record into [`RunState::cmd`] and
/// may [`RunState::flush`] when they need host-side results mid-queue.
pub(super) struct RunState<'a> {
    pub ctx: &'static VulkanContext,
    pub program: &'a VulkanProgram,
    pub buffers: Vec<VkBuffer>,
    command_pool: vk::CommandPool,
    cmd: Option<vk::CommandBuffer>,
    /// Transient objects destroyed on drop (after a final wait).
    pipelines: Vec<vk::Pipeline>,
    pipeline_layouts: Vec<vk::PipelineLayout>,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    shader_modules: Vec<vk::ShaderModule>,
    descriptor_pools: Vec<vk::DescriptorPool>,
}

impl<'a> RunState<'a> {
    pub fn new(
        ctx: &'static VulkanContext,
        program: &'a VulkanProgram,
    ) -> Result<Self, VulkanRuntimeError> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { ctx.device.create_command_pool(&pool_info, None) }
            .ctx("vkCreateCommandPool")?;
        Ok(Self {
            ctx,
            program,
            buffers: Vec::new(),
            command_pool,
            cmd: None,
            pipelines: Vec::new(),
            pipeline_layouts: Vec::new(),
            set_layouts: Vec::new(),
            shader_modules: Vec::new(),
            descriptor_pools: Vec::new(),
        })
    }

    /// Current command buffer, beginning one if needed.
    pub fn cmd(&mut self) -> Result<vk::CommandBuffer, VulkanRuntimeError> {
        if let Some(cmd) = self.cmd {
            return Ok(cmd);
        }
        let device = &self.ctx.device;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc) }
            .ctx("vkAllocateCommandBuffers")?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd, &begin) }.ctx("vkBeginCommandBuffer")?;
        self.cmd = Some(cmd);
        Ok(cmd)
    }

    /// Submit recorded work and wait for completion.
    pub fn flush(&mut self) -> Result<(), VulkanRuntimeError> {
        let Some(cmd) = self.cmd.take() else {
            return Ok(());
        };
        let device = &self.ctx.device;
        unsafe { device.end_command_buffer(cmd) }.ctx("vkEndCommandBuffer")?;
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .ctx("vkCreateFence")?;
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let result = unsafe { device.queue_submit(self.ctx.queue, &[submit], fence) }
            .ctx("vkQueueSubmit")
            .and_then(|()| {
                unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }
                    .ctx("vkWaitForFences")
            });
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &cmds);
        }
        result
    }

    /// Full-strength barrier: make prior compute writes and transfers visible
    /// to subsequent compute and transfer stages.
    pub fn full_barrier(&mut self) -> Result<(), VulkanRuntimeError> {
        let cmd = self.cmd()?;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            self.ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    pub fn track_descriptor_pool(&mut self, pool: vk::DescriptorPool) {
        self.descriptor_pools.push(pool);
    }
}

impl Drop for RunState<'_> {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        // Best-effort: never leave the GPU running while freeing resources.
        let _ = self.flush();
        unsafe {
            for &pipeline in &self.pipelines {
                device.destroy_pipeline(pipeline, None);
            }
            for &layout in &self.pipeline_layouts {
                device.destroy_pipeline_layout(layout, None);
            }
            for &layout in &self.set_layouts {
                device.destroy_descriptor_set_layout(layout, None);
            }
            for &module in &self.shader_modules {
                device.destroy_shader_module(module, None);
            }
            for &pool in &self.descriptor_pools {
                device.destroy_descriptor_pool(pool, None);
            }
            device.destroy_command_pool(self.command_pool, None);
            for buffer in &self.buffers {
                buffer.destroy(device);
            }
        }
    }
}

/// One compiled compute pipeline with its layouts (owned by the
/// [`RunState`] cleanup lists).
pub(super) struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
}

pub(super) fn create_compute_pipeline(
    state: &mut RunState<'_>,
    spec: &VkPipelineSpec,
) -> Result<ComputePipeline, VulkanRuntimeError> {
    let device = &state.ctx.device;

    let module_info = vk::ShaderModuleCreateInfo::default().code(&spec.spirv);
    let module = unsafe { device.create_shader_module(&module_info, None) }
        .ctx("vkCreateShaderModule")?;
    state.shader_modules.push(module);

    // Binding 0 = output, 1.. = args (matches the shared WGSL emission).
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..=spec.num_arg_bindings)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
        .ctx("vkCreateDescriptorSetLayout")?;
    state.set_layouts.push(set_layout);

    let set_layouts = [set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .ctx("vkCreatePipelineLayout")?;
    state.pipeline_layouts.push(layout);

    let entry = std::ffi::CString::new(spec.entry_point.as_str())
        .map_err(|_| VulkanRuntimeError::Message("bad entry point name".into()))?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipeline = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .map_err(|(_, result)| VulkanRuntimeError::call("vkCreateComputePipelines", result))?[0];
    state.pipelines.push(pipeline);

    Ok(ComputePipeline {
        pipeline,
        layout,
        set_layout,
    })
}

/// Run one program: upload params → execute steps → densify sinks to host.
pub fn run_program(
    ctx: &'static VulkanContext,
    program: &VulkanProgram,
    param_bytes: &[(usize, &[u8])],
    sink_out: &mut [(usize, &mut [u8])],
) -> Result<(), VulkanRuntimeError> {
    let mut state = RunState::new(ctx, program)?;

    // Geometry consumed by acceleration-structure builds needs device
    // addresses and AS-build-input usage.
    let geometry_buffers: std::collections::HashSet<usize> = if ctx.ray_query {
        program.trace_geometry_buffer_indices().collect()
    } else {
        Default::default()
    };

    for (index, spec) in program.buffers.iter().enumerate() {
        let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let device_address = geometry_buffers.contains(&index);
        if device_address {
            usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        }
        let buffer = create_buffer(ctx, spec.byte_len(), usage, device_address)?;
        if let Some(init) = &spec.init {
            write_buffer(ctx, &buffer, init)?;
        }
        state.buffers.push(buffer);
    }

    for &(index, bytes) in param_bytes {
        let spec = program
            .buffers
            .get(index)
            .ok_or_else(|| VulkanRuntimeError::Message(format!("bad param buffer {index}")))?;
        if bytes.len() as u64 != spec.nbytes {
            return Err(VulkanRuntimeError::Message(format!(
                "param buffer {index} size mismatch: expected {}, got {}",
                spec.nbytes,
                bytes.len()
            )));
        }
        write_buffer(ctx, &state.buffers[index], bytes)?;
    }

    // One pipeline + descriptor set per unique pipeline spec / dispatch.
    let pipelines: Vec<ComputePipeline> = program
        .pipelines
        .iter()
        .map(|spec| create_compute_pipeline(&mut state, spec))
        .collect::<Result<_, _>>()?;

    for step in &program.queue {
        match step {
            VkStep::Compute(dispatch) => {
                encode_compute(&mut state, dispatch, &pipelines)?;
            }
            VkStep::TraceRays(step) => super::trace::execute(&mut state, step)?,
            VkStep::Rasterize(step) => super::raster::execute(&mut state, step)?,
        }
    }

    state.flush()?;

    for entry in sink_out.iter_mut() {
        let (view_index, host_out) = entry;
        let view = &program.buffer_views[*view_index];
        let raw = read_buffer(ctx, &state.buffers[view.buffer_index])?;
        let densified = crate::backends::densify_view(&raw, view.offset, &view.shape, &view.pitch)
            .map_err(VulkanRuntimeError::Message)?;
        if densified.len() != host_out.len() {
            return Err(VulkanRuntimeError::Message(format!(
                "sink size mismatch: densified {} vs host {}",
                densified.len(),
                host_out.len()
            )));
        }
        host_out.copy_from_slice(&densified);
    }

    Ok(())
}

fn encode_compute(
    state: &mut RunState<'_>,
    dispatch: &VkDispatch,
    pipelines: &[ComputePipeline],
) -> Result<(), VulkanRuntimeError> {
    let spec = &state.program.pipelines[dispatch.pipeline_index];
    let pipeline = &pipelines[dispatch.pipeline_index];
    let device = &state.ctx.device;

    // Descriptor pool + set for this dispatch (naive: one pool per dispatch).
    let binding_count = spec.num_arg_bindings + 1;
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(binding_count)];
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

    let mut buffer_infos = Vec::with_capacity(binding_count as usize);
    buffer_infos.push(
        vk::DescriptorBufferInfo::default()
            .buffer(state.buffers[dispatch.output_buffer_index].buffer)
            .range(vk::WHOLE_SIZE),
    );
    for &view_index in &dispatch.arg_view_indices {
        let buffer_index = state.program.buffer_views[view_index].buffer_index;
        buffer_infos.push(
            vk::DescriptorBufferInfo::default()
                .buffer(state.buffers[buffer_index].buffer)
                .range(vk::WHOLE_SIZE),
        );
    }
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

    let cmd = state.cmd()?;
    if spec.clear_output_before_dispatch {
        unsafe {
            device.cmd_fill_buffer(
                cmd,
                state.buffers[dispatch.output_buffer_index].buffer,
                0,
                vk::WHOLE_SIZE,
                0,
            );
        }
    }
    state.full_barrier()?;
    let cmd = state.cmd()?;
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[set],
            &[],
        );
        let [x, y, z] = spec.dispatch_size;
        if x > 0 {
            device.cmd_dispatch(cmd, x, y, z);
        }
    }
    state.full_barrier()?;
    Ok(())
}

