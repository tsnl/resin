//! `rasterize` executor for the Vulkan backend: a render pass drawing
//! non-indexed triangles (vertex pulling from the clip-position storage
//! buffer) into an RGBA32Float visibility attachment with a D32 depth buffer,
//! then copied into the output storage buffer.
//!
//! The shared WGSL shaders are translated per stage through naga at execute
//! time (the graphics-pipeline analogue of the compute lowering). Vulkan's
//! NDC y points down; the node's convention is y up, so the viewport flips y
//! with a negative height (core since Vulkan 1.1) and naga's own
//! `ADJUST_COORDINATE_SPACE` flip is disabled to avoid double-flipping.
//! `cmd_copy_image_to_buffer` with `buffer_row_length = 0` is tightly packed,
//! landing exactly as dense `[H, W, 4]` f32 in the output buffer.

use ash::vk;

use super::error::{VkResultExt, VulkanRuntimeError};
use super::program::VkRasterStep;
use super::runtime::RunState;
use crate::backends::hwraster;

const COLOR_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Translate one stage of the shared raster WGSL to SPIR-V.
fn compile_stage(
    stage: naga::ShaderStage,
    entry_point: &'static str,
) -> Result<Vec<u32>, VulkanRuntimeError> {
    super::lower::wgsl_to_spirv_with(
        hwraster::RASTER_WGSL,
        super::lower::SpirvOptions {
            stage,
            entry_point,
            ray_query: false,
            adjust_coordinate_space: false,
        },
    )
    .map_err(|e| VulkanRuntimeError::Message(format!("raster shader: {e}")))
}

/// One render target image with its allocation and view.
#[derive(Default)]
struct ImageTarget {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// Everything created for one rasterize step. Filled progressively so a
/// mid-construction failure still destroys the partial state; `vkDestroy*` /
/// `vkFreeMemory` ignore null handles.
#[derive(Default)]
struct Resources {
    color: ImageTarget,
    depth: ImageTarget,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
}

impl Resources {
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_shader_module(self.vertex_module, None);
            device.destroy_shader_module(self.fragment_module, None);
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_render_pass(self.render_pass, None);
            for target in [&self.color, &self.depth] {
                device.destroy_image_view(target.view, None);
                device.destroy_image(target.image, None);
                device.free_memory(target.memory, None);
            }
        }
    }
}

/// Prefer device-local image memory, else anything the image accepts
/// (lavapipe is UMA — everything is host-visible).
fn image_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
) -> Result<u32, VulkanRuntimeError> {
    let pick = |flags: vk::MemoryPropertyFlags| {
        (0..props.memory_type_count).find(|&i| {
            (type_bits & (1 << i)) != 0
                && props.memory_types[i as usize].property_flags.contains(flags)
        })
    };
    pick(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| pick(vk::MemoryPropertyFlags::empty()))
        .ok_or_else(|| VulkanRuntimeError::Message("no memory type for raster images".into()))
}

fn create_target(
    ctx: &super::runtime::VulkanContext,
    extent: vk::Extent2D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    target: &mut ImageTarget,
) -> Result<(), VulkanRuntimeError> {
    let device = &ctx.device;
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    target.image = unsafe { device.create_image(&image_info, None) }.ctx("vkCreateImage")?;

    let requirements = unsafe { device.get_image_memory_requirements(target.image) };
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(image_memory_type(
            &ctx.memory_props,
            requirements.memory_type_bits,
        )?);
    target.memory = unsafe { device.allocate_memory(&alloc, None) }.ctx("vkAllocateMemory")?;
    unsafe { device.bind_image_memory(target.image, target.memory, 0) }
        .ctx("vkBindImageMemory")?;

    let view_info = vk::ImageViewCreateInfo::default()
        .image(target.image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    target.view = unsafe { device.create_image_view(&view_info, None) }.ctx("vkCreateImageView")?;
    Ok(())
}

pub(super) fn execute(
    state: &mut RunState<'_>,
    step: &VkRasterStep,
) -> Result<(), VulkanRuntimeError> {
    let mut res = Resources::default();
    let recorded = record(state, step, &mut res);
    // The queue must go idle before the local objects can die; flush even on
    // failure so partially recorded work never outlives them.
    let flushed = state.flush();
    res.destroy(&state.ctx.device);
    recorded.and(flushed)
}

fn record(
    state: &mut RunState<'_>,
    step: &VkRasterStep,
    res: &mut Resources,
) -> Result<(), VulkanRuntimeError> {
    let ctx = state.ctx;
    let device = &ctx.device;
    let extent = vk::Extent2D {
        width: step.width,
        height: step.height,
    };
    let positions = state.buffers[step.positions_buffer_index].buffer;
    let triangles = state.buffers[step.triangles_buffer_index].buffer;
    let output = state.buffers[step.output_buffer_index].buffer;

    let vs_spirv = compile_stage(naga::ShaderStage::Vertex, hwraster::VS_ENTRY)?;
    let fs_spirv = compile_stage(naga::ShaderStage::Fragment, hwraster::FS_ENTRY)?;
    res.vertex_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs_spirv), None)
    }
    .ctx("vkCreateShaderModule")?;
    res.fragment_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fs_spirv), None)
    }
    .ctx("vkCreateShaderModule")?;

    create_target(
        ctx,
        extent,
        COLOR_FORMAT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        &mut res.color,
    )?;
    create_target(
        ctx,
        extent,
        DEPTH_FORMAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
        &mut res.depth,
    )?;

    let attachments = [
        vk::AttachmentDescription::default()
            .format(COLOR_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(DEPTH_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)
        .depth_stencil_attachment(&depth_ref)];
    // Correctness-first, like the compute runtime's full barriers: make all
    // prior writes (compute-produced clip positions) visible to the pass,
    // and the attachment writes visible to the following copy.
    let full_dependency = |src: u32, dst: u32| {
        vk::SubpassDependency::default()
            .src_subpass(src)
            .dst_subpass(dst)
            .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
    };
    let dependencies = [
        full_dependency(vk::SUBPASS_EXTERNAL, 0),
        full_dependency(0, vk::SUBPASS_EXTERNAL),
    ];
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    res.render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }
        .ctx("vkCreateRenderPass")?;

    let framebuffer_views = [res.color.view, res.depth.view];
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(res.render_pass)
        .attachments(&framebuffer_views)
        .width(extent.width)
        .height(extent.height)
        .layers(1);
    res.framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None) }
        .ctx("vkCreateFramebuffer")?;

    // Bindings 0 (positions) and 1 (triangles): vertex-pulled storage.
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
        })
        .collect();
    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    res.set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
        .ctx("vkCreateDescriptorSetLayout")?;
    let set_layouts = [res.set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    res.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .ctx("vkCreatePipelineLayout")?;

    let vs_entry = std::ffi::CString::new(hwraster::VS_ENTRY)
        .map_err(|_| VulkanRuntimeError::Message("bad vertex entry point name".into()))?;
    let fs_entry = std::ffi::CString::new(hwraster::FS_ENTRY)
        .map_err(|_| VulkanRuntimeError::Message("bad fragment entry point name".into()))?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(res.vertex_module)
            .name(&vs_entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(res.fragment_module)
            .name(&fs_entry),
    ];
    // Vertex pulling: no vertex input state.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    // Negative height flips Vulkan's y-down NDC to the node's y-up convention.
    let viewports = [vk::Viewport {
        x: 0.0,
        y: extent.height as f32,
        width: extent.width as f32,
        height: -(extent.height as f32),
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    }];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Strict '<': the first-drawn (lowest prim) triangle wins ties.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(res.pipeline_layout)
        .render_pass(res.render_pass)
        .subpass(0);
    res.pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .map_err(|(_, result)| VulkanRuntimeError::call("vkCreateGraphicsPipelines", result))?[0];

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(2)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    res.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .ctx("vkCreateDescriptorPool")?;
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(res.descriptor_pool)
        .set_layouts(&set_layouts);
    let set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .ctx("vkAllocateDescriptorSets")?[0];
    let buffer_infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(positions)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(triangles)
            .range(vk::WHOLE_SIZE),
    ];
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

    // Record: render pass → tightly packed copy into the output buffer. All
    // fallible calls happened above, so the pass is never left open.
    let cmd = state.cmd()?;
    let clear_values = [
        vk::ClearValue {
            // Background pixels stay all-zero.
            color: vk::ClearColorValue { float32: [0.0; 4] },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];
    let begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(res.render_pass)
        .framebuffer(res.framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        })
        .clear_values(&clear_values);
    unsafe {
        device.cmd_begin_render_pass(cmd, &begin_info, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, res.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            res.pipeline_layout,
            0,
            &[set],
            &[],
        );
        device.cmd_draw(cmd, step.triangle_count * 3, 1, 0, 0);
        device.cmd_end_render_pass(cmd);

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            });
        device.cmd_copy_image_to_buffer(
            cmd,
            res.color.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            output,
            &[region],
        );
    }
    // Make the copy visible to subsequent compute/transfer steps.
    state.full_barrier()
}
