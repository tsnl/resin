use std::slice;

use ash::{Device, vk};

use super::{COLOR_FORMAT, ResinGpu, ResinStatus, vk_status};

pub struct ResinPipeline {
    pub(super) device: Device,
    pub(super) handle: vk::Pipeline,
    pub(super) bind_point: vk::PipelineBindPoint,
}

impl ResinGpu {
    /// # Safety
    /// The SPIR-V must be valid for this device and the runtime's entry point and push-constant interface. The GPU must outlive the pipeline.
    pub unsafe fn create_compute_pipeline(&self, spv: &[u8]) -> Result<ResinPipeline, ResinStatus> {
        let shader = ShaderModule::create(&self.device, spv)?;
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(shader.stage(vk::ShaderStageFlags::COMPUTE))
            .layout(self.push_layout);
        let result = unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(&info),
                None,
            )
        };
        ResinPipeline::create(&self.device, vk::PipelineBindPoint::COMPUTE, result)
    }

    /// # Safety
    /// Both SPIR-V modules must be valid for this device and the runtime's shader interface. The GPU must outlive the pipeline.
    pub unsafe fn create_graphics_pipeline(
        &self,
        vertex_spv: &[u8],
        fragment_spv: &[u8],
    ) -> Result<ResinPipeline, ResinStatus> {
        let vertex = ShaderModule::create(&self.device, vertex_spv)?;
        let fragment = ShaderModule::create(&self.device, fragment_spv)?;
        let stages = [
            vertex.stage(vk::ShaderStageFlags::VERTEX),
            fragment.stage(vk::ShaderStageFlags::FRAGMENT),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(slice::from_ref(&attachment));
        let dynamic = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(slice::from_ref(&COLOR_FORMAT));
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.push_layout)
            .push_next(&mut rendering);
        let result = unsafe {
            self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(&info),
                None,
            )
        };
        ResinPipeline::create(&self.device, vk::PipelineBindPoint::GRAPHICS, result)
    }
}

impl ResinPipeline {
    fn create(
        device: &Device,
        bind_point: vk::PipelineBindPoint,
        result: Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
    ) -> Result<Self, ResinStatus> {
        match result {
            Ok(pipelines) => Ok(Self {
                device: device.clone(),
                handle: pipelines[0],
                bind_point,
            }),
            Err((pipelines, error)) => {
                for pipeline in pipelines {
                    unsafe { device.destroy_pipeline(pipeline, None) };
                }
                Err(vk_status(error))
            }
        }
    }
}

impl Drop for ResinPipeline {
    fn drop(&mut self) {
        unsafe { self.device.destroy_pipeline(self.handle, None) };
    }
}

struct ShaderModule<'a> {
    device: &'a Device,
    handle: vk::ShaderModule,
}

impl<'a> ShaderModule<'a> {
    fn create(device: &'a Device, bytes: &[u8]) -> Result<Self, ResinStatus> {
        let words = spirv_words(bytes)?;
        let info = vk::ShaderModuleCreateInfo::default().code(&words);
        let handle = unsafe { device.create_shader_module(&info, None) }.map_err(vk_status)?;
        Ok(Self { device, handle })
    }

    fn stage(&self, stage: vk::ShaderStageFlags) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(stage)
            .module(self.handle)
            .name(c"main")
    }
}

impl Drop for ShaderModule<'_> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_shader_module(self.handle, None) };
    }
}

fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>, ResinStatus> {
    if bytes.len() < 20 || !bytes.len().is_multiple_of(4) {
        return Err(ResinStatus::InvalidArgument);
    }
    let words: Vec<_> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    if words[0] != 0x0723_0203 {
        return Err(ResinStatus::InvalidArgument);
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_header_is_checked() {
        for bytes in [&[][..], &[3, 2, 35, 7], &[0; 20], &[0; 21]] {
            assert_eq!(spirv_words(bytes), Err(ResinStatus::InvalidArgument));
        }
    }

    #[test]
    fn spirv_bytes_need_not_be_word_aligned() {
        let words = [0x0723_0203u32, 0x0001_0600, 0, 1, 0];
        let mut bytes = vec![0];
        bytes.extend(words.iter().flat_map(|word| word.to_le_bytes()));
        assert_eq!(spirv_words(&bytes[1..]).unwrap(), words);
    }
}
