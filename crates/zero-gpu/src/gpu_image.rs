use super::*;

pub struct GpuImage {
    device: Arc<GpuDevice>,
    handle: vk::Image,
    width: u32,
    height: u32,
    dtype: GpuPixelFormat,
}
pub struct GpuImageConfig {
    pub width: u32,
    pub height: u32,
    pub dtype: GpuPixelFormat,
    pub usage: GpuImageUsageFlags,
}
#[derive(Clone, Copy)]
pub enum GpuPixelFormat {
    Rgba8Unorm,
    Rgba32Float,
}
bitflags! {
    #[derive(Clone, Copy)]
    pub struct GpuImageUsageFlags: u32 {
        const COLOR_ATTACHMENT = 0x1;
        const DEPTH_STENCIL_ATTACHMENT = 0x2;
        const SAMPLED = 0x4;
    }
}
impl GpuImage {
    pub fn create(device: Arc<GpuDevice>, config: &GpuImageConfig) -> Arc<Self> {
        unsafe {
            let handle = device
                .ash_device()
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(config.dtype.into())
                        .extent(vk::Extent3D {
                            width: config.width,
                            height: config.height,
                            depth: 1,
                        })
                        .usage(config.usage.into()),
                    None,
                )
                .unwrap();

            let width = config.width;
            let height = config.height;
            let dtype = config.dtype;

            Arc::new(GpuImage {
                device,
                handle,
                width,
                height,
                dtype,
            })
        }
    }
}
impl Into<vk::Format> for GpuPixelFormat {
    fn into(self) -> vk::Format {
        match self {
            GpuPixelFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            GpuPixelFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}
impl Into<vk::ImageUsageFlags> for GpuImageUsageFlags {
    fn into(self) -> vk::ImageUsageFlags {
        let mut res = vk::ImageUsageFlags::empty();
        if self.contains(GpuImageUsageFlags::COLOR_ATTACHMENT) {
            res |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if self.contains(GpuImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            res |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if self.contains(GpuImageUsageFlags::SAMPLED) {
            res |= vk::ImageUsageFlags::SAMPLED;
        }
        res
    }
}
