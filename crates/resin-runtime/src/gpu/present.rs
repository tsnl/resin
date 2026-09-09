use ash::{Device, Instance, khr, vk};

use super::{ResinGpu, ResinImage, cmd_image_barrier, vk_status};
use crate::{ResinStatus, window::Surface};

pub(super) struct Presentation {
    swapchain: Option<Swapchain>,
    surface: Surface,
    instance: Instance,
    device: Device,
    physical: vk::PhysicalDevice,
}

struct Swapchain {
    device: Device,
    loader: khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    window_size: (u32, u32),
    acquired: vk::Fence,
    ready: vk::Semaphore,
    presented: vk::Fence,
    acquire_pending: bool,
    present_pending: bool,
}

impl ResinGpu {
    /// # Safety
    /// Call on the main thread. The initialized image must belong to this GPU;
    /// all GPU, window, and image operations must be externally synchronized.
    pub unsafe fn present(&mut self, image: &mut ResinImage) -> Result<(), ResinStatus> {
        if image.device.handle() != self.device.handle()
            || image.layout.get() == vk::ImageLayout::UNDEFINED
        {
            return Err(ResinStatus::InvalidArgument);
        }
        let Some(mut presentation) = self.presentation.take() else {
            return Err(ResinStatus::InvalidArgument);
        };
        let result = unsafe { presentation.present(self, image) };
        self.presentation = Some(presentation);
        result
    }
}

impl Presentation {
    pub fn new(
        instance: &Instance,
        device: &Device,
        physical: vk::PhysicalDevice,
        surface: Surface,
    ) -> Self {
        Self {
            swapchain: None,
            surface,
            instance: instance.clone(),
            device: device.clone(),
            physical,
        }
    }

    unsafe fn present(
        &mut self,
        gpu: &ResinGpu,
        image: &mut ResinImage,
    ) -> Result<(), ResinStatus> {
        let size = self.surface.window.framebuffer_size();
        if size.0 == 0 || size.1 == 0 {
            return Err(ResinStatus::Incomplete);
        }
        if self
            .swapchain
            .as_ref()
            .is_some_and(|swapchain| swapchain.window_size != size)
        {
            self.swapchain = None;
        }
        if self.swapchain.is_none() {
            self.swapchain = Some(self.create_swapchain(size)?);
        }
        let swapchain = self.swapchain.as_mut().unwrap();
        match unsafe { swapchain.present(gpu, image) } {
            Ok(suboptimal) => {
                if suboptimal {
                    self.swapchain = None;
                }
                Ok(())
            }
            Err(error) => {
                // Discard acquisition and semaphore state after any failed frame.
                self.swapchain = None;
                Err(error)
            }
        }
    }

    fn create_swapchain(&self, size: (u32, u32)) -> Result<Swapchain, ResinStatus> {
        let caps = unsafe {
            self.surface
                .loader
                .get_physical_device_surface_capabilities(self.physical, self.surface.handle)
        }
        .map_err(vk_status)?;
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(ResinStatus::Unsupported);
        }
        let formats = unsafe {
            self.surface
                .loader
                .get_physical_device_surface_formats(self.physical, self.surface.handle)
        }
        .map_err(vk_status)?;
        let format = choose_format(&formats).ok_or(ResinStatus::Unsupported)?;
        for (format, feature) in [
            (vk::Format::R8G8B8A8_UNORM, vk::FormatFeatureFlags::BLIT_SRC),
            (format.format, vk::FormatFeatureFlags::BLIT_DST),
        ] {
            let properties = unsafe {
                self.instance
                    .get_physical_device_format_properties(self.physical, format)
            };
            if !properties.optimal_tiling_features.contains(feature) {
                return Err(ResinStatus::Unsupported);
            }
        }
        let extent = choose_extent(&caps, size);
        if extent.width == 0 || extent.height == 0 {
            return Err(ResinStatus::Incomplete);
        }
        let alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|alpha| caps.supported_composite_alpha.contains(*alpha))
        .ok_or(ResinStatus::Unsupported)?;
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface.handle)
            .min_image_count(image_count(&caps))
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(alpha)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);
        let loader = khr::swapchain::Device::new(&self.instance, &self.device);
        let handle = unsafe { loader.create_swapchain(&info, None) }.map_err(vk_status)?;
        let mut swapchain = Swapchain {
            device: self.device.clone(),
            loader,
            handle,
            images: Vec::new(),
            extent,
            window_size: size,
            acquired: vk::Fence::null(),
            ready: vk::Semaphore::null(),
            presented: vk::Fence::null(),
            acquire_pending: false,
            present_pending: false,
        };
        unsafe {
            swapchain.images = swapchain
                .loader
                .get_swapchain_images(handle)
                .map_err(vk_status)?;
            swapchain.acquired = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(vk_status)?;
            swapchain.presented = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(vk_status)?;
            swapchain.ready = self
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(vk_status)?;
        }
        Ok(swapchain)
    }
}

impl Swapchain {
    unsafe fn present(
        &mut self,
        gpu: &ResinGpu,
        image: &mut ResinImage,
    ) -> Result<bool, ResinStatus> {
        let (index, suboptimal) = unsafe { self.acquire() }.map_err(frame_status)?;
        unsafe { self.blit(gpu, image, index)? };
        let changed = unsafe { self.queue(gpu.queue, index) }.map_err(frame_status)?;
        Ok(suboptimal || changed)
    }

    unsafe fn acquire(&mut self) -> Result<(u32, bool), vk::Result> {
        let (index, suboptimal) = unsafe {
            self.loader.acquire_next_image(
                self.handle,
                100_000_000,
                vk::Semaphore::null(),
                self.acquired,
            )?
        };
        self.acquire_pending = true;
        unsafe {
            self.device
                .wait_for_fences(&[self.acquired], true, u64::MAX)?
        };
        self.acquire_pending = false;
        unsafe { self.device.reset_fences(&[self.acquired])? };
        Ok((index, suboptimal))
    }

    unsafe fn blit(
        &self,
        gpu: &ResinGpu,
        image: &mut ResinImage,
        index: u32,
    ) -> Result<(), ResinStatus> {
        let mut commands = unsafe { gpu.start_command_recording()? };
        let previous = commands
            .layouts
            .transition(&image.layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        cmd_image_barrier(
            &self.device,
            commands.handle,
            image.image,
            previous,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let destination = self.images[index as usize];
        cmd_image_barrier(
            &self.device,
            commands.handle,
            destination,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let region = vk::ImageBlit::default()
            .src_subresource(layers)
            .dst_subresource(layers)
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: image.width as i32,
                    y: image.height as i32,
                    z: 1,
                },
            ])
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: self.extent.width as i32,
                    y: self.extent.height as i32,
                    z: 1,
                },
            ]);
        unsafe {
            self.device.cmd_blit_image(
                commands.handle,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                vk::Filter::NEAREST,
            )
        };
        cmd_image_barrier(
            &self.device,
            commands.handle,
            destination,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
        );
        unsafe { gpu.submit_signaling(commands, self.ready) }
    }

    unsafe fn queue(&mut self, queue: vk::Queue, index: u32) -> Result<bool, vk::Result> {
        let mut fence_info = vk::SwapchainPresentFenceInfoEXT::default()
            .fences(std::slice::from_ref(&self.presented));
        let swapchains = [self.handle];
        let indices = [index];
        let info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices)
            .wait_semaphores(std::slice::from_ref(&self.ready))
            .push_next(&mut fence_info);
        let result = unsafe { self.loader.queue_present(queue, &info) };
        if result.is_ok()
            || matches!(
                result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR
                    | vk::Result::ERROR_SURFACE_LOST_KHR
                    | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)
            )
        {
            self.present_pending = true;
            unsafe {
                self.device
                    .wait_for_fences(&[self.presented], true, u64::MAX)?
            };
            self.present_pending = false;
            unsafe { self.device.reset_fences(&[self.presented])? };
        }
        result
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            if self.acquire_pending {
                let _ = self
                    .device
                    .wait_for_fences(&[self.acquired], true, u64::MAX);
            }
            if self.present_pending {
                let _ = self
                    .device
                    .wait_for_fences(&[self.presented], true, u64::MAX);
            }
            self.device.destroy_semaphore(self.ready, None);
            self.device.destroy_fence(self.acquired, None);
            self.device.destroy_fence(self.presented, None);
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}

fn frame_status(error: vk::Result) -> ResinStatus {
    match error {
        vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::TIMEOUT | vk::Result::NOT_READY => {
            ResinStatus::Incomplete
        }
        _ => vk_status(error),
    }
}

fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
        return Some(vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: formats[0].color_space,
        });
    }
    [
        vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB,
        vk::Format::R8G8B8A8_SRGB,
    ]
    .into_iter()
    .find_map(|format| {
        formats
            .iter()
            .find(|entry| {
                entry.format == format && entry.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
    })
}

fn choose_extent(caps: &vk::SurfaceCapabilitiesKHR, size: (u32, u32)) -> vk::Extent2D {
    if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: size
                .0
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: size
                .1
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    }
}

fn image_count(caps: &vk::SurfaceCapabilitiesKHR) -> u32 {
    let desired = caps.min_image_count.saturating_add(1);
    if caps.max_image_count == 0 {
        desired
    } else {
        desired.min(caps.max_image_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swapchain_size_and_count_respect_surface_limits() {
        let mut caps = vk::SurfaceCapabilitiesKHR::default()
            .min_image_count(2)
            .max_image_count(2)
            .current_extent(vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            })
            .min_image_extent(vk::Extent2D {
                width: 16,
                height: 16,
            })
            .max_image_extent(vk::Extent2D {
                width: 1920,
                height: 1080,
            });
        let extent = choose_extent(&caps, (1, 2000));
        assert_eq!((extent.width, extent.height), (16, 1080));
        assert_eq!(image_count(&caps), 2);
        caps.max_image_count = 0;
        assert_eq!(image_count(&caps), 3);
        caps.current_extent = vk::Extent2D {
            width: 0,
            height: 0,
        };
        assert!(choose_extent(&caps, (640, 480)) == caps.current_extent);
    }

    #[test]
    fn surface_format_requires_a_supported_rgba_format() {
        assert!(choose_format(&[]).is_none());
        let format = vk::SurfaceFormatKHR {
            format: vk::Format::UNDEFINED,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        assert!(choose_format(&[format]).unwrap().format == vk::Format::B8G8R8A8_UNORM);
        assert!(
            choose_format(&[vk::SurfaceFormatKHR {
                format: vk::Format::R16G16B16A16_SFLOAT,
                ..format
            }])
            .is_none()
        );
    }
}
