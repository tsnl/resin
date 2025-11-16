use super::*;

pub struct GpuDevice {
    pub manager: Arc<GpuManager>,
    pub ash_device: ash::Device,
    pub queue_family_indices: GpuQueueFamilyIndices,
}
pub struct GpuDeviceConfig {
    pub enabled_extension_names: Vec<CString>,
}
impl GpuDevice {
    pub fn create(
        manager: Arc<GpuManager>,
        physical_device: GpuPhysicalDevice,
        surface: Option<vk::SurfaceKHR>,
        device_create_info: &GpuDeviceConfig,
    ) -> Option<Arc<GpuDevice>> {
        unsafe {
            let queue_family_indices = GpuQueueFamilyIndices::find(&physical_device, surface)?;

            let extension_names: Vec<_> = device_create_info
                .enabled_extension_names
                .iter()
                .map(|it| it.as_ptr())
                .collect();
            let vk_device = manager
                .ash_instance()
                .create_device(
                    physical_device.vk_physical_device,
                    &vk::DeviceCreateInfo::default().enabled_extension_names(&extension_names),
                    None,
                )
                .unwrap();

            Some(Arc::new(GpuDevice {
                manager,
                ash_device: vk_device,
                queue_family_indices,
            }))
        }
    }
    pub fn ash_device(&self) -> &ash::Device {
        &self.ash_device
    }
    pub fn create_image(self: &Arc<Self>, config: GpuImageConfig) -> Arc<GpuImage> {
        GpuImage::create(self.clone(), &config)
    }
}
