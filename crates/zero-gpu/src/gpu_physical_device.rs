use super::*;

pub struct GpuPhysicalDevice {
    pub(crate) manager: Arc<GpuManager>,
    pub(crate) vk_physical_device: vk::PhysicalDevice,
    pub(crate) vk_device_properties: vk::PhysicalDeviceProperties,
}
impl HasGpuManager for GpuPhysicalDevice {
    fn manager(&self) -> &Arc<GpuManager> {
        &self.manager
    }
}
impl GpuPhysicalDevice {
    pub fn name(&self) -> &CStr {
        let bytes = bytemuck::cast_slice(&self.vk_device_properties.device_name);
        CStr::from_bytes_until_nul(bytes).unwrap()
    }
    pub fn queue_families(&self) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.ash_instance()
                .get_physical_device_queue_family_properties(self.vk_physical_device)
        }
    }
}
