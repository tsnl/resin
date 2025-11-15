use super::*;

pub struct GpuDevice {
    pub(crate) manager: Arc<GpuManager>,
    pub(crate) ash_device: ash::Device,
    pub(crate) queue_family_indices: GpuQueueFamilyIndices,
}
pub struct GpuDeviceConfig {
    pub enabled_extension_names: Vec<CString>,
}
