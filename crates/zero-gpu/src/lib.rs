use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::{CStr, CString};
use std::sync::Arc;

/// Vulkan context (thin wrapper around Instance), meant to be owned by the engine and shared across various modules.
pub struct GpuManager {
    ash_entry: ash::Entry,
    ash_instance: ash::Instance,
    ash_khr_surface_instance: Option<ash::khr::surface::Instance>,
    provides_surface_support: bool,
}
pub struct GpuManagerConfig {
    pub extensions: Vec<CString>,
    pub layers: Vec<CString>,
    pub require_surface_support: bool,
}
impl GpuManager {
    pub fn new(config: GpuManagerConfig) -> Self {
        unsafe {
            let ash_entry = ash::Entry::linked();

            let ash_instance = {
                let extensions: Vec<_> = config.extensions.iter().map(|it| it.as_ptr()).collect();
                let layers: Vec<_> = config.layers.iter().map(|it| it.as_ptr()).collect();
                ash_entry
                    .create_instance(
                        &vk::InstanceCreateInfo::default()
                            .enabled_extension_names(&extensions)
                            .enabled_layer_names(&layers),
                        None,
                    )
                    .unwrap()
            };

            let provides_surface_support = config.require_surface_support;

            let ash_khr_surface_instance = if provides_surface_support {
                Some(ash::khr::surface::Instance::new(&ash_entry, &ash_instance))
            } else {
                None
            };

            Self {
                ash_entry,
                ash_instance,
                ash_khr_surface_instance,
                provides_surface_support,
            }
        }
    }
    pub fn entry(&self) -> &ash::Entry {
        &self.ash_entry
    }
    pub fn ash_instance(&self) -> &ash::Instance {
        &self.ash_instance
    }
    pub fn ash_khr_surface_instance(&self) -> &ash::khr::surface::Instance {
        self.ash_khr_surface_instance
            .as_ref()
            .expect("Surface support was not requested")
    }
    pub fn create_surface(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> vk::SurfaceKHR {
        unsafe {
            ash_window::create_surface(
                &self.ash_entry,
                &self.ash_instance,
                display_handle,
                window_handle,
                None,
            )
            .unwrap()
        }
    }
    pub fn enumerate_physical_devices(self: Arc<Self>) -> Vec<PhysicalDevice> {
        unsafe {
            self.ash_instance()
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .map(|vk_physical_device| {
                    let vk_device_properties = self
                        .ash_instance()
                        .get_physical_device_properties(vk_physical_device);
                    PhysicalDevice {
                        manager: self.clone(),
                        vk_physical_device,
                        vk_device_properties,
                    }
                })
                .collect()
        }
    }
    pub fn try_create_device(
        self: Arc<Self>,
        physical_device: PhysicalDevice,
        surface: Option<vk::SurfaceKHR>,
        device_create_info: DeviceCreateInfo,
    ) -> Option<Device> {
        unsafe {
            let queue_family_indices = QueueFamilyIndices::find(&physical_device, surface)?;

            let extension_names: Vec<_> = device_create_info
                .enabled_extension_names
                .iter()
                .map(|it| it.as_ptr())
                .collect();
            let vk_device = self
                .ash_instance()
                .create_device(
                    physical_device.vk_physical_device,
                    &vk::DeviceCreateInfo::default().enabled_extension_names(&extension_names),
                    None,
                )
                .unwrap();

            Some(Device {
                manager: self.clone(),
                ash_device: vk_device,
                queue_family_indices,
            })
        }
    }
}

pub trait HasManager {
    fn manager(&self) -> &Arc<GpuManager>;
}
pub trait HasManagerExt: HasManager {
    fn ash_instance(&self) -> &ash::Instance {
        self.manager().ash_instance()
    }
    fn ash_khr_surface_instance(&self) -> &ash::khr::surface::Instance {
        self.manager().ash_khr_surface_instance()
    }
}
impl<T: HasManager> HasManagerExt for T {}

pub struct PhysicalDevice {
    manager: Arc<GpuManager>,
    vk_physical_device: vk::PhysicalDevice,
    vk_device_properties: vk::PhysicalDeviceProperties,
}
impl HasManager for PhysicalDevice {
    fn manager(&self) -> &Arc<GpuManager> {
        &self.manager
    }
}
impl PhysicalDevice {
    pub fn name(&self) -> &CStr {
        let bytes = bytemuck::cast_slice(&self.vk_device_properties.device_name);
        CStr::from_bytes_until_nul(bytes).unwrap()
    }
    fn queue_families(&self) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.ash_instance()
                .get_physical_device_queue_family_properties(self.vk_physical_device)
        }
    }
}

#[derive(Default)]
pub struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    compute_family: Option<u32>,
    transfer_family: Option<u32>,
    present_family: Option<u32>,
}
impl QueueFamilyIndices {
    fn is_complete(&self, requires_present: bool) -> bool {
        if self.graphics_family.is_none() {
            return false;
        }
        if self.compute_family.is_none() {
            return false;
        }
        if self.transfer_family.is_none() {
            return false;
        }
        if requires_present && self.present_family.is_none() {
            return false;
        }
        true
    }
    fn find(
        physical_device: &PhysicalDevice,
        surface: Option<vk::SurfaceKHR>,
    ) -> Option<QueueFamilyIndices> {
        // First, try finding a queue family with exclusive access
        if let Some(res) = Self::find_with_exclusivity_constraint(physical_device, surface, true) {
            return Some(res);
        }

        // If that fails, try finding a queue family with shared access
        if let Some(res) = Self::find_with_exclusivity_constraint(physical_device, surface, false) {
            return Some(res);
        }

        // If that fails, give up:
        None
    }
    fn find_with_exclusivity_constraint(
        physical_device: &PhysicalDevice,
        surface: Option<vk::SurfaceKHR>,
        forbid_queue_family_sharing: bool,
    ) -> Option<QueueFamilyIndices> {
        unsafe {
            let mut result = QueueFamilyIndices::default();
            let requires_present = surface.is_some();

            for (queue_family_index, queue_family) in
                physical_device.queue_families().into_iter().enumerate()
            {
                // Graphics family:
                if result.graphics_family.is_none()
                    && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    result.graphics_family = Some(queue_family_index as u32);

                    // If we're not sharing queue families, we can't use this family for anything else.
                    if forbid_queue_family_sharing {
                        continue;
                    }
                }

                // Compute family:
                if result.compute_family.is_none()
                    && queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                {
                    result.compute_family = Some(queue_family_index as u32);

                    // If we're not sharing queue families, we can't use this family for anything else.
                    if forbid_queue_family_sharing {
                        continue;
                    }
                }

                // Transfer family:
                if result.transfer_family.is_none()
                    && queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                {
                    result.transfer_family = Some(queue_family_index as u32);

                    // If we're not sharing queue families, we can't use this family for anything else.
                    if forbid_queue_family_sharing {
                        continue;
                    }
                }

                // Present family:
                if let Some(surface) = surface
                    && physical_device
                        .ash_khr_surface_instance()
                        .get_physical_device_surface_support(
                            physical_device.vk_physical_device,
                            queue_family_index as u32,
                            surface,
                        )
                        .unwrap()
                {
                    result.present_family = Some(queue_family_index as u32);

                    // If we're not sharing queue families, we can't use this family for anything else.
                    if forbid_queue_family_sharing {
                        continue;
                    }
                }

                // If we've found all the families we need, we can stop searching.
                if result.is_complete(requires_present) {
                    break;
                }
            }

            if !result.is_complete(requires_present) {
                None
            } else {
                Some(result)
            }
        }
    }
}

pub struct Device {
    manager: Arc<GpuManager>,
    ash_device: ash::Device,
    queue_family_indices: QueueFamilyIndices,
}
pub struct DeviceCreateInfo {
    pub enabled_extension_names: Vec<CString>,
}
