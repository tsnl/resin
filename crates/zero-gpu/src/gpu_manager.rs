use super::*;

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

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
    pub fn create(config: GpuManagerConfig) -> Arc<Self> {
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

            Arc::new(Self {
                ash_entry,
                ash_instance,
                ash_khr_surface_instance,
                provides_surface_support,
            })
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

    /// Creates a new surface for the given display and window handles.
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

    /// Enumerates all physical devices available on the system.
    pub fn enumerate_physical_devices(self: Arc<Self>) -> Vec<GpuPhysicalDevice> {
        unsafe {
            self.ash_instance()
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .map(|vk_physical_device| {
                    let vk_device_properties = self
                        .ash_instance()
                        .get_physical_device_properties(vk_physical_device);
                    GpuPhysicalDevice {
                        manager: self.clone(),
                        vk_physical_device,
                        vk_device_properties,
                    }
                })
                .collect()
        }
    }

    /// Attempts to create a device from the given physical device and surface.
    /// If the physical device does not support the required features, we will return None.
    pub fn try_create_device(
        self: Arc<Self>,
        physical_device: GpuPhysicalDevice,
        surface: Option<vk::SurfaceKHR>,
        config: &GpuDeviceConfig,
    ) -> Option<Arc<GpuDevice>> {
        if surface.is_some() && !self.provides_surface_support {
            panic!(
                "This instance does not support surface creation, but a surface was provided when trying to create a device."
            );
        }
        GpuDevice::create(self, physical_device, surface, config)
    }
}

pub trait HasGpuManager {
    fn manager(&self) -> &Arc<GpuManager>;
}
pub trait HasGpuManagerExt: HasGpuManager {
    fn ash_instance(&self) -> &ash::Instance {
        self.manager().ash_instance()
    }
    fn ash_khr_surface_instance(&self) -> &ash::khr::surface::Instance {
        self.manager().ash_khr_surface_instance()
    }
}
impl<T: HasGpuManager> HasGpuManagerExt for T {}
