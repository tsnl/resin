use super::*;

#[derive(Default)]
pub struct GpuQueueFamilyIndices {
    graphics_family: Option<u32>,
    compute_family: Option<u32>,
    transfer_family: Option<u32>,
    present_family: Option<u32>,
}
impl GpuQueueFamilyIndices {
    pub fn find(
        physical_device: &GpuPhysicalDevice,
        surface: Option<vk::SurfaceKHR>,
    ) -> Option<GpuQueueFamilyIndices> {
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
        physical_device: &GpuPhysicalDevice,
        surface: Option<vk::SurfaceKHR>,
        forbid_queue_family_sharing: bool,
    ) -> Option<GpuQueueFamilyIndices> {
        unsafe {
            let mut result = GpuQueueFamilyIndices::default();
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
}
