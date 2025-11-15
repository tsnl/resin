//
// API:
//

mod gpu_manager;
pub use gpu_manager::{GpuManager, GpuManagerConfig};
use gpu_manager::{HasGpuManager, HasGpuManagerExt};

mod gpu_physical_device;
pub use gpu_physical_device::GpuPhysicalDevice;

mod gpu_queue_family_indices;
use gpu_queue_family_indices::GpuQueueFamilyIndices;

mod gpu_device;
pub use gpu_device::{GpuDevice, GpuDeviceConfig};

//
// Common prelude:
//

use ash::vk;
use std::ffi::{CStr, CString};
use std::sync::Arc;
