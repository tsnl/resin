//! Vulkan 1.3 device selection and feature negotiation.

use std::ffi::{CStr, c_char};
use std::slice;

use ash::{Device, Entry, Instance, vk};

use crate::{ResinStatus, ResinWindow, window::Surface};

use super::{ResinGpuDeviceInfo, ResinGpuDeviceType, vk_status};

pub struct DeviceContext {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub physical: vk::PhysicalDevice,
    pub surface: Option<Surface>,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub max_buffer_size: vk::DeviceSize,
    pub memory_priority: bool,
}

pub fn create_device() -> Result<DeviceContext, ResinStatus> {
    create(None, None)
}

pub fn create_device_at(index: u32) -> Result<DeviceContext, ResinStatus> {
    create(None, Some(index))
}

pub fn create_device_for_window(window: &ResinWindow) -> Result<DeviceContext, ResinStatus> {
    create(Some(window), None)
}

fn create(window: Option<&ResinWindow>, index: Option<u32>) -> Result<DeviceContext, ResinStatus> {
    let extensions = window
        .map(ResinWindow::extensions)
        .transpose()?
        .unwrap_or_default();
    let (entry, instance, devices) = create_instance(&extensions)?;
    let surface = window
        .map(|window| window.surface(&entry, &instance))
        .transpose()
        .inspect_err(|_| {
            unsafe { instance.destroy_instance(None) };
        })?;
    let selected = if let Some(index) = index {
        devices
            .get(index as usize)
            .ok_or(ResinStatus::InvalidArgument)
            .and_then(|&physical| {
                inspect_device(&instance, physical).ok_or(ResinStatus::Unsupported)
            })
    } else if devices.is_empty() {
        Err(ResinStatus::VulkanUnavailable)
    } else {
        select_device(&instance, &devices, surface.as_ref()).ok_or(ResinStatus::Unsupported)
    };
    match selected {
        Ok(selected) => finish_device(entry, instance, selected, surface),
        Err(err) => {
            drop(surface);
            unsafe { instance.destroy_instance(None) };
            Err(err)
        }
    }
}

pub fn enumerate_devices() -> Result<Vec<ResinGpuDeviceInfo>, ResinStatus> {
    let (_entry, instance, physical_devices) = create_instance(&[])?;
    let infos = physical_devices
        .iter()
        .enumerate()
        .map(|(index, &physical)| physical_device_info(&instance, physical, index as u32))
        .collect();
    unsafe { instance.destroy_instance(None) };
    Ok(infos)
}

fn create_instance(
    extensions: &[*const c_char],
) -> Result<(Entry, Instance, Vec<vk::PhysicalDevice>), ResinStatus> {
    let entry = unsafe { Entry::load() }.map_err(|_| ResinStatus::VulkanUnavailable)?;
    #[cfg(target_os = "macos")]
    let portability_extensions = {
        let mut names = extensions.to_vec();
        if !names
            .iter()
            .any(|&name| unsafe { CStr::from_ptr(name) } == vk::KHR_PORTABILITY_ENUMERATION_NAME)
        {
            names.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
        }
        names
    };
    #[cfg(target_os = "macos")]
    let extensions = portability_extensions.as_slice();
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"resin")
        .application_version(0)
        .engine_name(c"resin")
        .engine_version(0)
        .api_version(vk::API_VERSION_1_3);
    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(extensions);
    #[cfg(target_os = "macos")]
    let instance_info = instance_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    let instance = unsafe { entry.create_instance(&instance_info, None) }.map_err(vk_status)?;
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|err| {
        unsafe { instance.destroy_instance(None) };
        vk_status(err)
    })?;
    Ok((entry, instance, physical_devices))
}

fn finish_device(
    entry: Entry,
    instance: Instance,
    selected: SelectedDevice,
    mut surface: Option<Surface>,
) -> Result<DeviceContext, ResinStatus> {
    let queue_priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(selected.queue_family)
        .queue_priorities(&queue_priorities);

    let mut enabled_extensions = selected.optional_extensions.clone();
    if surface.is_some() {
        enabled_extensions.extend([
            vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            vk::EXT_SWAPCHAIN_MAINTENANCE1_NAME.as_ptr(),
        ]);
    }

    let mut vulkan11 = selected.vulkan11;
    let mut vulkan12 = selected.vulkan12;
    let mut vulkan13 = selected.vulkan13;
    let mut memory_priority = selected.memory_priority;
    let mut pageable = selected.pageable;
    let mut reconvergence = selected.reconvergence;
    let mut subgroup_rotate = selected.subgroup_rotate;
    let mut expect_assume = selected.expect_assume;
    let mut float_controls2 = selected.float_controls2;
    let mut quad_control = selected.quad_control;
    let mut workgroup_layout = selected.workgroup_layout;
    let mut shader_clock = selected.shader_clock;
    let mut atomic_float = selected.atomic_float;
    let mut swapchain =
        vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default().swapchain_maintenance1(true);

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(selected.features10)
        .push_next(&mut vulkan11)
        .push_next(&mut vulkan12)
        .push_next(&mut vulkan13);
    if selected.memory_priority_enabled {
        features2 = features2.push_next(&mut memory_priority);
        if selected.pageable_enabled {
            features2 = features2.push_next(&mut pageable);
        }
    }
    if selected.reconvergence_enabled {
        features2 = features2.push_next(&mut reconvergence);
    }
    if selected.subgroup_rotate_enabled {
        features2 = features2.push_next(&mut subgroup_rotate);
    }
    if selected.expect_assume_enabled {
        features2 = features2.push_next(&mut expect_assume);
    }
    if selected.float_controls2_enabled {
        features2 = features2.push_next(&mut float_controls2);
    }
    if selected.quad_control_enabled {
        features2 = features2.push_next(&mut quad_control);
    }
    if selected.workgroup_layout_enabled {
        features2 = features2.push_next(&mut workgroup_layout);
    }
    if selected.shader_clock_enabled {
        features2 = features2.push_next(&mut shader_clock);
    }
    if selected.atomic_float_enabled {
        features2 = features2.push_next(&mut atomic_float);
    }
    if surface.is_some() {
        features2 = features2.push_next(&mut swapchain);
    }

    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(slice::from_ref(&queue_info))
        .enabled_extension_names(&enabled_extensions)
        .push_next(&mut features2);

    let device = unsafe { instance.create_device(selected.physical, &device_info, None) }.map_err(
        |err| {
            drop(surface.take());
            unsafe { instance.destroy_instance(None) };
            vk_status(err)
        },
    )?;

    let queue = unsafe { device.get_device_queue(selected.queue_family, 0) };

    Ok(DeviceContext {
        entry,
        instance,
        device,
        queue,
        queue_family: selected.queue_family,
        physical: selected.physical,
        surface,
        memory_properties: selected.memory_properties,
        max_buffer_size: selected.max_buffer_size,
        memory_priority: selected.memory_priority_enabled,
    })
}

struct SelectedDevice {
    physical: vk::PhysicalDevice,
    queue_family: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    max_buffer_size: vk::DeviceSize,
    features10: vk::PhysicalDeviceFeatures,
    vulkan11: vk::PhysicalDeviceVulkan11Features<'static>,
    vulkan12: vk::PhysicalDeviceVulkan12Features<'static>,
    vulkan13: vk::PhysicalDeviceVulkan13Features<'static>,
    memory_priority: vk::PhysicalDeviceMemoryPriorityFeaturesEXT<'static>,
    pageable: vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT<'static>,
    reconvergence: vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR<'static>,
    subgroup_rotate: vk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR<'static>,
    expect_assume: vk::PhysicalDeviceShaderExpectAssumeFeaturesKHR<'static>,
    float_controls2: vk::PhysicalDeviceShaderFloatControls2FeaturesKHR<'static>,
    quad_control: vk::PhysicalDeviceShaderQuadControlFeaturesKHR<'static>,
    workgroup_layout: vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR<'static>,
    shader_clock: vk::PhysicalDeviceShaderClockFeaturesKHR<'static>,
    atomic_float: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT<'static>,
    memory_priority_enabled: bool,
    pageable_enabled: bool,
    reconvergence_enabled: bool,
    subgroup_rotate_enabled: bool,
    expect_assume_enabled: bool,
    float_controls2_enabled: bool,
    quad_control_enabled: bool,
    workgroup_layout_enabled: bool,
    shader_clock_enabled: bool,
    atomic_float_enabled: bool,
    optional_extensions: Vec<*const c_char>,
}

fn physical_device_info(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    index: u32,
) -> ResinGpuDeviceInfo {
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    let kind = match properties.device_type {
        vk::PhysicalDeviceType::INTEGRATED_GPU => ResinGpuDeviceType::Integrated,
        vk::PhysicalDeviceType::DISCRETE_GPU => ResinGpuDeviceType::Discrete,
        vk::PhysicalDeviceType::VIRTUAL_GPU => ResinGpuDeviceType::Virtual,
        vk::PhysicalDeviceType::CPU => ResinGpuDeviceType::Cpu,
        _ => ResinGpuDeviceType::Other,
    };
    let mut maint4 = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maint4);
    unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    let (device_local_bytes, host_visible_device_local_bytes) = heap_sizes(&memory);

    ResinGpuDeviceInfo {
        index,
        kind,
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        api_version: properties.api_version,
        driver_version: properties.driver_version,
        suitable: u32::from(inspect_device(instance, physical).is_some()),
        reserved: 0,
        device_local_bytes,
        host_visible_device_local_bytes,
        max_buffer_size: maint4.max_buffer_size,
        name: properties.device_name,
    }
}

fn heap_sizes(memory: &vk::PhysicalDeviceMemoryProperties) -> (u64, u64) {
    let mut device_local = 0u64;
    let mut host_visible_device_local = 0u64;
    for heap_index in 0..memory.memory_heap_count {
        let heap = memory.memory_heaps[heap_index as usize];
        if !heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            continue;
        }
        device_local += heap.size;
        let host_visible = (0..memory.memory_type_count).any(|type_index| {
            let ty = memory.memory_types[type_index as usize];
            ty.heap_index == heap_index
                && ty
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        });
        if host_visible {
            host_visible_device_local += heap.size;
        }
    }
    (device_local, host_visible_device_local)
}

fn select_device(
    instance: &Instance,
    devices: &[vk::PhysicalDevice],
    surface: Option<&Surface>,
) -> Option<SelectedDevice> {
    let mut best: Option<(u32, SelectedDevice)> = None;
    for &physical in devices {
        let Some(mut selected) = inspect_device(instance, physical) else {
            continue;
        };
        if let Some(surface) = surface {
            let Some(queue) = present_queue(instance, physical, surface) else {
                continue;
            };
            selected.queue_family = queue;
        }
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let score = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 4,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 1,
            _ => 0,
        };
        match best {
            Some((best_score, _)) if best_score >= score => {}
            _ => best = Some((score, selected)),
        }
    }
    best.map(|(_, selected)| selected)
}

fn inspect_device(instance: &Instance, physical: vk::PhysicalDevice) -> Option<SelectedDevice> {
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    if vk::api_version_major(properties.api_version) < 1
        || (vk::api_version_major(properties.api_version) == 1
            && vk::api_version_minor(properties.api_version) < 3)
    {
        return None;
    }
    if properties.limits.max_push_constants_size < 8 {
        return None;
    }

    let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }.ok()?;

    let queue_family = graphics_compute_queue_family(instance, physical)?;

    let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut memory_priority = vk::PhysicalDeviceMemoryPriorityFeaturesEXT::default();
    let mut pageable = vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT::default();
    let mut reconvergence = vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default();
    let mut subgroup_rotate = vk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR::default();
    let mut expect_assume = vk::PhysicalDeviceShaderExpectAssumeFeaturesKHR::default();
    let mut float_controls2 = vk::PhysicalDeviceShaderFloatControls2FeaturesKHR::default();
    let mut quad_control = vk::PhysicalDeviceShaderQuadControlFeaturesKHR::default();
    let mut workgroup_layout =
        vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default();
    let mut shader_clock = vk::PhysicalDeviceShaderClockFeaturesKHR::default();
    let mut atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();

    let has_memory_priority = has_extension(&extensions, vk::EXT_MEMORY_PRIORITY_NAME);
    let has_pageable = has_memory_priority
        && has_extension(&extensions, vk::EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_NAME);
    let has_reconvergence = has_extension(&extensions, vk::KHR_SHADER_MAXIMAL_RECONVERGENCE_NAME);
    let has_subgroup_rotate = has_extension(&extensions, vk::KHR_SHADER_SUBGROUP_ROTATE_NAME);
    let has_expect_assume = has_extension(&extensions, vk::KHR_SHADER_EXPECT_ASSUME_NAME);
    let has_float_controls2 = has_extension(&extensions, vk::KHR_SHADER_FLOAT_CONTROLS2_NAME);
    let has_quad_control = has_extension(&extensions, vk::KHR_SHADER_QUAD_CONTROL_NAME);
    let has_workgroup_layout =
        has_extension(&extensions, vk::KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_NAME);
    let has_shader_clock = has_extension(&extensions, vk::KHR_SHADER_CLOCK_NAME);
    let has_atomic_float = has_extension(&extensions, vk::EXT_SHADER_ATOMIC_FLOAT_NAME);

    let (shader_int64, shader_int16, shader_float64) = {
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13);
        if has_memory_priority {
            features2 = features2.push_next(&mut memory_priority);
            if has_pageable {
                features2 = features2.push_next(&mut pageable);
            }
        }
        if has_reconvergence {
            features2 = features2.push_next(&mut reconvergence);
        }
        if has_subgroup_rotate {
            features2 = features2.push_next(&mut subgroup_rotate);
        }
        if has_expect_assume {
            features2 = features2.push_next(&mut expect_assume);
        }
        if has_float_controls2 {
            features2 = features2.push_next(&mut float_controls2);
        }
        if has_quad_control {
            features2 = features2.push_next(&mut quad_control);
        }
        if has_workgroup_layout {
            features2 = features2.push_next(&mut workgroup_layout);
        }
        if has_shader_clock {
            features2 = features2.push_next(&mut shader_clock);
        }
        if has_atomic_float {
            features2 = features2.push_next(&mut atomic_float);
        }
        unsafe { instance.get_physical_device_features2(physical, &mut features2) };
        (
            features2.features.shader_int64,
            features2.features.shader_int16,
            features2.features.shader_float64,
        )
    };

    if shader_int64 != vk::TRUE
        || vulkan12.buffer_device_address != vk::TRUE
        || vulkan12.timeline_semaphore != vk::TRUE
        || vulkan13.synchronization2 != vk::TRUE
        || vulkan13.maintenance4 != vk::TRUE
        || vulkan13.dynamic_rendering != vk::TRUE
    {
        return None;
    }

    let mut maint4 = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maint4);
    unsafe { instance.get_physical_device_properties2(physical, &mut props2) };

    let mut optional_extensions = Vec::new();
    #[cfg(target_os = "macos")]
    if has_extension(&extensions, vk::KHR_PORTABILITY_SUBSET_NAME) {
        optional_extensions.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
    }
    let memory_priority_enabled =
        has_memory_priority && memory_priority.memory_priority == vk::TRUE;
    if memory_priority_enabled {
        optional_extensions.push(vk::EXT_MEMORY_PRIORITY_NAME.as_ptr());
    }
    let pageable_enabled = memory_priority_enabled
        && has_pageable
        && pageable.pageable_device_local_memory == vk::TRUE;
    if pageable_enabled {
        optional_extensions.push(vk::EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_NAME.as_ptr());
    }
    let reconvergence_enabled =
        has_reconvergence && reconvergence.shader_maximal_reconvergence == vk::TRUE;
    if reconvergence_enabled {
        optional_extensions.push(vk::KHR_SHADER_MAXIMAL_RECONVERGENCE_NAME.as_ptr());
    }
    let subgroup_rotate_enabled =
        has_subgroup_rotate && subgroup_rotate.shader_subgroup_rotate == vk::TRUE;
    if subgroup_rotate_enabled {
        optional_extensions.push(vk::KHR_SHADER_SUBGROUP_ROTATE_NAME.as_ptr());
    }
    let expect_assume_enabled = has_expect_assume && expect_assume.shader_expect_assume == vk::TRUE;
    if expect_assume_enabled {
        optional_extensions.push(vk::KHR_SHADER_EXPECT_ASSUME_NAME.as_ptr());
    }
    let float_controls2_enabled =
        has_float_controls2 && float_controls2.shader_float_controls2 == vk::TRUE;
    if float_controls2_enabled {
        optional_extensions.push(vk::KHR_SHADER_FLOAT_CONTROLS2_NAME.as_ptr());
    }
    let quad_control_enabled = has_quad_control && quad_control.shader_quad_control == vk::TRUE;
    if quad_control_enabled {
        optional_extensions.push(vk::KHR_SHADER_QUAD_CONTROL_NAME.as_ptr());
    }
    let workgroup_layout_enabled =
        has_workgroup_layout && workgroup_layout.workgroup_memory_explicit_layout == vk::TRUE;
    if workgroup_layout_enabled {
        optional_extensions.push(vk::KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_NAME.as_ptr());
    }
    let shader_clock_enabled = has_shader_clock
        && (shader_clock.shader_device_clock == vk::TRUE
            || shader_clock.shader_subgroup_clock == vk::TRUE);
    if shader_clock_enabled {
        optional_extensions.push(vk::KHR_SHADER_CLOCK_NAME.as_ptr());
    }
    let atomic_float_enabled =
        has_atomic_float && atomic_float.shader_buffer_float32_atomics == vk::TRUE;
    if atomic_float_enabled {
        optional_extensions.push(vk::EXT_SHADER_ATOMIC_FLOAT_NAME.as_ptr());
    }

    Some(SelectedDevice {
        physical,
        queue_family,
        memory_properties: unsafe { instance.get_physical_device_memory_properties(physical) },
        max_buffer_size: maint4.max_buffer_size,
        features10: vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .shader_int16(shader_int16 == vk::TRUE)
            .shader_float64(shader_float64 == vk::TRUE),
        vulkan11: vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(on(vulkan11.storage_buffer16_bit_access))
            .uniform_and_storage_buffer16_bit_access(on(
                vulkan11.uniform_and_storage_buffer16_bit_access
            ))
            .storage_push_constant16(on(vulkan11.storage_push_constant16)),
        vulkan12: vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .vulkan_memory_model(on(vulkan12.vulkan_memory_model))
            .vulkan_memory_model_device_scope(on(vulkan12.vulkan_memory_model_device_scope))
            .scalar_block_layout(on(vulkan12.scalar_block_layout))
            .shader_int8(on(vulkan12.shader_int8))
            .storage_buffer8_bit_access(on(vulkan12.storage_buffer8_bit_access))
            .uniform_and_storage_buffer8_bit_access(on(
                vulkan12.uniform_and_storage_buffer8_bit_access
            ))
            .storage_push_constant8(on(vulkan12.storage_push_constant8))
            .shader_float16(on(vulkan12.shader_float16))
            .shader_buffer_int64_atomics(on(vulkan12.shader_buffer_int64_atomics))
            .shader_shared_int64_atomics(on(vulkan12.shader_shared_int64_atomics))
            .host_query_reset(on(vulkan12.host_query_reset)),
        vulkan13: vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .maintenance4(true)
            .dynamic_rendering(true)
            .subgroup_size_control(on(vulkan13.subgroup_size_control))
            .compute_full_subgroups(on(vulkan13.compute_full_subgroups))
            .shader_integer_dot_product(on(vulkan13.shader_integer_dot_product))
            .shader_terminate_invocation(on(vulkan13.shader_terminate_invocation))
            .shader_zero_initialize_workgroup_memory(on(
                vulkan13.shader_zero_initialize_workgroup_memory
            ))
            .shader_demote_to_helper_invocation(on(vulkan13.shader_demote_to_helper_invocation))
            .pipeline_creation_cache_control(on(vulkan13.pipeline_creation_cache_control)),
        memory_priority: vk::PhysicalDeviceMemoryPriorityFeaturesEXT::default()
            .memory_priority(memory_priority_enabled),
        pageable: vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT::default()
            .pageable_device_local_memory(pageable_enabled),
        reconvergence: vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default()
            .shader_maximal_reconvergence(reconvergence_enabled),
        subgroup_rotate: vk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR::default()
            .shader_subgroup_rotate(subgroup_rotate_enabled)
            .shader_subgroup_rotate_clustered(on(subgroup_rotate.shader_subgroup_rotate_clustered)),
        expect_assume: vk::PhysicalDeviceShaderExpectAssumeFeaturesKHR::default()
            .shader_expect_assume(expect_assume_enabled),
        float_controls2: vk::PhysicalDeviceShaderFloatControls2FeaturesKHR::default()
            .shader_float_controls2(float_controls2_enabled),
        quad_control: vk::PhysicalDeviceShaderQuadControlFeaturesKHR::default()
            .shader_quad_control(quad_control_enabled),
        workgroup_layout: vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default()
            .workgroup_memory_explicit_layout(workgroup_layout_enabled)
            .workgroup_memory_explicit_layout8_bit_access(on(
                workgroup_layout.workgroup_memory_explicit_layout8_bit_access
            ))
            .workgroup_memory_explicit_layout16_bit_access(on(
                workgroup_layout.workgroup_memory_explicit_layout16_bit_access
            )),
        shader_clock: vk::PhysicalDeviceShaderClockFeaturesKHR::default()
            .shader_device_clock(on(shader_clock.shader_device_clock))
            .shader_subgroup_clock(on(shader_clock.shader_subgroup_clock)),
        atomic_float: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomics(atomic_float_enabled)
            .shader_buffer_float32_atomic_add(on(atomic_float.shader_buffer_float32_atomic_add)),
        memory_priority_enabled,
        pageable_enabled,
        reconvergence_enabled,
        subgroup_rotate_enabled,
        expect_assume_enabled,
        float_controls2_enabled,
        quad_control_enabled,
        workgroup_layout_enabled,
        shader_clock_enabled,
        atomic_float_enabled,
        optional_extensions,
    })
}

fn graphics_compute_queue_family(instance: &Instance, physical: vk::PhysicalDevice) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    families.iter().enumerate().find_map(|(index, family)| {
        if family
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        {
            Some(index as u32)
        } else {
            None
        }
    })
}

fn present_queue(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    surface: &Surface,
) -> Option<u32> {
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }.ok()?;
    if ![vk::KHR_SWAPCHAIN_NAME, vk::EXT_SWAPCHAIN_MAINTENANCE1_NAME]
        .iter()
        .all(|name| has_extension(&extensions, name))
    {
        return None;
    }
    let mut swapchain = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut swapchain);
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    if swapchain.swapchain_maintenance1 != vk::TRUE {
        return None;
    }
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    families.iter().enumerate().find_map(|(index, family)| {
        let graphics = family
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);
        let present = unsafe {
            surface.loader.get_physical_device_surface_support(
                physical,
                index as u32,
                surface.handle,
            )
        }
        .unwrap_or(false);
        (graphics && present).then_some(index as u32)
    })
}

fn has_extension(extensions: &[vk::ExtensionProperties], name: &CStr) -> bool {
    extensions
        .iter()
        .any(|extension| extension.extension_name_as_c_str() == Ok(name))
}

fn on(value: vk::Bool32) -> bool {
    value == vk::TRUE
}
