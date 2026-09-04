//! Vulkan 1.3 device bring-up with the modern extension set the runtime uses.

use std::ffi::{CStr, c_char};
use std::slice;

use ash::{Device, Entry, Instance, ext, khr, vk};

use crate::ResinStatus;

use super::vk_status;

pub struct DeviceContext {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub max_buffer_size: vk::DeviceSize,
    pub memory_priority: bool,
    pub shader_object: ext::shader_object::Device,
    pub map_memory2: khr::map_memory2::Device,
    pub maintenance6: khr::maintenance6::Device,
}

pub fn create_device() -> Result<DeviceContext, ResinStatus> {
    let entry = unsafe { Entry::load() }.map_err(|_| ResinStatus::VulkanUnavailable)?;

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"resin")
        .application_version(0)
        .engine_name(c"resin")
        .engine_version(0)
        .api_version(vk::API_VERSION_1_3);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None) }.map_err(vk_status)?;

    let physical_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|err| {
        unsafe { instance.destroy_instance(None) };
        vk_status(err)
    })?;
    if physical_devices.is_empty() {
        unsafe { instance.destroy_instance(None) };
        return Err(ResinStatus::VulkanUnavailable);
    }

    let Some(selected) = select_device(&instance, &physical_devices) else {
        unsafe { instance.destroy_instance(None) };
        return Err(ResinStatus::Unsupported);
    };

    let queue_priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(selected.queue_family)
        .queue_priorities(&queue_priorities);

    let mut enabled_extensions = vec![
        vk::EXT_SHADER_OBJECT_NAME.as_ptr(),
        vk::KHR_MAP_MEMORY2_NAME.as_ptr(),
        vk::KHR_MAINTENANCE5_NAME.as_ptr(),
        vk::KHR_MAINTENANCE6_NAME.as_ptr(),
    ];
    enabled_extensions.extend(selected.optional_extensions.iter().copied());

    let mut vulkan11 = selected.vulkan11;
    let mut vulkan12 = selected.vulkan12;
    let mut vulkan13 = selected.vulkan13;
    let mut shader_object = selected.shader_object;
    let mut maintenance5 = selected.maintenance5;
    let mut maintenance6 = selected.maintenance6;
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

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(selected.features10)
        .push_next(&mut vulkan11)
        .push_next(&mut vulkan12)
        .push_next(&mut vulkan13)
        .push_next(&mut shader_object)
        .push_next(&mut maintenance5)
        .push_next(&mut maintenance6);
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

    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(slice::from_ref(&queue_info))
        .enabled_extension_names(&enabled_extensions)
        .push_next(&mut features2);

    let device = unsafe { instance.create_device(selected.physical, &device_info, None) }.map_err(
        |err| {
            unsafe { instance.destroy_instance(None) };
            vk_status(err)
        },
    )?;

    let queue = unsafe { device.get_device_queue(selected.queue_family, 0) };
    let shader_object_fn = ext::shader_object::Device::new(&instance, &device);
    let map_memory2 = khr::map_memory2::Device::new(&instance, &device);
    let maintenance6_fn = khr::maintenance6::Device::new(&instance, &device);

    Ok(DeviceContext {
        entry,
        instance,
        device,
        queue,
        queue_family: selected.queue_family,
        memory_properties: selected.memory_properties,
        max_buffer_size: selected.max_buffer_size,
        memory_priority: selected.memory_priority_enabled,
        shader_object: shader_object_fn,
        map_memory2,
        maintenance6: maintenance6_fn,
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
    shader_object: vk::PhysicalDeviceShaderObjectFeaturesEXT<'static>,
    maintenance5: vk::PhysicalDeviceMaintenance5FeaturesKHR<'static>,
    maintenance6: vk::PhysicalDeviceMaintenance6FeaturesKHR<'static>,
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

fn select_device(instance: &Instance, devices: &[vk::PhysicalDevice]) -> Option<SelectedDevice> {
    let mut best: Option<(u32, SelectedDevice)> = None;
    for &physical in devices {
        let Some(selected) = inspect_device(instance, physical) else {
            continue;
        };
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
    for required in [
        vk::EXT_SHADER_OBJECT_NAME,
        vk::KHR_MAP_MEMORY2_NAME,
        vk::KHR_MAINTENANCE5_NAME,
        vk::KHR_MAINTENANCE6_NAME,
    ] {
        if !has_extension(&extensions, required) {
            return None;
        }
    }

    let queue_family = graphics_compute_queue_family(instance, physical)?;

    let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut shader_object = vk::PhysicalDeviceShaderObjectFeaturesEXT::default();
    let mut maintenance5 = vk::PhysicalDeviceMaintenance5FeaturesKHR::default();
    let mut maintenance6 = vk::PhysicalDeviceMaintenance6FeaturesKHR::default();
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
            .push_next(&mut vulkan13)
            .push_next(&mut shader_object)
            .push_next(&mut maintenance5)
            .push_next(&mut maintenance6);
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
        || shader_object.shader_object != vk::TRUE
        || maintenance5.maintenance5 != vk::TRUE
        || maintenance6.maintenance6 != vk::TRUE
    {
        return None;
    }

    let mut maint4 = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maint4);
    unsafe { instance.get_physical_device_properties2(physical, &mut props2) };

    let mut optional_extensions = Vec::new();
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
        shader_object: vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(true),
        maintenance5: vk::PhysicalDeviceMaintenance5FeaturesKHR::default().maintenance5(true),
        maintenance6: vk::PhysicalDeviceMaintenance6FeaturesKHR::default().maintenance6(true),
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

fn has_extension(extensions: &[vk::ExtensionProperties], name: &CStr) -> bool {
    extensions
        .iter()
        .any(|extension| extension.extension_name_as_c_str() == Ok(name))
}

fn on(value: vk::Bool32) -> bool {
    value == vk::TRUE
}
