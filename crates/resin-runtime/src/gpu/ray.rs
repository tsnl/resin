//! Immutable triangle scenes and depth-one ray pipelines. All builds complete before publication.
use super::{
    ResinCommandBuffer, ResinGpu, ResinPipeline, ResinStatus, cmd_memory_barrier, vk_status,
};
use ash::{khr, vk};
use std::{rc::Rc, slice};

pub(super) struct Device {
    acceleration: khr::acceleration_structure::Device,
    pipeline: khr::ray_tracing_pipeline::Device,
    properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    scratch_alignment: u64,
    max_primitives: u64,
    max_instances: u64,
    max_dimensions: [u64; 3],
}
impl Device {
    pub(super) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Self {
        let mut properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut acceleration = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut limits = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut properties)
            .push_next(&mut acceleration);
        unsafe { instance.get_physical_device_properties2(physical, &mut limits) };
        let max_dimensions = std::array::from_fn(|i| {
            u64::from(limits.properties.limits.max_compute_work_group_count[i])
                * u64::from(limits.properties.limits.max_compute_work_group_size[i])
        });
        properties.p_next = std::ptr::null_mut();
        Self {
            acceleration: khr::acceleration_structure::Device::new(instance, device),
            pipeline: khr::ray_tracing_pipeline::Device::new(instance, device),
            properties,
            scratch_alignment: acceleration
                .min_acceleration_structure_scratch_offset_alignment
                .into(),
            max_primitives: acceleration.max_primitive_count,
            max_instances: acceleration.max_instance_count,
            max_dimensions,
        }
    }
}

struct Buffer {
    device: ash::Device,
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    address: u64,
    bytes: usize,
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.handle, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
impl Buffer {
    fn new(
        gpu: &ResinGpu,
        bytes: usize,
        usage: vk::BufferUsageFlags,
        host: bool,
    ) -> Result<Self, ResinStatus> {
        let info = vk::BufferCreateInfo::default()
            .size(bytes.max(1) as u64)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let handle = unsafe { gpu.device.create_buffer(&info, None) }.map_err(vk_status)?;
        let result = Self::allocate(gpu, handle, bytes, host);
        if result.is_err() {
            unsafe { gpu.device.destroy_buffer(handle, None) };
        }
        result
    }
    fn allocate(
        gpu: &ResinGpu,
        handle: vk::Buffer,
        bytes: usize,
        host: bool,
    ) -> Result<Self, ResinStatus> {
        let requirements = unsafe { gpu.device.get_buffer_memory_requirements(handle) };
        let flags = if host {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let index = (0..gpu.memory_properties.memory_type_count)
            .find(|i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && gpu.memory_properties.memory_types[*i as usize]
                        .property_flags
                        .contains(flags)
            })
            .ok_or(ResinStatus::Unsupported)?;
        let mut address =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(handle);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index)
            .push_next(&mut address)
            .push_next(&mut dedicated);
        let memory = unsafe { gpu.device.allocate_memory(&info, None) }.map_err(vk_status)?;
        if let Err(error) = unsafe { gpu.device.bind_buffer_memory(handle, memory, 0) } {
            unsafe { gpu.device.free_memory(memory, None) };
            return Err(vk_status(error));
        }
        let address = unsafe {
            gpu.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(handle))
        };
        Ok(Self {
            device: gpu.device.clone(),
            handle,
            memory,
            address,
            bytes,
        })
    }
    fn upload(&self, bytes: &[u8]) -> Result<(), ResinStatus> {
        if bytes.len() > self.bytes {
            return Err(ResinStatus::InvalidArgument);
        }
        let mapped = unsafe {
            self.device
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(vk_status)?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast(), bytes.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }
}

struct Acceleration {
    loader: khr::acceleration_structure::Device,
    handle: vk::AccelerationStructureKHR,
    address: u64,
    _buffer: Buffer,
}
impl Drop for Acceleration {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_acceleration_structure(self.handle, None)
        };
    }
}
pub(crate) struct Scene {
    top: Acceleration,
    _bottom: Acceleration,
}
use crate::ResinRayScene;

impl ResinGpu {
    pub(crate) fn supports_ray_tracing(&self) -> bool {
        self.ray.is_some()
    }

    /// Build one opaque non-indexed triangle mesh with one or more affine instances.
    /// Inputs are copied; the completed scene retains only its acceleration structures.
    pub(crate) unsafe fn create_ray_scene(
        &self,
        vertices: &[f32],
        transforms: &[f32],
    ) -> Result<ResinRayScene, ResinStatus> {
        let ray = self.ray.as_ref().ok_or(ResinStatus::Unsupported)?;
        if vertices.is_empty()
            || !vertices.len().is_multiple_of(9)
            || transforms.is_empty()
            || !transforms.len().is_multiple_of(12)
            || vertices.len() / 3 > u32::MAX as usize
            || (vertices.len() / 9) as u64 > ray.max_primitives
            || (transforms.len() / 12) as u64 > ray.max_instances
            || transforms.len() / 12 > 0x1000000
            || vertices.iter().chain(transforms).any(|x| !x.is_finite())
        {
            return Err(ResinStatus::InvalidArgument);
        }
        let input = Buffer::new(
            self,
            std::mem::size_of_val(vertices),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        )?;
        input.upload(unsafe { bytes(vertices) })?;
        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: input.address,
            })
            .vertex_stride(12)
            .max_vertex((vertices.len() / 3 - 1) as u32)
            .index_type(vk::IndexType::NONE_KHR);
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles });
        let bottom = unsafe {
            self.build_acceleration(
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                geometry,
                (vertices.len() / 9) as u32,
            )
        }?;
        let instances: Vec<_> = transforms
            .chunks_exact(12)
            .enumerate()
            .map(|(index, transform)| vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: transform.try_into().unwrap(),
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(index as u32, 255),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: bottom.address,
                },
            })
            .collect();
        let instance_buffer = Buffer::new(
            self,
            std::mem::size_of_val(instances.as_slice()),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        )?;
        instance_buffer.upload(unsafe { bytes(&instances) })?;
        let data = vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
            vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer.address,
            },
        );
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances: data });
        let top = unsafe {
            self.build_acceleration(
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                geometry,
                instances.len() as u32,
            )
        }?;
        Ok(ResinRayScene {
            scene: Rc::new(Scene {
                top,
                _bottom: bottom,
            }),
        })
    }

    unsafe fn build_acceleration(
        &self,
        kind: vk::AccelerationStructureTypeKHR,
        geometry: vk::AccelerationStructureGeometryKHR<'_>,
        count: u32,
    ) -> Result<Acceleration, ResinStatus> {
        let ray = self.ray.as_ref().ok_or(ResinStatus::Unsupported)?;
        let geometries = [geometry];
        let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(kind)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            ray.acceleration.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &[count],
                &mut sizes,
            )
        };
        let buffer = Buffer::new(
            self,
            sizes.acceleration_structure_size as usize,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            false,
        )?;
        let create = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.handle)
            .size(sizes.acceleration_structure_size)
            .ty(kind);
        let handle = unsafe {
            ray.acceleration
                .create_acceleration_structure(&create, None)
        }
        .map_err(vk_status)?;
        let address = unsafe {
            ray.acceleration.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        };
        let acceleration = Acceleration {
            loader: ray.acceleration.clone(),
            handle,
            address,
            _buffer: buffer,
        };
        let scratch_size = sizes
            .build_scratch_size
            .checked_add(ray.scratch_alignment)
            .ok_or(ResinStatus::OutOfMemory)?;
        let scratch = Buffer::new(
            self,
            scratch_size as usize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
        )?;
        let aligned = align(scratch.address, ray.scratch_alignment)?;
        info = info
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: aligned,
            });
        let commands = unsafe { self.start_command_recording() }?;
        cmd_memory_barrier(&self.device, commands.handle);
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)];
        unsafe {
            ray.acceleration
                .cmd_build_acceleration_structures(commands.handle, &[info], &[&ranges])
        };
        cmd_memory_barrier(&self.device, commands.handle);
        unsafe { self.submit(commands) }?;
        Ok(acceleration)
    }

    pub(crate) unsafe fn create_ray_pipeline(
        &self,
        scene: &ResinRayScene,
        shaders: [&[u8]; 3],
    ) -> Result<ResinPipeline, ResinStatus> {
        let ray = self.ray.as_ref().ok_or(ResinStatus::Unsupported)?;
        if scene.scene.top._buffer.device.handle() != self.device.handle() {
            return Err(ResinStatus::InvalidArgument);
        }
        let modules =
            shaders.map(|shader| super::pipeline::ShaderModule::create(&self.device, shader));
        let [generation, miss, hit] = modules;
        let (generation, miss, hit) = (generation?, miss?, hit?);
        let stages = [
            generation.stage(vk::ShaderStageFlags::RAYGEN_KHR),
            miss.stage(vk::ShaderStageFlags::MISS_KHR),
            hit.stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        ];
        let empty = vk::RayTracingShaderGroupCreateInfoKHR::default()
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);
        let groups = [
            empty
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0),
            empty
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1),
            empty
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .closest_hit_shader(2),
        ];
        let create = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1)
            .layout(self.push_layout);
        let handle = match unsafe {
            ray.pipeline.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[create],
                None,
            )
        } {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, error)) => {
                for pipeline in pipelines {
                    unsafe { self.device.destroy_pipeline(pipeline, None) };
                }
                return Err(vk_status(error));
            }
        };
        let mut pipeline = ResinPipeline {
            device: self.device.clone(),
            handle,
            bind_point: vk::PipelineBindPoint::RAY_TRACING_KHR,
            ray: None,
        };
        let size = ray.properties.shader_group_handle_size as usize;
        let stride = align(
            size as u64,
            ray.properties.shader_group_handle_alignment.into(),
        )?;
        if stride > ray.properties.max_shader_group_stride.into() {
            return Err(ResinStatus::Unsupported);
        }
        let spacing = align(stride, ray.properties.shader_group_base_alignment.into())?;
        let handles = unsafe {
            ray.pipeline
                .get_ray_tracing_shader_group_handles(handle, 0, 3, size * 3)
        }
        .map_err(vk_status)?;
        let table = Buffer::new(
            self,
            (spacing * 4) as usize,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            true,
        )?;
        let base = align(
            table.address,
            ray.properties.shader_group_base_alignment.into(),
        )?;
        let offset = (base - table.address) as usize;
        let mut data = vec![0; table.bytes];
        for i in 0..3 {
            data[offset + i * spacing as usize..offset + i * spacing as usize + size]
                .copy_from_slice(&handles[i * size..(i + 1) * size]);
        }
        table.upload(&data)?;
        let regions = std::array::from_fn(|i| vk::StridedDeviceAddressRegionKHR {
            device_address: base + i as u64 * spacing,
            stride,
            size: stride,
        });
        pipeline.ray = Some(Dispatch {
            loader: ray.pipeline.clone(),
            resources: Rc::new(Resources {
                _table: table,
                scene: scene.scene.clone(),
            }),
            regions,
            max_count: ray.properties.max_ray_dispatch_invocation_count.into(),
            max_dimensions: ray.max_dimensions,
        });
        Ok(pipeline)
    }
}

struct Resources {
    _table: Buffer,
    scene: Rc<Scene>,
}
#[derive(Clone)]
pub(super) struct Dispatch {
    loader: khr::ray_tracing_pipeline::Device,
    resources: Rc<Resources>,
    regions: [vk::StridedDeviceAddressRegionKHR; 3],
    max_count: u64,
    max_dimensions: [u64; 3],
}
impl ResinCommandBuffer {
    pub(crate) unsafe fn trace_rays(
        &mut self,
        root: u64,
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), ResinStatus> {
        let ray = self.ray.as_ref().ok_or(ResinStatus::InvalidArgument)?;
        let dimensions = [x, y, z];
        let count = u64::from(x)
            .checked_mul(y.into())
            .and_then(|xy| xy.checked_mul(z.into()))
            .ok_or(ResinStatus::InvalidArgument)?;
        if self.rendering
            || count > ray.max_count
            || dimensions
                .iter()
                .zip(ray.max_dimensions)
                .any(|(value, max)| u64::from(*value) > max)
        {
            return Err(ResinStatus::InvalidArgument);
        }
        cmd_memory_barrier(&self.device, self.handle);
        let values = [root, ray.resources.scene.top.address];
        unsafe {
            self.device.cmd_push_constants(
                self.handle,
                self.push_layout,
                vk::ShaderStageFlags::ALL,
                0,
                bytes(&values),
            );
            ray.loader.cmd_trace_rays(
                self.handle,
                &ray.regions[0],
                &ray.regions[1],
                &ray.regions[2],
                &vk::StridedDeviceAddressRegionKHR::default(),
                x,
                y,
                z,
            );
        }
        Ok(())
    }
}
fn align(value: u64, alignment: u64) -> Result<u64, ResinStatus> {
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or(ResinStatus::OutOfMemory)
}
unsafe fn bytes<T>(values: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
