//! Vulkan compute, graphics, and presentation for the Resin C ABI.

mod device;
mod pipeline;
mod present;

pub use pipeline::ResinPipeline;

use std::cell::Cell;
use std::ffi::c_char;
use std::ops::Range;
use std::ptr;
use std::rc::Rc;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use ash::{Device, Entry, Instance, vk};

use crate::allocator::RangeAllocator;
use crate::{ResinMemory, ResinStatus, ResinWindow};

use device::{create_device, create_device_at};

pub const GPU_DEVICE_NAME_MAX: usize = 256;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResinGpuDeviceType {
    Other = 0,
    Integrated = 1,
    Discrete = 2,
    Virtual = 3,
    Cpu = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ResinGpuDeviceInfo {
    pub index: u32,
    pub kind: ResinGpuDeviceType,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub driver_version: u32,
    pub suitable: u32,
    pub reserved: u32,
    pub device_local_bytes: u64,
    pub host_visible_device_local_bytes: u64,
    pub max_buffer_size: u64,
    pub name: [c_char; GPU_DEVICE_NAME_MAX],
}

const DEFAULT_ALIGNMENT: usize = 16;
const HEAP_BLOCK_BYTES: usize = 16 * 1024 * 1024;
const PUSH_CONSTANT_SIZE: u32 = 8;
const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

pub struct ResinGpu {
    _entry: Entry,
    instance: Instance,
    device: Device,
    push_layout: vk::PipelineLayout,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    timeline: vk::Semaphore,
    timeline_value: AtomicU64,
    timestamp_valid_bits: u32,
    timestamp_period: f32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    max_buffer_size: vk::DeviceSize,
    memory_priority: bool,
    presentation: Option<present::Presentation>,
    heaps: [Vec<Option<HeapBlock>>; 3],
}

struct HeapBlock {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    host: *mut u8,
    device_address: u64,
    size: u64,
    ranges: RangeAllocator,
}

pub struct ResinAllocation {
    memory: ResinMemory,
    block: usize,
    range: Range<u64>,
    host: *mut u8,
    device_address: u64,
    size: usize,
    buffer: vk::Buffer,
    buffer_offset: vk::DeviceSize,
}

pub struct ResinImage {
    device: Device,
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
    layout: Rc<Cell<vk::ImageLayout>>,
}

pub struct ResinCommandBuffer {
    device: Device,
    push_layout: vk::PipelineLayout,
    pool: vk::CommandPool,
    handle: vk::CommandBuffer,
    pipeline_bound: bool,
    graphics: bool,
    rendering: bool,
    submitted: bool,
    timestamp_pool: vk::QueryPool,
    layouts: ImageLayouts,
}

/// Layouts predicted by this recording, separate from submitted image state.
#[derive(Default)]
struct ImageLayouts {
    changes: Vec<ImageLayoutChange>,
}

struct ImageLayoutChange {
    committed: Rc<Cell<vk::ImageLayout>>,
    initial: vk::ImageLayout,
    final_layout: vk::ImageLayout,
}

impl ImageLayouts {
    fn transition(
        &mut self,
        committed: &Rc<Cell<vk::ImageLayout>>,
        next: vk::ImageLayout,
    ) -> vk::ImageLayout {
        if let Some(change) = self
            .changes
            .iter_mut()
            .find(|change| Rc::ptr_eq(&change.committed, committed))
        {
            let previous = change.final_layout;
            change.final_layout = next;
            previous
        } else {
            let initial = committed.get();
            self.changes.push(ImageLayoutChange {
                committed: committed.clone(),
                initial,
                final_layout: next,
            });
            initial
        }
    }

    fn is_current(&self) -> bool {
        self.changes
            .iter()
            .all(|change| change.committed.get() == change.initial)
    }

    fn commit(&self) {
        for change in &self.changes {
            change.committed.set(change.final_layout);
        }
    }
}

impl ResinGpu {
    pub fn create() -> Result<Self, ResinStatus> {
        Self::from_context(create_device()?)
    }

    pub fn create_at(index: u32) -> Result<Self, ResinStatus> {
        Self::from_context(create_device_at(index)?)
    }

    /// # Safety
    /// Use and destroy this GPU and window on the process main thread.
    pub unsafe fn create_for_window(window: &ResinWindow) -> Result<Self, ResinStatus> {
        Self::from_context(device::create_device_for_window(window)?)
    }

    pub fn device_count() -> Result<u32, ResinStatus> {
        Ok(device::enumerate_devices()?.len() as u32)
    }

    pub fn enumerate_devices(infos: &mut [ResinGpuDeviceInfo]) -> Result<(), ResinStatus> {
        let devices = device::enumerate_devices()?;
        let written = infos.len().min(devices.len());
        infos[..written].copy_from_slice(&devices[..written]);
        if written < devices.len() {
            Err(ResinStatus::Incomplete)
        } else {
            Ok(())
        }
    }

    fn from_context(mut created: device::DeviceContext) -> Result<Self, ResinStatus> {
        let properties = unsafe {
            created
                .instance
                .get_physical_device_properties(created.physical)
        };
        let queues = unsafe {
            created
                .instance
                .get_physical_device_queue_family_properties(created.physical)
        };
        let timestamp_valid_bits = queues[created.queue_family as usize].timestamp_valid_bits;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(created.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { created.device.create_command_pool(&pool_info, None) }
            .map_err(|err| {
                drop(created.surface.take());
                destroy_partial(
                    &created.device,
                    &created.instance,
                    vk::CommandPool::null(),
                    vk::Semaphore::null(),
                );
                vk_status(err)
            })?;

        let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type);
        let timeline =
            unsafe { created.device.create_semaphore(&semaphore_info, None) }.map_err(|err| {
                drop(created.surface.take());
                destroy_partial(
                    &created.device,
                    &created.instance,
                    command_pool,
                    vk::Semaphore::null(),
                );
                vk_status(err)
            })?;

        let push_range = root_range();
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(slice::from_ref(&push_range));
        let push_layout = unsafe { created.device.create_pipeline_layout(&layout_info, None) }
            .map_err(|err| {
                drop(created.surface.take());
                destroy_partial(&created.device, &created.instance, command_pool, timeline);
                vk_status(err)
            })?;

        let presentation = created.surface.take().map(|surface| {
            present::Presentation::new(
                &created.instance,
                &created.device,
                created.physical,
                surface,
            )
        });
        Ok(Self {
            _entry: created.entry,
            instance: created.instance,
            device: created.device,
            push_layout,
            queue: created.queue,
            command_pool,
            timeline,
            timeline_value: AtomicU64::new(0),
            timestamp_valid_bits,
            timestamp_period: properties.limits.timestamp_period,
            memory_properties: created.memory_properties,
            max_buffer_size: created.max_buffer_size,
            memory_priority: created.memory_priority,
            presentation,
            heaps: [Vec::new(), Vec::new(), Vec::new()],
        })
    }

    /// # Safety
    /// The GPU must outlive the allocation. All GPU and child-object operations must be externally synchronized.
    pub unsafe fn malloc(
        &mut self,
        bytes: usize,
        alignment: usize,
        memory: ResinMemory,
    ) -> Result<ResinAllocation, ResinStatus> {
        if bytes == 0 {
            return Err(ResinStatus::InvalidArgument);
        }
        let alignment = if alignment == 0 {
            DEFAULT_ALIGNMENT
        } else {
            alignment
        };
        if !alignment.is_power_of_two() {
            return Err(ResinStatus::InvalidArgument);
        }

        let heap = heap_index(memory);
        for block_index in 0..self.heaps[heap].len() {
            let Some(block) = self.heaps[heap][block_index].as_mut() else {
                continue;
            };
            if let Some(allocation) = try_suballocate(block, block_index, memory, bytes, alignment)
            {
                return Ok(allocation);
            }
        }

        let tight = bytes
            .checked_add(alignment - 1)
            .ok_or(ResinStatus::OutOfMemory)?;
        let slab = tight.max(HEAP_BLOCK_BYTES);
        let mut block = match self.create_heap_block(slab, memory) {
            Ok(block) => block,
            Err(ResinStatus::OutOfMemory) if slab > tight => {
                self.create_heap_block(tight, memory)?
            }
            Err(status) => return Err(status),
        };
        let block_index = self.reserve_heap_slot(heap);
        let Some(allocation) = try_suballocate(&mut block, block_index, memory, bytes, alignment)
        else {
            return Err(ResinStatus::OutOfMemory);
        };
        self.heaps[heap][block_index] = Some(block);
        Ok(allocation)
    }

    /// # Safety
    /// The allocation must be live, belong to this GPU, and have no outstanding host borrows or GPU uses. It must not be used or freed again.
    pub unsafe fn free(&mut self, allocation: &ResinAllocation) {
        let heap = heap_index(allocation.memory);
        let release = {
            let Some(block) = self.heaps[heap]
                .get_mut(allocation.block)
                .and_then(|slot| slot.as_mut())
            else {
                return;
            };
            block.ranges.free(allocation.range.clone());
            block.size as usize > HEAP_BLOCK_BYTES && block.ranges.is_fully_free()
        };
        if release {
            self.heaps[heap][allocation.block] = None;
        }
    }

    pub fn host_to_device(&self, host_pointer: *const u8) -> Result<u64, ResinStatus> {
        if host_pointer.is_null() {
            return Err(ResinStatus::InvalidArgument);
        }
        let addr = host_pointer as usize;
        for heap in &self.heaps {
            for block in heap.iter().flatten() {
                if block.host.is_null() {
                    continue;
                }
                let start = block.host as usize;
                let end = start + block.size as usize;
                if addr >= start && addr < end {
                    return Ok(block.device_address + (addr - start) as u64);
                }
            }
        }
        Err(ResinStatus::InvalidArgument)
    }

    fn reserve_heap_slot(&mut self, heap: usize) -> usize {
        if let Some(index) = self.heaps[heap].iter().position(|slot| slot.is_none()) {
            index
        } else {
            self.heaps[heap].push(None);
            self.heaps[heap].len() - 1
        }
    }

    fn create_heap_block(
        &self,
        bytes: usize,
        memory: ResinMemory,
    ) -> Result<HeapBlock, ResinStatus> {
        if self.max_buffer_size != 0 && bytes as vk::DeviceSize > self.max_buffer_size {
            return Err(ResinStatus::OutOfMemory);
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(bytes as vk::DeviceSize)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            );
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }.map_err(vk_status)?;

        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let req_info = vk::BufferMemoryRequirementsInfo2::default().buffer(buffer);
        unsafe {
            self.device
                .get_buffer_memory_requirements2(&req_info, &mut requirements2);
        }
        let requirements = requirements2.memory_requirements;

        let request = memory_request(memory);
        let candidates = memory_type_indices(
            &self.memory_properties,
            requirements.memory_type_bits,
            request,
        );
        if candidates.is_empty() {
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(ResinStatus::Unsupported);
        }

        let use_dedicated = dedicated.requires_dedicated_allocation == vk::TRUE
            || dedicated.prefers_dedicated_allocation == vk::TRUE;

        let device_memory = allocate_memory_with(
            requirements,
            &candidates,
            true,
            if use_dedicated {
                DedicatedAllocation::Buffer(buffer)
            } else {
                DedicatedAllocation::None
            },
            self.memory_priority.then_some(match memory {
                ResinMemory::Gpu => 1.0,
                ResinMemory::Default => 0.75,
                ResinMemory::Readback => 0.25,
            }),
            |info| unsafe { self.device.allocate_memory(info, None) },
        )
        .inspect_err(|_| {
            unsafe { self.device.destroy_buffer(buffer, None) };
        })?;

        let bind = vk::BindBufferMemoryInfo::default()
            .buffer(buffer)
            .memory(device_memory)
            .memory_offset(0);
        if let Err(err) = unsafe { self.device.bind_buffer_memory2(slice::from_ref(&bind)) } {
            unsafe {
                self.device.free_memory(device_memory, None);
                self.device.destroy_buffer(buffer, None);
            }
            return Err(vk_status(err));
        }

        let host = if request.host_visible {
            match unsafe {
                self.device.map_memory(
                    device_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            } {
                Ok(ptr) => ptr.cast::<u8>(),
                Err(err) => {
                    unsafe {
                        self.device.destroy_buffer(buffer, None);
                        self.device.free_memory(device_memory, None);
                    }
                    return Err(vk_status(err));
                }
            }
        } else {
            ptr::null_mut()
        };

        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe { self.device.get_buffer_device_address(&address_info) };
        let size = bytes as u64;

        Ok(HeapBlock {
            device: self.device.clone(),
            buffer,
            memory: device_memory,
            host,
            device_address,
            size,
            ranges: RangeAllocator::new(device_address..device_address + size),
        })
    }

    /// # Safety
    /// The dimensions must satisfy the device limits. The GPU must outlive the image and all commands using it.
    pub unsafe fn create_image(&self, width: u32, height: u32) -> Result<ResinImage, ResinStatus> {
        if width == 0 || height == 0 {
            return Err(ResinStatus::InvalidArgument);
        }
        let (image, view, memory) = self.create_color_image(
            width,
            height,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        Ok(ResinImage {
            device: self.device.clone(),
            image,
            view,
            memory,
            width,
            height,
            layout: Rc::new(Cell::new(vk::ImageLayout::UNDEFINED)),
        })
    }

    fn create_color_image(
        &self,
        width: u32,
        height: u32,
        samples: vk::SampleCountFlags,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory), ResinStatus> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(COLOR_FORMAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&image_info, None) }.map_err(vk_status)?;

        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let req_info = vk::ImageMemoryRequirementsInfo2::default().image(image);
        unsafe {
            self.device
                .get_image_memory_requirements2(&req_info, &mut requirements2);
        }
        let requirements = requirements2.memory_requirements;
        let request = MemoryRequest {
            required: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            preferred: vk::MemoryPropertyFlags::empty(),
            avoid: vk::MemoryPropertyFlags::HOST_VISIBLE,
            host_visible: false,
        };
        let candidates = memory_type_indices(
            &self.memory_properties,
            requirements.memory_type_bits,
            request,
        );
        if candidates.is_empty() {
            unsafe { self.device.destroy_image(image, None) };
            return Err(ResinStatus::Unsupported);
        }

        let use_dedicated = dedicated.requires_dedicated_allocation == vk::TRUE
            || dedicated.prefers_dedicated_allocation == vk::TRUE;

        let device_memory = allocate_memory_with(
            requirements,
            &candidates,
            false,
            if use_dedicated {
                DedicatedAllocation::Image(image)
            } else {
                DedicatedAllocation::None
            },
            self.memory_priority.then_some(1.0),
            |info| unsafe { self.device.allocate_memory(info, None) },
        )
        .inspect_err(|_| {
            unsafe { self.device.destroy_image(image, None) };
        })?;

        let bind = vk::BindImageMemoryInfo::default()
            .image(image)
            .memory(device_memory)
            .memory_offset(0);
        if let Err(err) = unsafe { self.device.bind_image_memory2(slice::from_ref(&bind)) } {
            unsafe {
                self.device.free_memory(device_memory, None);
                self.device.destroy_image(image, None);
            }
            return Err(vk_status(err));
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(COLOR_FORMAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        let view = match unsafe { self.device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(err) => {
                unsafe {
                    self.device.destroy_image(image, None);
                    self.device.free_memory(device_memory, None);
                }
                return Err(vk_status(err));
            }
        };
        Ok((image, view, device_memory))
    }

    /// # Safety
    /// The GPU must outlive the recording. Operations on its command pool and queue must be externally synchronized.
    pub unsafe fn start_command_recording(&self) -> Result<ResinCommandBuffer, ResinStatus> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let allocated =
            unsafe { self.device.allocate_command_buffers(&alloc_info) }.map_err(vk_status)?;
        let handle = allocated[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(err) = unsafe { self.device.begin_command_buffer(handle, &begin_info) } {
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &[handle])
            };
            return Err(vk_status(err));
        }
        Ok(ResinCommandBuffer {
            device: self.device.clone(),
            push_layout: self.push_layout,
            pool: self.command_pool,
            handle,
            pipeline_bound: false,
            graphics: false,
            rendering: false,
            submitted: false,
            timestamp_pool: vk::QueryPool::null(),
            layouts: ImageLayouts::default(),
        })
    }

    /// # Safety
    /// All recorded resources and GPU addresses must remain valid through completion and belong to this GPU. Host accesses and queue operations must be synchronized.
    pub unsafe fn submit(&self, command_buffer: ResinCommandBuffer) -> Result<(), ResinStatus> {
        unsafe { self.submit_signaling(command_buffer, vk::Semaphore::null()) }
    }

    pub(crate) unsafe fn record_with_timestamps(&self) -> Result<ResinCommandBuffer, ResinStatus> {
        if self.timestamp_valid_bits == 0 {
            return Err(ResinStatus::Unsupported);
        }
        let mut commands = unsafe { self.start_command_recording() }?;
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);
        commands.timestamp_pool =
            unsafe { self.device.create_query_pool(&info, None) }.map_err(vk_status)?;
        unsafe {
            self.device
                .cmd_reset_query_pool(commands.handle, commands.timestamp_pool, 0, 2);
            self.device.cmd_write_timestamp2(
                commands.handle,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                commands.timestamp_pool,
                0,
            );
        }
        Ok(commands)
    }

    pub(crate) unsafe fn submit_with_timestamps(
        &self,
        mut commands: ResinCommandBuffer,
    ) -> Result<Duration, ResinStatus> {
        if commands.timestamp_pool == vk::QueryPool::null() {
            return Err(ResinStatus::InvalidArgument);
        }
        unsafe { self.submit_and_wait(&mut commands, vk::Semaphore::null()) }?;
        let mut timestamps = [0u64; 2];
        unsafe {
            self.device.get_query_pool_results(
                commands.timestamp_pool,
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(vk_status)?;
        Ok(timestamp_elapsed(
            timestamps,
            self.timestamp_valid_bits,
            self.timestamp_period,
        ))
    }

    unsafe fn submit_signaling(
        &self,
        mut command_buffer: ResinCommandBuffer,
        ready: vk::Semaphore,
    ) -> Result<(), ResinStatus> {
        unsafe { self.submit_and_wait(&mut command_buffer, ready) }
    }

    unsafe fn submit_and_wait(
        &self,
        command_buffer: &mut ResinCommandBuffer,
        ready: vk::Semaphore,
    ) -> Result<(), ResinStatus> {
        if command_buffer.rendering || !command_buffer.layouts.is_current() {
            return Err(ResinStatus::InvalidArgument);
        }
        // Submission consumes the recording; no ended buffer returns to callers.
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        let dependency = vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier));
        unsafe {
            self.device
                .cmd_pipeline_barrier2(command_buffer.handle, &dependency);
            if command_buffer.timestamp_pool != vk::QueryPool::null() {
                self.device.cmd_write_timestamp2(
                    command_buffer.handle,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    command_buffer.timestamp_pool,
                    1,
                );
            }
        }
        unsafe { self.device.end_command_buffer(command_buffer.handle) }.map_err(vk_status)?;

        let value = self.timeline_value.fetch_add(1, Ordering::Relaxed) + 1;
        let command_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer.handle);
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let signals = [
            signal,
            vk::SemaphoreSubmitInfo::default()
                .semaphore(ready)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        let signal_count = if ready == vk::Semaphore::null() { 1 } else { 2 };
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(slice::from_ref(&command_info))
            .signal_semaphore_infos(&signals[..signal_count]);
        if let Err(err) = unsafe {
            self.device
                .queue_submit2(self.queue, slice::from_ref(&submit), vk::Fence::null())
        } {
            return Err(vk_status(err));
        }
        command_buffer.submitted = true;
        command_buffer.layouts.commit();

        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(slice::from_ref(&self.timeline))
            .values(slice::from_ref(&value));
        match unsafe { self.device.wait_semaphores(&wait, u64::MAX) } {
            Ok(()) => {
                command_buffer.submitted = false;
                Ok(())
            }
            Err(err) => Err(vk_status(err)),
        }
    }
}

impl ResinAllocation {
    pub fn host_pointer(&self) -> *mut u8 {
        self.host
    }

    /// # Safety
    /// The allocation and its GPU must remain live for the returned slice's lifetime. Its bytes must be initialized, and neither the host nor device may mutate them while borrowed.
    pub unsafe fn host_bytes(&self) -> Option<&[u8]> {
        if self.host.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(self.host, self.size) })
        }
    }

    pub fn device_pointer(&self) -> u64 {
        self.device_address
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl ResinImage {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl ResinCommandBuffer {
    /// # Safety
    /// The pipeline must belong to this recording's GPU and remain live through command completion.
    pub unsafe fn set_pipeline(&mut self, pipeline: &ResinPipeline) -> Result<(), ResinStatus> {
        if self.device.handle() != pipeline.device.handle() {
            return Err(ResinStatus::InvalidArgument);
        }
        let graphics = pipeline.bind_point == vk::PipelineBindPoint::GRAPHICS;
        if self.rendering && !graphics {
            return Err(ResinStatus::InvalidArgument);
        }
        unsafe {
            self.device
                .cmd_bind_pipeline(self.handle, pipeline.bind_point, pipeline.handle);
        }
        self.graphics = graphics;
        self.pipeline_bound = true;
        Ok(())
    }

    /// # Safety
    /// The image must belong to this recording's GPU and remain live through command completion.
    pub unsafe fn begin_rendering(
        &mut self,
        image: &mut ResinImage,
        clear: [f32; 4],
    ) -> Result<(), ResinStatus> {
        if self.rendering {
            return Err(ResinStatus::InvalidArgument);
        }
        // BDA resources can alias: order all earlier accesses before graphics.
        cmd_memory_barrier(&self.device, self.handle);
        let old_layout = self
            .layouts
            .transition(&image.layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let (src_stage, src_access) = match old_layout {
            vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            _ => (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
        };
        cmd_image_barrier(
            &self.device,
            self.handle,
            image.image,
            old_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_stage,
            src_access,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue { float32: clear },
        };
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value);
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: image.width,
                    height: image.height,
                },
            })
            .layer_count(1)
            .color_attachments(slice::from_ref(&attachment));
        unsafe {
            self.device.cmd_begin_rendering(self.handle, &rendering);
        }
        set_viewport(&self.device, self.handle, image.width, image.height);
        self.rendering = true;
        Ok(())
    }

    /// # Safety
    /// The recording's GPU and resources must remain live and externally synchronized.
    pub unsafe fn end_rendering(&mut self) -> Result<(), ResinStatus> {
        if !self.rendering {
            return Err(ResinStatus::InvalidArgument);
        }
        unsafe {
            self.device.cmd_end_rendering(self.handle);
        }
        self.rendering = false;
        Ok(())
    }

    /// # Safety
    /// The root address and every shader-accessed address must be valid for the bound shaders. All resources must remain live through completion.
    pub unsafe fn draw(&mut self, root_data: u64, vertex_count: u32) -> Result<(), ResinStatus> {
        if !self.pipeline_bound || !self.graphics || !self.rendering {
            return Err(ResinStatus::InvalidArgument);
        }
        if vertex_count == 0 {
            return Err(ResinStatus::InvalidArgument);
        }
        self.push_root(root_data);
        unsafe {
            self.device.cmd_draw(self.handle, vertex_count, 1, 0, 0);
        }
        Ok(())
    }

    /// # Safety
    /// Both resources must belong to this recording's GPU and remain live through completion. The image contents must have been initialized.
    pub unsafe fn copy_image_to_buffer(
        &mut self,
        image: &mut ResinImage,
        dst: &ResinAllocation,
    ) -> Result<(), ResinStatus> {
        if self.rendering {
            return Err(ResinStatus::InvalidArgument);
        }
        let bytes = (image.width as usize)
            .checked_mul(image.height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(ResinStatus::InvalidArgument)?;
        if dst.size < bytes {
            return Err(ResinStatus::InvalidArgument);
        }
        if !dst.buffer_offset.is_multiple_of(4) {
            return Err(ResinStatus::InvalidArgument);
        }
        cmd_memory_barrier(&self.device, self.handle);
        let old_layout = self
            .layouts
            .transition(&image.layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        cmd_image_barrier(
            &self.device,
            self.handle,
            image.image,
            old_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let region = vk::BufferImageCopy::default()
            .buffer_offset(dst.buffer_offset)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: image.width,
                height: image.height,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.handle,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.buffer,
                slice::from_ref(&region),
            );
        }
        Ok(())
    }

    /// # Safety
    /// The workgroup counts must satisfy device limits. The root address and every shader-accessed address must be valid, synchronized, and live through completion.
    pub unsafe fn dispatch(
        &mut self,
        root_data: u64,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> Result<(), ResinStatus> {
        if !self.pipeline_bound || self.graphics || self.rendering {
            return Err(ResinStatus::InvalidArgument);
        }
        cmd_memory_barrier(&self.device, self.handle);
        self.push_root(root_data);
        unsafe {
            self.device
                .cmd_dispatch(self.handle, group_count_x, group_count_y, group_count_z);
        }
        Ok(())
    }

    fn push_root(&self, root_data: u64) {
        unsafe {
            self.device.cmd_push_constants(
                self.handle,
                self.push_layout,
                root_range().stage_flags,
                0,
                &root_data.to_ne_bytes(),
            );
        }
    }
}

fn root_range() -> vk::PushConstantRange {
    vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE
            | vk::ShaderStageFlags::VERTEX
            | vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: PUSH_CONSTANT_SIZE,
    }
}

impl Drop for ResinGpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            drop(self.presentation.take());
            for heap in &mut self.heaps {
                heap.clear();
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline_layout(self.push_layout, None);
            self.device.destroy_semaphore(self.timeline, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Drop for HeapBlock {
    fn drop(&mut self) {
        unsafe {
            if !self.host.is_null() {
                self.device.unmap_memory(self.memory);
                self.host = ptr::null_mut();
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl Drop for ResinImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl Drop for ResinCommandBuffer {
    fn drop(&mut self) {
        if self.handle == vk::CommandBuffer::null() {
            return;
        }
        if self.submitted && unsafe { self.device.device_wait_idle() }.is_err() {
            self.handle = vk::CommandBuffer::null();
            return;
        }
        unsafe {
            self.device
                .free_command_buffers(self.pool, slice::from_ref(&self.handle));
            if self.timestamp_pool != vk::QueryPool::null() {
                self.device.destroy_query_pool(self.timestamp_pool, None);
            }
        }
        self.handle = vk::CommandBuffer::null();
    }
}

fn timestamp_elapsed(timestamps: [u64; 2], valid_bits: u32, period_ns: f32) -> Duration {
    // Undefined high bits must not affect elapsed time on narrower hardware counters.
    let mask = u64::MAX >> (64 - valid_bits);
    let ticks = timestamps[1].wrapping_sub(timestamps[0]) & mask;
    Duration::from_secs_f64(ticks as f64 * f64::from(period_ns) * 1e-9)
}

#[derive(Clone, Copy)]
struct MemoryRequest {
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
    avoid: vk::MemoryPropertyFlags,
    host_visible: bool,
}

fn memory_request(memory: ResinMemory) -> MemoryRequest {
    match memory {
        ResinMemory::Default => MemoryRequest {
            required: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            preferred: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            avoid: vk::MemoryPropertyFlags::empty(),
            host_visible: true,
        },
        ResinMemory::Gpu => MemoryRequest {
            required: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            preferred: vk::MemoryPropertyFlags::empty(),
            avoid: vk::MemoryPropertyFlags::HOST_VISIBLE,
            host_visible: false,
        },
        ResinMemory::Readback => MemoryRequest {
            required: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            preferred: vk::MemoryPropertyFlags::HOST_CACHED,
            avoid: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            host_visible: true,
        },
    }
}

fn memory_type_indices(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    request: MemoryRequest,
) -> Vec<u32> {
    let mut scored = Vec::new();
    for index in 0..properties.memory_type_count {
        if type_bits & (1 << index) == 0 {
            continue;
        }
        let flags = properties.memory_types[index as usize].property_flags;
        if !flags.contains(request.required) {
            continue;
        }
        let mut score = 0u32;
        if !request.preferred.is_empty() && flags.contains(request.preferred) {
            score += 2;
        }
        if request.avoid.is_empty() || !flags.intersects(request.avoid) {
            score += 1;
        }
        scored.push((score, index));
    }
    scored.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    scored.into_iter().map(|(_, index)| index).collect()
}

#[derive(Clone, Copy)]
enum DedicatedAllocation {
    None,
    Buffer(vk::Buffer),
    Image(vk::Image),
}

fn allocate_memory_with(
    requirements: vk::MemoryRequirements,
    candidates: &[u32],
    device_address: bool,
    dedicated: DedicatedAllocation,
    priority: Option<f32>,
    mut allocate: impl FnMut(&vk::MemoryAllocateInfo<'_>) -> Result<vk::DeviceMemory, vk::Result>,
) -> Result<vk::DeviceMemory, ResinStatus> {
    for &memory_type_index in candidates {
        // push_next mutates each node, so a retry must start with an entirely fresh chain.
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut dedicated_info = match dedicated {
            DedicatedAllocation::None => vk::MemoryDedicatedAllocateInfo::default(),
            DedicatedAllocation::Buffer(buffer) => {
                vk::MemoryDedicatedAllocateInfo::default().buffer(buffer)
            }
            DedicatedAllocation::Image(image) => {
                vk::MemoryDedicatedAllocateInfo::default().image(image)
            }
        };
        let mut priority_info =
            vk::MemoryPriorityAllocateInfoEXT::default().priority(priority.unwrap_or(0.5));
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        if device_address {
            info = info.push_next(&mut flags);
        }
        if !matches!(dedicated, DedicatedAllocation::None) {
            info = info.push_next(&mut dedicated_info);
        }
        if priority.is_some() {
            info = info.push_next(&mut priority_info);
        }
        match allocate(&info) {
            Ok(memory) => return Ok(memory),
            Err(err) if vk_status(err) == ResinStatus::OutOfMemory => {}
            Err(err) => return Err(vk_status(err)),
        }
    }
    Err(ResinStatus::OutOfMemory)
}

fn try_suballocate(
    block: &mut HeapBlock,
    block_index: usize,
    memory: ResinMemory,
    bytes: usize,
    alignment: usize,
) -> Option<ResinAllocation> {
    let range = block.ranges.allocate(bytes as u64, alignment as u64).ok()?;
    let offset = range.start - block.device_address;
    let host = if block.host.is_null() {
        ptr::null_mut()
    } else {
        unsafe { block.host.add(offset as usize) }
    };
    Some(ResinAllocation {
        memory,
        block: block_index,
        range: range.clone(),
        host,
        device_address: range.start,
        size: bytes,
        buffer: block.buffer,
        buffer_offset: offset,
    })
}

fn heap_index(memory: ResinMemory) -> usize {
    match memory {
        ResinMemory::Default => 0,
        ResinMemory::Gpu => 1,
        ResinMemory::Readback => 2,
    }
}

fn set_viewport(device: &Device, cmd: vk::CommandBuffer, width: u32, height: u32) {
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: width as f32,
        height: height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width, height },
    };
    unsafe {
        device.cmd_set_viewport(cmd, 0, slice::from_ref(&viewport));
        device.cmd_set_scissor(cmd, 0, slice::from_ref(&scissor));
    }
}

/// Conservative ordering for arbitrary BDA aliases across compute, rendering,
/// copies, and submissions on the single queue. Called outside rendering only.
fn cmd_memory_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
    let dependency = vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier));
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dependency);
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_image_barrier(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );
    let dependency = vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&barrier));
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dependency);
    }
}

fn destroy_partial(
    device: &Device,
    instance: &Instance,
    command_pool: vk::CommandPool,
    timeline: vk::Semaphore,
) {
    unsafe {
        if command_pool != vk::CommandPool::null() {
            device.destroy_command_pool(command_pool, None);
        }
        if timeline != vk::Semaphore::null() {
            device.destroy_semaphore(timeline, None);
        }
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}

pub(crate) fn vk_status(err: vk::Result) -> ResinStatus {
    match err {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            ResinStatus::OutOfMemory
        }
        vk::Result::ERROR_INITIALIZATION_FAILED
        | vk::Result::ERROR_INCOMPATIBLE_DRIVER
        | vk::Result::ERROR_LAYER_NOT_PRESENT => ResinStatus::VulkanUnavailable,
        vk::Result::ERROR_FEATURE_NOT_PRESENT | vk::Result::ERROR_EXTENSION_NOT_PRESENT => {
            ResinStatus::Unsupported
        }
        _ => ResinStatus::VulkanError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::Handle;

    #[test]
    fn timestamp_elapsed_uses_device_period() {
        assert_eq!(
            timestamp_elapsed([10, 30], 64, 2.5),
            Duration::from_nanos(50)
        );
        assert_eq!(timestamp_elapsed([10, 10], 64, 1.0), Duration::ZERO);
    }

    #[test]
    fn timestamp_elapsed_masks_high_bits_and_handles_counter_wrap() {
        assert_eq!(
            timestamp_elapsed([0xab00_00f0, 0xcd00_0010], 8, 1.0),
            Duration::from_nanos(32),
        );
        assert_eq!(
            timestamp_elapsed([u64::MAX - 3, 5], 64, 1.0),
            Duration::from_nanos(9),
        );
    }
    #[test]
    fn allocation_retries_have_acyclic_complete_extension_chains() {
        for device_address in [false, true] {
            for dedicated in [
                DedicatedAllocation::None,
                DedicatedAllocation::Buffer(vk::Buffer::from_raw(1)),
                DedicatedAllocation::Image(vk::Image::from_raw(2)),
            ] {
                for priority in [None, Some(0.75)] {
                    let mut calls = 0;
                    let memory = allocate_memory_with(
                        vk::MemoryRequirements {
                            size: 4096,
                            alignment: 16,
                            memory_type_bits: 0b10101,
                        },
                        &[0, 2, 4],
                        device_address,
                        dedicated,
                        priority,
                        |info| {
                            assert_eq!(info.allocation_size, 4096);
                            assert_eq!(info.memory_type_index, calls * 2);
                            let mut types = Vec::new();
                            let mut node = info.p_next.cast::<vk::BaseInStructure<'_>>();
                            while !node.is_null() {
                                // SAFETY: the callback borrows the live allocation-info chain.
                                let current = unsafe { &*node };
                                assert!(
                                    !types.contains(&current.s_type),
                                    "cyclic or duplicate pNext node"
                                );
                                types.push(current.s_type);
                                node = current.p_next;
                            }
                            assert_eq!(
                                types.contains(&vk::StructureType::MEMORY_ALLOCATE_FLAGS_INFO),
                                device_address
                            );
                            assert_eq!(
                                types.contains(&vk::StructureType::MEMORY_DEDICATED_ALLOCATE_INFO),
                                !matches!(dedicated, DedicatedAllocation::None)
                            );
                            assert_eq!(
                                types.contains(
                                    &vk::StructureType::MEMORY_PRIORITY_ALLOCATE_INFO_EXT
                                ),
                                priority.is_some()
                            );
                            calls += 1;
                            if calls < 3 {
                                Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
                            } else {
                                Ok(vk::DeviceMemory::from_raw(7))
                            }
                        },
                    )
                    .unwrap();
                    assert_eq!(memory, vk::DeviceMemory::from_raw(7));
                    assert_eq!(calls, 3);
                }
            }
        }
    }

    #[test]
    fn allocation_does_not_retry_a_non_memory_error() {
        let mut calls = 0;
        let result = allocate_memory_with(
            vk::MemoryRequirements::default(),
            &[0, 1],
            true,
            DedicatedAllocation::None,
            None,
            |_| {
                calls += 1;
                Err(vk::Result::ERROR_DEVICE_LOST)
            },
        );
        assert_eq!(result, Err(ResinStatus::VulkanError));
        assert_eq!(calls, 1);
    }

    #[test]
    fn cancelling_a_recording_keeps_committed_layouts() {
        let image = Rc::new(Cell::new(vk::ImageLayout::UNDEFINED));
        let mut cancelled = ImageLayouts::default();
        assert_eq!(
            (cancelled.transition(&image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)).as_raw(),
            vk::ImageLayout::UNDEFINED.as_raw()
        );
        assert_eq!(
            (cancelled.transition(&image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)).as_raw(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw()
        );
        drop(cancelled);
        assert_eq!((image.get()).as_raw(), vk::ImageLayout::UNDEFINED.as_raw());
        let mut next = ImageLayouts::default();
        assert_eq!(
            (next.transition(&image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)).as_raw(),
            vk::ImageLayout::UNDEFINED.as_raw()
        );
        assert!(next.is_current());
        next.commit();
        assert_eq!(
            (image.get()).as_raw(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw()
        );
    }

    #[test]
    fn submitting_another_recording_invalidates_stale_layout_predictions() {
        let image = Rc::new(Cell::new(vk::ImageLayout::UNDEFINED));
        let mut first = ImageLayouts::default();
        let mut second = ImageLayouts::default();
        first.transition(&image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        second.transition(&image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert!(second.is_current());
        first.commit();
        assert!(!second.is_current());
        drop(second);
        assert_eq!(
            (image.get()).as_raw(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw()
        );
        let mut third = ImageLayouts::default();
        assert_eq!(
            (third.transition(&image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)).as_raw(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw()
        );
        assert!(third.is_current());
    }
}
