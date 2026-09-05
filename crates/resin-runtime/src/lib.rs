//! Host support and headless Vulkan runtime, exposed through C and unsafe Rust APIs.
//!
//! CPU-visible allocations have mapped host pointers and GPU addresses.
//! Shaders receive a 64-bit root address as a push constant.
//!
//! A GPU must outlive its resources; recorded resources must outlive completion
//! or cancellation. GPU and child-object operations require external synchronization.
//! Host borrows must not overlap GPU writes or deallocation.

mod allocator;
mod gpu;
mod host;
mod image;
mod print;

pub const INCLUDE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/include");

use std::ffi::c_void;
use std::os::raw::c_char;
use std::ptr;

pub use gpu::{
    GPU_DEVICE_NAME_MAX, ResinAllocation, ResinCommandBuffer, ResinGpu, ResinGpuDeviceInfo,
    ResinGpuDeviceType, ResinImage, ResinPipeline,
};

#[doc(hidden)]
pub mod testing {
    use std::fs::File;
    use std::os::unix::io::AsRawFd;

    pub struct GpuLock {
        file: File,
    }

    impl Drop for GpuLock {
        fn drop(&mut self) {
            unsafe {
                libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
            }
        }
    }

    /// Exclusive lock shared by unit and integration GPU tests.
    pub fn lock_gpu() -> GpuLock {
        let path = std::env::temp_dir().join("resin-runtime-gpu.lock");
        let file = File::create(&path).expect("create gpu lock file");
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        assert_eq!(rc, 0, "flock");
        GpuLock { file }
    }
}
pub use image::{PngImage, image_read_png, image_write_png};

pub type ResinDeviceAddress = u64;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResinStatus {
    Success = 0,
    InvalidArgument = 1,
    VulkanUnavailable = 2,
    Unsupported = 3,
    OutOfMemory = 4,
    VulkanError = 5,
    IoError = 6,
    Incomplete = 7,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResinMemory {
    /// Persistently mapped, coherent memory; device-local when the hardware permits.
    Default = 0,
    /// Device-local memory, which need not be visible to the host.
    Gpu = 1,
    /// Persistently mapped, coherent memory intended for device-to-host results.
    Readback = 2,
}

impl ResinStatus {
    fn from_raw(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Success),
            1 => Some(Self::InvalidArgument),
            2 => Some(Self::VulkanUnavailable),
            3 => Some(Self::Unsupported),
            4 => Some(Self::OutOfMemory),
            5 => Some(Self::VulkanError),
            6 => Some(Self::IoError),
            7 => Some(Self::Incomplete),
            _ => None,
        }
    }
}

impl ResinMemory {
    fn from_raw(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Default),
            1 => Some(Self::Gpu),
            2 => Some(Self::Readback),
            _ => None,
        }
    }
}

/// # Safety
/// `count` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_device_count(count: *mut u32) -> ResinStatus {
    if count.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *count = 0 };
    match ResinGpu::device_count() {
        Ok(n) => {
            unsafe { *count = n };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// Writes up to `count` entries; returns [`ResinStatus::Incomplete`] if truncated.
///
/// # Safety
/// `infos` must point to `count` elements when `count` is nonzero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_enumerate_devices(
    infos: *mut ResinGpuDeviceInfo,
    count: u32,
) -> ResinStatus {
    if count > 0 && infos.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let out = if count == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(infos, count as usize) }
    };
    match ResinGpu::enumerate_devices(out) {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// # Safety
/// `out_gpu` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_create(out_gpu: *mut *mut ResinGpu) -> ResinStatus {
    if out_gpu.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_gpu = ptr::null_mut() };
    match ResinGpu::create() {
        Ok(gpu) => {
            unsafe { *out_gpu = Box::into_raw(Box::new(gpu)) };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// `out_gpu` must be a valid pointer. `index` is a Vulkan physical-device index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_create_at(
    index: u32,
    out_gpu: *mut *mut ResinGpu,
) -> ResinStatus {
    if out_gpu.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_gpu = ptr::null_mut() };
    match ResinGpu::create_at(index) {
        Ok(gpu) => {
            unsafe { *out_gpu = Box::into_raw(Box::new(gpu)) };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// `gpu` must be null or a pointer from [`resin_gpu_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_destroy(gpu: *mut ResinGpu) {
    if !gpu.is_null() {
        drop(unsafe { Box::from_raw(gpu) });
    }
}

/// # Safety
/// `gpu` and `out_allocation` must be valid; `gpu` must outlive the allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_malloc(
    gpu: *mut ResinGpu,
    bytes: usize,
    alignment: usize,
    memory: i32,
    out_allocation: *mut *mut ResinAllocation,
) -> ResinStatus {
    if gpu.is_null() || out_allocation.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_allocation = ptr::null_mut() };
    let Some(memory) = ResinMemory::from_raw(memory) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { (&mut *gpu).malloc(bytes, alignment, memory) } {
        Ok(allocation) => {
            unsafe { *out_allocation = Box::into_raw(Box::new(allocation)) };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// `allocation` must be null or a pointer from [`resin_gpu_malloc`] for `gpu`.
/// A null `gpu` leaves the allocation untouched.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_free(gpu: *mut ResinGpu, allocation: *mut ResinAllocation) {
    if allocation.is_null() {
        return;
    }
    let Some(gpu) = (unsafe { gpu.as_mut() }) else {
        return;
    };
    unsafe { gpu.free(&*allocation) };
    drop(unsafe { Box::from_raw(allocation) });
}

/// # Safety
/// `allocation` must be null or a live allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_allocation_host_pointer(
    allocation: *const ResinAllocation,
) -> *mut c_void {
    unsafe { allocation.as_ref() }
        .map(ResinAllocation::host_pointer)
        .unwrap_or(ptr::null_mut())
        .cast()
}

/// # Safety
/// `allocation` must be null or a live allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_allocation_device_pointer(
    allocation: *const ResinAllocation,
) -> ResinDeviceAddress {
    unsafe { allocation.as_ref() }
        .map(ResinAllocation::device_pointer)
        .unwrap_or(0)
}

/// # Safety
/// `allocation` must be null or a live allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_allocation_size(allocation: *const ResinAllocation) -> usize {
    unsafe { allocation.as_ref() }
        .map(ResinAllocation::size)
        .unwrap_or(0)
}

/// # Safety
/// `gpu` and `out_device_pointer` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_host_to_device_pointer(
    gpu: *const ResinGpu,
    host_pointer: *const c_void,
    out_device_pointer: *mut ResinDeviceAddress,
) -> ResinStatus {
    if gpu.is_null() || out_device_pointer.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_device_pointer = 0 };
    match unsafe { (&*gpu).host_to_device(host_pointer.cast()) } {
        Ok(address) => {
            unsafe { *out_device_pointer = address };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// `spv_bytes` must point to `spv_length` bytes of SPIR-V, or be null when the
/// length is zero. `gpu` and `out_pipeline` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_create_compute_pipeline(
    gpu: *mut ResinGpu,
    spv_bytes: *const c_void,
    spv_length: usize,
    out_pipeline: *mut *mut ResinPipeline,
) -> ResinStatus {
    if gpu.is_null() || out_pipeline.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_pipeline = ptr::null_mut() };
    if spv_length == 0 || spv_bytes.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let spv = unsafe { std::slice::from_raw_parts(spv_bytes.cast::<u8>(), spv_length) };
    match unsafe { (&*gpu).create_compute_pipeline(spv) } {
        Ok(pipeline) => {
            unsafe { *out_pipeline = Box::into_raw(Box::new(pipeline)) };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// SPIR-V arguments must point to the given byte lengths. `gpu` and
/// `out_pipeline` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_create_graphics_pipeline(
    gpu: *mut ResinGpu,
    vertex_spv_bytes: *const c_void,
    vertex_spv_length: usize,
    fragment_spv_bytes: *const c_void,
    fragment_spv_length: usize,
    out_pipeline: *mut *mut ResinPipeline,
) -> ResinStatus {
    if gpu.is_null() || out_pipeline.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_pipeline = ptr::null_mut() };
    if vertex_spv_length == 0
        || vertex_spv_bytes.is_null()
        || fragment_spv_length == 0
        || fragment_spv_bytes.is_null()
    {
        return ResinStatus::InvalidArgument;
    }
    let vertex =
        unsafe { std::slice::from_raw_parts(vertex_spv_bytes.cast::<u8>(), vertex_spv_length) };
    let fragment =
        unsafe { std::slice::from_raw_parts(fragment_spv_bytes.cast::<u8>(), fragment_spv_length) };
    match unsafe { (&*gpu).create_graphics_pipeline(vertex, fragment) } {
        Ok(pipeline) => {
            unsafe { *out_pipeline = Box::into_raw(Box::new(pipeline)) };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// `gpu` and `out_image` must be valid; `gpu` must outlive the image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_create_image(
    gpu: *mut ResinGpu,
    width: u32,
    height: u32,
    out_image: *mut *mut ResinImage,
) -> ResinStatus {
    if gpu.is_null() || out_image.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_image = ptr::null_mut() };
    match unsafe { (&*gpu).create_image(width, height) } {
        Ok(image) => {
            unsafe { *out_image = Box::into_raw(Box::new(image)) };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// `image` must be null or a pointer from [`resin_gpu_create_image`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_free_image(gpu: *mut ResinGpu, image: *mut ResinImage) {
    let _ = gpu;
    if !image.is_null() {
        drop(unsafe { Box::from_raw(image) });
    }
}

/// # Safety
/// `image` must be null or a live image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_image_width(image: *const ResinImage) -> u32 {
    unsafe { image.as_ref() }
        .map(ResinImage::width)
        .unwrap_or(0)
}

/// # Safety
/// `image` must be null or a live image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_image_height(image: *const ResinImage) -> u32 {
    unsafe { image.as_ref() }
        .map(ResinImage::height)
        .unwrap_or(0)
}

/// # Safety
/// `pipeline` must be null or a pointer from a pipeline create function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_free_pipeline(gpu: *mut ResinGpu, pipeline: *mut ResinPipeline) {
    let _ = gpu;
    if !pipeline.is_null() {
        drop(unsafe { Box::from_raw(pipeline) });
    }
}

/// # Safety
/// `gpu` and `out_command_buffer` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_start_command_recording(
    gpu: *mut ResinGpu,
    out_command_buffer: *mut *mut ResinCommandBuffer,
) -> ResinStatus {
    if gpu.is_null() || out_command_buffer.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_command_buffer = ptr::null_mut() };
    match unsafe { (&*gpu).start_command_recording() } {
        Ok(command_buffer) => {
            unsafe { *out_command_buffer = Box::into_raw(Box::new(command_buffer)) };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// `command_buffer` must be recording; `pipeline` must be a live pipeline.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_set_pipeline(
    command_buffer: *mut ResinCommandBuffer,
    pipeline: *const ResinPipeline,
) -> ResinStatus {
    let Some(command_buffer) = (unsafe { command_buffer.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    let Some(pipeline) = (unsafe { pipeline.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { command_buffer.set_pipeline(pipeline) } {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// # Safety
/// `command_buffer` must be recording with a pipeline bound.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_dispatch(
    command_buffer: *mut ResinCommandBuffer,
    root_data: ResinDeviceAddress,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
) -> ResinStatus {
    let Some(command_buffer) = (unsafe { command_buffer.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { command_buffer.dispatch(root_data, group_count_x, group_count_y, group_count_z) }
    {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// # Safety
/// `command_buffer` must be recording; `color` must be a live image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_begin_rendering(
    command_buffer: *mut ResinCommandBuffer,
    color: *mut ResinImage,
    clear_r: f32,
    clear_g: f32,
    clear_b: f32,
    clear_a: f32,
) -> ResinStatus {
    let Some(command_buffer) = (unsafe { command_buffer.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    let Some(color) = (unsafe { color.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { command_buffer.begin_rendering(color, [clear_r, clear_g, clear_b, clear_a]) } {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// # Safety
/// `command_buffer` must be inside a rendering scope.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_end_rendering(
    command_buffer: *mut ResinCommandBuffer,
) -> ResinStatus {
    let Some(command_buffer) = (unsafe { command_buffer.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { command_buffer.end_rendering() } {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// # Safety
/// `command_buffer` must be rendering with a graphics pipeline bound.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_draw(
    command_buffer: *mut ResinCommandBuffer,
    root_data: ResinDeviceAddress,
    vertex_count: u32,
) -> ResinStatus {
    let Some(command_buffer) = (unsafe { command_buffer.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { command_buffer.draw(root_data, vertex_count) } {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// # Safety
/// `command_buffer` must be recording and not rendering. `dst` must be large
/// enough for `width * height * 4` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_copy_image_to_buffer(
    command_buffer: *mut ResinCommandBuffer,
    image: *mut ResinImage,
    dst: *const ResinAllocation,
) -> ResinStatus {
    let Some(command_buffer) = (unsafe { command_buffer.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    let Some(image) = (unsafe { image.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    let Some(dst) = (unsafe { dst.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { command_buffer.copy_image_to_buffer(image, dst) } {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// Submits, waits until this command buffer's work completes, then frees it.
///
/// # Safety
/// A non-null `command_buffer` is consumed even when the status is not success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_submit(
    gpu: *mut ResinGpu,
    command_buffer: *mut ResinCommandBuffer,
) -> ResinStatus {
    if command_buffer.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let command_buffer = unsafe { Box::from_raw(command_buffer) };
    let Some(gpu) = (unsafe { gpu.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { gpu.submit(*command_buffer) } {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// # Safety
/// `command_buffer` is consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_cancel_command_buffer(
    gpu: *mut ResinGpu,
    command_buffer: *mut ResinCommandBuffer,
) {
    let _ = gpu;
    if !command_buffer.is_null() {
        drop(unsafe { Box::from_raw(command_buffer) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn resin_status_string(status: i32) -> *const c_char {
    match ResinStatus::from_raw(status) {
        Some(ResinStatus::Success) => c"success".as_ptr(),
        Some(ResinStatus::InvalidArgument) => c"invalid argument".as_ptr(),
        Some(ResinStatus::VulkanUnavailable) => c"vulkan unavailable".as_ptr(),
        Some(ResinStatus::Unsupported) => c"unsupported".as_ptr(),
        Some(ResinStatus::OutOfMemory) => c"out of memory".as_ptr(),
        Some(ResinStatus::VulkanError) => c"vulkan error".as_ptr(),
        Some(ResinStatus::IoError) => c"io error".as_ptr(),
        Some(ResinStatus::Incomplete) => c"incomplete".as_ptr(),
        None => c"unknown".as_ptr(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{resin_image_free, resin_image_read_png, resin_image_write_png};
    use std::io::Write;
    use std::process::{Command, Stdio};
    use std::slice;

    struct Gpu {
        ptr: *mut ResinGpu,
        _lock: crate::testing::GpuLock,
    }

    impl Gpu {
        fn create() -> Option<Self> {
            let lock = crate::testing::lock_gpu();
            let mut gpu = ptr::null_mut();
            let status = unsafe { resin_gpu_create(&mut gpu) };
            match status {
                ResinStatus::Success => Some(Self {
                    ptr: gpu,
                    _lock: lock,
                }),
                ResinStatus::VulkanUnavailable | ResinStatus::Unsupported => None,
                other => panic!("resin_gpu_create failed: {other:?}"),
            }
        }
    }

    impl Drop for Gpu {
        fn drop(&mut self) {
            unsafe { resin_gpu_destroy(self.ptr) };
        }
    }

    fn require_gpu() -> Option<Gpu> {
        let Some(gpu) = Gpu::create() else {
            eprintln!("skipping: no suitable Vulkan device");
            return None;
        };
        Some(gpu)
    }

    #[test]
    fn status_strings_are_stable() {
        unsafe {
            assert_eq!(
                std::ffi::CStr::from_ptr(resin_status_string(ResinStatus::Success as i32)).to_str(),
                Ok("success")
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(resin_status_string(ResinStatus::InvalidArgument as i32))
                    .to_str(),
                Ok("invalid argument")
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(resin_status_string(ResinStatus::IoError as i32)).to_str(),
                Ok("io error")
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(resin_status_string(ResinStatus::Incomplete as i32))
                    .to_str(),
                Ok("incomplete")
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(resin_status_string(99)).to_str(),
                Ok("unknown")
            );
        }
    }

    #[test]
    fn create_rejects_null_out() {
        assert_eq!(
            unsafe { resin_gpu_create(ptr::null_mut()) },
            ResinStatus::InvalidArgument
        );
        assert_eq!(
            unsafe { resin_gpu_create_at(0, ptr::null_mut()) },
            ResinStatus::InvalidArgument
        );
        assert_eq!(
            unsafe { resin_gpu_device_count(ptr::null_mut()) },
            ResinStatus::InvalidArgument
        );
        assert_eq!(
            unsafe { resin_gpu_enumerate_devices(ptr::null_mut(), 1) },
            ResinStatus::InvalidArgument
        );
    }

    #[test]
    fn device_enumeration_and_create_at() {
        let _lock = crate::testing::lock_gpu();
        let mut count = 0;
        match unsafe { resin_gpu_device_count(&mut count) } {
            ResinStatus::Success => {}
            ResinStatus::VulkanUnavailable => {
                eprintln!("skipping: no Vulkan");
                return;
            }
            other => panic!("resin_gpu_device_count failed: {other:?}"),
        }

        let mut gpu = ptr::null_mut();
        assert_eq!(
            unsafe { resin_gpu_create_at(count, &mut gpu) },
            ResinStatus::InvalidArgument
        );
        assert!(gpu.is_null());

        if count == 0 {
            return;
        }

        let mut infos = vec![
            ResinGpuDeviceInfo {
                index: 0,
                kind: ResinGpuDeviceType::Other,
                vendor_id: 0,
                device_id: 0,
                api_version: 0,
                driver_version: 0,
                suitable: 0,
                reserved: 0,
                device_local_bytes: 0,
                host_visible_device_local_bytes: 0,
                max_buffer_size: 0,
                name: [0; GPU_DEVICE_NAME_MAX],
            };
            count as usize
        ];
        assert_eq!(
            unsafe { resin_gpu_enumerate_devices(infos.as_mut_ptr(), count) },
            ResinStatus::Success
        );
        assert_eq!(
            unsafe { resin_gpu_enumerate_devices(infos.as_mut_ptr(), 0) },
            ResinStatus::Incomplete
        );

        let mut chosen = None;
        for (i, info) in infos.iter().enumerate() {
            assert_eq!(info.index, i as u32);
            let name = unsafe { std::ffi::CStr::from_ptr(info.name.as_ptr()) };
            assert!(!name.to_bytes().is_empty(), "device {i} has empty name");
            if info.suitable != 0 {
                assert!(
                    info.device_local_bytes > 0,
                    "suitable device reports no VRAM"
                );
                chosen = Some(info.index);
            }
        }

        let Some(index) = chosen else {
            eprintln!("skipping: no suitable Vulkan device");
            return;
        };
        assert_eq!(
            unsafe { resin_gpu_create_at(index, &mut gpu) },
            ResinStatus::Success
        );
        assert!(!gpu.is_null());
        unsafe { resin_gpu_destroy(gpu) };
    }

    #[test]
    fn destroy_null_is_a_noop() {
        unsafe { resin_gpu_destroy(ptr::null_mut()) };
    }

    #[test]
    fn malloc_rejects_bad_arguments() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let mut allocation = ptr::null_mut();
        assert_eq!(
            unsafe {
                resin_gpu_malloc(gpu.ptr, 0, 16, ResinMemory::Default as i32, &mut allocation)
            },
            ResinStatus::InvalidArgument
        );
        assert_eq!(
            unsafe {
                resin_gpu_malloc(gpu.ptr, 64, 3, ResinMemory::Default as i32, &mut allocation)
            },
            ResinStatus::InvalidArgument
        );
        assert_eq!(
            unsafe { resin_gpu_malloc(gpu.ptr, 64, 16, 99, &mut allocation) },
            ResinStatus::InvalidArgument
        );
        assert!(allocation.is_null());
    }

    #[test]
    fn cancel_command_buffer_without_submit() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let mut command_buffer = ptr::null_mut();
        assert_eq!(
            unsafe { resin_gpu_start_command_recording(gpu.ptr, &mut command_buffer) },
            ResinStatus::Success
        );
        assert_eq!(
            unsafe { resin_gpu_dispatch(command_buffer, 0, 1, 1, 1) },
            ResinStatus::InvalidArgument
        );
        unsafe { resin_gpu_cancel_command_buffer(gpu.ptr, command_buffer) };
    }

    #[test]
    fn malloc_and_pointer_translation() {
        let Some(gpu) = require_gpu() else {
            return;
        };

        let mut allocation = ptr::null_mut();
        let status = unsafe {
            resin_gpu_malloc(
                gpu.ptr,
                1024,
                16,
                ResinMemory::Default as i32,
                &mut allocation,
            )
        };
        assert_eq!(status, ResinStatus::Success);
        assert!(!allocation.is_null());

        let host = unsafe { resin_allocation_host_pointer(allocation) };
        assert!(!host.is_null());
        assert_eq!(unsafe { resin_allocation_size(allocation) }, 1024);

        let base_device = unsafe { resin_allocation_device_pointer(allocation) };
        assert_ne!(base_device, 0);

        let mut translated = 0;
        let status = unsafe { resin_gpu_host_to_device_pointer(gpu.ptr, host, &mut translated) };
        assert_eq!(status, ResinStatus::Success);
        assert_eq!(translated, base_device);

        let interior = unsafe { host.byte_add(128) };
        let status =
            unsafe { resin_gpu_host_to_device_pointer(gpu.ptr, interior, &mut translated) };
        assert_eq!(status, ResinStatus::Success);
        assert_eq!(translated, base_device + 128);

        let status = unsafe {
            resin_gpu_host_to_device_pointer(
                gpu.ptr,
                ptr::dangling::<c_void>().cast(),
                &mut translated,
            )
        };
        assert_eq!(status, ResinStatus::InvalidArgument);

        unsafe { resin_gpu_free(gpu.ptr, allocation) };

        let mut aligned = ptr::null_mut();
        let status =
            unsafe { resin_gpu_malloc(gpu.ptr, 8, 256, ResinMemory::Default as i32, &mut aligned) };
        assert_eq!(status, ResinStatus::Success);
        let aligned_host = unsafe { resin_allocation_host_pointer(aligned) };
        let aligned_device = unsafe { resin_allocation_device_pointer(aligned) };
        assert_eq!(aligned_device % 256, 0);
        let mut translated = 0;
        assert_eq!(
            unsafe { resin_gpu_host_to_device_pointer(gpu.ptr, aligned_host, &mut translated) },
            ResinStatus::Success
        );
        assert_eq!(translated, aligned_device);
        unsafe { resin_gpu_free(gpu.ptr, aligned) };

        let mut reused = ptr::null_mut();
        let status =
            unsafe { resin_gpu_malloc(gpu.ptr, 8, 256, ResinMemory::Default as i32, &mut reused) };
        assert_eq!(status, ResinStatus::Success);
        assert_eq!(unsafe { resin_allocation_device_pointer(reused) } % 256, 0);
        unsafe { resin_gpu_free(gpu.ptr, reused) };

        let mut gpu_only = ptr::null_mut();
        let status =
            unsafe { resin_gpu_malloc(gpu.ptr, 256, 256, ResinMemory::Gpu as i32, &mut gpu_only) };
        assert_eq!(status, ResinStatus::Success);
        assert!(unsafe { resin_allocation_host_pointer(gpu_only) }.is_null());
        let gpu_only_device = unsafe { resin_allocation_device_pointer(gpu_only) };
        assert_ne!(gpu_only_device, 0);
        assert_eq!(gpu_only_device % 256, 0);
        unsafe { resin_gpu_free(gpu.ptr, gpu_only) };

        const SLAB: usize = 16 * 1024 * 1024;
        let mut slab = ptr::null_mut();
        let status = unsafe {
            resin_gpu_malloc(gpu.ptr, SLAB, 4096, ResinMemory::Default as i32, &mut slab)
        };
        assert_eq!(status, ResinStatus::Success);
        assert_eq!(unsafe { resin_allocation_device_pointer(slab) } % 4096, 0);
        assert_eq!(unsafe { resin_allocation_size(slab) }, SLAB);
        unsafe { resin_gpu_free(gpu.ptr, slab) };
    }

    #[test]
    fn compute_dispatch_writes_through_root_pointer() {
        let Some(gpu) = require_gpu() else {
            return;
        };

        let Some(spv) = compile_compute(
            r#"
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 64) in;

layout(buffer_reference, std430, buffer_reference_align = 8) buffer Root {
    uint count;
    uint _pad;
    uint64_t dst;
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32Array {
    uint values[];
};

layout(push_constant) uniform Push {
    uint64_t root;
};

void main() {
    Root r = Root(root);
    uint i = gl_GlobalInvocationID.x;
    if (i >= r.count) return;
    U32Array dst = U32Array(r.dst);
    dst.values[i] = i + 1u;
}
"#,
        ) else {
            return;
        };

        let mut pipeline = ptr::null_mut();
        let status = unsafe {
            resin_gpu_create_compute_pipeline(
                gpu.ptr,
                spv.as_ptr().cast(),
                spv.len(),
                &mut pipeline,
            )
        };
        assert_eq!(status, ResinStatus::Success);

        const COUNT: u32 = 256;
        let mut values = ptr::null_mut();
        let status = unsafe {
            resin_gpu_malloc(
                gpu.ptr,
                (COUNT as usize) * 4,
                4,
                ResinMemory::Default as i32,
                &mut values,
            )
        };
        assert_eq!(status, ResinStatus::Success);

        let mut root = ptr::null_mut();
        let status =
            unsafe { resin_gpu_malloc(gpu.ptr, 16, 8, ResinMemory::Default as i32, &mut root) };
        assert_eq!(status, ResinStatus::Success);

        let values_host = unsafe { resin_allocation_host_pointer(values) }.cast::<u32>();
        unsafe { std::ptr::write_bytes(values_host, 0, COUNT as usize) };

        let mut dst_device = 0;
        let status = unsafe {
            resin_gpu_host_to_device_pointer(gpu.ptr, values_host.cast(), &mut dst_device)
        };
        assert_eq!(status, ResinStatus::Success);

        #[repr(C)]
        struct Root {
            count: u32,
            pad: u32,
            dst: u64,
        }
        let root_host = unsafe { resin_allocation_host_pointer(root) }.cast::<Root>();
        unsafe {
            root_host.write(Root {
                count: COUNT,
                pad: 0,
                dst: dst_device,
            });
        }

        let mut root_device = 0;
        let status = unsafe {
            resin_gpu_host_to_device_pointer(gpu.ptr, root_host.cast(), &mut root_device)
        };
        assert_eq!(status, ResinStatus::Success);

        let mut command_buffer = ptr::null_mut();
        assert_eq!(
            unsafe { resin_gpu_start_command_recording(gpu.ptr, &mut command_buffer) },
            ResinStatus::Success
        );
        assert_eq!(
            unsafe { resin_gpu_set_pipeline(command_buffer, pipeline) },
            ResinStatus::Success
        );
        assert_eq!(
            unsafe { resin_gpu_dispatch(command_buffer, root_device, COUNT.div_ceil(64), 1, 1) },
            ResinStatus::Success
        );
        assert_eq!(
            unsafe { resin_gpu_submit(gpu.ptr, command_buffer) },
            ResinStatus::Success
        );

        for i in 0..COUNT {
            assert_eq!(unsafe { *values_host.add(i as usize) }, i + 1, "index {i}");
        }

        unsafe {
            resin_gpu_free(gpu.ptr, root);
            resin_gpu_free(gpu.ptr, values);
            resin_gpu_free_pipeline(gpu.ptr, pipeline);
        }
    }

    #[test]
    fn png_roundtrip_and_channel_convert() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("resin-png-roundtrip-{}.png", std::process::id()));
        let path_c = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

        let width = 2u32;
        let height = 2u32;
        let pixels: [u8; 16] = [
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 10, 20, 30, 40,
        ];
        assert_eq!(
            unsafe {
                resin_image_write_png(path_c.as_ptr(), width, height, 4, pixels.as_ptr().cast(), 0)
            },
            ResinStatus::Success
        );

        let mut out_w = 0;
        let mut out_h = 0;
        let mut out_ch = 0;
        let mut out_pixels = ptr::null_mut();
        assert_eq!(
            unsafe {
                resin_image_read_png(
                    path_c.as_ptr(),
                    0,
                    &mut out_w,
                    &mut out_h,
                    &mut out_ch,
                    &mut out_pixels,
                )
            },
            ResinStatus::Success
        );
        assert_eq!((out_w, out_h, out_ch), (2, 2, 4));
        let decoded = unsafe { slice::from_raw_parts(out_pixels.cast::<u8>(), 16) };
        assert_eq!(decoded, pixels);
        unsafe { resin_image_free(out_pixels) };

        out_pixels = ptr::null_mut();
        assert_eq!(
            unsafe {
                resin_image_read_png(
                    path_c.as_ptr(),
                    3,
                    &mut out_w,
                    &mut out_h,
                    &mut out_ch,
                    &mut out_pixels,
                )
            },
            ResinStatus::Success
        );
        assert_eq!(out_ch, 3);
        let rgb = unsafe { slice::from_raw_parts(out_pixels.cast::<u8>(), 12) };
        assert_eq!(&rgb[..3], &[255, 0, 0]);
        unsafe { resin_image_free(out_pixels) };

        let _ = std::fs::remove_file(&path);
        assert_eq!(
            unsafe {
                resin_image_read_png(
                    path_c.as_ptr(),
                    0,
                    &mut out_w,
                    &mut out_h,
                    &mut out_ch,
                    &mut out_pixels,
                )
            },
            ResinStatus::IoError
        );
        assert!(out_pixels.is_null());
        assert_eq!(
            unsafe { resin_image_write_png(path_c.as_ptr(), 0, 1, 4, pixels.as_ptr().cast(), 0) },
            ResinStatus::InvalidArgument
        );
        unsafe { resin_image_free(ptr::null_mut()) };
    }

    #[test]
    fn pipeline_rejects_non_spirv() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let mut pipeline = ptr::null_mut();
        let bytes = [0u8; 8];
        let status = unsafe {
            resin_gpu_create_compute_pipeline(
                gpu.ptr,
                bytes.as_ptr().cast(),
                bytes.len(),
                &mut pipeline,
            )
        };
        assert_eq!(status, ResinStatus::InvalidArgument);
        assert!(pipeline.is_null());
    }

    fn compile_compute(src: &str) -> Option<Vec<u8>> {
        let mut child = match Command::new("glslc")
            .args([
                "-fshader-stage=comp",
                "--target-env=vulkan1.2",
                "-O",
                "-o",
                "-",
                "-",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("skipping: glslc not found");
                return None;
            }
            Err(err) => panic!("failed to spawn glslc: {err}"),
        };
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(src.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "glslc failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Some(output.stdout)
    }
}
