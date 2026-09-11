//! Owning GPU views and the checks used by compiler-generated host accesses.

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::shared::{ResinArc, resin_arc_data, resin_arc_new, resin_arc_release, resin_arc_retain};
use crate::{
    RESIN_GPU_ACCESS_READ, RESIN_GPU_ACCESS_WRITE, ResinAllocation, ResinCommandBuffer, ResinGpu,
    ResinGpuPtr, ResinGpuSpan, ResinImage, ResinMemory, ResinStatus,
};

struct AllocationOwner {
    gpu: *mut ResinGpu,
    gpu_owner: *mut ResinArc,
    allocation: ResinAllocation,
    // Logical size differs from physical storage only for empty allocations.
    bytes: usize,
    gpu_uses: AtomicUsize,
}

struct ProjectionOwner {
    root: ResinGpuPtr,
    dependencies: Vec<*mut ResinArc>,
}

/// One command's CPU-access exclusion. `owner` retains every allocation listed
/// here, either directly or through the immutable projection payload.
pub(crate) struct GpuUse {
    owner: *mut ResinArc,
    allocations: Vec<*mut ResinArc>,
}

impl GpuUse {
    unsafe fn new(owner: *mut ResinArc, allocations: Vec<*mut ResinArc>) -> Self {
        unsafe { resin_arc_retain(owner) };
        for &allocation in &allocations {
            let payload = unsafe { &*resin_arc_data(allocation).cast::<AllocationOwner>() };
            if payload.gpu_uses.fetch_add(1, Ordering::AcqRel) == usize::MAX {
                crate::host::fail("GPU allocation use count overflow");
            }
        }
        Self { owner, allocations }
    }

    unsafe fn projection(owner: *mut ResinArc, projection: &ProjectionOwner) -> Self {
        let allocations = std::iter::once(projection.root.owner)
            .chain(projection.dependencies.iter().copied())
            .collect();
        unsafe { Self::new(owner, allocations) }
    }
}

impl Drop for GpuUse {
    fn drop(&mut self) {
        for &allocation in &self.allocations {
            let payload = unsafe { &*resin_arc_data(allocation).cast::<AllocationOwner>() };
            payload.gpu_uses.fetch_sub(1, Ordering::Release);
        }
        unsafe { resin_arc_release(self.owner) };
    }
}

impl Drop for ProjectionOwner {
    fn drop(&mut self) {
        unsafe { resin_arc_release(self.root.owner) };
        for &owner in &self.dependencies {
            unsafe { resin_arc_release(owner) };
        }
    }
}

unsafe extern "C" fn destroy_allocation(payload: *mut c_void) {
    let owner = unsafe { &mut *payload.cast::<AllocationOwner>() };
    unsafe {
        (*owner.gpu).free(&owner.allocation);
        resin_arc_release(owner.gpu_owner);
        ptr::drop_in_place(owner);
    }
}

pub(crate) unsafe fn allocate(
    gpu: *mut ResinGpu,
    gpu_owner: *mut ResinArc,
    bytes: usize,
    alignment: usize,
    memory: i32,
    out: *mut ResinGpuPtr,
) -> ResinStatus {
    if out.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe {
        *out = ResinGpuPtr {
            owner: ptr::null_mut(),
            offset: 0,
            access: 0,
        };
    }
    if gpu.is_null() || gpu_owner.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let Some(memory) = ResinMemory::from_raw(memory) else {
        return ResinStatus::InvalidArgument;
    };
    let allocation = match unsafe { (*gpu).malloc(bytes.max(1), alignment, memory) } {
        Ok(allocation) => allocation,
        Err(status) => return status,
    };
    let owner = resin_arc_new(
        size_of::<AllocationOwner>(),
        align_of::<AllocationOwner>(),
        destroy_allocation,
    );
    unsafe {
        resin_arc_retain(gpu_owner);
        resin_arc_data(owner)
            .cast::<AllocationOwner>()
            .write(AllocationOwner {
                gpu,
                gpu_owner,
                allocation,
                bytes,
                gpu_uses: AtomicUsize::new(0),
            });
        *out = ResinGpuPtr {
            owner,
            offset: 0,
            access: RESIN_GPU_ACCESS_READ | RESIN_GPU_ACCESS_WRITE,
        };
    }
    ResinStatus::Success
}

unsafe fn allocation_owner<'a>(value: ResinGpuPtr) -> Result<&'a AllocationOwner, &'static str> {
    if value.owner.is_null() {
        return Err("access through an empty GpuPtr");
    }
    Ok(unsafe { &*resin_arc_data(value.owner).cast::<AllocationOwner>() })
}

fn check_range(offset: usize, bytes: usize, length: usize) -> Result<(), &'static str> {
    if offset > length || bytes > length - offset {
        return Err("GPU pointer access out of bounds");
    }
    Ok(())
}

fn check_alignment(address: usize, alignment: usize) -> Result<(), &'static str> {
    if !alignment.is_power_of_two() || !address.is_multiple_of(alignment) {
        return Err("misaligned GPU pointer access");
    }
    Ok(())
}

fn check_access(access: u32, required: u32) -> Result<(), &'static str> {
    let valid = RESIN_GPU_ACCESS_READ | RESIN_GPU_ACCESS_WRITE;
    if required == 0 || required & !valid != 0 || access & required != required {
        return Err("GPU pointer access permission denied");
    }
    Ok(())
}

unsafe fn checked_host(
    value: ResinGpuPtr,
    bytes: usize,
    alignment: usize,
    required_access: u32,
) -> Result<*mut c_void, &'static str> {
    let owner = unsafe { allocation_owner(value) }?;
    check_range(value.offset, bytes, owner.bytes)?;
    check_access(value.access, required_access)?;
    if owner.gpu_uses.load(Ordering::Acquire) != 0 {
        return Err("CPU access to a GPU allocation retained by a recording");
    }
    let host = owner.allocation.host_pointer();
    if host.is_null() {
        return Err("GPU allocation is not mapped for CPU access");
    }
    let address = unsafe { host.add(value.offset) };
    check_alignment(address as usize, alignment)?;
    Ok(address.cast())
}

pub(crate) unsafe fn host(
    value: ResinGpuPtr,
    bytes: usize,
    alignment: usize,
    required_access: u32,
) -> *mut c_void {
    unsafe { checked_host(value, bytes, alignment, required_access) }
        .unwrap_or_else(|message| crate::host::fail(message))
}

unsafe fn checked_offset(
    value: ResinGpuPtr,
    byte_offset: usize,
    bytes: usize,
    alignment: usize,
) -> Result<ResinGpuPtr, &'static str> {
    let offset = value
        .offset
        .checked_add(byte_offset)
        .ok_or("GPU pointer offset overflow")?;
    let owner = unsafe { allocation_owner(value) }?;
    check_range(offset, bytes, owner.bytes)?;
    let address = owner
        .allocation
        .device_pointer()
        .checked_add(offset as u64)
        .ok_or("GPU pointer offset overflow")?;
    check_alignment(address as usize, alignment)?;
    Ok(ResinGpuPtr { offset, ..value })
}

pub(crate) unsafe fn offset(
    value: ResinGpuPtr,
    byte_offset: usize,
    bytes: usize,
    alignment: usize,
) -> ResinGpuPtr {
    unsafe { checked_offset(value, byte_offset, bytes, alignment) }
        .unwrap_or_else(|message| crate::host::fail(message))
}

unsafe extern "C" fn destroy_projection(payload: *mut c_void) {
    unsafe { ptr::drop_in_place(payload.cast::<ProjectionOwner>()) };
}

pub(crate) unsafe fn projection_new(root: ResinGpuPtr) -> *mut ResinArc {
    let allocation =
        unsafe { allocation_owner(root) }.unwrap_or_else(|message| crate::host::fail(message));
    let bytes = allocation
        .bytes
        .checked_sub(root.offset)
        .unwrap_or_else(|| crate::host::fail("GPU projection root out of bounds"));
    unsafe {
        host(
            root,
            bytes,
            1,
            RESIN_GPU_ACCESS_READ | RESIN_GPU_ACCESS_WRITE,
        )
    };
    let owner = resin_arc_new(
        size_of::<ProjectionOwner>(),
        align_of::<ProjectionOwner>(),
        destroy_projection,
    );
    unsafe {
        resin_arc_retain(root.owner);
        resin_arc_data(owner)
            .cast::<ProjectionOwner>()
            .write(ProjectionOwner {
                root,
                dependencies: Vec::new(),
            });
    }
    owner
}

unsafe fn projection_owner<'a>(owner: *mut ResinArc) -> &'a mut ProjectionOwner {
    if owner.is_null() {
        crate::host::fail("access through empty GPU arguments");
    }
    unsafe { &mut *resin_arc_data(owner).cast::<ProjectionOwner>() }
}

pub(crate) unsafe fn projection_root(projection: *mut ResinArc) -> *mut c_void {
    let root = unsafe { projection_owner(projection) }.root;
    let allocation =
        unsafe { allocation_owner(root) }.unwrap_or_else(|message| crate::host::fail(message));
    unsafe {
        host(
            root,
            allocation.bytes - root.offset,
            1,
            RESIN_GPU_ACCESS_WRITE,
        )
    }
}

unsafe fn checked_device_pointer(
    value: ResinGpuPtr,
    bytes: usize,
    alignment: usize,
    gpu: *mut ResinGpu,
) -> Result<u64, &'static str> {
    let owner = unsafe { allocation_owner(value) }?;
    if owner.gpu != gpu {
        return Err("GPU pointer belongs to a different device");
    }
    check_range(value.offset, bytes, owner.bytes)?;
    // Ordinary shader Ptr permits both reads and writes. Narrowed host views
    // cannot be projected until shader access qualifiers exist.
    check_access(value.access, RESIN_GPU_ACCESS_READ | RESIN_GPU_ACCESS_WRITE)?;
    let address = owner
        .allocation
        .device_pointer()
        .checked_add(value.offset as u64)
        .ok_or("GPU pointer offset overflow")?;
    check_alignment(address as usize, alignment)?;
    Ok(address)
}

unsafe fn checked_projection_pointer(
    projection: &mut ProjectionOwner,
    value: ResinGpuPtr,
    bytes: usize,
    alignment: usize,
) -> Result<u64, &'static str> {
    let gpu = unsafe { allocation_owner(projection.root) }?.gpu;
    let address = unsafe { checked_device_pointer(value, bytes, alignment, gpu) }?;
    if value.owner != projection.root.owner && !projection.dependencies.contains(&value.owner) {
        unsafe { resin_arc_retain(value.owner) };
        projection.dependencies.push(value.owner);
    }
    Ok(address)
}

pub(crate) unsafe fn projection_pointer(
    projection: *mut ResinArc,
    value: ResinGpuPtr,
    bytes: usize,
    alignment: usize,
) -> u64 {
    unsafe { checked_projection_pointer(projection_owner(projection), value, bytes, alignment) }
        .unwrap_or_else(|message| crate::host::fail(message))
}

unsafe fn projected_root(
    commands: &ResinCommandBuffer,
    projection: &ProjectionOwner,
) -> Result<u64, ResinStatus> {
    let owner =
        unsafe { allocation_owner(projection.root) }.map_err(|_| ResinStatus::InvalidArgument)?;
    if !commands.belongs_to_gpu(unsafe { &*owner.gpu }) {
        return Err(ResinStatus::InvalidArgument);
    }
    Ok(owner.allocation.device_pointer() + projection.root.offset as u64)
}

pub(crate) unsafe fn projected_dispatch(
    commands: *mut ResinCommandBuffer,
    projection: *mut ResinArc,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
) -> ResinStatus {
    let Some(commands) = (unsafe { commands.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    if projection.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let value = unsafe { projection_owner(projection) };
    let result = unsafe { projected_root(commands, value) }.and_then(|root| unsafe {
        commands.dispatch(root, group_count_x, group_count_y, group_count_z)
    });
    if let Err(status) = result {
        return status;
    }
    commands.retain_gpu_use(unsafe { GpuUse::projection(projection, value) });
    ResinStatus::Success
}

pub(crate) unsafe fn projected_draw(
    commands: *mut ResinCommandBuffer,
    projection: *mut ResinArc,
    vertex_count: u32,
) -> ResinStatus {
    let Some(commands) = (unsafe { commands.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    if projection.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let value = unsafe { projection_owner(projection) };
    let result = unsafe { projected_root(commands, value) }
        .and_then(|root| unsafe { commands.draw(root, vertex_count) });
    if let Err(status) = result {
        return status;
    }
    commands.retain_gpu_use(unsafe { GpuUse::projection(projection, value) });
    ResinStatus::Success
}

pub(crate) unsafe fn copy_image_to_span(
    commands: *mut ResinCommandBuffer,
    image: *mut ResinImage,
    destination: ResinGpuSpan,
) -> ResinStatus {
    let (Some(commands), Some(image)) = (unsafe { commands.as_mut() }, unsafe { image.as_mut() })
    else {
        return ResinStatus::InvalidArgument;
    };
    let owner = match unsafe { allocation_owner(destination.data) } {
        Ok(owner) => owner,
        Err(_) => return ResinStatus::InvalidArgument,
    };
    let bytes = (image.width() as usize)
        .checked_mul(image.height() as usize)
        .and_then(|pixels| pixels.checked_mul(4));
    if !commands.belongs_to_gpu(unsafe { &*owner.gpu })
        || !image.belongs_to_gpu(unsafe { &*owner.gpu })
        || bytes.is_none_or(|bytes| bytes > destination.length)
        || check_range(destination.data.offset, destination.length, owner.bytes).is_err()
        || check_access(destination.data.access, RESIN_GPU_ACCESS_WRITE).is_err()
    {
        return ResinStatus::InvalidArgument;
    }
    if let Err(status) = unsafe {
        commands.copy_image_to_buffer_offset(image, &owner.allocation, destination.data.offset)
    } {
        return status;
    }
    commands.retain_gpu_use(unsafe {
        GpuUse::new(destination.data.owner, vec![destination.data.owner])
    });
    ResinStatus::Success
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResinGpuSpan;
    use crate::shared::{resin_weak_release, resin_weak_retain, resin_weak_upgrade};

    struct TestGpu {
        gpu: *mut ResinGpu,
        owner: *mut ResinArc,
        _lock: crate::testing::GpuLock,
    }

    impl TestGpu {
        fn new() -> Option<Self> {
            let lock = crate::testing::lock_gpu();
            let gpu = match ResinGpu::create() {
                Ok(gpu) => Box::into_raw(Box::new(gpu)),
                Err(ResinStatus::VulkanUnavailable | ResinStatus::Unsupported) => {
                    assert!(
                        !matches!(
                            std::env::var("RESIN_REQUIRE_GPU").as_deref(),
                            Ok("1" | "true")
                        ),
                        "RESIN_REQUIRE_GPU is set but no suitable Vulkan device is available"
                    );
                    return None;
                }
                Err(status) => panic!("GPU creation failed: {status:?}"),
            };
            unsafe extern "C" fn destroy_gpu(payload: *mut c_void) {
                unsafe { drop(Box::from_raw(*payload.cast::<*mut ResinGpu>())) };
            }
            let owner = resin_arc_new(
                size_of::<*mut ResinGpu>(),
                align_of::<*mut ResinGpu>(),
                destroy_gpu,
            );
            unsafe { resin_arc_data(owner).cast::<*mut ResinGpu>().write(gpu) };
            Some(Self {
                gpu,
                owner,
                _lock: lock,
            })
        }

        unsafe fn allocate(&self, bytes: usize, memory: ResinMemory) -> ResinGpuPtr {
            let mut value = ResinGpuPtr {
                owner: ptr::null_mut(),
                offset: 0,
                access: 0,
            };
            assert_eq!(
                unsafe { allocate(self.gpu, self.owner, bytes, 8, memory as i32, &mut value) },
                ResinStatus::Success
            );
            value
        }
    }

    impl Drop for TestGpu {
        fn drop(&mut self) {
            unsafe { resin_arc_release(self.owner) };
        }
    }

    #[test]
    fn view_abi_preserves_owner_offset_and_access() {
        assert_eq!(size_of::<ResinGpuPtr>(), 24);
        assert_eq!(align_of::<ResinGpuPtr>(), 8);
        assert_eq!(std::mem::offset_of!(ResinGpuPtr, offset), 8);
        assert_eq!(std::mem::offset_of!(ResinGpuPtr, access), 16);
        assert_eq!(size_of::<ResinGpuSpan>(), 32);
        assert_eq!(std::mem::offset_of!(ResinGpuSpan, length), 24);
    }

    #[test]
    fn ranges_allow_empty_tails_and_reject_overflow() {
        assert!(check_range(16, 0, 16).is_ok());
        assert!(check_range(16, 1, 16).is_err());
        assert!(check_range(17, 0, 16).is_err());
        assert!(check_range(8, usize::MAX, 16).is_err());
        assert!(check_range(0, 0, 0).is_ok());
    }

    #[test]
    fn access_requires_every_requested_permission() {
        assert!(check_access(3, 1).is_ok());
        assert!(check_access(3, 2).is_ok());
        assert!(check_access(3, 3).is_ok());
        assert!(check_access(1, 2).is_err());
        assert!(check_access(2, 1).is_err());
        assert!(check_access(3, 0).is_err());
        assert!(check_access(3, 4).is_err());
    }

    #[test]
    fn alignment_rejects_invalid_requirements_and_addresses() {
        assert!(check_alignment(8, 8).is_ok());
        assert!(check_alignment(9, 8).is_err());
        assert!(check_alignment(8, 0).is_err());
        assert!(check_alignment(9, 3).is_err());
    }

    #[test]
    fn interior_views_retain_the_allocation_and_device() {
        let Some(mut gpu) = TestGpu::new() else {
            return;
        };
        unsafe {
            let value = gpu.allocate(16, ResinMemory::Default);
            let interior = checked_offset(value, 8, 8, 8).unwrap();
            assert_eq!(interior.owner, value.owner);
            resin_arc_retain(interior.owner);
            resin_arc_release(value.owner);
            checked_host(interior, 8, 8, RESIN_GPU_ACCESS_WRITE)
                .unwrap()
                .cast::<u64>()
                .write(42);
            let device_owner = gpu.owner;
            resin_weak_retain(device_owner);
            resin_arc_release(gpu.owner);
            gpu.owner = ptr::null_mut();
            let device = resin_weak_upgrade(device_owner);
            assert!(!device.is_null());
            resin_arc_release(device);
            assert_eq!(
                checked_host(interior, 8, 8, RESIN_GPU_ACCESS_READ)
                    .unwrap()
                    .cast::<u64>()
                    .read(),
                42
            );
            resin_arc_release(interior.owner);
            assert!(resin_weak_upgrade(device_owner).is_null());
            resin_weak_release(device_owner);
        }
    }

    #[test]
    fn view_permissions_and_recording_state_control_cpu_access() {
        let Some(gpu) = TestGpu::new() else { return };
        unsafe {
            let value = gpu.allocate(16, ResinMemory::Default);
            let readonly = ResinGpuPtr {
                access: RESIN_GPU_ACCESS_READ,
                ..value
            };
            let writeonly = ResinGpuPtr {
                access: RESIN_GPU_ACCESS_WRITE,
                ..value
            };
            assert!(checked_host(readonly, 8, 8, RESIN_GPU_ACCESS_READ).is_ok());
            assert!(checked_host(readonly, 8, 8, RESIN_GPU_ACCESS_WRITE).is_err());
            assert!(checked_host(writeonly, 8, 8, RESIN_GPU_ACCESS_READ).is_err());
            assert!(checked_host(writeonly, 8, 8, RESIN_GPU_ACCESS_WRITE).is_ok());
            let owner = allocation_owner(value).unwrap();
            owner.gpu_uses.store(1, Ordering::Release);
            assert!(checked_host(value, 8, 8, RESIN_GPU_ACCESS_READ).is_err());
            assert!(checked_host(value, 8, 8, RESIN_GPU_ACCESS_WRITE).is_err());
            owner.gpu_uses.store(0, Ordering::Release);
            assert!(checked_host(value, 8, 8, RESIN_GPU_ACCESS_WRITE).is_ok());
            resin_arc_release(value.owner);
        }
    }

    #[test]
    fn empty_and_unmapped_views_reject_host_payload_access() {
        let Some(gpu) = TestGpu::new() else { return };
        unsafe {
            let empty = gpu.allocate(0, ResinMemory::Default);
            assert!(checked_offset(empty, 0, 0, 8).is_ok());
            assert!(checked_host(empty, 1, 1, RESIN_GPU_ACCESS_READ).is_err());
            resin_arc_release(empty.owner);
            let unmapped = gpu.allocate(16, ResinMemory::Gpu);
            assert!(checked_offset(unmapped, 8, 8, 8).is_ok());
            assert!(checked_host(unmapped, 8, 8, RESIN_GPU_ACCESS_READ).is_err());
            resin_arc_release(unmapped.owner);
        }
    }

    #[test]
    fn projected_commands_retain_dependencies_and_unlock_after_completion_or_cancel() {
        let Some(gpu) = TestGpu::new() else { return };
        let Some(shader) = crate::tests::compile_compute(
            r#"
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Root { uint64_t destination; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Value { uint number; };
layout(push_constant) uniform Push { uint64_t root; };
void main() { Value(Root(root).destination).number += 1; }
"#,
        ) else {
            assert!(!matches!(
                std::env::var("RESIN_REQUIRE_GLSLC").as_deref(),
                Ok("1" | "true")
            ));
            return;
        };
        unsafe {
            let pipeline = (*gpu.gpu).create_compute_pipeline(&shader).unwrap();
            let allocation = gpu.allocate(16, ResinMemory::Default);
            let value = checked_offset(allocation, 8, 4, 4).unwrap();
            checked_host(value, 4, 4, RESIN_GPU_ACCESS_WRITE)
                .unwrap()
                .cast::<u32>()
                .write(0);
            let root = gpu.allocate(8, ResinMemory::Default);
            let projection = projection_new(root);
            let address = projection_pointer(projection, value, 4, 4);
            let base_address = checked_device_pointer(allocation, 16, 8, gpu.gpu).unwrap();
            assert_eq!(address, base_address + 8);
            assert_eq!(projection_pointer(projection, value, 4, 4), address);
            assert_eq!(projection_owner(projection).dependencies.len(), 1);
            projection_root(projection).cast::<u64>().write(address);
            resin_arc_release(root.owner);

            let mut cancelled = (*gpu.gpu).start_command_recording().unwrap();
            assert_eq!(
                projected_dispatch(&mut cancelled, projection, 1, 1, 1),
                ResinStatus::InvalidArgument
            );
            assert!(checked_host(value, 4, 4, RESIN_GPU_ACCESS_READ).is_ok());
            cancelled.set_pipeline(&pipeline).unwrap();
            assert_eq!(
                projected_dispatch(&mut cancelled, projection, 1, 1, 1),
                ResinStatus::Success
            );
            assert!(checked_host(value, 4, 4, RESIN_GPU_ACCESS_READ).is_err());
            drop(cancelled);
            assert_eq!(
                checked_host(value, 4, 4, RESIN_GPU_ACCESS_READ)
                    .unwrap()
                    .cast::<u32>()
                    .read(),
                0
            );

            let mut commands = (*gpu.gpu).start_command_recording().unwrap();
            commands.set_pipeline(&pipeline).unwrap();
            assert_eq!(
                projected_dispatch(&mut commands, projection, 1, 1, 1),
                ResinStatus::Success
            );
            assert_eq!(
                projected_dispatch(&mut commands, projection, 1, 1, 1),
                ResinStatus::Success
            );
            assert_eq!(
                allocation_owner(value)
                    .unwrap()
                    .gpu_uses
                    .load(Ordering::Acquire),
                2
            );
            resin_weak_retain(projection);
            resin_arc_release(projection);
            let retained = resin_weak_upgrade(projection);
            assert!(!retained.is_null());
            resin_arc_release(retained);
            (*gpu.gpu).submit(commands).unwrap();
            assert!(resin_weak_upgrade(projection).is_null());
            resin_weak_release(projection);
            assert_eq!(
                checked_host(value, 4, 4, RESIN_GPU_ACCESS_READ)
                    .unwrap()
                    .cast::<u32>()
                    .read(),
                2
            );
            resin_arc_release(allocation.owner);
        }
    }

    #[test]
    fn projection_checks_fail_without_retaining_invalid_dependencies() {
        let Some(gpu) = TestGpu::new() else { return };
        unsafe {
            let root = gpu.allocate(8, ResinMemory::Default);
            let projection = projection_new(root);
            let value = gpu.allocate(16, ResinMemory::Default);
            let payload = projection_owner(projection);
            assert!(checked_projection_pointer(payload, value, 17, 1).is_err());
            assert!(
                checked_projection_pointer(payload, ResinGpuPtr { offset: 1, ..value }, 8, 8)
                    .is_err()
            );
            assert!(
                checked_projection_pointer(
                    payload,
                    ResinGpuPtr {
                        access: RESIN_GPU_ACCESS_READ,
                        ..value
                    },
                    8,
                    8
                )
                .is_err()
            );
            assert!(checked_device_pointer(value, 8, 8, ptr::null_mut()).is_err());
            assert!(payload.dependencies.is_empty());
            assert!(checked_host(value, 8, 8, RESIN_GPU_ACCESS_READ).is_ok());
            resin_arc_release(projection);
            resin_arc_release(root.owner);
            resin_arc_release(value.owner);
        }
    }

    #[test]
    fn image_copy_uses_the_span_offset_and_unlocks_after_submission() {
        let Some(gpu) = TestGpu::new() else { return };
        unsafe {
            let allocation = gpu.allocate(80, ResinMemory::Readback);
            checked_host(allocation, 80, 1, RESIN_GPU_ACCESS_WRITE)
                .unwrap()
                .cast::<u8>()
                .write_bytes(0xaa, 80);
            let data = checked_offset(allocation, 8, 64, 4).unwrap();
            let mut image = (*gpu.gpu).create_image(4, 4).unwrap();
            let mut commands = (*gpu.gpu).start_command_recording().unwrap();
            commands
                .begin_rendering(&mut image, [0.0, 1.0, 0.0, 1.0])
                .unwrap();
            commands.end_rendering().unwrap();
            assert_eq!(
                copy_image_to_span(&mut commands, &mut image, ResinGpuSpan { data, length: 63 }),
                ResinStatus::InvalidArgument
            );
            assert!(checked_host(allocation, 80, 1, RESIN_GPU_ACCESS_READ).is_ok());
            assert_eq!(
                copy_image_to_span(&mut commands, &mut image, ResinGpuSpan { data, length: 64 }),
                ResinStatus::Success
            );
            assert!(checked_host(allocation, 80, 1, RESIN_GPU_ACCESS_READ).is_err());
            (*gpu.gpu).submit(commands).unwrap();
            let bytes = std::slice::from_raw_parts(
                checked_host(allocation, 80, 1, RESIN_GPU_ACCESS_READ)
                    .unwrap()
                    .cast::<u8>(),
                80,
            );
            assert_eq!(&bytes[..8], &[0xaa; 8]);
            assert_eq!(&bytes[72..], &[0xaa; 8]);
            for pixel in bytes[8..72].chunks_exact(4) {
                assert_eq!(pixel, &[0, 255, 0, 255]);
            }
            resin_arc_release(allocation.owner);
        }
    }
}
