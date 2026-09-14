//! Atomic ownership for compiler-generated shared payloads. Payload access is unsynchronized.

use std::{
    alloc::{Layout, alloc, dealloc},
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering, fence},
};

pub struct ResinArc {
    strong: AtomicUsize,
    // Includes one implicit weak reference while the payload is alive.
    weak: AtomicUsize,
    payload: *mut u8,
    layout: Layout,
    length: usize,
    stride: usize,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
}

fn retain(count: &AtomicUsize) {
    if count.fetch_add(1, Ordering::Relaxed) >= isize::MAX as usize {
        eprintln!("reference count overflow");
        std::process::abort();
    }
}

/// Allocate uninitialized payload storage. The caller must initialize it before release.
/// Allocation failure terminates the process, like other infallible host allocation.
#[unsafe(no_mangle)]
pub extern "C" fn resin_arc_new(
    size: usize,
    align: usize,
    destroy: unsafe extern "C" fn(*mut c_void),
) -> *mut ResinArc {
    let owner = try_allocate(1, size, align, 0, Some(destroy));
    if owner.is_null() {
        crate::host::fail("shared allocation failed");
    }
    owner
}

/// Allocate storage for `length` elements, returning null on overflow or allocation failure.
/// The caller must initialize every element before publishing or releasing the owner.
/// Final release destroys elements in reverse order with the optional element callback.
#[unsafe(no_mangle)]
pub extern "C" fn resin_arc_span_try_new(
    length: usize,
    stride: usize,
    align: usize,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
) -> *mut ResinArc {
    try_allocate(length, stride, align, 0, destroy)
}

// Strings reserve a trailing NUL outside the sequence's logical element count.
pub(crate) fn new_string(length: usize) -> *mut ResinArc {
    let owner = try_allocate(length, 1, 1, 1, None);
    if owner.is_null() {
        crate::host::fail("string allocation failed");
    }
    owner
}

fn try_allocate(
    length: usize,
    stride: usize,
    align: usize,
    extra: usize,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
) -> *mut ResinArc {
    let Some(size) = length
        .checked_mul(stride)
        .and_then(|size| size.checked_add(extra))
    else {
        return std::ptr::null_mut();
    };
    let Ok(layout) = Layout::from_size_align(size.max(1), align) else {
        return std::ptr::null_mut();
    };
    allocate_control(layout, length, stride, destroy)
}

fn allocate_control(
    layout: Layout,
    length: usize,
    stride: usize,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
) -> *mut ResinArc {
    let payload = unsafe { alloc(layout) };
    if payload.is_null() {
        return std::ptr::null_mut();
    }
    let owner = unsafe { alloc(Layout::new::<ResinArc>()) }.cast::<ResinArc>();
    if owner.is_null() {
        unsafe { dealloc(payload, layout) };
        return std::ptr::null_mut();
    }
    unsafe {
        owner.write(ResinArc {
            strong: AtomicUsize::new(1),
            weak: AtomicUsize::new(1),
            payload,
            layout,
            length,
            stride,
            destroy,
        })
    };
    owner
}

/// # Safety
/// `owner` must identify a live strong reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_arc_data(owner: *mut ResinArc) -> *mut c_void {
    if owner.is_null() {
        eprintln!("access through an empty shared owner");
        std::process::abort();
    }
    unsafe { (*owner).payload.cast() }
}

/// # Safety
/// `owner` must identify a live strong reference to a sequence allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_arc_span_length(owner: *mut ResinArc) -> usize {
    if owner.is_null() {
        crate::host::fail("access through an empty shared owner");
    }
    unsafe { (*owner).length }
}

/// # Safety
/// A non-null `owner` must identify a live strong reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_arc_retain(owner: *mut ResinArc) {
    if let Some(owner) = unsafe { owner.as_ref() } {
        retain(&owner.strong);
    }
}

/// # Safety
/// A non-null `owner` must identify an owned strong reference being consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_arc_release(owner: *mut ResinArc) {
    if owner.is_null() {
        return;
    }
    let control = unsafe { &*owner };
    if control.strong.fetch_sub(1, Ordering::Release) == 1 {
        fence(Ordering::Acquire);
        unsafe {
            if let Some(destroy) = control.destroy {
                for index in (0..control.length).rev() {
                    destroy(control.payload.add(index * control.stride).cast());
                }
            }
            dealloc(control.payload, control.layout);
            resin_weak_release(owner);
        }
    }
}

/// # Safety
/// A non-null `owner` must identify a live strong or weak reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_weak_retain(owner: *mut ResinArc) {
    if let Some(owner) = unsafe { owner.as_ref() } {
        retain(&owner.weak);
    }
}

/// # Safety
/// A non-null `owner` must identify an owned weak reference being consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_weak_release(owner: *mut ResinArc) {
    if owner.is_null() {
        return;
    }
    if unsafe { (*owner).weak.fetch_sub(1, Ordering::Release) } == 1 {
        fence(Ordering::Acquire);
        unsafe {
            std::ptr::drop_in_place(owner);
            dealloc(owner.cast(), Layout::new::<ResinArc>());
        }
    }
}

/// # Safety
/// A non-null `owner` must identify a live weak reference. The result owns one strong
/// reference, or is null if the payload has expired.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_weak_upgrade(owner: *mut ResinArc) -> *mut ResinArc {
    let Some(control) = (unsafe { owner.as_ref() }) else {
        return std::ptr::null_mut();
    };
    let mut count = control.strong.load(Ordering::Relaxed);
    loop {
        if count == 0 {
            return std::ptr::null_mut();
        }
        if count >= isize::MAX as usize {
            eprintln!("reference count overflow");
            std::process::abort();
        }
        match control.strong.compare_exchange_weak(
            count,
            count + 1,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => return owner,
            Err(actual) => count = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    unsafe extern "C" fn destroy(value: *mut c_void) {
        let counter = unsafe { *(value.cast::<*const AtomicUsize>()) };
        unsafe {
            (*counter).fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn sequence_allocation_rejects_overflow_and_invalid_layout() {
        for (length, stride, align) in
            [(usize::MAX, 2, 1), (1, usize::MAX, 1), (1, 1, 0), (1, 1, 3)]
        {
            assert!(resin_arc_span_try_new(length, stride, align, None).is_null());
        }
        assert!(try_allocate(usize::MAX, 1, 1, 1, None).is_null());
    }

    #[test]
    fn empty_sequence_has_aligned_storage_and_zero_length() {
        unsafe {
            let owner = resin_arc_span_try_new(0, 4, 64, None);
            assert!(!owner.is_null());
            assert_eq!(resin_arc_span_length(owner), 0);
            assert!(!resin_arc_data(owner).is_null());
            assert_eq!(resin_arc_data(owner) as usize % 64, 0);
            resin_arc_release(owner);
        }
    }

    #[test]
    fn zero_sized_elements_retain_count_and_each_receive_destruction() {
        static DESTROYED: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn destroy(_: *mut c_void) {
            DESTROYED.fetch_add(1, Ordering::Relaxed);
        }
        unsafe {
            let owner = resin_arc_span_try_new(13, 0, 1, Some(destroy));
            assert!(!owner.is_null());
            assert_eq!(resin_arc_span_length(owner), 13);
            resin_arc_release(owner);
        }
        assert_eq!(DESTROYED.load(Ordering::Relaxed), 13);
    }

    #[test]
    fn sequence_weak_upgrade_preserves_length_and_final_drop_order() {
        struct Element {
            index: usize,
            destroyed: *const std::sync::Mutex<Vec<usize>>,
        }
        unsafe extern "C" fn destroy(value: *mut c_void) {
            let element = unsafe { &*value.cast::<Element>() };
            unsafe { &*element.destroyed }
                .lock()
                .unwrap()
                .push(element.index);
        }
        let destroyed = std::sync::Mutex::new(Vec::new());
        unsafe {
            let owner = resin_arc_span_try_new(
                3,
                size_of::<Element>(),
                align_of::<Element>(),
                Some(destroy),
            );
            assert!(!owner.is_null());
            let elements = resin_arc_data(owner).cast::<Element>();
            for index in 0..3 {
                elements.add(index).write(Element {
                    index,
                    destroyed: &destroyed,
                });
            }
            resin_weak_retain(owner);
            let other = resin_weak_upgrade(owner);
            assert_eq!(resin_arc_span_length(other), 3);
            resin_arc_release(owner);
            assert!(destroyed.lock().unwrap().is_empty());
            resin_arc_release(other);
            assert_eq!(*destroyed.lock().unwrap(), [2, 1, 0]);
            assert!(resin_weak_upgrade(owner).is_null());
            resin_weak_release(owner);
        }
    }
    #[test]
    fn weak_reference_does_not_keep_payload_alive() {
        let destroyed = AtomicUsize::new(0);
        unsafe {
            let owner = resin_arc_new(
                size_of::<*const AtomicUsize>(),
                align_of::<*const AtomicUsize>(),
                destroy,
            );
            *resin_arc_data(owner).cast::<*const AtomicUsize>() = &destroyed;
            resin_weak_retain(owner);
            let other = resin_weak_upgrade(owner);
            assert_eq!(other, owner);
            resin_arc_release(owner);
            assert_eq!(destroyed.load(Ordering::Relaxed), 0);
            resin_arc_release(other);
            assert_eq!(destroyed.load(Ordering::Relaxed), 1);
            assert!(resin_weak_upgrade(owner).is_null());
            resin_weak_release(owner);
        }
    }
    #[test]
    fn concurrent_upgrade_and_last_release_destroy_once() {
        let destroyed = AtomicUsize::new(0);
        unsafe {
            let owner = resin_arc_new(
                size_of::<*const AtomicUsize>(),
                align_of::<*const AtomicUsize>(),
                destroy,
            );
            *resin_arc_data(owner).cast::<*const AtomicUsize>() = &destroyed;
            resin_weak_retain(owner);
            let address = owner as usize;
            std::thread::scope(|scope| {
                for _ in 0..8 {
                    resin_weak_retain(owner);
                    scope.spawn(move || {
                        let weak = address as *mut ResinArc;
                        for _ in 0..1000 {
                            let strong = resin_weak_upgrade(weak);
                            if !strong.is_null() {
                                resin_arc_retain(strong);
                                resin_arc_release(strong);
                                resin_arc_release(strong);
                            }
                        }
                        resin_weak_release(weak);
                    });
                }
                resin_arc_release(owner);
            });
            assert_eq!(destroyed.load(Ordering::Relaxed), 1);
            assert!(resin_weak_upgrade(owner).is_null());
            resin_weak_release(owner);
        }
    }
}
