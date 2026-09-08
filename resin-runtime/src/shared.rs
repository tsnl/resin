//! Atomic ownership for compiler-generated shared payloads. Payload access is unsynchronized.

use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering, fence},
};

pub struct ResinArc {
    strong: AtomicUsize,
    // Includes one implicit weak reference while the payload is alive.
    weak: AtomicUsize,
    payload: *mut u8,
    layout: Layout,
    destroy: unsafe extern "C" fn(*mut c_void),
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
    let layout =
        Layout::from_size_align(size.max(1), align).unwrap_or_else(|_| std::process::abort());
    let payload = unsafe { alloc(layout) };
    if payload.is_null() {
        handle_alloc_error(layout);
    }
    Box::into_raw(Box::new(ResinArc {
        strong: AtomicUsize::new(1),
        weak: AtomicUsize::new(1),
        payload,
        layout,
        destroy,
    }))
}

/// # Safety
/// `owner` must identify a live strong reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_arc_data(owner: *mut ResinArc) -> *mut c_void {
    if owner.is_null() {
        eprintln!("access through an empty Arc");
        std::process::abort();
    }
    unsafe { (*owner).payload.cast() }
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
            (control.destroy)(control.payload.cast());
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
            drop(Box::from_raw(owner));
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
