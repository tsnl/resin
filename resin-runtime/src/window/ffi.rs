use std::{
    ffi::{CStr, c_char},
    ptr,
};

use super::ResinWindow;
use crate::{ResinGpu, ResinImage, ResinStatus};

/// # Safety
/// Use only on the main thread. `title` must be a NUL-terminated UTF-8 string and
/// `out_window` must be writable. Do not initialize or terminate GLFW elsewhere.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_create(
    width: u32,
    height: u32,
    title: *const c_char,
    out_window: *mut *mut ResinWindow,
) -> ResinStatus {
    if out_window.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_window = ptr::null_mut() };
    if title.is_null() {
        return ResinStatus::InvalidArgument;
    }
    match unsafe { ResinWindow::create(width, height, CStr::from_ptr(title)) } {
        Ok(window) => {
            unsafe { *out_window = Box::into_raw(Box::new(window)) };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// `window` must be null or a live owned window. Call only on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_destroy(window: *mut ResinWindow) {
    if !window.is_null() {
        drop(unsafe { Box::from_raw(window) });
    }
}

/// # Safety
/// `window` must be a live window on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_poll_events(window: *const ResinWindow) -> ResinStatus {
    let Some(window) = (unsafe { window.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    window.poll_events();
    ResinStatus::Success
}

/// # Safety
/// `window` must be null or a live window on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_should_close(window: *const ResinWindow) -> i32 {
    i32::from(unsafe { window.as_ref() }.is_none_or(ResinWindow::should_close))
}

/// # Safety
/// `window` must be a live window on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_set_should_close(
    window: *const ResinWindow,
    close: i32,
) -> ResinStatus {
    let Some(window) = (unsafe { window.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    window.set_should_close(close != 0);
    ResinStatus::Success
}

/// # Safety
/// `window` must be a live window on the main thread; width and height must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_framebuffer_size(
    window: *const ResinWindow,
    width: *mut u32,
    height: *mut u32,
) -> ResinStatus {
    if width.is_null() || height.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe {
        *width = 0;
        *height = 0;
    }
    let Some(window) = (unsafe { window.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    let size = window.framebuffer_size();
    unsafe {
        *width = size.0;
        *height = size.1;
    }
    ResinStatus::Success
}

/// # Safety
/// `window` must be a live window on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_set_size(
    window: *const ResinWindow,
    width: u32,
    height: u32,
) -> ResinStatus {
    let Some(window) = (unsafe { window.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    match window.set_size(width, height) {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// # Safety
/// `window` must be null or a live window on the main thread. `key` is a GLFW key code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_key_pressed(window: *const ResinWindow, key: i32) -> i32 {
    i32::from(unsafe { window.as_ref() }.is_some_and(|window| window.key_pressed(key)))
}

/// # Safety
/// `window` must be null or a live window on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_key_state(window: *const ResinWindow, key: i32) -> u32 {
    unsafe { window.as_ref() }.map_or(0, |window| window.key_state(key))
}

/// # Safety
/// `window` must be null or a live window on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_mouse_button_state(
    window: *const ResinWindow,
    button: i32,
) -> u32 {
    unsafe { window.as_ref() }.map_or(0, |window| window.mouse_button_state(button))
}

/// # Safety
/// `window` must be live on the main thread; x and y must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_cursor_position(
    window: *const ResinWindow,
    x: *mut f64,
    y: *mut f64,
) -> ResinStatus {
    unsafe { coordinates(window, x, y, ResinWindow::cursor_position) }
}

/// # Safety
/// `window` must be live on the main thread; x and y must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_scroll_delta(
    window: *const ResinWindow,
    x: *mut f64,
    y: *mut f64,
) -> ResinStatus {
    unsafe { coordinates(window, x, y, ResinWindow::scroll_delta) }
}

unsafe fn coordinates(
    window: *const ResinWindow,
    x: *mut f64,
    y: *mut f64,
    query: fn(&ResinWindow) -> (f64, f64),
) -> ResinStatus {
    if x.is_null() || y.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe {
        *x = 0.0;
        *y = 0.0;
    }
    let Some(window) = (unsafe { window.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    let value = query(window);
    unsafe {
        *x = value.0;
        *y = value.1;
    }
    ResinStatus::Success
}

/// # Safety
/// `window` must be null or a live window on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_focused(window: *const ResinWindow) -> i32 {
    i32::from(unsafe { window.as_ref() }.is_some_and(ResinWindow::focused))
}

/// # Safety
/// `window` must be live on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_window_capture_cursor(
    window: *const ResinWindow,
    capture: i32,
) -> ResinStatus {
    let Some(window) = (unsafe { window.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    window.capture_cursor(capture != 0);
    ResinStatus::Success
}

/// # Safety
/// `window` must be live on the main thread and `out_gpu` must be writable.
/// Use and destroy the returned GPU and its resources on the main thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_create_for_window(
    window: *const ResinWindow,
    out_gpu: *mut *mut ResinGpu,
) -> ResinStatus {
    if out_gpu.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe { *out_gpu = ptr::null_mut() };
    let Some(window) = (unsafe { window.as_ref() }) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { ResinGpu::create_for_window(window) } {
        Ok(gpu) => {
            unsafe { *out_gpu = Box::into_raw(Box::new(gpu)) };
            ResinStatus::Success
        }
        Err(status) => status,
    }
}

/// # Safety
/// GPU and initialized image must be live, on the main thread, and externally
/// synchronized. The image must belong to the GPU.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gpu_present(
    gpu: *mut ResinGpu,
    image: *mut ResinImage,
) -> ResinStatus {
    let (Some(gpu), Some(image)) = (unsafe { gpu.as_mut() }, unsafe { image.as_mut() }) else {
        return ResinStatus::InvalidArgument;
    };
    match unsafe { gpu.present(image) } {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_arguments_do_not_initialize_glfw() {
        unsafe {
            let mut window = ptr::dangling_mut();
            assert_eq!(
                resin_window_create(0, 10, c"test".as_ptr(), &mut window),
                ResinStatus::InvalidArgument
            );
            assert!(window.is_null());
            assert_eq!(
                resin_window_create(u32::MAX, 10, c"test".as_ptr(), &mut window),
                ResinStatus::InvalidArgument
            );
            assert_eq!(
                resin_window_create(10, 10, ptr::null(), &mut window),
                ResinStatus::InvalidArgument
            );
            assert_eq!(
                resin_window_create(10, 10, c"test".as_ptr(), ptr::null_mut()),
                ResinStatus::InvalidArgument
            );
            assert_eq!(resin_window_should_close(ptr::null()), 1);
            assert_eq!(resin_window_key_pressed(ptr::null(), 256), 0);
            assert_eq!(resin_window_key_state(ptr::null(), 256), 0);
            assert_eq!(resin_window_mouse_button_state(ptr::null(), 0), 0);
            assert_eq!(resin_window_focused(ptr::null()), 0);
            assert_eq!(
                resin_window_capture_cursor(ptr::null(), 1),
                ResinStatus::InvalidArgument
            );
            let (mut x, mut y) = (1.0, 1.0);
            assert_eq!(
                resin_window_cursor_position(ptr::null(), &mut x, &mut y),
                ResinStatus::InvalidArgument
            );
            assert_eq!((x, y), (0.0, 0.0));
            assert_eq!(
                resin_window_scroll_delta(ptr::null(), &mut x, &mut y),
                ResinStatus::InvalidArgument
            );
            assert_eq!(
                resin_window_cursor_position(ptr::null(), ptr::null_mut(), &mut y),
                ResinStatus::InvalidArgument
            );

            assert_eq!(
                resin_window_poll_events(ptr::null()),
                ResinStatus::InvalidArgument
            );
            assert_eq!(
                resin_window_set_should_close(ptr::null(), 1),
                ResinStatus::InvalidArgument
            );
            assert_eq!(
                resin_window_set_size(ptr::null(), 10, 10),
                ResinStatus::InvalidArgument
            );
            let (mut width, mut height) = (1, 1);
            assert_eq!(
                resin_window_framebuffer_size(ptr::null(), &mut width, &mut height),
                ResinStatus::InvalidArgument
            );
            assert_eq!((width, height), (0, 0));
            let mut gpu = ptr::dangling_mut();
            assert_eq!(
                resin_gpu_create_for_window(ptr::null(), &mut gpu),
                ResinStatus::InvalidArgument
            );
            assert!(gpu.is_null());
            assert_eq!(
                resin_gpu_present(ptr::null_mut(), ptr::null_mut()),
                ResinStatus::InvalidArgument
            );
            resin_window_destroy(ptr::null_mut());
        }
    }
}
