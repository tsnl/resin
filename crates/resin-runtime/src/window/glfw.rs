use std::{
    cell::RefCell,
    ffi::{c_char, c_void},
    rc::{Rc, Weak},
};

use ash::vk;
use libloading::Library;

use crate::ResinStatus;

type Handle = *mut c_void;

pub(super) struct Glfw {
    _library: Library,
    terminate: unsafe extern "C" fn(),
    pub default_window_hints: unsafe extern "C" fn(),
    pub window_hint: unsafe extern "C" fn(i32, i32),
    pub create_window: unsafe extern "C" fn(i32, i32, *const c_char, Handle, Handle) -> Handle,
    pub destroy_window: unsafe extern "C" fn(Handle),
    pub poll_events: unsafe extern "C" fn(),
    pub window_should_close: unsafe extern "C" fn(Handle) -> i32,
    pub set_window_should_close: unsafe extern "C" fn(Handle, i32),
    pub get_framebuffer_size: unsafe extern "C" fn(Handle, *mut i32, *mut i32),
    pub set_window_size: unsafe extern "C" fn(Handle, i32, i32),
    pub get_key: unsafe extern "C" fn(Handle, i32) -> i32,
    pub get_required_instance_extensions: unsafe extern "C" fn(*mut u32) -> *const *const c_char,
    pub create_window_surface: unsafe extern "C" fn(
        vk::Instance,
        Handle,
        *const vk::AllocationCallbacks<'_>,
        *mut vk::SurfaceKHR,
    ) -> vk::Result,
}

thread_local! {
    static GLFW: RefCell<Weak<Glfw>> = const { RefCell::new(Weak::new()) };
}

impl Glfw {
    // Call only on the main thread, without another owner of GLFW initialization.
    pub unsafe fn acquire() -> Result<Rc<Self>, ResinStatus> {
        GLFW.with(|slot| {
            if let Some(glfw) = slot.borrow().upgrade() {
                return Ok(glfw);
            }
            let glfw = Rc::new(unsafe { Self::load()? });
            *slot.borrow_mut() = Rc::downgrade(&glfw);
            Ok(glfw)
        })
    }

    unsafe fn load() -> Result<Self, ResinStatus> {
        #[cfg(target_os = "windows")]
        let name = "glfw3.dll";
        #[cfg(target_os = "macos")]
        let name = "libglfw.3.dylib";
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let name = "libglfw.so.3";
        let library = unsafe { Library::new(name) }.map_err(|_| ResinStatus::WindowUnavailable)?;
        unsafe {
            let init: unsafe extern "C" fn() -> i32 = symbol(&library, b"glfwInit\0")?;
            let api = Self {
                terminate: symbol(&library, b"glfwTerminate\0")?,
                default_window_hints: symbol(&library, b"glfwDefaultWindowHints\0")?,
                window_hint: symbol(&library, b"glfwWindowHint\0")?,
                create_window: symbol(&library, b"glfwCreateWindow\0")?,
                destroy_window: symbol(&library, b"glfwDestroyWindow\0")?,
                poll_events: symbol(&library, b"glfwPollEvents\0")?,
                window_should_close: symbol(&library, b"glfwWindowShouldClose\0")?,
                set_window_should_close: symbol(&library, b"glfwSetWindowShouldClose\0")?,
                get_framebuffer_size: symbol(&library, b"glfwGetFramebufferSize\0")?,
                set_window_size: symbol(&library, b"glfwSetWindowSize\0")?,
                get_key: symbol(&library, b"glfwGetKey\0")?,
                get_required_instance_extensions: symbol(
                    &library,
                    b"glfwGetRequiredInstanceExtensions\0",
                )?,
                create_window_surface: symbol(&library, b"glfwCreateWindowSurface\0")?,
                _library: library,
            };
            if init() == 0 {
                return Err(ResinStatus::WindowUnavailable);
            }
            Ok(api)
        }
    }
}

impl Drop for Glfw {
    fn drop(&mut self) {
        unsafe { (self.terminate)() };
    }
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, ResinStatus> {
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|_| ResinStatus::WindowUnavailable)
}
