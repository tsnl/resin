use std::{
    cell::RefCell,
    ffi::{CStr, c_char, c_void},
    io::Write,
    ptr,
    rc::{Rc, Weak},
};

use ash::vk;
use libloading::Library;

use crate::ResinStatus;

type Handle = *mut c_void;

pub(super) struct Glfw {
    _library: Library,
    terminate: unsafe extern "C" fn(),
    get_error: unsafe extern "C" fn(*mut *const c_char) -> i32,
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
        let library = unsafe { Library::new(name) }.map_err(|error| {
            let _ = writeln!(
                std::io::stderr().lock(),
                "resin: could not load {name}: {error}"
            );
            ResinStatus::WindowUnavailable
        })?;
        unsafe {
            let init: unsafe extern "C" fn() -> i32 = symbol(&library, c"glfwInit")?;
            let api = Self {
                terminate: symbol(&library, c"glfwTerminate")?,
                get_error: symbol(&library, c"glfwGetError")?,
                default_window_hints: symbol(&library, c"glfwDefaultWindowHints")?,
                window_hint: symbol(&library, c"glfwWindowHint")?,
                create_window: symbol(&library, c"glfwCreateWindow")?,
                destroy_window: symbol(&library, c"glfwDestroyWindow")?,
                poll_events: symbol(&library, c"glfwPollEvents")?,
                window_should_close: symbol(&library, c"glfwWindowShouldClose")?,
                set_window_should_close: symbol(&library, c"glfwSetWindowShouldClose")?,
                get_framebuffer_size: symbol(&library, c"glfwGetFramebufferSize")?,
                set_window_size: symbol(&library, c"glfwSetWindowSize")?,
                get_key: symbol(&library, c"glfwGetKey")?,
                get_required_instance_extensions: symbol(
                    &library,
                    c"glfwGetRequiredInstanceExtensions",
                )?,
                create_window_surface: symbol(&library, c"glfwCreateWindowSurface")?,
                _library: library,
            };
            if init() == 0 {
                api.print_error("glfwInit");
                return Err(ResinStatus::WindowUnavailable);
            }
            Ok(api)
        }
    }

    pub fn print_error(&self, operation: &str) {
        let mut description = ptr::null();
        let code = unsafe { (self.get_error)(&mut description) };
        let description = if description.is_null() {
            c"no error description"
        } else {
            unsafe { CStr::from_ptr(description) }
        };
        let _ = writeln!(
            std::io::stderr().lock(),
            "resin: {operation} failed: GLFW error {code:#010x}: {}",
            description.to_string_lossy()
        );
    }
}

impl Drop for Glfw {
    fn drop(&mut self) {
        unsafe { (self.terminate)() };
    }
}

unsafe fn symbol<T: Copy>(library: &Library, name: &CStr) -> Result<T, ResinStatus> {
    unsafe { library.get::<T>(name.to_bytes_with_nul()) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            let _ = writeln!(
                std::io::stderr().lock(),
                "resin: could not load {}: {error}",
                name.to_string_lossy()
            );
            ResinStatus::WindowUnavailable
        })
}
