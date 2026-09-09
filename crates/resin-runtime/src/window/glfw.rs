use std::{
    cell::RefCell,
    ffi::CStr,
    io::Write,
    ptr,
    rc::{Rc, Weak},
};

use crate::ResinStatus;

pub(super) struct Glfw;

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
            let glfw = Rc::new(Self);
            if unsafe { glfw_sys::glfwInit() } == glfw_sys::GLFW_FALSE {
                glfw.print_error("glfwInit");
                return Err(ResinStatus::WindowUnavailable);
            }
            *slot.borrow_mut() = Rc::downgrade(&glfw);
            Ok(glfw)
        })
    }

    pub fn print_error(&self, operation: &str) {
        let mut description = ptr::null();
        let code = unsafe { glfw_sys::glfwGetError(&mut description) };
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
        unsafe { glfw_sys::glfwTerminate() };
    }
}
