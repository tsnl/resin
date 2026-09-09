pub mod ffi;
mod glfw;
mod input;

use std::{
    cell::{Cell, RefCell},
    ffi::{CStr, c_char},
    ptr,
    rc::Rc,
};

use ash::vk::Handle;
use ash::{Entry, Instance, khr, vk};
use glfw_sys as sys;

use crate::{ResinStatus, gpu::vk_status};
use glfw::Glfw;

pub struct ResinWindow {
    pub(crate) native: Rc<NativeWindow>,
}

pub(crate) struct NativeWindow {
    glfw: Rc<Glfw>,
    handle: *mut sys::GLFWwindow,
    attached: Cell<bool>,
    input: RefCell<input::Input>,
}

pub(crate) struct Surface {
    pub loader: khr::surface::Instance,
    pub handle: vk::SurfaceKHR,
    pub window: Rc<NativeWindow>,
}

impl ResinWindow {
    /// # Safety
    /// Create, use, and destroy windows and their GPUs on the process main thread.
    /// The runtime must be the only owner of GLFW initialization and termination.
    pub unsafe fn create(width: u32, height: u32, title: &CStr) -> Result<Self, ResinStatus> {
        if !valid_size(width, height) {
            return Err(ResinStatus::InvalidArgument);
        }
        let glfw = unsafe { Glfw::acquire()? };
        let handle = unsafe {
            sys::glfwDefaultWindowHints();
            sys::glfwWindowHint(sys::GLFW_CLIENT_API, sys::GLFW_NO_API);
            sys::glfwCreateWindow(
                width as i32,
                height as i32,
                title.as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if handle.is_null() {
            glfw.print_error("glfwCreateWindow");
            return Err(ResinStatus::WindowUnavailable);
        }
        let native = Rc::new(NativeWindow {
            glfw,
            handle,
            attached: Cell::new(false),
            input: RefCell::default(),
        });
        // The Rc allocation stays at a stable address until after GLFW destroys
        // the native window, including when a GPU retains it.
        unsafe {
            sys::glfwSetWindowUserPointer(handle, Rc::as_ptr(&native).cast_mut().cast());
            sys::glfwSetKeyCallback(handle, Some(key_callback));
            sys::glfwSetMouseButtonCallback(handle, Some(mouse_button_callback));
            sys::glfwSetScrollCallback(handle, Some(scroll_callback));
        }
        Ok(Self { native })
    }

    pub fn poll_events(&self) {
        unsafe { sys::glfwPollEvents() };
        self.native.input.borrow_mut().commit();
    }

    pub fn should_close(&self) -> bool {
        unsafe { sys::glfwWindowShouldClose(self.native.handle) != sys::GLFW_FALSE }
    }

    pub fn set_should_close(&self, close: bool) {
        unsafe { sys::glfwSetWindowShouldClose(self.native.handle, i32::from(close)) };
    }

    pub fn framebuffer_size(&self) -> (u32, u32) {
        self.native.framebuffer_size()
    }

    pub fn set_size(&self, width: u32, height: u32) -> Result<(), ResinStatus> {
        if !valid_size(width, height) {
            return Err(ResinStatus::InvalidArgument);
        }
        unsafe { sys::glfwSetWindowSize(self.native.handle, width as i32, height as i32) };
        Ok(())
    }

    pub fn key_pressed(&self, key: i32) -> bool {
        self.key_state(key) & input::DOWN != 0
    }

    pub fn key_state(&self, key: i32) -> u32 {
        self.native.input.borrow().key_state(key)
    }

    pub fn mouse_button_state(&self, button: i32) -> u32 {
        self.native.input.borrow().mouse_button_state(button)
    }

    pub fn cursor_position(&self) -> (f64, f64) {
        let (mut x, mut y) = (0.0, 0.0);
        unsafe { sys::glfwGetCursorPos(self.native.handle, &mut x, &mut y) };
        (x, y)
    }

    pub fn scroll_delta(&self) -> (f64, f64) {
        self.native.input.borrow().scroll_delta()
    }

    pub fn focused(&self) -> bool {
        unsafe { sys::glfwGetWindowAttrib(self.native.handle, sys::GLFW_FOCUSED) == sys::GLFW_TRUE }
    }

    pub fn capture_cursor(&self, capture: bool) {
        unsafe {
            sys::glfwSetInputMode(
                self.native.handle,
                sys::GLFW_CURSOR,
                if capture {
                    sys::GLFW_CURSOR_DISABLED
                } else {
                    sys::GLFW_CURSOR_NORMAL
                },
            );
        }
    }

    pub(crate) fn extensions(&self) -> Result<Vec<*const c_char>, ResinStatus> {
        let mut count = 0;
        let names = unsafe { sys::glfwGetRequiredInstanceExtensions(&mut count) };
        if names.is_null() || count == 0 {
            self.native
                .glfw
                .print_error("glfwGetRequiredInstanceExtensions");
            return Err(ResinStatus::VulkanUnavailable);
        }
        let mut names = unsafe { std::slice::from_raw_parts(names, count as usize) }.to_vec();
        for extension in [
            vk::KHR_GET_SURFACE_CAPABILITIES2_NAME,
            vk::EXT_SURFACE_MAINTENANCE1_NAME,
        ] {
            if !names
                .iter()
                .any(|&name| unsafe { CStr::from_ptr(name) } == extension)
            {
                names.push(extension.as_ptr());
            }
        }
        Ok(names)
    }

    pub(crate) fn surface(
        &self,
        entry: &Entry,
        instance: &Instance,
    ) -> Result<Surface, ResinStatus> {
        if self.native.attached.get() {
            return Err(ResinStatus::InvalidArgument);
        }
        let mut handle = ptr::null_mut();
        let result = unsafe {
            sys::glfwCreateWindowSurface(
                instance.handle().as_raw() as sys::VkInstance,
                self.native.handle,
                ptr::null(),
                &mut handle,
            )
        };
        vk::Result::from_raw(result)
            .result()
            .inspect_err(|_| self.native.glfw.print_error("glfwCreateWindowSurface"))
            .map_err(vk_status)?;
        self.native.attached.set(true);
        Ok(Surface {
            loader: khr::surface::Instance::new(entry, instance),
            handle: vk::SurfaceKHR::from_raw(handle as u64),
            window: self.native.clone(),
        })
    }
}

impl NativeWindow {
    pub fn framebuffer_size(&self) -> (u32, u32) {
        let (mut width, mut height) = (0, 0);
        unsafe { sys::glfwGetFramebufferSize(self.handle, &mut width, &mut height) };
        (width.max(0) as u32, height.max(0) as u32)
    }
}

impl Drop for NativeWindow {
    fn drop(&mut self) {
        unsafe { sys::glfwDestroyWindow(self.handle) };
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.handle, None) };
        self.window.attached.set(false);
    }
}

fn valid_size(width: u32, height: u32) -> bool {
    (1..=i32::MAX as u32).contains(&width) && (1..=i32::MAX as u32).contains(&height)
}

unsafe extern "C" fn key_callback(
    window: *mut sys::GLFWwindow,
    key: i32,
    _scancode: i32,
    action: i32,
    _mods: i32,
) {
    let native = unsafe { &*sys::glfwGetWindowUserPointer(window).cast::<NativeWindow>() };
    native.input.borrow_mut().key(key, action);
}

unsafe extern "C" fn mouse_button_callback(
    window: *mut sys::GLFWwindow,
    button: i32,
    action: i32,
    _mods: i32,
) {
    let native = unsafe { &*sys::glfwGetWindowUserPointer(window).cast::<NativeWindow>() };
    native.input.borrow_mut().mouse_button(button, action);
}

unsafe extern "C" fn scroll_callback(window: *mut sys::GLFWwindow, x: f64, y: f64) {
    let native = unsafe { &*sys::glfwGetWindowUserPointer(window).cast::<NativeWindow>() };
    native.input.borrow_mut().scroll(x, y);
}
