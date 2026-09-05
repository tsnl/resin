pub mod ffi;
mod glfw;

use std::{
    cell::Cell,
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
        Ok(Self {
            native: Rc::new(NativeWindow {
                glfw,
                handle,
                attached: Cell::new(false),
            }),
        })
    }

    pub fn poll_events(&self) {
        unsafe { sys::glfwPollEvents() };
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
        (sys::GLFW_KEY_SPACE..=sys::GLFW_KEY_LAST).contains(&key)
            && unsafe { sys::glfwGetKey(self.native.handle, key) == sys::GLFW_PRESS }
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
