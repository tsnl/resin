__all__ = ["Window"]

import glfw

from .basic import BaseResource
from .excepts import GlfwError
from .gpu import GpuContext, GpuSurface
from .typed_vulkan import raw_ffi


class WindowContext(BaseResource):
    def __init__(
        self,
        *,
        gpu_context: GpuContext,
        parent: BaseResource | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self.gpu_context = gpu_context

        ok = glfw.init()
        if not ok:
            raise GlfwError("Failed to initialize GLFW")

    def _on_dispose(self) -> None:
        glfw.terminate()


class Window(BaseResource):
    context: WindowContext
    width: int
    height: int
    title: str
    glfw_window_handle: glfw._GLFWwindow
    gpu_surface: GpuSurface

    def __init__(
        self,
        *,
        context: WindowContext,
        width: int,
        height: int,
        title: str,
    ) -> None:
        super().__init__(parent=context)
        self.context = context
        self.width = width
        self.height = height
        self.title = title
        self.glfw_window_handle = self._new_glfw_window()
        self.gpu_surface = self._new_gpu_surface()

    def _new_glfw_window(self) -> glfw._GLFWwindow:
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        glfw_window = glfw.create_window(
            width=self.width,
            height=self.height,
            title=self.title,
            monitor=None,
            share=None,
        )
        if not glfw_window:
            raise GlfwError("Failed to create GLFW window")

        return glfw_window

    def _new_gpu_surface(self) -> GpuSurface:
        if not self.context.gpu_context.enable_present_support:
            raise RuntimeError("GPU context does not support presentation")

        surface_ptr = raw_ffi.new("VkSurfaceKHR[1]")
        result = glfw.create_window_surface(
            instance=self.context.gpu_context.vk_instance,
            window=self.glfw_window_handle,
            allocator=None,
            surface=surface_ptr,
        )
        if result != 0:
            raise RuntimeError(f"Failed to create window surface: VkResult: {result}")
        width, height = glfw.get_framebuffer_size(self.glfw_window_handle)
        return GpuSurface(
            context=self.context.gpu_context,
            parent=self,
            vk_surface=surface_ptr[0],
            width=width,
            height=height,
        )

    def _on_dispose(self) -> None:
        glfw.destroy_window(self.glfw_window_handle)

    def should_close(self) -> bool:
        return glfw.window_should_close(self.glfw_window_handle)

    def show(self):
        glfw.show_window(self.glfw_window_handle)

    def hide(self):
        glfw.hide_window(self.glfw_window_handle)

    @staticmethod
    def poll_events():
        glfw.poll_events()
