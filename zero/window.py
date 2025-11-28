__all__ = ["Window"]

from typing import TypeAlias

import glfw

from .core import BaseContext, BaseContextResource
from .excepts import GlfwError
from .gpu import GpuContext, GpuSurface


class WindowContext(BaseContext["WindowContext"]):
    def __init__(self, gpu_context: GpuContext) -> None:
        super().__init__()

        self.gpu_context = gpu_context

        ok = glfw.init()
        if not ok:
            raise GlfwError("Failed to initialize GLFW")

    def _on_dispose(self) -> None:
        glfw.terminate()

    def create_window(
        self,
        width: int,
        height: int,
        title: str,
    ) -> "Window":
        # Configure window hints
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        # Create GLFW window
        glfw_window = glfw.create_window(
            width=width,
            height=height,
            title=title,
            monitor=None,
            share=None,
        )
        if not glfw_window:
            raise GlfwError("Failed to create GLFW window")

        # Return Window resource
        return Window(context=self, glfw_window=glfw_window)


WindowResource: TypeAlias = BaseContextResource["WindowContext"]


class Window(WindowResource):
    glfw_window_handle: glfw._GLFWwindow

    def __init__(
        self,
        *,
        context: WindowContext,
        glfw_window: glfw._GLFWwindow,
    ) -> None:
        super().__init__(parent=context)
        self.glfw_window_handle = glfw_window

    def _on_dispose(self) -> None:
        glfw.destroy_window(self.glfw_window_handle)

    def create_surface(self) -> GpuSurface:
        width, height = glfw.get_framebuffer_size(
            self.glfw_window_handle,
        )
        return self.context.gpu_context.create_surface_from_raw_glfw_window_handle(
            raw_glfw_window_handle=self.glfw_window_handle,
            framebuffer_width=width,
            framebuffer_height=height,
        )

    def should_close(self) -> bool:
        return glfw.window_should_close(self.glfw_window_handle)

    def show(self):
        glfw.show_window(self.glfw_window_handle)

    def hide(self):
        glfw.hide_window(self.glfw_window_handle)

    @staticmethod
    def poll_events():
        glfw.poll_events()
