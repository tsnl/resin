__all__ = ["Window"]

import atexit
import functools

import glfw


class Window:
    _all = []

    def __init__(self, width: int, height: int, title: str):
        super().__init__()

        _lazy_glfw_init()

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self._glfw_window = glfw.create_window(
            width=width,
            height=height,
            title=title,
            monitor=None,
            share=None,
        )
        if not self._glfw_window:
            raise GlfwError("Failed to create GLFW window")

        Window._all.append(self)

    def should_close(self) -> bool:
        return glfw.window_should_close(self._glfw_window)

    def show(self):
        glfw.show_window(self._glfw_window)

    def hide(self):
        glfw.hide_window(self._glfw_window)

    @staticmethod
    def all() -> list["Window"]:
        return Window._all

    @staticmethod
    def update_all():
        glfw.poll_events()


@functools.cache
def _lazy_glfw_init():
    ok = bool(glfw.init())
    if not ok:
        raise GlfwError("Failed to initialize GLFW")

    atexit.register(glfw.terminate)


class GlfwError(RuntimeError):
    def __init__(self, message: str):
        error_code, error_message = glfw.get_error()
        super().__init__(f"{message}: {error_message} (error=0x{error_code:X})")
