__all__ = ["Window"]

from typing import TypeAlias, Iterator

import glfw

from .excepts import GlfwError
from .core import BaseContext, BaseContextResource


class WindowContext(BaseContext["WindowContext"]):
    def __init__(self) -> None:
        ok = bool(glfw.init())
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
        return Window(context=self, width=width, height=height, title=title)


WindowResource: TypeAlias = BaseContextResource["WindowContext"]


class Window(WindowResource):
    _all: set["Window"] = set()

    def __init__(self, context: WindowContext, *, width: int, height: int, title: str):
        super().__init__(parent=context)

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

        Window._all.add(self)

    def _on_dispose(self) -> None:
        glfw.destroy_window(self._glfw_window)
        Window._all.remove(self)

    def should_close(self) -> bool:
        return glfw.window_should_close(self._glfw_window)

    def show(self):
        glfw.show_window(self._glfw_window)

    def hide(self):
        glfw.hide_window(self._glfw_window)

    @staticmethod
    def all() -> Iterator["Window"]:
        return iter(Window._all)

    @staticmethod
    def update_all():
        glfw.poll_events()
