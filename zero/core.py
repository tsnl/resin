import atexit
import functools

import glfw

from .excepts import GlfwError


@functools.cache
def ensure_glfw_init():
    ok = bool(glfw.init())
    if not ok:
        raise GlfwError("Failed to initialize GLFW")

    atexit.register(glfw.terminate)
