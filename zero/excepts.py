__all__ = ["GlfwError"]

import glfw


class GlfwError(RuntimeError):
    def __init__(self, message: str):
        error_code, error_message = glfw.get_error()
        super().__init__(f"{message}: {error_message} (error=0x{error_code:X})")
