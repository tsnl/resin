__all__ = [
    "LogicError",
    "GlfwError",
    "PlatformSupportError",
]

import glfw


class LogicError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


class GlfwError(RuntimeError):
    def __init__(self, message: str):
        error_code, error_message = glfw.get_error()
        super().__init__(f"{message}: {error_message} (error=0x{error_code:X})")


class PlatformSupportError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
