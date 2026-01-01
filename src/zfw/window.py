"""
Window management and GLFW integration.

This module provides a lightweight abstraction over GLFW for window management,
separating windowing concerns from GUI and rendering logic.
"""

__all__ = [
    "Window",
    "WindowContext",
]

from typing import Callable

import glfw

from .basic import (
    BaseResource,
    logger,
    Key,
    KeyModifier,
    MouseButton,
    ButtonAction,
)
from .excepts import GlfwError
from .gpu import GpuContext, GpuSurface
from .typed_vulkan import raw_ffi


LOG = logger(__name__)


type KeyCallback = Callable[[Key | None, int, ButtonAction, list[KeyModifier]], None]
type MouseButtonCallback = Callable[
    [MouseButton, ButtonAction, list[KeyModifier]], None
]
type CursorPosCallback = Callable[[float, float], None]
type FramebufferSizeCallback = Callable[[int, int], None]


class WindowContext(BaseResource):
    """
    Context for window management, manages GLFW initialization/termination.
    """

    def __init__(
        self,
        *,
        parent_resource: BaseResource | None = None,
    ) -> None:
        super().__init__(parent_resource=parent_resource)

        ok = glfw.init()
        if not ok:
            raise GlfwError("Failed to initialize GLFW")

    def _on_dispose(self) -> None:
        glfw.terminate()


class Window(BaseResource):
    """
    A platform window backed by GLFW.

    This class handles raw window creation, event callbacks, and GPU surface management.
    It does not include any GUI or rendering logic.
    """

    _width_dip: int
    _height_dip: int
    _width_px: int
    _height_px: int
    _title: str
    _glfw_window_handle: glfw._GLFWwindow
    _gpu_surface: GpuSurface
    _last_mouse_x: float
    _last_mouse_y: float

    # Callbacks
    _key_callback: KeyCallback | None
    _mouse_button_callback: MouseButtonCallback | None
    _cursor_pos_callback: CursorPosCallback | None
    _framebuffer_size_callback: FramebufferSizeCallback | None

    def __init__(
        self,
        *,
        gpu_context: GpuContext,
        window_context: WindowContext,
        width_dip: int,
        height_dip: int,
        title: str,
        resizable: bool = True,
        min_width_dip: int = 640,
        min_height_dip: int = 540,
    ) -> None:
        super().__init__(parent_resource=window_context)

        self._gpu_context = gpu_context
        self._window_context = window_context
        self._width_dip = width_dip
        self._height_dip = height_dip
        self._width_px = 0
        self._height_px = 0
        self._min_width_dip = min_width_dip
        self._min_height_dip = min_height_dip
        self._title = title
        self._resizable = resizable

        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Initialize callbacks to None
        self._key_callback = None
        self._mouse_button_callback = None
        self._cursor_pos_callback = None
        self._framebuffer_size_callback = None

        # Create GLFW window and GPU surface
        self._glfw_window_handle = self._new_glfw_window()
        self._gpu_surface = self._new_gpu_surface()

    def _new_glfw_window(self) -> glfw._GLFWwindow:
        # Create GLFW window:
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, int(self._resizable))
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw_window = glfw.create_window(
            width=self._width_dip,
            height=self._height_dip,
            title=self._title,
            monitor=None,
            share=None,
        )
        if not glfw_window:
            raise GlfwError("Failed to create GLFW window")

        # Set the minimum size:
        glfw.set_window_size_limits(
            window=glfw_window,
            minwidth=max(self._min_width_dip, self._width_dip),
            minheight=max(self._min_height_dip, self._height_dip),
            maxwidth=glfw.DONT_CARE,
            maxheight=glfw.DONT_CARE,
        )

        # If raw mouse motion is supported, enable it by default.
        if glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(
                glfw_window,
                glfw.RAW_MOUSE_MOTION,
                glfw.TRUE,
            )
        else:
            LOG.warning(
                "Raw mouse motion is not supported on this window: 'joystick' cursor "
                "mode may be less accurate. See GLFW documentation for details.",
            )

        # Bind event handlers:
        glfw.set_key_callback(
            window=glfw_window,
            cbfun=self._on_glfw_key_event,
        )
        glfw.set_mouse_button_callback(
            window=glfw_window,
            cbfun=self._on_glfw_mouse_button_event,
        )
        glfw.set_cursor_pos_callback(
            window=glfw_window,
            cbfun=self._on_glfw_cursor_pos_event,
        )
        glfw.set_framebuffer_size_callback(
            window=glfw_window,
            cbfun=self._on_framebuffer_resize_event,
        )

        # Update window sizes:
        width_px, height_px = glfw.get_framebuffer_size(glfw_window)
        self._width_px = width_px
        self._height_px = height_px

        return glfw_window

    def _new_gpu_surface(self) -> GpuSurface:
        if not self._gpu_context.enable_present_support:
            raise RuntimeError("GPU context does not support presentation")

        surface_ptr = raw_ffi.new("VkSurfaceKHR[1]")
        result = glfw.create_window_surface(
            instance=self._gpu_context.vk_instance,
            window=self._glfw_window_handle,
            allocator=None,
            surface=surface_ptr,
        )
        if result != 0:
            raise RuntimeError(f"Failed to create window surface: VkResult: {result}")
        width, height = glfw.get_framebuffer_size(self._glfw_window_handle)
        return GpuSurface(
            context=self._gpu_context,
            parent_resource=self,
            vk_surface=surface_ptr[0],
            width=width,
            height=height,
        )

    def _recreate_gpu_surface(self) -> None:
        """Recreate the GPU surface after a window resize event."""
        self._gpu_surface.dispose()
        self._gpu_surface = self._new_gpu_surface()

    #
    # Resource disposal:
    #

    def _on_dispose(self) -> None:
        super()._on_dispose()
        glfw.destroy_window(self._glfw_window_handle)

    #
    # Properties:
    #

    @property
    def width_dip(self) -> int:
        return self._width_dip

    @property
    def height_dip(self) -> int:
        return self._height_dip

    @property
    def width_px(self) -> int:
        return self._width_px

    @property
    def height_px(self) -> int:
        return self._height_px

    @property
    def gpu_surface(self) -> GpuSurface:
        return self._gpu_surface

    @property
    def glfw_window(self) -> glfw._GLFWwindow:
        return self._glfw_window_handle

    @property
    def content_scale(self) -> tuple[float, float]:
        return glfw.get_window_content_scale(self._glfw_window_handle)

    #
    # Window state management:
    #

    def should_close(self) -> bool:
        return glfw.window_should_close(self._glfw_window_handle)

    def show(self) -> None:
        glfw.show_window(self._glfw_window_handle)

    def hide(self) -> None:
        glfw.hide_window(self._glfw_window_handle)

    def set_cursor_mode(self, mode: str) -> None:
        """
        Sets the mouse input mode for the window.
        - "cursor": cursor input, mouse movement handled by the OS.
        - "joystick": cursor hidden, mouse movement captured by the window.
        """
        match mode:
            case "joystick":
                glfw.set_input_mode(
                    self._glfw_window_handle,
                    glfw.CURSOR,
                    glfw.CURSOR_DISABLED,
                )
            case "cursor":
                glfw.set_input_mode(
                    self._glfw_window_handle,
                    glfw.CURSOR,
                    glfw.CURSOR_NORMAL,
                )
            case _:
                raise ValueError(f"Invalid cursor mode: {mode!r}")

    @staticmethod
    def poll_events() -> None:
        glfw.poll_events()

    #
    # Callback setters:
    #

    def set_key_callback(self, callback: KeyCallback | None) -> None:
        self._key_callback = callback

    def set_mouse_button_callback(self, callback: MouseButtonCallback | None) -> None:
        self._mouse_button_callback = callback

    def set_cursor_pos_callback(self, callback: CursorPosCallback | None) -> None:
        self._cursor_pos_callback = callback

    def set_framebuffer_size_callback(
        self, callback: FramebufferSizeCallback | None
    ) -> None:
        self._framebuffer_size_callback = callback

    #
    # GLFW event handlers:
    #

    def _on_glfw_key_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        key: int,
        scancode: int,
        action: int,
        mods: int,
    ) -> None:
        if self._key_callback is None:
            return
        decoded_key = _decode_glfw_key(key)
        decoded_action = _decode_glfw_action(action)
        decoded_mods = _decode_glfw_mods(mods)
        self._key_callback(decoded_key, scancode, decoded_action, decoded_mods)

    def _on_glfw_mouse_button_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        glfw_button: int,
        glfw_action: int,
        mods: int,
    ) -> None:
        if self._mouse_button_callback is None:
            return
        button = _decode_glfw_mouse_button(glfw_button)
        action = _decode_glfw_action(glfw_action)
        decoded_mods = _decode_glfw_mods(mods)
        self._mouse_button_callback(button, action, decoded_mods)

    def _on_glfw_cursor_pos_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        x: float,
        y: float,
    ) -> None:
        dx, self._last_mouse_x = x - self._last_mouse_x, x
        dy, self._last_mouse_y = y - self._last_mouse_y, y

        _ = dx, dy  # Currently unused

        if self._cursor_pos_callback is not None:
            self._cursor_pos_callback(x, y)

    def _on_framebuffer_resize_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        width_px: int,
        height_px: int,
    ) -> None:
        xs, ys = glfw.get_window_content_scale(self._glfw_window_handle)
        self._width_dip = int(round(width_px / xs))
        self._height_dip = int(round(height_px / ys))
        self._width_px = width_px
        self._height_px = height_px

        if self._framebuffer_size_callback is not None:
            self._framebuffer_size_callback(width_px, height_px)

    def handle_resize_for_gpu_surface(self) -> bool:
        """
        Check if the window was resized and recreate the GPU surface if needed.

        Returns True if a resize occurred and the surface was recreated, False otherwise.
        """
        current_width = self._gpu_surface.width
        current_height = self._gpu_surface.height
        framebuffer_width = self._width_px
        framebuffer_height = self._height_px

        # Ignore resize if dimensions are zero (window minimized or not yet sized)
        if framebuffer_width <= 0 or framebuffer_height <= 0:
            return False

        # Check if size has changed
        if current_width != framebuffer_width or current_height != framebuffer_height:
            self._recreate_gpu_surface()
            return True

        return False


#
# GLFW decoding helpers:
#


def _decode_glfw_action(action: int) -> ButtonAction:
    d: dict[int, ButtonAction] = {
        glfw.PRESS: "press",
        glfw.RELEASE: "release",
        glfw.REPEAT: "repeat",
    }
    return d[action]


def _decode_glfw_mods(mods: int) -> list[KeyModifier]:
    result: list[KeyModifier] = []
    if mods & glfw.MOD_SHIFT:
        result.append("shift")
    if mods & glfw.MOD_CONTROL:
        result.append("control")
    if mods & glfw.MOD_ALT:
        result.append("alt")
    if mods & glfw.MOD_SUPER:
        result.append("super")
    return result


def _decode_glfw_key(key: int) -> Key | None:
    glfw_key_map: dict[int, Key] = {
        # Printable keys (US layout)
        glfw.KEY_SPACE: "space",
        glfw.KEY_APOSTROPHE: "apostrophe",
        glfw.KEY_COMMA: "comma",
        glfw.KEY_MINUS: "minus",
        glfw.KEY_PERIOD: "period",
        glfw.KEY_SLASH: "slash",
        glfw.KEY_0: "0",
        glfw.KEY_1: "1",
        glfw.KEY_2: "2",
        glfw.KEY_3: "3",
        glfw.KEY_4: "4",
        glfw.KEY_5: "5",
        glfw.KEY_6: "6",
        glfw.KEY_7: "7",
        glfw.KEY_8: "8",
        glfw.KEY_9: "9",
        glfw.KEY_SEMICOLON: "semicolon",
        glfw.KEY_EQUAL: "equal",
        glfw.KEY_A: "a",
        glfw.KEY_B: "b",
        glfw.KEY_C: "c",
        glfw.KEY_D: "d",
        glfw.KEY_E: "e",
        glfw.KEY_F: "f",
        glfw.KEY_G: "g",
        glfw.KEY_H: "h",
        glfw.KEY_I: "i",
        glfw.KEY_J: "j",
        glfw.KEY_K: "k",
        glfw.KEY_L: "l",
        glfw.KEY_M: "m",
        glfw.KEY_N: "n",
        glfw.KEY_O: "o",
        glfw.KEY_P: "p",
        glfw.KEY_Q: "q",
        glfw.KEY_R: "r",
        glfw.KEY_S: "s",
        glfw.KEY_T: "t",
        glfw.KEY_U: "u",
        glfw.KEY_V: "v",
        glfw.KEY_W: "w",
        glfw.KEY_X: "x",
        glfw.KEY_Y: "y",
        glfw.KEY_Z: "z",
        glfw.KEY_LEFT_BRACKET: "left-bracket",
        glfw.KEY_BACKSLASH: "backslash",
        glfw.KEY_RIGHT_BRACKET: "right-bracket",
        glfw.KEY_GRAVE_ACCENT: "grave-accent",
        glfw.KEY_WORLD_1: "world-1",
        glfw.KEY_WORLD_2: "world-2",
        # Function keys and special keys
        glfw.KEY_ESCAPE: "escape",
        glfw.KEY_ENTER: "enter",
        glfw.KEY_TAB: "tab",
        glfw.KEY_BACKSPACE: "backspace",
        glfw.KEY_INSERT: "insert",
        glfw.KEY_DELETE: "delete",
        glfw.KEY_RIGHT: "right",
        glfw.KEY_LEFT: "left",
        glfw.KEY_DOWN: "down",
        glfw.KEY_UP: "up",
        glfw.KEY_PAGE_UP: "page-up",
        glfw.KEY_PAGE_DOWN: "page-down",
        glfw.KEY_HOME: "home",
        glfw.KEY_END: "end",
        glfw.KEY_CAPS_LOCK: "caps-lock",
        glfw.KEY_SCROLL_LOCK: "scroll-lock",
        glfw.KEY_NUM_LOCK: "num-lock",
        glfw.KEY_PRINT_SCREEN: "print-screen",
        glfw.KEY_PAUSE: "pause",
        glfw.KEY_F1: "f1",
        glfw.KEY_F2: "f2",
        glfw.KEY_F3: "f3",
        glfw.KEY_F4: "f4",
        glfw.KEY_F5: "f5",
        glfw.KEY_F6: "f6",
        glfw.KEY_F7: "f7",
        glfw.KEY_F8: "f8",
        glfw.KEY_F9: "f9",
        glfw.KEY_F10: "f10",
        glfw.KEY_F11: "f11",
        glfw.KEY_F12: "f12",
        glfw.KEY_F13: "f13",
        glfw.KEY_F14: "f14",
        glfw.KEY_F15: "f15",
        glfw.KEY_F16: "f16",
        glfw.KEY_F17: "f17",
        glfw.KEY_F18: "f18",
        glfw.KEY_F19: "f19",
        glfw.KEY_F20: "f20",
        glfw.KEY_F21: "f21",
        glfw.KEY_F22: "f22",
        glfw.KEY_F23: "f23",
        glfw.KEY_F24: "f24",
        glfw.KEY_F25: "f25",
        # Keypad keys
        glfw.KEY_KP_0: "kp-0",
        glfw.KEY_KP_1: "kp-1",
        glfw.KEY_KP_2: "kp-2",
        glfw.KEY_KP_3: "kp-3",
        glfw.KEY_KP_4: "kp-4",
        glfw.KEY_KP_5: "kp-5",
        glfw.KEY_KP_6: "kp-6",
        glfw.KEY_KP_7: "kp-7",
        glfw.KEY_KP_8: "kp-8",
        glfw.KEY_KP_9: "kp-9",
        glfw.KEY_KP_DECIMAL: "kp-decimal",
        glfw.KEY_KP_DIVIDE: "kp-divide",
        glfw.KEY_KP_MULTIPLY: "kp-multiply",
        glfw.KEY_KP_SUBTRACT: "kp-subtract",
        glfw.KEY_KP_ADD: "kp-add",
        glfw.KEY_KP_ENTER: "kp-enter",
        glfw.KEY_KP_EQUAL: "kp-equal",
        # Modifier keys
        glfw.KEY_LEFT_SHIFT: "left-shift",
        glfw.KEY_LEFT_CONTROL: "left-control",
        glfw.KEY_LEFT_ALT: "left-alt",
        glfw.KEY_LEFT_SUPER: "left-super",
        glfw.KEY_RIGHT_SHIFT: "right-shift",
        glfw.KEY_RIGHT_CONTROL: "right-control",
        glfw.KEY_RIGHT_ALT: "right-alt",
        glfw.KEY_RIGHT_SUPER: "right-super",
        glfw.KEY_MENU: "menu",
    }
    return glfw_key_map.get(key, None)


def _decode_glfw_mouse_button(button: int) -> MouseButton:
    glfw_mouse_button_map: dict[int, MouseButton] = {
        glfw.MOUSE_BUTTON_1: "left",
        glfw.MOUSE_BUTTON_2: "right",
        glfw.MOUSE_BUTTON_3: "middle",
        glfw.MOUSE_BUTTON_4: "button-4",
        glfw.MOUSE_BUTTON_5: "button-5",
        glfw.MOUSE_BUTTON_6: "button-6",
        glfw.MOUSE_BUTTON_7: "button-7",
        glfw.MOUSE_BUTTON_8: "button-8",
    }
    return glfw_mouse_button_map[button]
