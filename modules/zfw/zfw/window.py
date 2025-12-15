__all__ = ["Window"]

from dataclasses import dataclass
from typing import TypeAlias, Literal
import warnings

import glfw

from .basic import BaseResource, KeyAction, KeyModifier, Key
from .excepts import GlfwError
from .gpu import GpuContext, GpuSurface
from .typed_vulkan import raw_ffi
from .events import EventRouter, Event


class WindowContext(BaseResource):
    def __init__(
        self,
        *,
        gpu_context: GpuContext,
        parent_resource: BaseResource | None = None,
    ) -> None:
        super().__init__(parent_resource=parent_resource)

        self.gpu_context = gpu_context

        ok = glfw.init()
        if not ok:
            raise GlfwError("Failed to initialize GLFW")

    def _on_dispose_resource(self) -> None:
        glfw.terminate()


class Window(EventRouter, BaseResource):
    window_context: WindowContext
    window_width: int
    window_height: int
    window_title: str
    window_glfw_window_handle: glfw._GLFWwindow
    window_gpu_surface: GpuSurface

    def __init__(
        self,
        *,
        window_context: WindowContext,
        window_width: int,
        window_height: int,
        window_title: str,
    ) -> None:
        super().__init__(parent_resource=window_context)

        self.window_context = window_context
        self.window_width = window_width
        self.window_height = window_height
        self.window_title = window_title
        self.window_glfw_window_handle = self._new_glfw_window()
        self.window_gpu_surface = self._new_gpu_surface()

        self.last_mouse_x: float = 0.0
        self.last_mouse_y: float = 0.0

    def _new_glfw_window(self) -> glfw._GLFWwindow:
        # Create GLFW window:
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw_window = glfw.create_window(
            width=self.window_width,
            height=self.window_height,
            title=self.window_title,
            monitor=None,
            share=None,
        )
        if not glfw_window:
            raise GlfwError("Failed to create GLFW window")

        # If raw mouse motion is supported, enable it by default.
        # > If supported, raw mouse motion can be enabled or disabled per-window and at
        # > any time but it will only be provided when the cursor is disabled.
        # If raw mouse motion is supported, it should be enabled whenever the cursor
        # mode is set to "joystick".
        if glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(
                glfw_window,
                glfw.RAW_MOUSE_MOTION,
                glfw.TRUE,
            )
        else:
            warnings.warn(
                "Raw mouse motion is not supported on this system: 'joystick' cursor "
                "mode may be less accurate."
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

        return glfw_window

    def _new_gpu_surface(self) -> GpuSurface:
        if not self.window_context.gpu_context.enable_present_support:
            raise RuntimeError("GPU context does not support presentation")

        surface_ptr = raw_ffi.new("VkSurfaceKHR[1]")
        result = glfw.create_window_surface(
            instance=self.window_context.gpu_context.vk_instance,
            window=self.window_glfw_window_handle,
            allocator=None,
            surface=surface_ptr,
        )
        if result != 0:
            raise RuntimeError(f"Failed to create window surface: VkResult: {result}")
        width, height = glfw.get_framebuffer_size(self.window_glfw_window_handle)
        return GpuSurface(
            context=self.window_context.gpu_context,
            parent_resource=self,
            vk_surface=surface_ptr[0],
            width=width,
            height=height,
        )

    def _on_dispose_resource(self) -> None:
        glfw.destroy_window(self.window_glfw_window_handle)

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window_glfw_window_handle)

    def show(self):
        glfw.show_window(self.window_glfw_window_handle)

    def hide(self):
        glfw.hide_window(self.window_glfw_window_handle)

    def set_cursor_mode(self, cursor_mode: "WindowCursorMode"):
        """
        Sets the mouse input mode for the window.
        - "cursor": cursor-style input, mouse movement handled by the OS.
        - "joystick": cursor hidden, mouse movement captured by the window.

        :param self: Description
        :param mouse_input_mode: Description
        :type mouse_input_mode: "WindowMouseInputMode"
        """

        match cursor_mode:
            case "joystick":
                glfw.set_input_mode(
                    self.window_glfw_window_handle,
                    glfw.CURSOR,
                    glfw.CURSOR_DISABLED,
                )
            case "cursor":
                glfw.set_input_mode(
                    self.window_glfw_window_handle,
                    glfw.CURSOR,
                    glfw.CURSOR_NORMAL,
                )
            case _:
                raise ValueError(f"Invalid cursor mode: {cursor_mode!r}")

    @property
    def content_scale(self) -> tuple[float, float]:
        return glfw.get_window_content_scale(self.window_glfw_window_handle)

    @staticmethod
    def poll_events():
        glfw.poll_events()

    def _on_glfw_key_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        key: int,
        scancode: int,
        action: int,
        mods: int,
    ) -> None:
        self.publish(
            WindowKeyEvent(
                window=self,
                key=_decode_glfw_key(key),
                action=_decode_glfw_action(action),
                mods=_decode_glfw_mods(mods),
                raw_scancode=scancode,
            )
        )

    def _on_glfw_mouse_button_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        button: int,
        action: int,
        mods: int,
    ) -> None:
        self.publish(
            WindowMouseButtonEvent(
                window=self,
                button=_decode_glfw_mouse_button(button),
                action=_decode_glfw_action(action),
                mods=_decode_glfw_mods(mods),
            )
        )

    def _on_glfw_cursor_pos_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        x: float,
        y: float,
    ) -> None:
        dx, self.last_mouse_x = x - self.last_mouse_x, x
        dy, self.last_mouse_y = y - self.last_mouse_y, y

        self.publish(WindowCursorPosEvent(window=self, x=x, y=y, dx=dx, dy=dy))


WindowCursorMode: TypeAlias = Literal["cursor", "joystick"]


@dataclass
class WindowKeyEvent(Event):
    window: Window
    key: Key | None
    action: "KeyAction"
    mods: list["KeyModifier"]
    raw_scancode: int


@dataclass
class WindowMouseButtonEvent(Event):
    window: Window
    button: str
    action: "KeyAction"
    mods: list["KeyModifier"]


@dataclass
class WindowCursorPosEvent(Event):
    window: Window
    x: float
    y: float
    dx: float
    dy: float


def _decode_glfw_action(action: int) -> KeyAction:
    d: dict[int, KeyAction] = {
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
    # Map GLFW key codes to key names matching the Key literal type
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


def _decode_glfw_mouse_button(button: int) -> str:
    glfw_mouse_button_map: dict[int, str] = {
        glfw.MOUSE_BUTTON_1: "left",
        glfw.MOUSE_BUTTON_2: "right",
        glfw.MOUSE_BUTTON_3: "middle",
        glfw.MOUSE_BUTTON_4: "button-4",
        glfw.MOUSE_BUTTON_5: "button-5",
        glfw.MOUSE_BUTTON_6: "button-6",
        glfw.MOUSE_BUTTON_7: "button-7",
        glfw.MOUSE_BUTTON_8: "button-8",
    }
    return glfw_mouse_button_map.get(button, "button_" + str(button))
