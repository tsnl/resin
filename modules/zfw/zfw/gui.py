"""
GUI widgets and window management.

Each widget...
- Is an event router for GuiEvents: it can receive and publish GuiEvents.
- Has an axis-aligned position and size in device-independent pixels (DIP) relative to
  its parent widget.
- Can render itself on a Canvas.
- May have child widgets that extend outside its bounds.

Widget stacking order:
- parent always below children
- among siblings, later added always above earlier added
"""

__all__ = [
    "DEFAULT_THEME",
    "GuiContext",
    "GuiCursorMode",
    "GuiTheme",
    "GuiWidget",
    "GuiWidgetStyle",
    "GuiWindow",
]

from dataclasses import dataclass
from typing import Literal
import warnings

from cassowary import SimplexSolver, Variable, STRONG

import glfw

from .basic import (
    BaseResource,
    MouseButton,
    ButtonAction,
    Font,
    KeyModifier,
    Key,
    HorizontalAlignment,
    VerticalAlignment,
    JsonObject,
    LogicError,
)
from .renderer import Canvas, RendererImage
from .excepts import GlfwError
from .gpu import GpuContext, GpuSurface
from .typed_vulkan import raw_ffi


@dataclass
class GuiWidgetStyle:
    font: Font = "sans-serif"
    font_size_dip: int = 14
    font_weight: int = 400
    bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    bg_image: RendererImage | None = None
    bg_hover_color: tuple[float, float, float, float] | None = None
    bg_hover_image: RendererImage | None = None
    fg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    fg_hover_color: tuple[float, float, float, float] | None = None
    border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    border_thickness: tuple[int, int, int, int] = (0, 0, 0, 0)
    hover_border_color: tuple[float, float, float, float] | None = None
    hover_border_thickness: tuple[int, int, int, int] | None = None
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    text_horizontal_alignment: HorizontalAlignment = "center"
    text_vertical_alignment: VerticalAlignment = "middle"
    wrap: bool = False


type GuiTheme = dict[str, JsonObject]


DEFAULT_THEME: GuiTheme = {
    "label": {
        "bg_color": (0.925, 0.925, 0.925, 1.0),  # Light gray background (Windows XP)
        "fg_color": (0.0, 0.0, 0.0, 1.0),  # Black text
    },
    "button": {
        "bg_color": (0.85, 0.87, 0.92, 1.0),  # Light blue-gray (Windows XP button)
        "fg_color": (0.0, 0.0, 0.0, 1.0),  # Black text
        "bg_hover_color": (0.78, 0.84, 0.95, 1.0),  # Lighter blue on hover
        "border_color": (0.0, 0.33, 0.65, 1.0),  # Windows XP blue border
        "border_thickness": (1, 1, 1, 1),
        "hover_border_color": (0.0, 0.45, 0.85, 1.0),  # Brighter blue on hover
        "hover_border_thickness": (1, 1, 1, 1),
        "padding": (5, 10, 5, 10),
    },
    "h1": {
        "font_size_dip": 32,
        "font_weight": 1000,
        "fg_color": (1.0, 1.0, 1.0, 1.0),  # White text
        "bg_color": (0.0, 0.33, 0.65, 1.0),  # Windows XP title bar blue
        "border_color": (0.0, 0.2, 0.5, 1.0),
        "border_thickness": (0, 0, 2, 0),
    },
    "h2": {
        "font_size_dip": 24,
        "font_weight": 800,
        "fg_color": (0.0, 0.0, 0.0, 1.0),
    },
}


def _eval_style(theme: GuiTheme, class_names: list[str]) -> GuiWidgetStyle:
    for class_name in class_names:
        if class_name not in theme:
            raise LogicError(f"Style class name not found in theme: {class_name!r}")

    d = {}
    for class_name in class_names:
        style_data = theme.get(class_name)
        if style_data is None:
            raise LogicError(f"Style class name not found in theme: {class_name!r}")
        d |= style_data
    return GuiWidgetStyle(**d)


class GuiContext(BaseResource):
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


type GuiCursorMode = Literal["cursor", "joystick"]


class GuiWindow(BaseResource):
    width: int
    height: int
    title: str
    theme: GuiTheme
    glfw_window_handle: glfw._GLFWwindow
    gpu_surface: GpuSurface
    last_mouse_x: float
    last_mouse_y: float
    _gui_context: GuiContext
    _central_widget: "GuiWidget"

    def __init__(
        self,
        *,
        gui_context: GuiContext,
        width: int,
        height: int,
        title: str,
        theme: GuiTheme | None = None,
        num_grid_rows: int = 1,
        num_grid_cols: int = 1,
        grid_row_sizes: tuple[int, ...] | None = None,
        grid_col_sizes: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(parent_resource=gui_context)

        self._gui_context = gui_context
        self.width = width
        self.height = height
        self.title = title
        self.theme = theme or DEFAULT_THEME

        self.glfw_window_handle = self._new_glfw_window()
        self.gpu_surface = self._new_gpu_surface()

        self.last_mouse_x: float = 0.0
        self.last_mouse_y: float = 0.0

        # Create central widget that occupies the full window
        self._central_widget = GuiWidget(
            parent_widget=None,
            gui_context=gui_context,
            gui_window=self,
            local_xywh_dip=(0, 0, width, height),
            theme=self.theme,
            num_child_grid_rows=num_grid_rows,
            num_child_grid_cols=num_grid_cols,
            child_grid_row_size=grid_row_sizes,
            child_grid_col_size=grid_col_sizes,
            parent_resource=self,
        )

    def _new_glfw_window(self) -> glfw._GLFWwindow:
        # Create GLFW window:
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

        # If raw mouse motion is supported, enable it by default.
        if glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(
                glfw_window,
                glfw.RAW_MOUSE_MOTION,
                glfw.TRUE,
            )
        else:
            warnings.warn(
                "Raw mouse motion is not supported on this system: 'joystick' cursor "
                "mode may be less accurate.",
                category=RuntimeWarning,
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

        # Return the created GLFW window handle
        return glfw_window

    def _new_gpu_surface(self) -> GpuSurface:
        if not self._gui_context.gpu_context.enable_present_support:
            raise RuntimeError("GPU context does not support presentation")

        surface_ptr = raw_ffi.new("VkSurfaceKHR[1]")
        result = glfw.create_window_surface(
            instance=self._gui_context.gpu_context.vk_instance,
            window=self.glfw_window_handle,
            allocator=None,
            surface=surface_ptr,
        )
        if result != 0:
            raise RuntimeError(f"Failed to create window surface: VkResult: {result}")
        width, height = glfw.get_framebuffer_size(self.glfw_window_handle)
        return GpuSurface(
            context=self._gui_context.gpu_context,
            parent_resource=self,
            vk_surface=surface_ptr[0],
            width=width,
            height=height,
        )

    def _on_dispose_resource(self) -> None:
        super()._on_dispose_resource()
        glfw.destroy_window(self.glfw_window_handle)

    def should_close(self) -> bool:
        return glfw.window_should_close(self.glfw_window_handle)

    def show(self):
        glfw.show_window(self.glfw_window_handle)

    def hide(self):
        glfw.hide_window(self.glfw_window_handle)

    def set_cursor_mode(self, cursor_mode: "GuiCursorMode"):
        """
        Sets the mouse input mode for the window.
        - "cursor": cursor input, mouse movement handled by the OS.
        - "joystick": cursor hidden, mouse movement captured by the window.
        """
        match cursor_mode:
            case "joystick":
                glfw.set_input_mode(
                    self.glfw_window_handle,
                    glfw.CURSOR,
                    glfw.CURSOR_DISABLED,
                )
            case "cursor":
                glfw.set_input_mode(
                    self.glfw_window_handle,
                    glfw.CURSOR,
                    glfw.CURSOR_NORMAL,
                )
            case _:
                raise ValueError(f"Invalid cursor mode: {cursor_mode!r}")

    @property
    def content_scale(self) -> tuple[float, float]:
        return glfw.get_window_content_scale(self.glfw_window_handle)

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
        # TODO: Handle key events in GuiWidget if needed
        pass

    def _on_glfw_mouse_button_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        button: int,
        action: int,
        mods: int,
    ) -> None:
        self._central_widget._receive_mouse_button_action(
            button=_decode_glfw_mouse_button(button),
            action=_decode_glfw_action(action),
        )

    def _on_glfw_cursor_pos_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        x: float,
        y: float,
    ) -> None:
        dx, self.last_mouse_x = x - self.last_mouse_x, x
        dy, self.last_mouse_y = y - self.last_mouse_y, y

        _ = dx, dy  # Currently unused

        self._central_widget._receive_mouse_position(
            mouse_x_dip=int(round(x)),
            mouse_y_dip=int(round(y)),
        )

    def render(self, canvas: Canvas) -> None:
        self._central_widget.update_layout()
        self._central_widget._render(canvas)

    @property
    def central_widget(self) -> "GuiWidget":
        """Get the central widget that occupies the full window area."""
        return self._central_widget


class GuiWidget(BaseResource):
    _parent_widget: "GuiWidget | None"
    _gui_window: "GuiWindow | None"
    _gui_context: "GuiContext"
    _local_xywh_dip: tuple[int, int, int, int]
    _child_widget_list: list["GuiWidget"]
    _mouse_over: bool
    _latest_local_mouse_pos: tuple[int, int]
    _latest_global_mouse_pos: tuple[int, int]

    # Layout params (position in parent)
    _layout_row: int
    _layout_col: int
    _layout_row_span: int
    _layout_col_span: int

    # Grid params (for children)
    _num_child_grid_rows: int
    _num_child_grid_cols: int
    _child_grid_row_size: tuple[int, ...]
    _child_grid_col_size: tuple[int, ...]

    # Content:
    _text: str
    _style_classes: list[str]
    _is_enabled: bool
    _theme: GuiTheme

    def __init__(
        self,
        *,
        parent_widget: "GuiWidget | None",
        gui_context: GuiContext | None = None,
        gui_window: "GuiWindow | None" = None,
        local_xywh_dip: tuple[int, int, int, int] | None = None,
        theme: GuiTheme | None = None,
        row: int = 0,
        col: int = 0,
        row_span: int = 1,
        col_span: int = 1,
        num_child_grid_rows: int = 1,
        num_child_grid_cols: int = 1,
        child_grid_row_size: tuple[int, ...] | None = None,
        child_grid_col_size: tuple[int, ...] | None = None,
        text: str = "",
        style_classes: list[str] | None = None,
        is_enabled: bool = True,
        parent_resource: "BaseResource | None" = None,
    ) -> None:
        super().__init__(parent_resource=(parent_resource or parent_widget))

        self._parent_widget = parent_widget
        self._gui_window = (
            gui_window if parent_widget is None else parent_widget._gui_window
        )

        # Infer gui_context from parent if not provided
        if gui_context is None:
            if parent_widget is not None:
                gui_context = parent_widget._gui_context
            else:
                raise ValueError(
                    "gui_context must be provided when parent_widget is None"
                )
        self._gui_context = gui_context
        self._local_xywh_dip = local_xywh_dip or (0, 0, 0, 0)
        self._child_widget_list = []
        self._mouse_over = False
        self._latest_local_mouse_pos = (0, 0)
        self._latest_global_mouse_pos = (0, 0)

        self._layout_row = row
        self._layout_col = col
        self._layout_row_span = row_span
        self._layout_col_span = col_span

        self._num_child_grid_rows = num_child_grid_rows
        self._num_child_grid_cols = num_child_grid_cols
        self._child_grid_row_size = child_grid_row_size or tuple(
            [-1] * num_child_grid_rows
        )
        self._child_grid_col_size = child_grid_col_size or tuple(
            [-1] * num_child_grid_cols
        )

        # Inherit theme from parent widget, or use provided theme
        if parent_widget is not None:
            self._theme = parent_widget._theme
        elif theme is not None:
            self._theme = theme
        else:
            self._theme = DEFAULT_THEME

        self._text = text
        self._style_classes = style_classes or ["label"]
        self._cached_style = _eval_style(self._theme, self._style_classes)
        self._is_enabled = is_enabled

        if self._parent_widget is not None:
            self._parent_widget._add_child_widget(self)

    def set_grid_config(
        self,
        num_rows: int,
        num_cols: int,
        row_sizes: tuple[int, ...] | None = None,
        col_sizes: tuple[int, ...] | None = None,
    ) -> None:
        self._num_child_grid_rows = num_rows
        self._num_child_grid_cols = num_cols
        self._child_grid_row_size = row_sizes or tuple([-1] * num_rows)
        self._child_grid_col_size = col_sizes or tuple([-1] * num_cols)

    def update_layout(self) -> None:
        if self._child_widget_list:
            self._solve_layout()

        for child in self._child_widget_list:
            child.update_layout()

    def _solve_layout(self) -> None:
        solver = SimplexSolver()

        # Variables for grid lines
        row_vars = [Variable(f"row_{i}") for i in range(self._num_child_grid_rows + 1)]
        col_vars = [Variable(f"col_{i}") for i in range(self._num_child_grid_cols + 1)]

        # Unit size for stretch
        unit_w = Variable("unit_w")
        unit_h = Variable("unit_h")

        # Constraints
        _, _, w, h = self._local_xywh_dip

        # Boundaries
        solver.add_constraint(row_vars[0] == 0)
        solver.add_constraint(row_vars[self._num_child_grid_rows] == h)
        solver.add_constraint(col_vars[0] == 0)
        solver.add_constraint(col_vars[self._num_child_grid_cols] == w)

        # Ordering
        for i in range(self._num_child_grid_rows):
            solver.add_constraint(row_vars[i + 1] >= row_vars[i])
        for i in range(self._num_child_grid_cols):
            solver.add_constraint(col_vars[i + 1] >= col_vars[i])

        # Row sizes
        for i in range(self._num_child_grid_rows):
            size = (
                self._child_grid_row_size[i]
                if i < len(self._child_grid_row_size)
                else -1
            )
            if size >= 0:
                solver.add_constraint(
                    row_vars[i + 1] - row_vars[i] == size,
                    strength=STRONG,
                )
            else:
                weight = -size
                solver.add_constraint(
                    row_vars[i + 1] - row_vars[i] == weight * unit_h,
                    strength=STRONG,
                )

        # Col sizes
        for i in range(self._num_child_grid_cols):
            size = (
                self._child_grid_col_size[i]
                if i < len(self._child_grid_col_size)
                else -1
            )
            if size >= 0:
                solver.add_constraint(
                    col_vars[i + 1] - col_vars[i] == size,
                    strength=STRONG,
                )
            else:
                weight = -size
                solver.add_constraint(
                    col_vars[i + 1] - col_vars[i] == weight * unit_w,
                    strength=STRONG,
                )

        # Update children
        for child in self._child_widget_list:
            r = child._layout_row
            c = child._layout_col
            rs = child._layout_row_span
            cs = child._layout_col_span

            # Clamp to grid
            r = max(0, min(r, self._num_child_grid_rows - 1))
            c = max(0, min(c, self._num_child_grid_cols - 1))
            rs = max(1, min(rs, self._num_child_grid_rows - r))
            cs = max(1, min(cs, self._num_child_grid_cols - c))

            y1 = row_vars[r].value
            y2 = row_vars[r + rs].value
            x1 = col_vars[c].value
            x2 = col_vars[c + cs].value

            child._local_xywh_dip = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def _add_child_widget(self, child_widget: "GuiWidget") -> None:
        self._child_widget_list.append(child_widget)

    @property
    def mouse_over(self) -> bool:
        return self._mouse_over

    @property
    def local_xywh_dip(self) -> tuple[int, int, int, int]:
        return self._local_xywh_dip

    @property
    def global_xywh_dip(self) -> tuple[int, int, int, int]:
        if self._parent_widget is None:
            return self._local_xywh_dip
        else:
            parent_x, parent_y, _, _ = self._parent_widget.global_xywh_dip
            x, y, w, h = self._local_xywh_dip
            return (parent_x + x, parent_y + y, w, h)

    def _receive_mouse_position(self, mouse_x_dip: int, mouse_y_dip: int):
        # OPTIMIZATION: early out if mouse position hasn't changed.
        if (mouse_x_dip, mouse_y_dip) == self._latest_global_mouse_pos:
            return

        # Update `self._mouse_over`
        is_over = self._intersect_point(mouse_x_dip, mouse_y_dip)
        if is_over != self._mouse_over:
            self._mouse_over = is_over
            self._on_mouse_over_changed(mouse_x_dip, mouse_y_dip)

        # If mouse is over, call `_on_mouse_move`.
        if self._mouse_over:
            self._on_mouse_move(mouse_x_dip, mouse_y_dip)

        # If mouse is over, update `self._local_mouse_pos`.
        if self._mouse_over:
            x, y, _, _ = self.global_xywh_dip
            self._latest_global_mouse_pos = (mouse_x_dip, mouse_y_dip)
            self._latest_local_mouse_pos = (mouse_x_dip - x, mouse_y_dip - y)

        # Regardless of whether mouse is over, propagate to children.
        # Children may be outside parent's bounds.
        for child in self._child_widget_list:
            child._receive_mouse_position(mouse_x_dip, mouse_y_dip)

    def _receive_mouse_button_action(
        self,
        button: MouseButton,
        action: ButtonAction,
    ) -> bool:
        # Only handle mouse button release events for now.
        if action != "release":
            return False

        # Propagate to children first (topmost first).
        for child in reversed(self._child_widget_list):
            if child._receive_mouse_button_action(button=button, action=action):
                return True
        else:
            # No child handled it, try to handle it ourselves.

            # Only handle if mouse is over.
            # This means a widget cannot be clicked if the mouse is outside its bounds,
            # even if it has children outside its bounds.
            if not self._mouse_over:
                return False

            # Invoke the click handler.
            return self._on_click(button=button)

    def _intersect_point(self, x: int, y: int) -> bool:
        rx, ry, rw, rh = self.global_xywh_dip
        return rx <= x < rx + rw and ry <= y < ry + rh

    def _on_mouse_over_changed(self, x_dip: int, y_dip: int) -> None:
        pass

    def _on_mouse_move(self, x_dip: int, y_dip: int) -> None:
        pass

    def _on_click(self, button: MouseButton) -> bool:
        return False

    def _render(self, canvas: Canvas) -> None:
        # Render self.
        self._render_self(canvas)

        # Render children, in order, after self.
        for child in self._child_widget_list:
            child._render(canvas)

    @staticmethod
    def _parse_style_class_names(raw: str | list[str]) -> list[str]:
        if isinstance(raw, list):
            return raw
        return [tag.strip() for tag in raw.split(" ") if tag.strip()]

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        self._is_enabled = value

    def _render_self(self, canvas: Canvas) -> None:
        style = self._cached_style

        x, y, w, h = self.global_xywh_dip

        is_hover = self.mouse_over and self.is_enabled

        # Determine colors
        bg_color = (
            style.bg_hover_color
            if (is_hover and style.bg_hover_color is not None)
            else style.bg_color
        )
        bg_image = (
            style.bg_hover_image
            if (is_hover and style.bg_hover_image is not None)
            else style.bg_image
        )

        border_color = (
            style.hover_border_color
            if (is_hover and style.hover_border_color is not None)
            else style.border_color
        )
        border_thickness = (
            style.hover_border_thickness
            if (is_hover and style.hover_border_thickness is not None)
            else style.border_thickness
        )

        fg_color = (
            style.fg_hover_color
            if (is_hover and style.fg_hover_color is not None)
            else style.fg_color
        )

        # Draw background quad:
        canvas.add_quad(
            dst_xy=(x, y),
            dst_wh=(w, h),
            color=bg_color,
            image=bg_image,
            border_color=border_color,
            border_thickness=border_thickness,
        )

        # Draw text:
        pt, pr, pb, pl = style.padding
        canvas.add_text(
            text=self._text,
            font=style.font,
            font_size_px=style.font_size_dip,
            font_weight=style.font_weight,
            dst_xy=(x + pl, y + pt),
            dst_wh=(w - pl - pr, h - pt - pb),
            color=fg_color,
            wrap=style.wrap,
            horizontal_alignment=style.text_horizontal_alignment,
            vertical_alignment=style.text_vertical_alignment,
        )


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
