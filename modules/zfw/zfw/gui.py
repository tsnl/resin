"""
GUI widgets and window management.

Each widget...
-   Is multipurpose: usable as label, button, container, etc based on styling.
-   Has a "state": default/hover/unclickable affecting style.
    Default is normal state and is a base for other state-dependent styles.
-   Contains a background image, text, borders, children in a grid layout.
-   CSS-like styling via style classes and overrides (hover, unclickable, etc).
-   Grid layout via constraint solver (kiwisolver).
-   Event processing loop order:
    1.  Update style based on previous state (hover, clicked, etc).
        The margin, border, and padding may affect the state (hover, clicked, etc) by
        affecting the bounding box of the widget or its layout children.
    2.  Update layout constraints based on style (margin, border, padding, etc). Solve.
        This ensures the widget and its children have up-to-date positions and sizes
        given the current layout.
    3.  Update state (hover, clicked, etc) using input events and style: mouse move,
        mouse button, key press, etc.
    4.  Render self and children.

Widget stacking order:
- parent always below children.
- among siblings, later added always above earlier added.
"""

__all__ = [
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

from kiwisolver import (
    Solver as KiwiSolver,
    Variable as KiwiVariable,
    Expression as KiwiExpression,
    Term as KiwiTerm,
)


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
from .gpu import GpuContext, GpuDevice, GpuSurface, GpuSwapChain
from .typed_vulkan import raw_ffi
from .events import EventHub


type GuiImageLayout = Literal["fit", "crop", "stretch"]


def _compute_image_src_xy_wh(
    dst_wh: tuple[int, int],
    image: RendererImage | None,
    layout: GuiImageLayout,
    user_src_xy: tuple[int, int] = (0, 0),
    user_src_wh: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """
    Compute the src_xy and src_wh to pass to Canvas.add_quad() based on the layout mode.

    Args:
        dst_wh: The destination widget size in pixels
        image: The image to layout, or None
        layout: The layout mode ("fit", "crop", or "stretch")
        user_src_xy: User-specified source rectangle origin (default: top-left)
        user_src_wh: User-specified source rectangle size, or None to use full image

    Returns:
        A tuple of (src_xy, src_wh) to pass to Canvas.add_quad()
    """
    if image is None:
        # No image: return defaults
        return user_src_xy, user_src_wh

    # Determine the source rectangle
    src_w = user_src_wh[0] if user_src_wh is not None else image.px_width
    src_h = user_src_wh[1] if user_src_wh is not None else image.px_height
    src_xy = user_src_xy
    src_wh = (src_w, src_h)

    dst_w, dst_h = dst_wh

    if layout == "stretch":
        # Stretch: use the source rectangle as-is
        return src_xy, src_wh
    elif layout == "fit":
        # Fit: scale the source rectangle so the entire image fits within the destination
        # We adjust src_wh to match the aspect ratio of dst_wh
        src_aspect = src_w / src_h
        dst_aspect = dst_w / dst_h

        if src_aspect > dst_aspect:
            # Source is wider: limit by destination width
            new_src_w = int(src_h * dst_aspect)
            offset = (src_w - new_src_w) // 2
            return (src_xy[0] + offset, src_xy[1]), (new_src_w, src_h)
        else:
            # Source is taller: limit by destination height
            new_src_h = int(src_w / dst_aspect)
            offset = (src_h - new_src_h) // 2
            return (src_xy[0], src_xy[1] + offset), (src_w, new_src_h)
    elif layout == "crop":
        # Crop: trim the minimum to fit the center
        src_aspect = src_w / src_h
        dst_aspect = dst_w / dst_h

        if src_aspect > dst_aspect:
            # Source is wider: crop left and right
            new_src_w = int(src_h * dst_aspect)
            offset = (src_w - new_src_w) // 2
            return (src_xy[0] + offset, src_xy[1]), (new_src_w, src_h)
        else:
            # Source is taller: crop top and bottom
            new_src_h = int(src_w / dst_aspect)
            offset = (src_h - new_src_h) // 2
            return (src_xy[0], src_xy[1] + offset), (src_w, new_src_h)
    else:
        raise ValueError(f"Invalid image layout: {layout}")


@dataclass
class GuiWidgetStyle:
    font: Font = "sans-serif"
    font_size_dip: int = 14
    font_weight: int = 400
    bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    fg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    border_thickness: tuple[int, int, int, int] = (0, 0, 0, 0)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    margin: tuple[int, int, int, int] = (0, 0, 0, 0)
    text_horizontal_alignment: HorizontalAlignment = "center"
    text_vertical_alignment: VerticalAlignment = "middle"
    wrap: bool = False
    image_layout: GuiImageLayout = "fit"


type GuiWidgetState = Literal["default", "hover", "unclickable", "pressed", "cancelled"]
type GuiTheme = dict[str, dict[GuiWidgetState, JsonObject]]


_DEFAULT_THEME: GuiTheme = {
    "central": {
        "default": {
            "bg_color": (
                0.925,
                0.925,
                0.925,
                1.0,
            ),  # Light gray background (Windows XP)
            "border_color": (0.0, 0.0, 0.0, 0.0),
            "border_thickness": (0, 0, 0, 0),
            "padding": (0, 0, 0, 0),
        }
    },
    "label": {
        "default": {
            "bg_color": (
                0.925,
                0.925,
                0.925,
                1.0,
            ),
            "fg_color": (0.0, 0.0, 0.0, 1.0),
        },
    },
    "button": {
        "default": {
            "bg_color": (0.85, 0.87, 0.92, 1.0),  # Light blue-gray (Windows XP button)
            "fg_color": (0.0, 0.0, 0.0, 1.0),  # Black text
            "border_color": (0.0, 0.33, 0.65, 1.0),  # Windows XP blue border
            "border_thickness": (1, 1, 1, 1),
            "padding": (5, 5, 5, 5),
            "margin": (10, 10, 10, 10),
        },
        "hover": {
            "bg_color": (0.78, 0.84, 0.95, 1.0),  # Lighter blue on hover
            "border_color": (0.0, 0.45, 0.85, 1.0),  # Brighter blue on hover
        },
        "pressed": {
            "bg_color": (0.4, 0.6, 0.85, 1.0),  # Dark blue when pressed
            "border_color": (0.0, 0.2, 0.5, 1.0),  # Darker blue border when pressed
        },
        "cancelled": {
            "bg_color": (0.75, 0.75, 0.75, 1.0),  # Grey when cancelled
            "border_color": (0.5, 0.5, 0.5, 1.0),  # Darker grey border when cancelled
        },
        "unclickable": {
            "fg_color": (0.35, 0.35, 0.35, 1.0),
            "bg_color": (0.82, 0.82, 0.82, 1.0),
            "border_color": (0.65, 0.65, 0.65, 1.0),
            "border_thickness": (1, 1, 1, 1),
        },
    },
    "h1": {
        "default": {
            "font_size_dip": 32,
            "font_weight": 1000,
            "fg_color": (1.0, 1.0, 1.0, 1.0),  # White text
            "bg_color": (0.0, 0.33, 0.65, 1.0),  # Windows XP title bar blue
        },
    },
    "h2": {
        "default": {
            "font_size_dip": 24,
            "font_weight": 800,
            "fg_color": (0.0, 0.0, 0.0, 1.0),
        },
    },
}


def _eval_theme(theme: GuiTheme, override: GuiTheme) -> GuiTheme:
    res = {}

    for class_name in set(theme.keys()) | set(override.keys()):
        res[class_name] = {}

        theme_dicts = theme.get(class_name, {})
        override_dicts = override.get(class_name, {})

        res[class_name] = {
            state_name: {
                **theme_dicts.get(state_name, {}),
                **override_dicts.get(state_name, {}),
            }
            for state_name in set(theme_dicts.keys()) | set(override_dicts.keys())
        }

    return res


def _eval_style(
    theme: GuiTheme,
    class_names: list[str],
    state: GuiWidgetState,
) -> GuiWidgetStyle:
    for class_name in class_names:
        if class_name not in theme:
            raise LogicError(f"Style class name not found in theme: {class_name!r}")

    d = {}
    for class_name in class_names:
        per_state_style_dicts = theme[class_name]
        d |= per_state_style_dicts.get("default", {})
        d |= per_state_style_dicts.get(state, {})

    return GuiWidgetStyle(**d)


class GuiContext(BaseResource):
    def __init__(
        self,
        *,
        gpu_context: GpuContext,
        parent_resource: BaseResource | None = None,
    ) -> None:
        super().__init__(parent_resource=parent_resource)

        self._gpu_context = gpu_context
        self._kiwi_solver = KiwiSolver()

        ok = glfw.init()
        if not ok:
            raise GlfwError("Failed to initialize GLFW")

    def _on_dispose_resource(self) -> None:
        glfw.terminate()


type GuiCursorMode = Literal["cursor", "joystick"]


class GuiWindow(BaseResource):
    _width_dip: int
    _height_dip: int
    _width_px: int
    _height_px: int
    _title: str
    _theme: GuiTheme
    _glfw_window_handle: glfw._GLFWwindow
    _gpu_surface: GpuSurface
    _gpu_device: GpuDevice | None
    _gpu_swap_chain: GpuSwapChain | None
    _swapchain_image_count: int
    _last_mouse_x: float
    _last_mouse_y: float
    _gui_context: GuiContext
    _central_widget: "GuiWidget | None"
    _central_widget_stack: list["GuiWidget"]
    _kiwi_solver: KiwiSolver

    def __init__(
        self,
        *,
        gui_context: GuiContext,
        width_dip: int,
        height_dip: int,
        title: str,
        resizable: bool = True,
        theme: GuiTheme | None = None,
    ) -> None:
        super().__init__(parent_resource=gui_context)

        self._gui_context = gui_context
        self._width_dip = width_dip
        self._height_dip = height_dip
        self._width_px = 0
        self._height_px = 0
        self._title = title
        self._resizable = resizable
        self._theme = theme or _DEFAULT_THEME

        self._glfw_window_handle = self._new_glfw_window()
        self._gpu_surface = self._new_gpu_surface()
        self._gpu_device = None
        self._gpu_swap_chain = None
        self._swapchain_image_count = 3

        self._last_mouse_x: float = 0.0
        self._last_mouse_y: float = 0.0

        # Create central widget that occupies the full window
        self._central_widget = None
        self._central_widget_stack = []

        # For Kiwi solver: window size variables
        self._w_var = KiwiVariable("window_width")
        self._h_var = KiwiVariable("window_height")

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
        glfw.set_framebuffer_size_callback(
            window=glfw_window,
            cbfun=self._on_framebuffer_resize_event,
        )

        # Update window sizes:
        width_px, height_px = glfw.get_framebuffer_size(glfw_window)
        self._width_px = width_px
        self._height_px = height_px

        # Return the created GLFW window handle
        return glfw_window

    def _new_gpu_surface(self) -> GpuSurface:
        if not self._gui_context._gpu_context.enable_present_support:
            raise RuntimeError("GPU context does not support presentation")

        surface_ptr = raw_ffi.new("VkSurfaceKHR[1]")
        result = glfw.create_window_surface(
            instance=self._gui_context._gpu_context.vk_instance,
            window=self._glfw_window_handle,
            allocator=None,
            surface=surface_ptr,
        )
        if result != 0:
            raise RuntimeError(f"Failed to create window surface: VkResult: {result}")
        width, height = glfw.get_framebuffer_size(self._glfw_window_handle)
        return GpuSurface(
            context=self._gui_context._gpu_context,
            parent_resource=self,
            vk_surface=surface_ptr[0],
            width=width,
            height=height,
        )

    def _recreate_gpu_surface(self) -> None:
        """Recreate the GPU surface after a window resize event."""
        # Dispose the old surface
        self._gpu_surface.dispose_resource()
        # Create a new surface with the current framebuffer size
        self._gpu_surface = self._new_gpu_surface()

    #
    # Resource disposal:
    #

    def _on_dispose_resource(self) -> None:
        if self._gpu_swap_chain is not None:
            self._gpu_swap_chain.dispose_resource()
        super()._on_dispose_resource()
        glfw.destroy_window(self._glfw_window_handle)

    #
    # Swapchain management, setting a GPU device:
    #

    def set_gpu_device(
        self,
        gpu_device: GpuDevice,
        swapchain_image_count: int = 3,
    ) -> None:
        """Set the GPU device and create the swapchain."""
        self._gpu_device = gpu_device
        self._swapchain_image_count = swapchain_image_count
        self._create_swapchain()

    def _create_swapchain(self) -> None:
        """Create the GPU swapchain."""
        if self._gpu_device is None:
            return

        # Dispose old swapchain if it exists
        if self._gpu_swap_chain is not None:
            self._gpu_device.wait_idle()
            self._gpu_swap_chain.dispose_resource()

        # Create new swapchain
        self._gpu_swap_chain = GpuSwapChain(
            device=self._gpu_device,
            surface=self._gpu_surface,
            image_count=self._swapchain_image_count,
        )

    #
    # Central widget management:
    #

    @property
    def central_widget(self) -> "GuiWidget":
        """Get the central widget that occupies the full window area."""
        assert self._central_widget is not None
        return self._central_widget

    def set_central_widget(self, widget: "GuiWidget") -> None:
        """Set the central widget that occupies the full window area."""
        self._central_widget = widget
        self.update_layout()

    def push_central_widget(self, widget: "GuiWidget") -> None:
        """Push a new central widget onto the stack."""
        if self._central_widget is not None:
            self._central_widget_stack.append(self._central_widget)
        self._central_widget = widget
        self.update_layout()

    def pop_central_widget(self) -> None:
        """Pop the current central widget and restore the previous one."""
        if self._central_widget_stack:
            self._central_widget = self._central_widget_stack.pop()
            self.update_layout()
        else:
            self._central_widget = None

    #
    # Window management:
    #

    def should_close(self) -> bool:
        return glfw.window_should_close(self._glfw_window_handle)

    def show(self):
        glfw.show_window(self._glfw_window_handle)

    def hide(self):
        glfw.hide_window(self._glfw_window_handle)

    def set_cursor_mode(self, cursor_mode: "GuiCursorMode"):
        """
        Sets the mouse input mode for the window.
        - "cursor": cursor input, mouse movement handled by the OS.
        - "joystick": cursor hidden, mouse movement captured by the window.
        """
        match cursor_mode:
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
                raise ValueError(f"Invalid cursor mode: {cursor_mode!r}")

    @property
    def content_scale(self) -> tuple[float, float]:
        return glfw.get_window_content_scale(self._glfw_window_handle)

    #
    # Phase 1: update style:
    #

    def update_style(self) -> None:
        if self._central_widget is None:
            return
        self._central_widget._update_style()

    #
    # Phase 2: update layout
    #

    def update_layout(self):
        if self._central_widget is None:
            return

        solver = self._gui_context._kiwi_solver

        solver.reset()

        solver.addEditVariable(self._w_var, "strong")
        solver.addEditVariable(self._h_var, "strong")
        solver.suggestValue(self._w_var, self._width_dip)
        solver.suggestValue(self._h_var, self._height_dip)

        self._central_widget._update_layout_constraints(
            solver,
            0.0,
            0.0,
            self._w_var,
            self._h_var,
        )

        solver.updateVariables()

    #
    # Phase 3: input events:
    #

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
        if not self._central_widget:
            return

        # TODO: Handle key events in GuiWidget if needed
        pass

    def _on_glfw_mouse_button_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        glfw_button: int,
        glfw_action: int,
        mods: int,
    ) -> None:
        if not self._central_widget:
            return

        button = _decode_glfw_mouse_button(glfw_button)
        action = _decode_glfw_action(glfw_action)

        self._central_widget._receive_mouse_button_action(
            button=button,
            action=action,
            click_handled=False,
        )

    def _on_glfw_cursor_pos_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        x: float,
        y: float,
    ) -> None:
        if not self._central_widget:
            return

        dx, self._last_mouse_x = x - self._last_mouse_x, x
        dy, self._last_mouse_y = y - self._last_mouse_y, y

        _ = dx, dy  # Currently unused

        self._central_widget._receive_mouse_position_change(
            mouse_x_dip=int(round(x)),
            mouse_y_dip=int(round(y)),
        )

    def _on_framebuffer_resize_event(
        self,
        _glfw_window_handle: glfw._GLFWwindow,
        width_px: int,
        height_px: int,
    ):
        xs, ys = glfw.get_window_content_scale(self._glfw_window_handle)
        self._width_dip = int(round(width_px / xs))
        self._height_dip = int(round(height_px / ys))
        self._width_px = width_px
        self._height_px = height_px
        self._handle_resize_for_gpu_surface()

    def _handle_resize_for_gpu_surface(self) -> bool:
        """
        Check if the window was resized and recreate the GPU surface if needed.

        Returns True if a resize occurred and the surface was recreated, False otherwise.
        """
        # Get current framebuffer size
        current_width = self._gpu_surface.width
        current_height = self._gpu_surface.height
        framebuffer_width = self._width_px
        framebuffer_height = self._height_px

        # Ignore resize if dimensions are zero (window minimized or not yet sized)
        if framebuffer_width <= 0 or framebuffer_height <= 0:
            return False

        # Check if size has changed
        if current_width != framebuffer_width or current_height != framebuffer_height:
            # Recreate the GPU surface with the new size
            self._recreate_gpu_surface()
            # Recreate the swapchain if device is set
            if self._gpu_device is not None:
                self._create_swapchain()
            return True

        return False

    #
    # Render:
    #

    def render(self, canvas: Canvas) -> None:
        if self._central_widget is None:
            return
        self._central_widget._render(canvas)


class GuiWidget(BaseResource):
    _parent_widget: "GuiWidget | None"
    _window: "GuiWindow"
    _gui_context: "GuiContext"
    _child_widget_list: list["GuiWidget"]
    _mouse_over: bool
    _latest_local_mouse_pos: tuple[int, int]
    _latest_global_mouse_pos: tuple[int, int]
    _mouse_button_pressed_locally: bool

    # Layout params (position in parent)
    _row: int
    _col: int
    _row_span: int
    _col_span: int

    # Grid params (for children)
    _grid_row_size_hints: tuple[int, ...]
    _grid_col_size_hints: tuple[int, ...]

    # Content:
    _text: str | None
    _image: RendererImage | None
    _image_src_xy: tuple[int, int]
    _image_src_wh: tuple[int, int] | None
    _image_layout: GuiImageLayout
    _image_hover: RendererImage | None
    _image_hover_src_xy: tuple[int, int]
    _image_hover_src_wh: tuple[int, int] | None
    _image_hover_layout: GuiImageLayout
    _style_classes: list[str]
    _clickable: bool
    _theme: GuiTheme

    # Events
    _click_event_hub: EventHub["MouseButton"]
    _mouse_over_changed_event_hub: EventHub[bool]
    _mouse_move_event_hub: EventHub[tuple[int, int]]

    # Bounding box in DIP (computed during layout)
    _x: KiwiVariable
    _y: KiwiVariable
    _w: KiwiVariable
    _h: KiwiVariable

    def __init__(
        self,
        *,
        parent_widget: "GuiWidget | None" = None,
        window: "GuiWindow | None" = None,
        theme: GuiTheme | None = None,
        row: int = 0,
        col: int = 0,
        row_span: int = 1,
        col_span: int = 1,
        grid_rows: tuple[int, ...] | None = None,
        grid_cols: tuple[int, ...] | None = None,
        text: str | None = None,
        image: RendererImage | None = None,
        image_src_xy: tuple[int, int] = (0, 0),
        image_src_wh: tuple[int, int] | None = None,
        image_layout: GuiImageLayout = "fit",
        image_hover: RendererImage | None = None,
        image_hover_src_xy: tuple[int, int] = (0, 0),
        image_hover_src_wh: tuple[int, int] | None = None,
        image_hover_layout: GuiImageLayout | None = None,
        style_classes: list[str] | None = None,
        clickable: bool = True,
        parent_resource: "BaseResource | None" = None,
    ) -> None:
        if not window and not parent_widget:
            raise ValueError("Either parent_window or parent_widget must be provided")
        if window and parent_widget:
            raise ValueError("Either parent_window or parent_widget should be provided")

        super().__init__(parent_resource=(parent_resource or parent_widget))

        self._parent_widget, self._window = GuiWidget._resolve_parent_widget_and_window(
            parent_widget=parent_widget,
            window=window,
        )
        self._gui_context = self._window._gui_context
        self._theme = GuiWidget._resolve_theme(theme=theme, parent_widget=parent_widget)

        self._child_widget_list = []
        self._mouse_over = False
        self._latest_local_mouse_pos = (0, 0)
        self._latest_global_mouse_pos = (0, 0)
        self._mouse_button_pressed_locally = False

        self._row = row
        self._col = col
        self._row_span = row_span
        self._col_span = col_span

        self._grid_row_size_hints = grid_rows or (-1,)
        self._grid_col_size_hints = grid_cols or (-1,)

        self._text = text
        self._image = image
        self._image_src_xy = image_src_xy
        self._image_src_wh = image_src_wh
        self._image_layout = image_layout
        self._image_hover = image_hover
        self._image_hover_src_xy = image_hover_src_xy
        self._image_hover_src_wh = image_hover_src_wh
        self._image_hover_layout = image_hover_layout or image_layout
        self._style_classes = style_classes or ["label"]
        self._style = _eval_style(self._theme, self._style_classes, "default")
        self._clickable = clickable

        if self._parent_widget is not None:
            self._parent_widget._add_child_widget(self)

        self._click_event_hub = EventHub["MouseButton"]()
        self._mouse_over_changed_event_hub = EventHub[bool]()
        self._mouse_move_event_hub = EventHub[tuple[int, int]]()

        # Layout variables:
        self._x = KiwiVariable(f"{repr(self)}::x")
        self._y = KiwiVariable(f"{repr(self)}::y")
        self._w = KiwiVariable(f"{repr(self)}::w")
        self._h = KiwiVariable(f"{repr(self)}::h")
        self._grid_row_unit_var = KiwiVariable(f"{repr(self)}::grid_row_unit")
        self._grid_col_unit_var = KiwiVariable(f"{repr(self)}::grid_col_unit")
        self._grid_row_size_vars: list[KiwiVariable] = [
            KiwiVariable(f"{repr(self)}::grid_row_{i}")
            for i in range(len(self._grid_row_size_hints))
        ]
        self._grid_col_size_vars: list[KiwiVariable] = [
            KiwiVariable(f"{repr(self)}::grid_col_{i}")
            for i in range(len(self._grid_col_size_hints))
        ]

    @staticmethod
    def _resolve_parent_widget_and_window(
        parent_widget: "GuiWidget | None",
        window: "GuiWindow | None",
    ) -> tuple["GuiWidget | None", "GuiWindow"]:
        if not window and not parent_widget:
            raise ValueError("Either parent_window or parent_widget must be provided")
        if window and parent_widget:
            raise ValueError("Either parent_window or parent_widget should be provided")
        if parent_widget:
            return parent_widget, parent_widget._window
        else:
            assert window is not None
            return None, window

    @staticmethod
    def _resolve_theme(
        theme: GuiTheme | None,
        parent_widget: "GuiWidget | None",
    ) -> GuiTheme:
        return _eval_theme(
            parent_widget._theme if parent_widget is not None else _DEFAULT_THEME,
            theme or {},
        )

    def _add_child_widget(self, child_widget: "GuiWidget") -> None:
        self._child_widget_list.append(child_widget)

    #
    # Dispose resources:
    #

    def _on_dispose_resource(self) -> None:
        for child in self._child_widget_list:
            child.dispose_resource()

    #
    # Event hubs:
    #

    @property
    def click_event(self) -> EventHub["MouseButton"]:
        return self._click_event_hub

    @property
    def mouse_over_changed_event(self) -> EventHub[bool]:
        return self._mouse_over_changed_event_hub

    #
    # Layout accessors:
    #

    @property
    def _xywh(self) -> tuple[int, int, int, int]:
        return (
            int(round(self._x.value())),
            int(round(self._y.value())),
            int(round(self._w.value())),
            int(round(self._h.value())),
        )

    #
    # Phase 1: update style based on previous state:
    #

    def _update_style(self) -> None:
        self._style = _eval_style(
            theme=self._theme,
            class_names=self._style_classes,
            state=self._compute_style_state(),
        )

        # Recursively update children
        for child in self._child_widget_list:
            child._update_style()

    def _compute_style_state(self) -> GuiWidgetState:
        if not self._clickable:
            return "unclickable"
        elif self._mouse_button_pressed_locally:
            if self._mouse_over:
                return "pressed"
            else:
                return "cancelled"
        elif self._mouse_over:
            return "hover"
        else:
            return "default"

    #
    # Phase 2: update layout constraints:
    #

    def _update_layout_constraints(
        self,
        solver: KiwiSolver,
        x: KiwiExpression | KiwiTerm | KiwiVariable | float,
        y: KiwiExpression | KiwiTerm | KiwiVariable | float,
        w: KiwiExpression | KiwiTerm | KiwiVariable | float,
        h: KiwiExpression | KiwiTerm | KiwiVariable | float,
    ) -> None:
        # Setup own position constraints:
        self._update_layout_constraints_for_xywh(solver=solver, x=x, y=y, w=w, h=h)

        # Setup grid layout constraints for children:
        self._update_layout_constraints_for_grid_dim(
            grid_hints=self._grid_row_size_hints,
            grid_vars=self._grid_row_size_vars,
            unit_var=self._grid_row_unit_var,
            total_var=self._h,
            solver=solver,
        )
        self._update_layout_constraints_for_grid_dim(
            grid_hints=self._grid_col_size_hints,
            grid_vars=self._grid_col_size_vars,
            unit_var=self._grid_col_unit_var,
            total_var=self._w,
            solver=solver,
        )

        # Setup children's constraints:
        self._update_layout_constraints_for_children(solver=solver)

    def _update_layout_constraints_for_xywh(
        self,
        solver: KiwiSolver,
        x: KiwiExpression | KiwiTerm | KiwiVariable | float,
        y: KiwiExpression | KiwiTerm | KiwiVariable | float,
        w: KiwiExpression | KiwiTerm | KiwiVariable | float,
        h: KiwiExpression | KiwiTerm | KiwiVariable | float,
    ) -> None:
        # Position constraints:
        solver.addConstraint(self._x == x)
        solver.addConstraint(self._y == y)

        # Size constraints:
        solver.addConstraint(self._w == w)
        solver.addConstraint(self._h == h)

    @staticmethod
    def _update_layout_constraints_for_grid_dim(
        grid_hints: tuple[int, ...],
        grid_vars: list[KiwiVariable],
        unit_var: KiwiVariable,
        total_var: KiwiVariable,
        solver: KiwiSolver,
    ) -> None:
        solver.addConstraint(unit_var >= 0)
        for grid_var, size_hint in zip(grid_vars, grid_hints):
            solver.addConstraint(grid_var >= 0)
            if size_hint < 0:
                solver.addConstraint(grid_var == -size_hint * unit_var)
            else:
                solver.addConstraint(grid_var == size_hint)

        solver.addConstraint(total_var == sum(grid_vars, 0.0))

    def _update_layout_constraints_for_children(self, solver: KiwiSolver) -> None:
        for child in self._child_widget_list:
            # Compute child's x, y, w, h based on grid layout:
            child_x = self._x + sum(
                self._grid_col_size_vars[i] for i in range(child._col)
            )
            child_y = self._y + sum(
                self._grid_row_size_vars[i] for i in range(child._row)
            )
            child_w = sum(
                (
                    self._grid_col_size_vars[i]
                    for i in range(child._col, child._col + child._col_span)
                ),
                0.0,
            )
            child_h = sum(
                (
                    self._grid_row_size_vars[i]
                    for i in range(child._row, child._row + child._row_span)
                ),
                0.0,
            )

            # Setup child's constraints recursively:
            child._update_layout_constraints(
                solver=solver,
                x=child_x,
                y=child_y,
                w=child_w,
                h=child_h,
            )

    #
    # Phase 3: process input events:
    #

    def _receive_mouse_position_change(self, mouse_x_dip: int, mouse_y_dip: int):
        # OPTIMIZATION: early out if mouse position hasn't changed.
        if (mouse_x_dip, mouse_y_dip) == self._latest_global_mouse_pos:
            return

        # Update `self._mouse_over`
        is_over = self._intersect_point(mouse_x_dip, mouse_y_dip)
        if is_over != self._mouse_over:
            self._mouse_over = is_over
            self._on_mouse_over_changed()

        # If mouse is over, call `_on_mouse_move`.
        if self._mouse_over:
            self._on_mouse_move(mouse_x_dip, mouse_y_dip)

        # If mouse is over, update `self._local_mouse_pos`.
        if self._mouse_over:
            x, y, _, _ = self._xywh
            self._latest_global_mouse_pos = (mouse_x_dip, mouse_y_dip)
            self._latest_local_mouse_pos = (mouse_x_dip - x, mouse_y_dip - y)

        # Regardless of whether mouse is over, propagate to children.
        # Children may be outside parent's bounds.
        for child in self._child_widget_list:
            child._receive_mouse_position_change(mouse_x_dip, mouse_y_dip)

    def _receive_mouse_button_action(
        self,
        button: MouseButton,
        action: ButtonAction,
        click_handled: bool,
    ) -> bool:
        if action == "press":
            assert not click_handled, "click_handled must be False for press action"

            # Set pressed state if mouse is over this widget
            self._mouse_button_pressed_locally = self._mouse_over

            # Propagate press events to children first (topmost first)
            for child in reversed(self._child_widget_list):
                child._receive_mouse_button_action(
                    button=button,
                    action=action,
                    click_handled=False,
                )

            # Press action does not handle clicks, so always return False
            return False
        elif action == "release":
            # Clear pressed state
            was_pressed_locally = self._mouse_button_pressed_locally
            self._mouse_button_pressed_locally = False

            # Propagate to children first (topmost first) if not click_handled:
            for child in reversed(self._child_widget_list):
                click_handled = child._receive_mouse_button_action(
                    button=button,
                    action=action,
                    click_handled=click_handled,
                )

            # Invoke the click handler if...
            # - click not yet handled
            # - mouse is over this widget
            # - this widget was pressed locally (otherwise, it's not a valid click)
            if not click_handled and self._mouse_over and was_pressed_locally:
                click_handled = self._on_click(button=button)

            # Return whether click was handled
            return click_handled
        else:
            raise NotImplementedError(f"Unknown button action: {action!r}")

    def _intersect_point(self, x: int, y: int) -> bool:
        rx, ry, rw, rh = self._xywh
        mt, mr, mb, ml = self._style.margin
        return rx + ml <= x < rx + rw - mr and ry + mt <= y < ry + rh - mb

    def _on_mouse_over_changed(self) -> None:
        self._mouse_over_changed_event_hub.publish(self._mouse_over)

    def _on_mouse_move(self, x_dip: int, y_dip: int) -> None:
        self._mouse_move_event_hub.publish((x_dip, y_dip))

    def _on_click(self, button: MouseButton) -> bool:
        if not self._clickable:
            return False
        self._click_event_hub.publish(button)
        return True

    #
    # Render:
    #

    def _render(self, canvas: Canvas) -> None:
        # Render self.
        self._render_self(canvas)

        # Render children, in order, after self.
        for child in self._child_widget_list:
            child._render(canvas)

    def _render_self(self, canvas: Canvas) -> None:
        # Get the latest style:
        style = self._style

        # Get position and size:
        x, y, w, h = self._xywh
        pt, pr, pb, pl = style.padding
        bt, br, bb, bl = style.border_thickness
        mt, mr, mb, ml = style.margin

        # Determine image and image layout
        # TODO: move this image into style, so it's resolved during style eval.
        if self._mouse_over and self._image_hover is not None:
            bg_image = self._image_hover
            image_src_xy = self._image_hover_src_xy
            image_src_wh = self._image_hover_src_wh
            image_layout = self._image_hover_layout
        else:
            bg_image = self._image
            image_src_xy = self._image_src_xy
            image_src_wh = self._image_src_wh
            image_layout = self._image_layout

        # Compute src_xy and src_wh based on layout mode
        dst_wh = (w - ml - mr - bl - br, h - mt - mb - bt - bb)
        src_xy, src_wh = _compute_image_src_xy_wh(
            dst_wh=dst_wh,
            image=bg_image,
            layout=image_layout,
            user_src_xy=image_src_xy,
            user_src_wh=image_src_wh,
        )

        # Draw background quad:
        canvas.add_quad(
            dst_xy=(
                x + ml + bl,
                y + mt + bt,
            ),
            dst_wh=dst_wh,
            src_xy=src_xy,
            src_wh=src_wh,
            color=style.bg_color,
            image=bg_image,
            border_color=style.border_color,
            border_thickness=style.border_thickness,
        )

        # Draw text:
        if self._text is not None:
            canvas.add_text(
                text=self._text,
                font=style.font,
                font_size_px=style.font_size_dip,
                font_weight=style.font_weight,
                dst_xy=(
                    x + ml + bl + pl,
                    y + mt + bt + pt,
                ),
                dst_wh=(
                    w - ml - mr - bl - br - pl - pr,
                    h - mt - mb - bt - bb - pt - pb,
                ),
                color=style.fg_color,
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
