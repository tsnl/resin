"""
GUI widgets and window management.

Philosophy: each widget is a multipurpose element that can be styled to behave as a
label, button, container, etc. The key difference between these roles is the style,
which may depend on the widget's "state" (default, hover, unclickable, etc), and which
callbacks the user decides to connect to its events.

Each widget has a grid layout for its children, with constraints solved via kiwisolver.

Styles are CSS-like, with style classes and state-dependent overrides.

Order of operations per frame:
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

FIXME: Currently, 3D viewport rendering is broken. We rewrote the 2D renderer such that
we can render-to-texture and then display as a quad. This is the right way to handle
viewports. Need to rewrite this module after rewriting the 3D renderer.
"""

__all__ = [
    "GuiCursorMode",
    "GuiTheme",
    "GuiWidget",
    "GuiWidgetStyle",
    "GuiWindow",
]

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Callable
import time

import numpy as np
from kiwisolver import (
    Solver as KiwiSolver,
    Variable as KiwiVariable,
    Expression as KiwiExpression,
    Term as KiwiTerm,
)

from .basic import (
    BaseResource,
    logger,
    MouseButton,
    ButtonAction,
    Font,
    FontSize,
    FontWeight,
    Key,
    KeyModifier,
    HorizontalAlignment,
    VerticalAlignment,
    JsonObject,
    LogicError,
)
from .draw_2d import Draw2dTarget
from .draw_2d_ext import (
    Draw2dExtBasePrimitive,
    Draw2dExtQuadPrimitive,
    Draw2dExtTextPrimitive,
    Draw2dExtCanvas,
)
from .draw_3d import (
    Draw3dContext,
    Draw3dRenderer,
    Draw3dCameraIntrinsics,
    Draw3dGeometry,
    Draw3dMaterial,
)
from .bundled_data import BUNDLED_DATA_PATH
from .gpu import (
    GpuContext,
    GpuDescriptorSet,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuGraphicsPipeline,
    GpuImage,
    GpuPipelineLayout,
    GpuSampler,
    GpuShader,
    GpuSwapChain,
    GpuCommandEncoder,
)
from .window import Window

from .events import EventHub


LOG = logger(__name__)


type GuiImageLayout = Literal["fit", "crop", "stretch"]
type GuiCursorMode = Literal["cursor", "joystick"]


def _compute_image_src_xy_wh(
    dst_wh: tuple[int, int],
    image: GpuImage | None,
    layout: GuiImageLayout,
    user_src_xy: tuple[int, int] = (0, 0),
    user_src_wh: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """
    Compute the src_xy and src_wh to pass to Draw2dRenderer.add_quad() based on the layout mode.

    Args:
        dst_wh: The destination widget size in pixels
        image: The image to layout, or None
        layout: The layout mode ("fit", "crop", or "stretch")
        user_src_xy: User-specified source rectangle origin (default: top-left)
        user_src_wh: User-specified source rectangle size, or None to use full image

    Returns:
        A tuple of (src_xy, src_wh) to pass to Draw2dRenderer.add_quad()
    """
    if image is None:
        # No image: return defaults
        return user_src_xy, user_src_wh

    # Determine the source rectangle
    src_w = user_src_wh[0] if user_src_wh is not None else image.width
    src_h = user_src_wh[1] if user_src_wh is not None else image.height
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
    font_size: FontSize = "regular"
    font_weight: FontWeight = "regular"
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
            "font_size": "extra-large",
            "font_weight": "bold",
            "fg_color": (1.0, 1.0, 1.0, 1.0),  # White text
            "bg_color": (0.0, 0.33, 0.65, 1.0),  # Windows XP title bar blue
        },
    },
    "h2": {
        "default": {
            "font_size": "large",
            "font_weight": "bold",
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


class GuiWindow(BaseResource):
    """
    A GUI window that wraps a Window and provides 2D/3D rendering plus widget management.

    Takes a Window and GpuDevice as arguments instead of managing GLFW directly.
    """

    _window: Window
    _gpu_device: GpuDevice
    _gpu_swap_chain: GpuSwapChain | None
    _swapchain_image_count: int
    _theme: GuiTheme
    _last_mouse_x: float
    _last_mouse_y: float
    _central_widget: "GuiWidget | None"
    _central_widget_stack: list["GuiWidget"]
    _kiwi_solver: KiwiSolver

    # Renderers
    _draw_2d_renderer: Draw2dExtCanvas
    _draw_2d_targets: list[Draw2dTarget]  # One per swapchain image
    _draw_3d_context: Draw3dContext
    _draw_3d_renderer: Draw3dRenderer

    # Present pipeline (for blitting 2D target to swapchain)
    _present_descriptor_set_layout: GpuDescriptorSetLayout
    _present_pipeline_layout: GpuPipelineLayout
    _present_vertex_shader: GpuShader
    _present_fragment_shader: GpuShader
    _present_pipeline: GpuGraphicsPipeline
    _present_sampler: GpuSampler
    _present_descriptor_sets: list[GpuDescriptorSet]  # One per swapchain image

    # 3D viewport camera
    _camera_transform: np.ndarray | None
    _camera_intrinsics: Draw3dCameraIntrinsics | None
    _environment_map: GpuImage | None

    # 3D mesh collection for current frame
    _meshes: dict[tuple[Draw3dGeometry, Draw3dMaterial], np.ndarray]

    # Key event callback
    _key_event_callback: (
        Callable[[Key | None, int, ButtonAction, list[KeyModifier]], None] | None
    )

    # Frame timing
    _last_frame_time: float

    def __init__(
        self,
        *,
        window: Window,
        gpu_context: GpuContext,
        gpu_device: GpuDevice,
        draw_3d_context: Draw3dContext,
        swapchain_image_count: int = 3,
        theme: GuiTheme | None = None,
    ) -> None:
        super().__init__(parent_resource=window)

        self._window = window
        self._gpu_device = gpu_device
        self._theme = theme or _DEFAULT_THEME
        self._swapchain_image_count = swapchain_image_count

        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Create central widget stack
        self._central_widget = None
        self._central_widget_stack = []

        # For Kiwi solver: window size variables
        self._kiwi_solver = KiwiSolver()
        self._w_var = KiwiVariable("window_width")
        self._h_var = KiwiVariable("window_height")

        # Create swapchain
        self._gpu_swap_chain = None
        self._draw_2d_targets = []
        self._create_swapchain()

        # Create 2D renderer
        scale_x, _ = window.content_scale
        self._draw_2d_renderer = Draw2dExtCanvas(
            gpu_device=gpu_device,
            target_width_px=int(window.width_dip * scale_x),
            target_height_px=int(window.height_dip * scale_x),
            clear_color="transparent",
        )

        # Create one Draw2dTarget per swapchain image for multi-frame-in-flight
        for _ in range(swapchain_image_count):
            self._draw_2d_targets.append(
                Draw2dTarget(renderer=self._draw_2d_renderer.inner)
            )

        # Create present pipeline for blitting 2D target to swapchain
        self._present_descriptor_sets = []
        self._create_present_pipeline()

        # Create 3D renderer
        self._draw_3d_context = draw_3d_context
        self._draw_3d_renderer = Draw3dRenderer(
            context=draw_3d_context,
            gpu_device=gpu_device,
        )

        # 3D camera state
        self._camera_transform = None
        self._camera_intrinsics = None
        self._environment_map = None
        self._meshes = {}

        # Key event callback
        self._key_event_callback = None

        # Frame timing
        self._last_frame_time = 0.0

        # Set up window callbacks
        self._window.set_key_callback(self._on_key_event)
        self._window.set_mouse_button_callback(self._on_mouse_button_event)
        self._window.set_cursor_pos_callback(self._on_cursor_pos_event)
        self._window.set_framebuffer_size_callback(self._on_framebuffer_resize_event)

    def _create_swapchain(self) -> None:
        """Create the GPU swapchain."""
        # Dispose old swapchain if it exists
        if self._gpu_swap_chain is not None:
            self._gpu_device.wait_idle()
            self._gpu_swap_chain.dispose()

        # Dispose old targets
        for target in self._draw_2d_targets:
            target.dispose()
        self._draw_2d_targets.clear()

        # Create new swapchain
        self._gpu_swap_chain = GpuSwapChain(
            device=self._gpu_device,
            surface=self._window.gpu_surface,
            image_count=self._swapchain_image_count,
        )

        # Recreate targets if renderer exists
        if hasattr(self, "_draw_2d_renderer") and self._draw_2d_renderer is not None:
            for _ in range(self._swapchain_image_count):
                self._draw_2d_targets.append(
                    Draw2dTarget(renderer=self._draw_2d_renderer.inner)
                )

        # Recreate present descriptor sets
        if hasattr(self, "_present_descriptor_set_layout"):
            self._create_present_descriptor_sets()

    def _create_present_pipeline(self) -> None:
        """Create the present pipeline for blitting 2D target to swapchain."""

        self._present_descriptor_set_layout = GpuDescriptorSetLayout(
            device=self._gpu_device,
            bindings=OrderedDict(
                {
                    "sourceTexture": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "sourceSampler": GpuDescriptorSetLayoutBinding(
                        type="sampler",
                        stages=["fragment"],
                    ),
                }.items()
            ),
        )
        self._present_pipeline_layout = GpuPipelineLayout(
            device=self._gpu_device,
            descriptor_set_layouts=[self._present_descriptor_set_layout],
        )
        self._present_vertex_shader = GpuShader(
            device=self._gpu_device,
            spirv_path=(BUNDLED_DATA_PATH / "shaders/gui_present.vert.spv"),
            stage="vertex",
        )
        self._present_fragment_shader = GpuShader(
            device=self._gpu_device,
            spirv_path=(BUNDLED_DATA_PATH / "shaders/gui_present.frag.spv"),
            stage="fragment",
        )

        # Get swapchain dimensions for viewport
        assert self._gpu_swap_chain is not None
        swapchain_width = self._gpu_swap_chain.width
        swapchain_height = self._gpu_swap_chain.height

        self._present_pipeline = GpuGraphicsPipeline(
            device=self._gpu_device,
            vertex_shader=self._present_vertex_shader,
            fragment_shader=self._present_fragment_shader,
            enable_depth_test=False,
            enable_alpha_blending=False,
            viewport_width=swapchain_width,
            viewport_height=swapchain_height,
            layout=self._present_pipeline_layout,
            vk_color_format=self._gpu_swap_chain.vk_format,
        )
        self._present_sampler = GpuSampler(
            device=self._gpu_device,
            min_filter="nearest",
            mag_filter="nearest",
        )

        # Create descriptor sets for each target
        self._create_present_descriptor_sets()

    def _create_present_descriptor_sets(self) -> None:
        """Create descriptor sets for the present pipeline, one per target."""
        # Dispose old descriptor sets
        for ds in self._present_descriptor_sets:
            ds.dispose()
        self._present_descriptor_sets.clear()

        # Create new descriptor sets
        for target in self._draw_2d_targets:
            ds = GpuDescriptorSet(
                device=self._gpu_device,
                bindings={
                    "sourceTexture": target.color_image,
                    "sourceSampler": self._present_sampler,
                },
                layout=self._present_descriptor_set_layout,
            )
            self._present_descriptor_sets.append(ds)

    #
    # Resource disposal:
    #

    def _on_dispose(self) -> None:
        if self._draw_3d_renderer is not None:
            self._draw_3d_renderer.dispose()
        for ds in self._present_descriptor_sets:
            ds.dispose()
        self._present_descriptor_sets.clear()
        self._present_sampler.dispose()
        self._present_pipeline.dispose()
        self._present_fragment_shader.dispose()
        self._present_vertex_shader.dispose()
        self._present_pipeline_layout.dispose()
        self._present_descriptor_set_layout.dispose()
        for target in self._draw_2d_targets:
            target.dispose()
        self._draw_2d_targets.clear()
        if self._draw_2d_renderer is not None:
            self._draw_2d_renderer.dispose()
        if self._gpu_swap_chain is not None:
            self._gpu_swap_chain.dispose()
        super()._on_dispose()

    #
    # Properties:
    #

    @property
    def window(self) -> Window:
        return self._window

    @property
    def gpu_device(self) -> GpuDevice:
        return self._gpu_device

    @property
    def draw_2d_renderer(self) -> Draw2dExtCanvas:
        return self._draw_2d_renderer

    @property
    def draw_3d_renderer(self) -> Draw3dRenderer:
        return self._draw_3d_renderer

    @property
    def width_dip(self) -> int:
        return self._window.width_dip

    @property
    def height_dip(self) -> int:
        return self._window.height_dip

    @property
    def content_scale(self) -> tuple[float, float]:
        return self._window.content_scale

    @property
    def theme(self) -> GuiTheme:
        return self._theme

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
    # Window management (delegated to Window):
    #

    def should_close(self) -> bool:
        return self._window.should_close()

    def show(self) -> None:
        self._window.show()

    def hide(self) -> None:
        self._window.hide()

    def set_cursor_mode(self, cursor_mode: GuiCursorMode) -> None:
        """
        Sets the mouse input mode for the window.
        - "cursor": cursor input, mouse movement handled by the OS.
        - "joystick": cursor hidden, mouse movement captured by the window.
        """
        self._window.set_cursor_mode(cursor_mode)

    @staticmethod
    def poll_events() -> None:
        Window.poll_events()

    #
    # 3D viewport camera and rendering:
    #

    def set_3d_camera(
        self,
        transform: np.ndarray,
        intrinsics: Draw3dCameraIntrinsics,
    ) -> None:
        """Set the camera for 3D viewport rendering."""
        self._camera_transform = transform
        self._camera_intrinsics = intrinsics

    def set_environment_map(self, environment_map: GpuImage | None) -> None:
        """Set the environment map for IBL lighting."""
        self._environment_map = environment_map

    def add_3d_mesh(
        self,
        geometry: Draw3dGeometry,
        material: Draw3dMaterial,
        model_matrices: np.ndarray,
    ) -> None:
        """
        Add mesh instances to render in the 3D viewport.

        :param geometry: The geometry to render.
        :param material: The material to apply.
        :param model_matrices: A (N, 4, 4) array of model transforms in row-major layout.
        """
        key = (geometry, material)
        if key in self._meshes:
            # Concatenate with existing matrices
            self._meshes[key] = np.concatenate(
                [self._meshes[key], model_matrices], axis=0
            )
        else:
            self._meshes[key] = model_matrices

    def clear_3d_meshes(self) -> None:
        """Clear all 3D meshes for the next frame."""
        self._meshes = {}

    def set_key_event_callback(
        self,
        callback: "Callable[[Key | None, int, ButtonAction, list[KeyModifier]], None] | None",
    ) -> None:
        """Set a callback for key events."""
        self._key_event_callback = callback

    #
    # Update phase 1: update style:
    #

    def update_style(self) -> None:
        if self._central_widget is None:
            return
        self._central_widget._update_style()

    #
    # Update phase 2: update layout:
    #

    def update_layout(self) -> None:
        # TODO: Can we get rid of Kiwi here?

        if self._central_widget is None:
            return

        solver = self._kiwi_solver

        # Reset solver:
        solver.reset()

        # Setup window size constraints:
        solver.addEditVariable(self._w_var, "strong")
        solver.addEditVariable(self._h_var, "strong")
        solver.suggestValue(self._w_var, self._window.width_dip)
        solver.suggestValue(self._h_var, self._window.height_dip)

        # Setup widget layout constraints:
        self._central_widget._update_layout_constraints(
            solver,
            0.0,
            0.0,
            self._w_var,
            self._h_var,
        )

        # Solve:
        solver.updateVariables()

    #
    # Update phase 3: process input events:
    #

    def _on_key_event(
        self,
        key: Key | None,
        scancode: int,
        action: ButtonAction,
        mods: list[KeyModifier],
    ) -> None:
        if self._key_event_callback is not None:
            self._key_event_callback(key, scancode, action, mods)

        # TODO: Handle key events in GuiWidget if needed

    def _on_mouse_button_event(
        self,
        button: MouseButton,
        action: ButtonAction,
        mods: list[KeyModifier],
    ) -> None:
        if not self._central_widget:
            return

        self._central_widget._receive_mouse_button_action(
            button=button,
            action=action,
            click_handled=False,
        )

    def _on_cursor_pos_event(
        self,
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
        width_px: int,
        height_px: int,
    ) -> None:
        # Handle resize for GPU surface
        if self._window.handle_resize_for_gpu_surface():
            # Recreate swapchain if surface was recreated
            self._create_swapchain()

    #
    # Update and Render:
    #

    def update(self) -> None:
        """Update style, layout, poll input events, and call widget update hooks."""

        # Compute delta time
        current_time = time.perf_counter()
        if self._last_frame_time == 0.0:
            dt = 0.0
        else:
            dt = current_time - self._last_frame_time
        self._last_frame_time = current_time

        # This specific update order is important, and is documented in the docstring
        # for `GuiWidget`.

        # Update style:
        self.update_style()

        # Update layout:
        self.update_layout()

        # Receive input events:
        self.poll_events()

        # Call widget update hooks:
        if self._central_widget is not None:
            self._central_widget._update(dt)

        # Ready to 'render()'.

    def render(self) -> None:
        """Render the GUI window with 3D viewport and 2D widgets."""
        if self._gpu_swap_chain is None:
            return

        with self._gpu_swap_chain.present() as swapchain_target:
            command_encoder = GpuCommandEncoder(
                device=self._gpu_device,
                queue_type="graphics",
            )

            # Get the Draw2dTarget for this swapchain image
            draw_2d_target = self._draw_2d_targets[swapchain_target.image_index]

            # Create primitives list for this frame
            primitives: list[Draw2dExtBasePrimitive] = []
            scale_x, _ = self._window.content_scale

            # Draw to primitives:
            if self._central_widget is not None:
                self._central_widget._render(primitives)

            self._draw_2d_renderer.quads(
                command_encoder=command_encoder,
                target=draw_2d_target,
                primitives=primitives,
                scale=scale_x,
            )

            # Present the 2D target to swapchain using present pipeline
            command_encoder.transition_image_layout(
                image=swapchain_target.image,
                layout="color-attachment-optimal",
            )
            command_encoder.transition_image_layout(
                image=draw_2d_target.color_image,
                layout="texture-binding",
            )
            present_descriptor_set = self._present_descriptor_sets[
                swapchain_target.image_index
            ]
            with command_encoder.render(
                color_attachment=swapchain_target.image,
                clear_color="black",
            ) as rp:
                rp.bind_pipeline(pipeline=self._present_pipeline)
                rp.bind_descriptor_set(set_=present_descriptor_set, set_index=0)
                rp.draw(vertex_count=3, first_instance=0, instance_count=1)

            # Transition image for presentation
            command_encoder.transition_image_layout(
                image=swapchain_target.image,
                layout=(
                    "present-src"
                    if self._gpu_device.present_support_enabled
                    else "transfer-src-optimal"
                ),
            )
            command_encoder.submit(
                fence=swapchain_target.render_done_fence,
                wait_semaphores=[swapchain_target.render_wait_semaphore],
                signal_semaphores=[swapchain_target.render_done_semaphore],
            )


class GuiWidget(BaseResource):
    _parent_widget: "GuiWidget | None"
    _gui_window: "GuiWindow"
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
    _image: GpuImage | None
    _image_src_xy: tuple[int, int]
    _image_src_wh: tuple[int, int] | None
    _image_layout: GuiImageLayout
    _image_hover: GpuImage | None
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
        gui_window: "GuiWindow | None" = None,
        theme: GuiTheme | None = None,
        row: int = 0,
        col: int = 0,
        row_span: int = 1,
        col_span: int = 1,
        grid_rows: tuple[int, ...] | None = None,
        grid_cols: tuple[int, ...] | None = None,
        text: str | None = None,
        image: GpuImage | None = None,
        image_src_xy: tuple[int, int] = (0, 0),
        image_src_wh: tuple[int, int] | None = None,
        image_layout: GuiImageLayout = "fit",
        image_hover: GpuImage | None = None,
        image_hover_src_xy: tuple[int, int] = (0, 0),
        image_hover_src_wh: tuple[int, int] | None = None,
        image_hover_layout: GuiImageLayout | None = None,
        style_classes: list[str] | None = None,
        clickable: bool = True,
        parent_resource: "BaseResource | None" = None,
    ) -> None:
        if not gui_window and not parent_widget:
            raise ValueError("Either gui_window or parent_widget must be provided")
        if gui_window and parent_widget:
            raise ValueError("Either gui_window or parent_widget should be provided")

        super().__init__(parent_resource=(parent_resource or parent_widget))

        self._parent_widget, self._gui_window = (
            GuiWidget._resolve_parent_widget_and_window(
                parent_widget=parent_widget,
                gui_window=gui_window,
            )
        )
        self._theme = GuiWidget._resolve_theme(
            theme=theme, parent_widget=parent_widget, gui_window=gui_window
        )

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
        gui_window: "GuiWindow | None",
    ) -> tuple["GuiWidget | None", "GuiWindow"]:
        if not gui_window and not parent_widget:
            raise ValueError("Either gui_window or parent_widget must be provided")
        if gui_window and parent_widget:
            raise ValueError("Either gui_window or parent_widget should be provided")
        if parent_widget:
            return parent_widget, parent_widget._gui_window
        else:
            assert gui_window is not None
            return None, gui_window

    @staticmethod
    def _resolve_theme(
        theme: GuiTheme | None,
        parent_widget: "GuiWidget | None",
        gui_window: "GuiWindow | None",
    ) -> GuiTheme:
        if parent_widget is not None:
            base_theme = parent_widget._theme
        elif gui_window is not None:
            base_theme = gui_window.theme
        else:
            base_theme = _DEFAULT_THEME
        return _eval_theme(base_theme, theme or {})

    def _add_child_widget(self, child_widget: "GuiWidget") -> None:
        self._child_widget_list.append(child_widget)

    #
    # Dispose resources:
    #

    def _on_dispose(self) -> None:
        for child in self._child_widget_list:
            child.dispose()

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
    # Update:
    #

    def _update(self, dt: float) -> None:
        """Update this widget and its children. Called once per frame."""
        # Update self first.
        self._update_self(dt)

        # Update children, in order, after self.
        for child in self._child_widget_list:
            child._update(dt)

    def _update_self(self, dt: float) -> None:
        """Override this method to add custom per-frame update logic."""
        pass

    #
    # Render:
    #

    def _render(self, primitives: list[Draw2dExtBasePrimitive]) -> None:
        # Render self.
        self._render_self(primitives)

        # Render children, in order, after self.
        for child in self._child_widget_list:
            child._render(primitives)

    def _render_self(self, primitives: list[Draw2dExtBasePrimitive]) -> None:
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

        # Compute src_xywh_px for quad
        src_xywh_px: tuple[int, int, int, int] | None = None
        if src_xy is not None and src_wh is not None:
            src_xywh_px = (src_xy[0], src_xy[1], src_wh[0], src_wh[1])

        # Draw background quad:
        primitives.append(
            Draw2dExtQuadPrimitive(
                dst_xywh_dip=(
                    x + ml + bl,
                    y + mt + bt,
                    dst_wh[0],
                    dst_wh[1],
                ),
                src_xywh_px=src_xywh_px,
                fill_color=style.bg_color,
                fill_image=bg_image,
                border_color=style.border_color,
                border_thickness_dip=style.border_thickness,
            )
        )

        # Draw text:
        if self._text is not None:
            primitives.append(
                Draw2dExtTextPrimitive(
                    text=self._text,
                    font=style.font,
                    font_size=style.font_size,
                    font_weight=style.font_weight,
                    dst_xy_dip=(
                        x + ml + bl + pl,
                        y + mt + bt + pt,
                    ),
                    dst_wh_dip=(
                        w - ml - mr - bl - br - pl - pr,
                        h - mt - mb - bt - bb - pt - pb,
                    ),
                    color=style.fg_color,
                    wrap=style.wrap,
                    horizontal_alignment=style.text_horizontal_alignment,
                    vertical_alignment=style.text_vertical_alignment,
                )
            )
