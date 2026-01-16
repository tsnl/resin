"""
Immediate-mode GUI built on draw_2d.

This module provides an imgui-style API where widgets are "called" each frame
and state is stored externally or via unique IDs. No persistent widget objects.

Usage:
    input_state = gui.InputState()
    # ... wire up window callbacks to input_state methods ...

    with gui.window(width=800, height=600, input_state=input_state) as g:
        g.label("Hello World")
        if g.button("Click Me"):
            print("Clicked!")
        value = g.slider_float("Value", value, 0.0, 1.0)

    # Render g.primitives via Draw2dExtCanvas
"""

__all__ = [
    "Gui",
    "GuiStyle",
    "GuiWindow",
    "InputState",
    "window",
]

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Generator

import wgpu

from . import trace
from .basic import (
    ButtonAction,
    Font,
    FontSize,
    FontWeight,
    HorizontalAlignment,
    Key,
    KeyModifier,
    MouseButton,
    VerticalAlignment,
)
from .draw_2d import Draw2dRenderer
from .draw_2d_ext import (
    Draw2dExtBasePrimitive,
    Draw2dExtCanvas,
    Draw2dExtQuadPrimitive,
    Draw2dExtTextPrimitive,
)
from .gpu import BlitRenderer

if TYPE_CHECKING:
    from .draw_3d import Draw3dRenderer
    from .window import Window


LOG = logging.getLogger(__name__)


#
# Type aliases
#

type Color = tuple[float, float, float, float]
type LayoutDirection = Literal["vertical", "horizontal"]


#
# Input State
#


@dataclass
class InputState:
    """
    Mutable input state collected from window callbacks.

    Wire up window callbacks to the helper methods, then call begin_frame()
    at the start of each frame to reset per-frame state.
    """

    mouse_x: float = 0.0
    mouse_y: float = 0.0
    mouse_down: dict[MouseButton, bool] = field(default_factory=dict)
    mouse_clicked: dict[MouseButton, bool] = field(default_factory=dict)
    mouse_released: dict[MouseButton, bool] = field(default_factory=dict)
    keys_down: set[Key] = field(default_factory=set)
    keys_pressed: set[Key] = field(default_factory=set)
    keys_released: set[Key] = field(default_factory=set)
    text_input: str = ""
    scroll_x: float = 0.0
    scroll_y: float = 0.0

    def begin_frame(self) -> None:
        """Call at start of frame to clear per-frame state."""
        self.mouse_clicked.clear()
        self.mouse_released.clear()
        self.keys_pressed.clear()
        self.keys_released.clear()
        self.text_input = ""
        self.scroll_x = 0.0
        self.scroll_y = 0.0

    def on_mouse_button(
        self, button: MouseButton, action: ButtonAction, mods: list[KeyModifier]
    ) -> None:
        """Handle mouse button callback from Window."""
        _ = mods
        if action == "press":
            self.mouse_down[button] = True
            self.mouse_clicked[button] = True
        elif action == "release":
            self.mouse_down[button] = False
            self.mouse_released[button] = True

    def on_cursor_pos(self, x: float, y: float) -> None:
        """Handle cursor position callback from Window."""
        self.mouse_x = x
        self.mouse_y = y

    def on_key(
        self,
        key: Key | None,
        scancode: int,
        action: ButtonAction,
        mods: list[KeyModifier],
    ) -> None:
        """Handle key callback from Window."""
        _ = scancode, mods
        if key is None:
            return
        if action == "press":
            self.keys_down.add(key)
            self.keys_pressed.add(key)
        elif action == "release":
            self.keys_down.discard(key)
            self.keys_released.add(key)

    def on_char(self, char: str) -> None:
        """Handle character input callback from Window."""
        self.text_input += char

    def on_scroll(self, xoffset: float, yoffset: float) -> None:
        """Handle scroll callback from Window."""
        self.scroll_x += xoffset
        self.scroll_y += yoffset


#
# Style
#


@dataclass
class GuiStyle:
    """Style configuration for GUI widgets."""

    # Colors
    window_bg_color: Color | None = (0.1, 0.1, 0.1, 0.85)  # None to disable
    bg_color: Color = (0.15, 0.15, 0.15, 0.95)
    bg_color_hover: Color = (0.25, 0.25, 0.25, 1.0)
    bg_color_active: Color = (0.12, 0.12, 0.12, 1.0)
    fg_color: Color = (1.0, 1.0, 1.0, 1.0)
    fg_color_disabled: Color = (0.5, 0.5, 0.5, 1.0)
    border_color: Color = (0.4, 0.4, 0.4, 1.0)
    accent_color: Color = (0.26, 0.59, 0.98, 1.0)
    slider_grab_color: Color = (0.4, 0.4, 0.4, 1.0)
    slider_grab_color_active: Color = (0.5, 0.5, 0.5, 1.0)
    input_bg_color: Color = (0.1, 0.1, 0.1, 1.0)
    combo_dropdown_bg: Color = (0.18, 0.18, 0.18, 0.98)
    separator_color: Color = (0.4, 0.4, 0.4, 1.0)
    scrollbar_bg_color: Color = (0.1, 0.1, 0.1, 0.5)
    scrollbar_grab_color: Color = (0.3, 0.3, 0.3, 1.0)

    # Typography
    font: Font = "sans-serif"
    font_size: FontSize = "regular"
    font_weight: FontWeight = "regular"

    # Spacing
    item_spacing: int = 4
    window_padding: int = 8

    # Sizing
    button_padding_x: int = 12
    button_padding_y: int = 6
    input_height: int = 24
    slider_height: int = 20
    slider_grab_width: int = 12
    combo_arrow_size: int = 16
    scrollbar_width: int = 12
    separator_height: int = 1

    # Borders
    border_thickness: int = 1

    # Label width for labeled widgets (slider, combo, input)
    label_width: int = 100


#
# Internal State Storage
#


@dataclass
class _TextInputState:
    """Persistent state for a text input widget."""

    cursor_pos: int = 0
    text_buffer: str = ""


@dataclass
class _ComboState:
    """Persistent state for a combo box widget."""

    is_open: bool = False


@dataclass
class _ScrollState:
    """Persistent state for a scrollable region."""

    offset_y: float = 0.0


type _WidgetState = _TextInputState | _ComboState | _ScrollState

# Module-level state storage
_widget_states: dict[str, _WidgetState] = {}
_active_id: str | None = None  # Widget with keyboard focus (text input)


def _get_widget_state[T: _WidgetState](widget_id: str, factory: type[T]) -> T:
    """Get or create widget state."""
    state = _widget_states.get(widget_id)
    if state is None or not isinstance(state, factory):
        state = factory()
        _widget_states[widget_id] = state
    return state


#
# Layout Context
#


@dataclass
class _LayoutContext:
    """Layout context for a group of widgets."""

    direction: LayoutDirection
    origin_x: int
    origin_y: int
    width: int
    cursor_x: int
    cursor_y: int
    spacing: int
    max_height: int = 0  # Track max height in horizontal layout


#
# Gui Class
#


class Gui:
    """Immediate-mode GUI context for a single frame."""

    primitives: list[Draw2dExtBasePrimitive]
    _deferred_primitives: list[Draw2dExtBasePrimitive]

    _width: int
    _height: int
    _input: InputState
    _style: GuiStyle
    _layout_stack: list[_LayoutContext]
    _gui_window: "GuiWindow | None"
    _default_direction: LayoutDirection

    def __init__(
        self,
        *,
        width: int,
        height: int,
        input_state: InputState,
        style: GuiStyle,
        gui_window: "GuiWindow | None" = None,
        default_direction: LayoutDirection = "vertical",
    ) -> None:
        self._width = width
        self._height = height
        self._input = input_state
        self._style = style
        self._gui_window = gui_window
        self._default_direction = default_direction
        self.primitives = []
        self._deferred_primitives = []
        self._layout_stack = []

    def _begin_frame(self) -> None:
        """Initialize frame state."""
        self.primitives = []
        self._deferred_primitives = []
        padding = self._style.window_padding
        self._layout_stack = [
            _LayoutContext(
                direction=self._default_direction,
                origin_x=padding,
                origin_y=padding,
                width=self._width - 2 * padding,
                cursor_x=padding,
                cursor_y=padding,
                spacing=self._style.item_spacing,
            )
        ]

    def _end_frame(self) -> None:
        """Finalize frame - draw window background and append deferred primitives."""
        # Draw window background if configured
        if self._style.window_bg_color is not None:
            layout = self._current_layout()
            padding = self._style.window_padding

            # Calculate content bounds
            content_width = layout.width + padding
            content_height = layout.cursor_y  # cursor_y is at the bottom of content

            # Insert background at the beginning of primitives
            bg_prim = Draw2dExtQuadPrimitive(
                dst_xywh_dip=(0, 0, content_width, content_height),
                fill_color=self._style.window_bg_color,
            )
            self.primitives.insert(0, bg_prim)

        # Append deferred primitives (dropdowns, etc) on top
        self.primitives.extend(self._deferred_primitives)

    def _current_layout(self) -> _LayoutContext:
        """Get current layout context."""
        return self._layout_stack[-1]

    def _advance_cursor(self, width: int, height: int) -> None:
        """Advance cursor after drawing a widget."""
        layout = self._current_layout()
        if layout.direction == "vertical":
            layout.cursor_y += height + layout.spacing
        else:
            layout.cursor_x += width + layout.spacing
            layout.max_height = max(layout.max_height, height)

    def _is_hovered(self, x: int, y: int, w: int, h: int) -> bool:
        """Check if mouse is hovering over a rectangle."""
        mx, my = self._input.mouse_x, self._input.mouse_y
        return x <= mx < x + w and y <= my < y + h

    def _is_clicked(self, x: int, y: int, w: int, h: int) -> bool:
        """Check if a rectangle was clicked this frame."""
        return self._is_hovered(x, y, w, h) and self._input.mouse_clicked.get(
            "left", False
        )

    def _draw_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Color,
        *,
        border_color: Color | None = None,
        border_thickness: int = 0,
        deferred: bool = False,
    ) -> None:
        """Draw a filled rectangle."""
        prim = Draw2dExtQuadPrimitive(
            dst_xywh_dip=(x, y, w, h),
            fill_color=color,
            border_color=border_color or (0, 0, 0, 0),
            border_thickness_dip=(
                border_thickness,
                border_thickness,
                border_thickness,
                border_thickness,
            ),
        )
        if deferred:
            self._deferred_primitives.append(prim)
        else:
            self.primitives.append(prim)

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Color,
        *,
        h_align: HorizontalAlignment = "left",
        v_align: VerticalAlignment = "middle",
        wrap: bool = False,
        deferred: bool = False,
    ) -> None:
        """Draw text within a bounding box."""
        prim = Draw2dExtTextPrimitive(
            text=text,
            font=self._style.font,
            font_size=self._style.font_size,
            font_weight=self._style.font_weight,
            dst_xy_dip=(x, y),
            dst_wh_dip=(w, h),
            color=color,
            wrap=wrap,
            horizontal_alignment=h_align,
            vertical_alignment=v_align,
        )
        if deferred:
            self._deferred_primitives.append(prim)
        else:
            self.primitives.append(prim)

    #
    # Layout Methods
    #

    @contextmanager
    def horizontal(
        self,
        width: int | None = None,
        spacing: int | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager for horizontal layout group.

        Args:
            width: Fixed width for the group. None = fill available space.
            spacing: Spacing between items. None = use style default.

        Example:
            with g.horizontal():
                g.button("A")
                g.button("B")
                g.button("C")
        """
        layout = self._current_layout()
        if width is None:
            w = layout.width - (layout.cursor_x - layout.origin_x)
        else:
            w = width
        self._layout_stack.append(
            _LayoutContext(
                direction="horizontal",
                origin_x=layout.cursor_x,
                origin_y=layout.cursor_y,
                width=w,
                cursor_x=layout.cursor_x,
                cursor_y=layout.cursor_y,
                spacing=spacing if spacing is not None else self._style.item_spacing,
            )
        )
        try:
            yield
        finally:
            if len(self._layout_stack) > 1:
                finished = self._layout_stack.pop()
                # Calculate total width and height of the horizontal group
                total_width = finished.cursor_x - finished.origin_x - finished.spacing
                total_height = finished.max_height
                if total_width < 0:
                    total_width = 0
                # Advance parent layout
                self._advance_cursor(total_width, total_height)

    def same_line(self, spacing: int | None = None) -> None:
        """Place next widget on same line as previous."""
        layout = self._current_layout()
        if layout.direction == "vertical":
            # Switch to temporary horizontal mode
            # Undo the last vertical advance
            layout.cursor_y -= layout.spacing
            layout.direction = "horizontal"
            if spacing is not None:
                layout.spacing = spacing

    def separator(self) -> None:
        """Draw a horizontal separator line."""
        layout = self._current_layout()
        x = layout.cursor_x
        y = layout.cursor_y
        w = layout.width - (x - layout.origin_x)
        h = self._style.separator_height

        self._draw_rect(x, y, w, h, self._style.separator_color)
        self._advance_cursor(w, h)

    def space(self, size: int) -> None:
        """Add vertical or horizontal space."""
        layout = self._current_layout()
        if layout.direction == "vertical":
            layout.cursor_y += size
        else:
            layout.cursor_x += size

    def indent(self, width: int = 16) -> None:
        """Indent subsequent widgets."""
        layout = self._current_layout()
        layout.cursor_x += width
        layout.origin_x += width
        layout.width -= width

    def unindent(self, width: int = 16) -> None:
        """Remove indentation."""
        layout = self._current_layout()
        layout.cursor_x -= width
        layout.origin_x -= width
        layout.width += width

    #
    # Widget Methods
    #

    def label(
        self,
        text: str,
        *,
        wrap: bool = False,
        id: str | None = None,
    ) -> None:
        """Display a text label."""
        _ = id  # Labels don't need ID for now
        layout = self._current_layout()
        x = layout.cursor_x
        y = layout.cursor_y
        w = layout.width - (x - layout.origin_x)
        h = self._style.input_height

        self._draw_text(text, x, y, w, h, self._style.fg_color, wrap=wrap)
        self._advance_cursor(w, h)

    def label_multiline(
        self,
        text: str,
        *,
        width: int,
        height: int,
        scroll: bool = True,
        id: str | None = None,
    ) -> None:
        """Display a multiline text label with optional scrollbar."""
        widget_id = id if id is not None else f"label_ml_{text[:20]}"
        layout = self._current_layout()
        x = layout.cursor_x
        y = layout.cursor_y

        # Background
        self._draw_rect(
            x,
            y,
            width,
            height,
            self._style.input_bg_color,
            border_color=self._style.border_color,
            border_thickness=self._style.border_thickness,
        )

        # Handle scroll if enabled
        scroll_offset = 0.0
        if scroll:
            state = _get_widget_state(widget_id, _ScrollState)
            if self._is_hovered(x, y, width, height):
                state.offset_y -= self._input.scroll_y * 20
                state.offset_y = max(0.0, state.offset_y)
            scroll_offset = state.offset_y

        # Text content area (with padding)
        pad = 4
        text_x = x + pad
        text_y = y + pad - int(scroll_offset)
        text_w = width - 2 * pad
        if scroll:
            text_w -= self._style.scrollbar_width
        text_h = height - 2 * pad

        # Draw text (would need clipping for proper implementation)
        self._draw_text(
            text,
            text_x,
            text_y,
            text_w,
            text_h + int(scroll_offset),
            self._style.fg_color,
            wrap=True,
            v_align="top",
        )

        self._advance_cursor(width, height)

    def button(
        self,
        label: str,
        *,
        width: int | None = None,
        enabled: bool = True,
        id: str | None = None,
    ) -> bool:
        """
        Display a button.

        Returns True if clicked this frame.
        """
        _ = id  # Buttons use label as implicit ID
        layout = self._current_layout()
        x = layout.cursor_x
        y = layout.cursor_y

        # Calculate button size
        pad_x = self._style.button_padding_x
        h = self._style.input_height
        w = width if width is not None else len(label) * 8 + 2 * pad_x

        # Determine state
        hovered = self._is_hovered(x, y, w, h) and enabled
        pressed = hovered and self._input.mouse_down.get("left", False)
        clicked = self._is_clicked(x, y, w, h) and enabled

        # Choose color
        if not enabled:
            bg_color = self._style.bg_color
            fg_color = self._style.fg_color_disabled
        elif pressed:
            bg_color = self._style.bg_color_active
            fg_color = self._style.fg_color
        elif hovered:
            bg_color = self._style.bg_color_hover
            fg_color = self._style.fg_color
        else:
            bg_color = self._style.bg_color
            fg_color = self._style.fg_color

        # Draw button
        self._draw_rect(
            x,
            y,
            w,
            h,
            bg_color,
            border_color=self._style.border_color,
            border_thickness=self._style.border_thickness,
        )
        self._draw_text(label, x + pad_x, y, w - 2 * pad_x, h, fg_color, h_align="left")

        self._advance_cursor(w, h)
        return clicked

    def slider_int(
        self,
        label: str,
        value: int,
        min_val: int,
        max_val: int,
        *,
        width: int | None = None,
        id: str | None = None,
    ) -> int:
        """
        Display an integer slider.

        Returns the (potentially modified) value.
        """
        float_val = self._slider_impl(
            label, float(value), float(min_val), float(max_val), width, id, is_int=True
        )
        return int(round(float_val))

    def slider_float(
        self,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        *,
        width: int | None = None,
        id: str | None = None,
    ) -> float:
        """
        Display a float slider.

        Returns the (potentially modified) value.
        """
        return self._slider_impl(
            label, value, min_val, max_val, width, id, is_int=False
        )

    def _slider_impl(
        self,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        width: int | None,
        id: str | None,
        is_int: bool,
    ) -> float:
        """Internal slider implementation."""
        global _active_id
        widget_id = id if id is not None else f"slider_{label}"
        layout = self._current_layout()
        x = layout.cursor_x
        y = layout.cursor_y

        label_w = self._style.label_width
        slider_w = (
            width
            if width is not None
            else layout.width - (x - layout.origin_x) - label_w
        )
        h = self._style.slider_height
        grab_w = self._style.slider_grab_width

        # Draw label
        self._draw_text(label, x, y, label_w, h, self._style.fg_color)

        # Slider track position
        track_x = x + label_w
        track_y = y
        track_w = slider_w
        track_h = h

        # Draw track background
        self._draw_rect(
            track_x,
            track_y,
            track_w,
            track_h,
            self._style.input_bg_color,
            border_color=self._style.border_color,
            border_thickness=self._style.border_thickness,
        )

        # Calculate grab position
        range_val = max_val - min_val
        if range_val <= 0:
            range_val = 1.0
        t = (value - min_val) / range_val
        t = max(0.0, min(1.0, t))
        grab_x = track_x + int(t * (track_w - grab_w))
        grab_y = track_y

        # Handle interaction
        hovered = self._is_hovered(track_x, track_y, track_w, track_h)
        if hovered and self._input.mouse_clicked.get("left", False):
            _active_id = widget_id
        if _active_id == widget_id:
            if self._input.mouse_down.get("left", False):
                # Calculate new value from mouse position
                rel_x = self._input.mouse_x - track_x - grab_w / 2
                new_t = rel_x / (track_w - grab_w)
                new_t = max(0.0, min(1.0, new_t))
                value = min_val + new_t * range_val
                if is_int:
                    value = round(value)
            else:
                _active_id = None

        # Determine grab color
        is_active = _active_id == widget_id
        grab_color = (
            self._style.slider_grab_color_active
            if is_active or hovered
            else self._style.slider_grab_color
        )

        # Draw grab
        self._draw_rect(grab_x, grab_y, grab_w, track_h, grab_color)

        # Draw value text
        if is_int:
            value_text = str(int(round(value)))
        else:
            value_text = f"{value:.2f}"
        self._draw_text(
            value_text,
            track_x,
            track_y,
            track_w,
            track_h,
            self._style.fg_color,
            h_align="center",
        )

        total_w = label_w + slider_w
        self._advance_cursor(total_w, h)
        return value

    def text_input(
        self,
        label: str,
        value: str,
        *,
        width: int | None = None,
        id: str | None = None,
    ) -> str:
        """
        Display a text input field.

        Returns the (potentially modified) string value.
        """
        global _active_id
        widget_id = id if id is not None else f"text_{label}"
        layout = self._current_layout()
        x = layout.cursor_x
        y = layout.cursor_y

        label_w = self._style.label_width
        input_w = (
            width
            if width is not None
            else layout.width - (x - layout.origin_x) - label_w
        )
        h = self._style.input_height

        # Draw label
        self._draw_text(label, x, y, label_w, h, self._style.fg_color)

        # Input field position
        field_x = x + label_w
        field_y = y

        # Get or create state
        state = _get_widget_state(widget_id, _TextInputState)

        # Handle activation
        is_active = _active_id == widget_id
        if self._is_clicked(field_x, field_y, input_w, h):
            _active_id = widget_id
            state.text_buffer = value
            state.cursor_pos = len(value)
            is_active = True

        # Handle deactivation when clicking elsewhere
        if is_active and self._input.mouse_clicked.get("left", False):
            if not self._is_hovered(field_x, field_y, input_w, h):
                _active_id = None
                is_active = False
                value = state.text_buffer

        # Handle keyboard input when active
        if is_active:
            # Text input
            for char in self._input.text_input:
                state.text_buffer = (
                    state.text_buffer[: state.cursor_pos]
                    + char
                    + state.text_buffer[state.cursor_pos :]
                )
                state.cursor_pos += 1

            # Backspace
            if "backspace" in self._input.keys_pressed and state.cursor_pos > 0:
                state.text_buffer = (
                    state.text_buffer[: state.cursor_pos - 1]
                    + state.text_buffer[state.cursor_pos :]
                )
                state.cursor_pos -= 1

            # Delete
            if "delete" in self._input.keys_pressed:
                state.text_buffer = (
                    state.text_buffer[: state.cursor_pos]
                    + state.text_buffer[state.cursor_pos + 1 :]
                )

            # Arrow keys
            if "left" in self._input.keys_pressed:
                state.cursor_pos = max(0, state.cursor_pos - 1)
            if "right" in self._input.keys_pressed:
                state.cursor_pos = min(len(state.text_buffer), state.cursor_pos + 1)

            # Home/End
            if "home" in self._input.keys_pressed:
                state.cursor_pos = 0
            if "end" in self._input.keys_pressed:
                state.cursor_pos = len(state.text_buffer)

            # Enter to confirm
            if "enter" in self._input.keys_pressed:
                _active_id = None
                is_active = False
                value = state.text_buffer

            # Escape to cancel
            if "escape" in self._input.keys_pressed:
                _active_id = None
                is_active = False
                state.text_buffer = value

        # Determine colors
        bg_color = self._style.input_bg_color
        border_color = (
            self._style.accent_color if is_active else self._style.border_color
        )

        # Draw input background
        self._draw_rect(
            field_x,
            field_y,
            input_w,
            h,
            bg_color,
            border_color=border_color,
            border_thickness=self._style.border_thickness,
        )

        # Draw text
        display_text = state.text_buffer if is_active else value
        pad = 4
        self._draw_text(
            display_text,
            field_x + pad,
            field_y,
            input_w - 2 * pad,
            h,
            self._style.fg_color,
        )

        # Draw cursor when active (simple blinking could be added)
        if is_active:
            # Simple cursor representation - would need font metrics for accuracy
            cursor_x_offset = state.cursor_pos * 7  # Approximate character width
            cursor_x = field_x + pad + cursor_x_offset
            self._draw_rect(cursor_x, field_y + 4, 1, h - 8, self._style.fg_color)

        total_w = label_w + input_w
        self._advance_cursor(total_w, h)
        return state.text_buffer if is_active else value

    def input_int(
        self,
        label: str,
        value: int,
        *,
        width: int | None = None,
        id: str | None = None,
    ) -> int:
        """
        Display an integer input field.

        Returns the (potentially modified) value.
        """
        result = self.text_input(label, str(value), width=width, id=id)
        try:
            return int(result)
        except ValueError:
            return value

    def input_float(
        self,
        label: str,
        value: float,
        *,
        width: int | None = None,
        id: str | None = None,
    ) -> float:
        """
        Display a float input field.

        Returns the (potentially modified) value.
        """
        result = self.text_input(label, f"{value:.3f}", width=width, id=id)
        try:
            return float(result)
        except ValueError:
            return value

    def combo(
        self,
        label: str,
        current_index: int,
        items: list[str],
        *,
        width: int | None = None,
        id: str | None = None,
    ) -> int:
        """
        Display a combo box (dropdown).

        Returns the (potentially modified) selected index.
        """
        widget_id = id if id is not None else f"combo_{label}"
        layout = self._current_layout()
        x = layout.cursor_x
        y = layout.cursor_y

        label_w = self._style.label_width
        combo_w = (
            width
            if width is not None
            else layout.width - (x - layout.origin_x) - label_w
        )
        h = self._style.input_height

        # Draw label
        self._draw_text(label, x, y, label_w, h, self._style.fg_color)

        # Combo box position
        box_x = x + label_w
        box_y = y

        # Get state
        state = _get_widget_state(widget_id, _ComboState)

        # Handle click on combo box
        if self._is_clicked(box_x, box_y, combo_w, h):
            state.is_open = not state.is_open

        # Determine colors
        hovered = self._is_hovered(box_x, box_y, combo_w, h)
        bg_color = self._style.bg_color_hover if hovered else self._style.bg_color
        border_color = (
            self._style.accent_color if state.is_open else self._style.border_color
        )

        # Draw combo box
        self._draw_rect(
            box_x,
            box_y,
            combo_w,
            h,
            bg_color,
            border_color=border_color,
            border_thickness=self._style.border_thickness,
        )

        # Draw current selection
        current_text = items[current_index] if 0 <= current_index < len(items) else ""
        pad = 4
        arrow_w = self._style.combo_arrow_size
        self._draw_text(
            current_text,
            box_x + pad,
            box_y,
            combo_w - 2 * pad - arrow_w,
            h,
            self._style.fg_color,
        )

        # Draw dropdown arrow
        arrow_x = box_x + combo_w - arrow_w - pad
        self._draw_text("\u25bc", arrow_x, box_y, arrow_w, h, self._style.fg_color)

        # Draw dropdown list if open (deferred to render on top)
        new_index = current_index
        if state.is_open:
            dropdown_y = box_y + h
            item_h = h
            dropdown_h = len(items) * item_h

            # Background
            self._draw_rect(
                box_x,
                dropdown_y,
                combo_w,
                dropdown_h,
                self._style.combo_dropdown_bg,
                border_color=self._style.border_color,
                border_thickness=self._style.border_thickness,
                deferred=True,
            )

            # Items
            for i, item in enumerate(items):
                item_y = dropdown_y + i * item_h
                item_hovered = self._is_hovered(box_x, item_y, combo_w, item_h)

                if item_hovered:
                    self._draw_rect(
                        box_x,
                        item_y,
                        combo_w,
                        item_h,
                        self._style.accent_color,
                        deferred=True,
                    )

                self._draw_text(
                    item,
                    box_x + pad,
                    item_y,
                    combo_w - 2 * pad,
                    item_h,
                    self._style.fg_color,
                    deferred=True,
                )

                if item_hovered and self._input.mouse_clicked.get("left", False):
                    new_index = i
                    state.is_open = False

            # Close if clicked outside
            total_h = h + dropdown_h
            if self._input.mouse_clicked.get("left", False):
                if not self._is_hovered(box_x, box_y, combo_w, total_h):
                    state.is_open = False

        total_w = label_w + combo_w
        self._advance_cursor(total_w, h)
        return new_index

    def viewport(
        self,
        id: str,
        renderer: "Draw3dRenderer",
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> InputState:
        """
        Draw a 3D viewport and return input state filtered for this viewport.

        The returned InputState only contains events if this viewport is active
        (has keyboard focus) or hovered (for mouse events).

        Args:
            id: Unique identifier for this viewport.
            renderer: The Draw3dRenderer whose output will be displayed.
            width: Width in DIP. None = fill available horizontal space.
            height: Height in DIP. None = fill available vertical space.

        Returns:
            InputState containing events filtered for this viewport.
        """
        layout = self._current_layout()
        x = layout.cursor_x
        y = layout.cursor_y

        # Calculate size - fill available space if not specified
        if width is None:
            w = layout.width - (x - layout.origin_x)
        else:
            w = width

        if height is None:
            h = self._height - y - self._style.window_padding
        else:
            h = height

        # Resize renderer to match viewport size in pixels
        # (resize() early-outs if size unchanged)
        if self._gui_window is not None:
            scale = self._gui_window._window.content_scale[0]
            target_size_px = (int(w * scale), int(h * scale))
            renderer.resize(target_size_px)

        # Draw viewport quad with renderer's output texture
        prim = Draw2dExtQuadPrimitive(
            dst_xywh_dip=(x, y, w, h),
            fill_texture=renderer.get_output_image(),
        )
        self.primitives.append(prim)

        self._advance_cursor(w, h)

        # Create filtered input state for this viewport
        hovered = self._is_hovered(x, y, w, h)
        clicked = self._is_clicked(x, y, w, h)

        # Track viewport in GuiWindow if available
        if self._gui_window is not None:
            self._gui_window._register_viewport(id, x, y, w, h)
            if hovered:
                self._gui_window._hovered_viewport_id = id
            if clicked:
                self._gui_window._active_viewport_id = id

            return self._gui_window._get_viewport_input(id)

        # Fallback: return filtered input based on hover/focus
        return self._create_filtered_input(hovered)

    def _create_filtered_input(self, active: bool) -> InputState:
        """Create a filtered InputState based on whether viewport is active."""
        if not active:
            # Return empty input state
            return InputState()

        # Return copy of current input
        return InputState(
            mouse_x=self._input.mouse_x,
            mouse_y=self._input.mouse_y,
            mouse_down=dict(self._input.mouse_down),
            mouse_clicked=dict(self._input.mouse_clicked),
            mouse_released=dict(self._input.mouse_released),
            keys_down=set(self._input.keys_down),
            keys_pressed=set(self._input.keys_pressed),
            keys_released=set(self._input.keys_released),
            text_input=self._input.text_input,
            scroll_x=self._input.scroll_x,
            scroll_y=self._input.scroll_y,
        )

    @contextmanager
    def vertical(
        self,
        width: int | None = None,
        spacing: int | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager for vertical layout group.

        Args:
            width: Fixed width for the group. None = fill available space.
            spacing: Spacing between items. None = use style default.

        Example:
            with g.vertical(width=200):
                g.label("Line 1")
                g.label("Line 2")
        """
        layout = self._current_layout()
        if width is None:
            w = layout.width - (layout.cursor_x - layout.origin_x)
        else:
            w = width
        self._layout_stack.append(
            _LayoutContext(
                direction="vertical",
                origin_x=layout.cursor_x,
                origin_y=layout.cursor_y,
                width=w,
                cursor_x=layout.cursor_x,
                cursor_y=layout.cursor_y,
                spacing=spacing if spacing is not None else self._style.item_spacing,
            )
        )
        try:
            yield
        finally:
            if len(self._layout_stack) > 1:
                finished = self._layout_stack.pop()
                # Calculate total height of the vertical group
                total_height = finished.cursor_y - finished.origin_y - finished.spacing
                total_width = finished.width
                if total_height < 0:
                    total_height = 0
                # Advance parent layout
                self._advance_cursor(total_width, total_height)


#
# Context Manager
#


@contextmanager
def window(
    *,
    width: int,
    height: int,
    input_state: InputState,
    style: GuiStyle | None = None,
) -> Generator[Gui, None, None]:
    """
    Context manager for immediate-mode GUI frame.

    Yields a Gui object for drawing widgets.
    After the context exits, access gui.primitives for rendering.

    Example:
        with gui.window(width=800, height=600, input_state=input_state) as g:
            g.label("Hello")
            if g.button("Click"):
                do_something()

        quads = canvas.quads(primitives=g.primitives, scale=scale)
    """
    start = time.perf_counter()
    g = Gui(
        width=width,
        height=height,
        input_state=input_state,
        style=style or GuiStyle(),
    )
    g._begin_frame()
    try:
        yield g
    finally:
        g._end_frame()
        end = time.perf_counter()
        trace.add_time_span("gui/window", start, end)


#
# GuiWindow Class
#


class GuiWindow:
    """
    High-level GUI window that encapsulates rendering and input handling.

    Manages Draw2dRenderer, handles window callbacks, and tracks viewport focus.
    Provides a frame() context manager that yields a Gui with horizontal default layout.

    Example:
        gui_window = GuiWindow(window, device, queue)

        while running:
            with gui_window.frame() as g:
                # Horizontal layout by default - widgets stack left-to-right
                with g.vertical():
                    g.label("Settings")
                    g.slider_float("Value", value, 0, 1)

                viewport_input = g.viewport("main", renderer_3d)
                camera.update(viewport_input, dt)

            gui_window.present()
    """

    def __init__(
        self,
        window: "Window",
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
        *,
        style: GuiStyle | None = None,
    ) -> None:
        self._window = window
        self._device = device
        self._queue = queue
        self._style = style or GuiStyle()

        # Create renderers
        self._framebuffer_size = (window.width_px, window.height_px)
        self._draw_2d_renderer = Draw2dRenderer(
            device,
            queue,
            self._framebuffer_size,
            target_format="rgba8unorm-srgb",
        )
        self._draw_2d_canvas = Draw2dExtCanvas(device=device, queue=queue)
        self._blit_renderer = BlitRenderer(device=device)

        # Input state
        self._input = InputState()
        self._active_viewport_id: str | None = None
        self._hovered_viewport_id: str | None = None
        self._viewport_bounds: dict[str, tuple[int, int, int, int]] = {}

        # Current frame state
        self._current_primitives: list[Draw2dExtBasePrimitive] = []

        # Register window callbacks
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        """Register window input callbacks."""
        self._window.set_mouse_button_callback(self._input.on_mouse_button)
        self._window.set_cursor_pos_callback(self._input.on_cursor_pos)
        self._window.set_key_callback(self._input.on_key)
        self._window.set_char_callback(self._input.on_char)
        self._window.set_scroll_callback(self._input.on_scroll)
        self._window.set_framebuffer_size_callback(self._on_framebuffer_resize)

    def _on_framebuffer_resize(self, width: int, height: int) -> None:
        """Handle framebuffer resize."""
        if width == 0 or height == 0:
            return

        new_size = (width, height)
        if new_size == self._framebuffer_size:
            return

        LOG.info(f"GuiWindow resizing to {width}x{height}")
        self._framebuffer_size = new_size

        # Resize 2D renderer
        self._draw_2d_renderer.resize(new_size)

    def _register_viewport(self, id: str, x: int, y: int, w: int, h: int) -> None:
        """Register viewport bounds for input tracking."""
        self._viewport_bounds[id] = (x, y, w, h)

    def _get_viewport_input(self, viewport_id: str) -> InputState:
        """Get filtered input state for a viewport."""
        is_hovered = self._hovered_viewport_id == viewport_id
        is_active = self._active_viewport_id == viewport_id

        # Mouse events go to hovered viewport
        # Keyboard events go to active viewport
        filtered = InputState()

        if is_hovered:
            # Provide mouse state
            filtered.mouse_x = self._input.mouse_x
            filtered.mouse_y = self._input.mouse_y
            filtered.mouse_down = dict(self._input.mouse_down)
            filtered.mouse_clicked = dict(self._input.mouse_clicked)
            filtered.mouse_released = dict(self._input.mouse_released)
            filtered.scroll_x = self._input.scroll_x
            filtered.scroll_y = self._input.scroll_y

        if is_active:
            # Provide keyboard state
            filtered.keys_down = set(self._input.keys_down)
            filtered.keys_pressed = set(self._input.keys_pressed)
            filtered.keys_released = set(self._input.keys_released)
            filtered.text_input = self._input.text_input

        return filtered

    @property
    def input(self) -> InputState:
        """Get the raw input state (for non-viewport widgets)."""
        return self._input

    @property
    def active_viewport_id(self) -> str | None:
        """Get the ID of the currently active (focused) viewport."""
        return self._active_viewport_id

    @property
    def hovered_viewport_id(self) -> str | None:
        """Get the ID of the currently hovered viewport."""
        return self._hovered_viewport_id

    @contextmanager
    def frame(self) -> Generator[Gui, None, None]:
        """
        Context manager for a GUI frame.

        Default layout is horizontal - widgets stack left-to-right.
        Use g.vertical() for vertical sections.

        Example:
            with gui_window.frame() as g:
                with g.vertical():
                    g.label("Panel")
                viewport_input = g.viewport("main", renderer)
        """
        start = time.perf_counter()

        # Begin frame
        self._input.begin_frame()
        self._window.poll_events()
        self._hovered_viewport_id = None
        self._viewport_bounds.clear()

        g = Gui(
            width=self._window.width_dip,
            height=self._window.height_dip,
            input_state=self._input,
            style=self._style,
            gui_window=self,
            default_direction="horizontal",
        )
        g._begin_frame()

        try:
            yield g
        finally:
            g._end_frame()
            self._current_primitives = g.primitives
            end = time.perf_counter()
            trace.add_time_span("GuiWindow/frame", start, end)

    def render(self, command_encoder: wgpu.GPUCommandEncoder) -> None:
        """
        Render the GUI primitives to the internal texture.

        Call this after frame() context exits to render the GUI.
        """
        with trace.span("GuiWindow/render", "gui"):
            scale = self._window.content_scale[0]

            quads = self._draw_2d_canvas.quads(
                primitives=self._current_primitives,
                scale=scale,
            )

            self._draw_2d_renderer.record(
                quads=quads,
                command_encoder=command_encoder,
            )

    def present(self, command_encoder: wgpu.GPUCommandEncoder | None = None) -> None:
        """
        Render GUI and present to screen.

        If command_encoder is None, creates one and submits immediately.
        """
        with trace.span("GuiWindow/present", "gui"):
            current_texture = self._window.canvas_context.get_current_texture()
            if current_texture is None:
                return

            owns_encoder = command_encoder is None
            if owns_encoder:
                command_encoder = self._device.create_command_encoder()

            assert command_encoder is not None

            # Render GUI
            self.render(command_encoder)

            # Blit to screen
            self._blit_renderer.record(
                input_texture=self._draw_2d_renderer.get_output_image(),
                output_texture=current_texture,
                command_encoder=command_encoder,
            )

            if owns_encoder:
                self._queue.submit([command_encoder.finish()])
                self._window.canvas_context.present()

    def get_output_image(self) -> wgpu.GPUTexture:
        """Get the rendered GUI texture (for compositing)."""
        return self._draw_2d_renderer.get_output_image()

    def dispose(self) -> None:
        """Clean up resources."""
        self._draw_2d_canvas.dispose()
