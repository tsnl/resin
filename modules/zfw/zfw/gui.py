"""
GUI widgets.

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from .basic import BaseResource, MouseButton, ButtonAction, Font
from .renderer import Canvas, RendererImage
from .events import Event, EventRouter
from .window import Window, WindowCursorPosEvent, WindowEvent, WindowMouseButtonEvent


class GuiWidget(BaseResource, ABC):
    _parent_widget: "GuiWidget | None"
    _local_xywh_dip: tuple[int, int, int, int]
    _child_widget_list: list["GuiWidget"]
    _mouse_over: bool
    _latest_local_mouse_pos: tuple[int, int]
    _latest_global_mouse_pos: tuple[int, int]

    def __init__(
        self,
        *,
        parent_widget: "GuiWidget | None",
        local_xywh_dip: tuple[int, int, int, int],
        parent_resource: "BaseResource | None" = None,
    ) -> None:
        super().__init__(parent_resource=(parent_resource or parent_widget))

        self._parent_widget = parent_widget
        self._local_xywh_dip = local_xywh_dip
        self._child_widget_list = []
        self._mouse_over = False
        self._latest_local_mouse_pos = (0, 0)
        self._latest_global_mouse_pos = (0, 0)

        if self._parent_widget is not None:
            self._parent_widget._add_child_widget(self)

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

    @abstractmethod
    def _render_self(self, canvas: Canvas) -> None: ...


class GuiWindow(GuiWidget):
    _window: Window

    def __init__(self, *, window: Window) -> None:
        super().__init__(
            parent_widget=None,
            local_xywh_dip=(0, 0, window.width, window.height),
        )
        self._window = window

        self._bind_window_event_handlers()

    def _bind_window_event_handlers(self) -> None:
        @self._window.event_router.subscribe()
        def window_cursor_pos_event(event: WindowCursorPosEvent) -> None:
            self._receive_mouse_position(
                mouse_x_dip=int(round(event.x)),
                mouse_y_dip=int(round(event.y)),
            )

        @self._window.event_router.subscribe()
        def window_mouse_button_event(event: WindowMouseButtonEvent) -> None:
            self._receive_mouse_button_action(
                button=event.button,
                action=event.action,
            )

        self._window_cursor_pos_event_handler = window_cursor_pos_event
        self._window_mouse_button_event_handler = window_mouse_button_event

    def _on_dispose_resource(self) -> None:
        super()._on_dispose_resource()

        # Unsubscribe from window events.
        self._window.event_router.unsubscribe(self._window_cursor_pos_event_handler)
        self._window.event_router.unsubscribe(self._window_mouse_button_event_handler)

    def render(self, canvas: Canvas) -> None:
        self._render(canvas)

    def _render_self(self, _: Canvas) -> None:
        pass


class GuiLabel(GuiWidget):
    _text: str
    _font: Font
    _font_size_dip: int
    _bg_color: tuple[float, float, float, float]
    _bg_image: RendererImage | None
    _bg_hover_color: tuple[float, float, float, float]
    _bg_hover_image: RendererImage | None
    _fg_color: tuple[float, float, float, float]
    _fg_hover_color: tuple[float, float, float, float]
    _border_color: tuple[float, float, float, float]
    _border_thickness: tuple[int, int, int, int]
    _hover_border_color: tuple[float, float, float, float]
    _hover_border_thickness: tuple[int, int, int, int]
    _padding: tuple[int, int, int, int]
    _wrap: bool

    def __init__(
        self,
        *,
        parent_widget: GuiWidget,
        xywh_dip: tuple[int, int, int, int],
        text: str,
        font: Font = "sans-serif",
        font_size_dip: int = 18,
        bg_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bg_image: RendererImage | None = None,
        bg_hover_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bg_hover_image: RendererImage | None = None,
        fg_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        fg_hover_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        border_thickness: tuple[int, int, int, int] = (0, 0, 0, 0),
        hover_border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        hover_border_thickness: tuple[int, int, int, int] = (0, 0, 0, 0),
        padding: tuple[int, int, int, int] = (5, 5, 5, 5),
        wrap: bool = False,
    ) -> None:
        super().__init__(parent_widget=parent_widget, local_xywh_dip=xywh_dip)
        self._text = text
        self._font = font
        self._font_size_dip = font_size_dip
        self._bg_color = bg_color
        self._bg_image = bg_image
        self._bg_hover_color = bg_hover_color
        self._bg_hover_image = bg_hover_image
        self._fg_color = fg_color
        self._fg_hover_color = fg_hover_color or fg_color
        self._border_color = border_color
        self._border_thickness = border_thickness
        self._hover_border_color = hover_border_color or border_color
        self._hover_border_thickness = hover_border_thickness or border_thickness
        self._padding = padding
        self._wrap = wrap

    def _render_self(self, canvas: Canvas) -> None:
        x, y, w, h = self.global_xywh_dip

        # Draw background quad:
        canvas.add_quad(
            dst_xy=(x, y),
            dst_wh=(w, h),
            color=(self._bg_color if not self._mouse_over else self._bg_hover_color),
            image=(self._bg_image if not self._mouse_over else self._bg_hover_image),
            border_color=(
                self._border_color if not self._mouse_over else self._hover_border_color
            ),
            border_thickness=(
                self._border_thickness
                if not self._mouse_over
                else self._hover_border_thickness
            ),
        )

        # Draw text:
        pt, pr, pb, pl = self._padding
        canvas.add_text(
            text=self._text,
            font=self._font,
            font_size_px=self._font_size_dip,
            dst_xy=(x + pl, y + pt),
            dst_wh=(w - pl - pr, h - pt - pb),
            color=(self._fg_color if not self._mouse_over else self._fg_hover_color),
            wrap=self._wrap,
        )
