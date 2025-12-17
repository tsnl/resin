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

from .basic import BaseResource, MouseButton, ButtonAction
from .renderer import Canvas
from .events import Event, EventRouter
from .window import Window, WindowCursorPosEvent, WindowEvent, WindowMouseButtonEvent


class GuiWidget(BaseResource):
    _parent_widget: "GuiWidget | None"
    _xywh_dip: tuple[int, int, int, int]
    _child_widget_list: list["GuiWidget"]
    _mouse_over: bool
    _latest_local_mouse_pos: tuple[int, int]
    _latest_global_mouse_pos: tuple[int, int]

    def __init__(
        self,
        *,
        parent_widget: "GuiWidget | None",
        xywh_dip: tuple[int, int, int, int],
        parent_resource: "BaseResource | None" = None,
    ) -> None:
        super().__init__(parent_resource=(parent_resource or parent_widget))

        self._parent_widget = parent_widget
        self._xywh_dip = xywh_dip
        self._child_widget_list = []
        self._mouse_over = False
        self._latest_local_mouse_pos = (0, 0)
        self._latest_global_mouse_pos = (0, 0)

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
            x, y, _, _ = self._xywh_dip
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
        x, y, w, h = self._xywh_dip
        return x <= x < x + w and y <= y < y + h

    @property
    def mouse_over(self) -> bool:
        return self._mouse_over

    def _on_mouse_over_changed(self, x_dip: int, y_dip: int) -> None:
        pass

    def _on_mouse_move(self, x_dip: int, y_dip: int) -> None:
        pass

    def _on_click(self, button: MouseButton) -> bool:
        return False


class GuiWindow(GuiWidget):
    _window: Window

    def __init__(self, *, window: Window) -> None:
        super().__init__(
            parent_widget=None,
            xywh_dip=(0, 0, window.width, window.height),
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
