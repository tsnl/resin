"""
GUI widgets.

Each widget...
- Is an event router for GuiEvents: it can receive and publish GuiEvents.
- Has an axis-aligned position and size in device-independent pixels (DIP) relative to
  its parent widget.
- Can render itself on a Canvas.
- May have child widgets which are stacked on top of it, bottom to top in order.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from .basic import MouseButton, ButtonAction
from .renderer import Canvas
from .events import Event, EventRouter
from .window import Window, WindowCursorPosEvent, WindowEvent


class GuiWidget(ABC):
    _parent_widget: "GuiWidget | None"
    _xywh_dip: tuple[int, int, int, int]
    _child_widget_list: list["GuiWidget"]
    _mouse_over: bool

    def __init__(
        self,
        *,
        parent_widget: "GuiWidget | None",
        xywh_dip: tuple[int, int, int, int],
    ) -> None:
        super().__init__()

        self._parent_widget = parent_widget
        self._xywh_dip = xywh_dip
        self._child_widget_list = []
        self._mouse_over = False

        self._bind_event_handlers()

    def _bind_event_handlers(self) -> None:
        @self._event_router.subscribe()
        def gui_mouse_enter_event(event: GuiMouseEnterEvent) -> None:
            self.on_mouse_enter(event.x_dip, event.y_dip)

        @self._event_router.subscribe()
        def gui_mouse_move_event(event: GuiMouseMoveEvent) -> None:
            if not self._intersect_point(event.x_dip, event.y_dip):
                return

        @self._event_router.subscribe()
        def gui_mouse_button_event(event: GuiMouseButtonEvent) -> None:
            if not self._intersect_point(event.x_dip, event.y_dip):
                return
            if event.button == "left" and event.action == "release":
                for child_widget in reversed(self._child_widget_list):
                    handled = child_widget.on_click(
                        button=event.button,
                        x_dip=event.x_dip,
                        y_dip=event.y_dip,
                    )
                    if handled:
                        break
                else:
                    handled = self.on_click(
                        button=event.button,
                        x_dip=event.x_dip,
                        y_dip=event.y_dip,
                    )

    def _intersect_point(self, x: int, y: int) -> bool:
        x, y, w, h = self._xywh_dip
        return x <= x < x + w and y <= y < y + h

    @property
    def mouse_over(self) -> bool:
        return self._mouse_over

    def mouse_leave(self) -> None:
        self.on_mouse_leave()

    def on_mouse_leave(self) -> None:
        pass

    def mouse_enter(self, x_dip: int, y_dip: int) -> None:
        self.on_mouse_enter(x_dip, y_dip)

    def on_mouse_enter(self, x_dip: int, y_dip: int) -> None:
        pass

    def on_mouse_enter(self, x_dip: int, y_dip: int) -> None:
        pass

    def on_mouse_move(self, x_dip: int, y_dip: int) -> None:
        pass

    def on_click(self, button: MouseButton, x_dip: int, y_dip: int) -> bool:
        return False


class GuiWindow(GuiWidget):
    _window: Window

    def __init__(
        self,
        *,
        window: Window,
    ) -> None:
        super().__init__(
            parent_widget=None,
            xywh_dip=(0, 0, window.width, window.height),
        )
        self._window = window

        @self._window.event_router.subscribe()
        def window_cursor_pos_event(event: WindowCursorPosEvent) -> None:
            if self._intersect_point(int(event.x), int(event.y)):
                self._event_router.publish(
                    GuiMouseMoveEvent(
                        x_dip=int(event.x),
                        y_dip=int(event.y),
                    )
                )
