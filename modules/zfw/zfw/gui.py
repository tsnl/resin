from abc import ABC, abstractmethod

from .renderer import Canvas


class GuiWidget(ABC):
    @abstractmethod
    def on_render(self, renderer: Canvas) -> None: ...

    @abstractmethod
    def on_window_event(self, event: str, **kwargs) -> None: ...
