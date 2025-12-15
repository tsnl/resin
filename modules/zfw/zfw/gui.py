from abc import ABC, abstractmethod

from .renderer import RendererCanvas


class GuiWidget(ABC):
    @abstractmethod
    def on_render(self, renderer: RendererCanvas) -> None: ...

    @abstractmethod
    def on_window_event(self, event: str, **kwargs) -> None: ...
