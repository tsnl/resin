from typing import TypeAlias

from .core import BaseContext, BaseContextResource

#
# GuiContext
#


class GuiContext(BaseContext["GuiContext"]):
    def _on_dispose(self) -> None:
        pass


GuiResource: TypeAlias = BaseContextResource[GuiContext]
