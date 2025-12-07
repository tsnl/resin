from typing import TypeAlias

from .basic import BaseResource

#
# GuiContext
#


class GuiContext(BaseResource):
    def _on_dispose(self) -> None:
        pass


GuiResource: TypeAlias = BaseResource
