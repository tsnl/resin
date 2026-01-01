from .basic import BaseResource

from .gpu import GpuDevice


class Draw3dRenderer(BaseResource):
    _gpu_device: "GpuDevice"
    _target_width_px: int
    _target_height_px: int

    def __init__(self, gpu_device: "GpuDevice") -> None:
        super().__init__(parent_resource=None)

        self._gpu_device = gpu_device


class Draw3dTarget(BaseResource):
    _renderer: "Draw3dRenderer"
    _width_px: int
    _height_px: int

    def __init__(
        self,
        renderer: "Draw3dRenderer",
        width_px: int,
        height_px: int,
    ) -> None:
        super().__init__(parent_resource=renderer)

        self._renderer = renderer
        self._width_px = width_px
        self._height_px = height_px
