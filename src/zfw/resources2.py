"""
(WIP: NOT YET READY FOR USE)

Resources are bulky immutable data that must be carefully managed.
E.g. images, meshes, audio, etc.

This module defines all resources. Facets of these resources can be managed by using
resource classes as handles, i.e. hashmap keys.
"""

import numpy as np
import numpy.typing as npt

from .basic import BaseDisposable, ColorSpace
from .events import EventHub


class BaseResource(BaseDisposable):
    dispose_event: EventHub

    def __init__(self):
        super().__init__()
        self.dispose_event = EventHub()

    def _on_dispose(self) -> None:
        self.dispose_event.publish(self)
        return super()._on_dispose()


class GeometryResource(BaseResource):
    v_p_array: npt.NDArray[np.float32]  # (nv, 3)
    v_n_array: npt.NDArray[np.float32]  # (nv, 3)
    v_t_array: npt.NDArray[np.float32]  # (nv, 2)
    t_indices: npt.NDArray[np.uint32]  # (nt, 3)

    def __init__(
        self,
        *,
        v_p_array: npt.NDArray[np.float32],
        v_n_array: npt.NDArray[np.float32],
        v_t_array: npt.NDArray[np.float32],
        t_indices: npt.NDArray[np.uint32],
    ):
        super().__init__()

        self.v_p_array = v_p_array
        self.v_n_array = v_n_array
        self.v_t_array = v_t_array
        self.t_indices = t_indices


class MaterialResource(BaseResource):
    color_map: "ImageResource"
    color_factor: npt.NDArray[np.float32]  # (4,)

    normal_map: "ImageResource"

    metalness_map: "ImageResource"
    metalness_factor: float

    roughness_map: "ImageResource"
    roughness_factor: float

    def __init__(
        self,
        *,
        color_map: "ImageResource",
        color_factor: npt.NDArray[np.float32],
        normal_map: "ImageResource",
        metalness_map: "ImageResource",
        metalness_factor: float,
        roughness_map: "ImageResource",
        roughness_factor: float,
    ):
        super().__init__()

        self.color_map = color_map
        self.color_factor = color_factor
        self.normal_map = normal_map
        self.metalness_map = metalness_map
        self.metalness_factor = metalness_factor
        self.roughness_map = roughness_map
        self.roughness_factor = roughness_factor


class ImageResource(BaseResource):
    size_wh_px: tuple[int, int]
    channels: int
    color_space: ColorSpace | None
    data: np.ndarray

    def __init__(
        self,
        *,
        size_wh_px: tuple[int, int],
        channels: int,
        color_space: ColorSpace | None,
        data: np.ndarray,
    ):
        super().__init__()

        self.size_wh_px = size_wh_px
        self.channels = channels
        self.color_space = color_space
        self.data = data
