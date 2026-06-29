"""Reusable GPU forward-render session for interactive 3DGS viewing."""

import struct

import resin_rt_pybind

from resin import dsl
from resin.core.etype import F4, U4
from resin.interp import ProgramId
from resin.lib.gaussians.blend import gaussian_blend
from resin.lib.gaussians.reference import PreprocessResult, pad_preprocess_result
from resin.lib.gaussians.tiling import (
    DEFAULT_TILE_SIZE,
    build_tiled_layout,
    gaussian_blend_tiled,
)
from resin.runtime import compile_program
from resin.runtime.compile import ParamBinding


class GpuForwardSession:
    """Compile once per gaussian / instance capacity, then replay frames.

    Uses **tiled** blending (Phase 3.5) by default. Pass ``tiled=False`` for the
    legacy global-sort path. ``fixed_count`` pins the gaussian buffer size so
    culling does not re-admit programs; instance buffers are sized as
    ``fixed_count * max_tiles_per_gaussian``.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        interp: resin_rt_pybind.Interp | None = None,
        fixed_count: int | None = None,
        tiled: bool = True,
        tile_size: int = DEFAULT_TILE_SIZE,
        max_tiles_per_gaussian: int = 32,
    ) -> None:
        self.width: int = width
        self.height: int = height
        self._fixed_count: int | None = fixed_count
        self._tiled: bool = tiled
        self._tile_size: int = tile_size
        self._max_tiles_per_gaussian: int = max_tiles_per_gaussian
        self._interp: resin_rt_pybind.Interp = interp or resin_rt_pybind.Interp("wgpu")
        self._program_id: ProgramId | None = None
        self._binding: ParamBinding | None = None
        self._count: int = -1
        self._max_instances: int = -1
        if fixed_count is not None:
            self._admit(fixed_count)

    def render(self, pre: PreprocessResult) -> tuple[float, ...]:
        if self._fixed_count is not None:
            pre = pad_preprocess_result(pre, self._fixed_count)
        n = len(pre["depths"])
        if n == 0:
            return tuple(0.0 for _ in range(self.width * self.height * 3))
        if n != self._count:
            self._admit(n)
        assert self._binding is not None
        assert self._program_id is not None

        payload: dict[str, bytes] = {
            "means2d": struct.pack(
                f"<{n * 2}f", *(v for m in pre["means2d"] for v in m)
            ),
            "conics": struct.pack(
                f"<{n * 3}f", *(v for c in pre["conics"] for v in c)
            ),
            "colors": struct.pack(
                f"<{n * 3}f", *(v for c in pre["colors"] for v in c)
            ),
            "opacities": struct.pack(f"<{n}f", *pre["opacities"]),
        }

        if self._tiled:
            layout = build_tiled_layout(
                pre,
                width=self.width,
                height=self.height,
                tile_size=self._tile_size,
            )
            inst = list(layout.instance_ids)
            if len(inst) > self._max_instances:
                raise ValueError(
                    f"instance count {len(inst)} exceeds capacity {self._max_instances}; "
                    "increase max_tiles_per_gaussian"
                )
            # Pad instances with 0 and empty tile ranges for unused slots.
            inst_pad = inst + [0] * (self._max_instances - len(inst))
            n_tiles = layout.n_tiles
            ranges_flat: list[int] = []
            for start, end in layout.tile_ranges:
                ranges_flat.extend([start, end])
            # Ensure we always write n_tiles * 2 entries (layout has exactly n_tiles).
            assert len(ranges_flat) == n_tiles * 2
            payload["instances"] = struct.pack(
                f"<{self._max_instances}I", *inst_pad
            )
            payload["tile_ranges"] = struct.pack(f"<{n_tiles * 2}I", *ranges_flat)
        else:
            payload["depths"] = struct.pack(f"<{n}f", *pre["depths"])

        self._binding.write(payload)
        self._interp.run(self._program_id)
        raw = self._binding.read_sink("image")
        return struct.unpack(f"<{self.width * self.height * 3}f", raw)

    def _admit(self, count: int) -> None:
        if self._count != -1:
            self._interp = resin_rt_pybind.Interp("wgpu")

        means2d = dsl.param(shape=(count, 2), etype=F4, name="means2d")
        conics = dsl.param(shape=(count, 3), etype=F4, name="conics")
        colors = dsl.param(shape=(count, 3), etype=F4, name="colors")
        opacities = dsl.param(shape=(count,), etype=F4, name="opacities")

        if self._tiled:
            ntx = (self.width + self._tile_size - 1) // self._tile_size
            nty = (self.height + self._tile_size - 1) // self._tile_size
            max_inst = count * self._max_tiles_per_gaussian
            instances = dsl.param(shape=(max_inst,), etype=U4, name="instances")
            tile_ranges = dsl.param(
                shape=(ntx * nty * 2,), etype=U4, name="tile_ranges"
            )
            image = gaussian_blend_tiled(
                width=self.width,
                height=self.height,
                means2d=means2d,
                conics=conics,
                colors=colors,
                opacities=opacities,
                instances=instances,
                tile_ranges=tile_ranges,
                tile_size=self._tile_size,
                n_tiles_x=ntx,
                n_tiles_y=nty,
            )
            params: dict[str, dsl.View] = {
                "means2d": means2d,
                "conics": conics,
                "colors": colors,
                "opacities": opacities,
                "instances": instances,
                "tile_ranges": tile_ranges,
            }
            self._max_instances = max_inst
        else:
            depths = dsl.param(shape=(count,), etype=F4, name="depths")
            _values, order = depths.sort()
            _ = _values
            image = gaussian_blend(
                width=self.width,
                height=self.height,
                means2d=means2d,
                conics=conics,
                colors=colors,
                opacities=opacities,
                order=order,
            )
            params = {
                "depths": depths,
                "means2d": means2d,
                "conics": conics,
                "colors": colors,
                "opacities": opacities,
            }
            self._max_instances = count

        compiled = compile_program(params=params, sinks={"image": image})
        self._program_id = compiled.admit(self._interp)
        self._binding = compiled.binding(self._interp, self._program_id)
        self._count = count
