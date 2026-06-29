"""Reusable GPU forward-render session for interactive 3DGS viewing."""

import struct

import resin_rt_pybind

from resin import dsl
from resin.core.etype import F4
from resin.interp import ProgramId
from resin.lib.gaussians.blend import gaussian_blend
from resin.lib.gaussians.reference import PreprocessResult, pad_preprocess_result
from resin.lib.gaussians.tiling import DEFAULT_TILE_SIZE
from resin.lib.gaussians.tiling_gpu import build_gpu_tiled_blend_graph
from resin.runtime import compile_program
from resin.runtime.compile import ParamBinding


class GpuForwardSession:
    """Compile once; each frame uploads preprocess buffers and runs on GPU.

    **Tiled path (default):** binning uses GPU ``TileCount`` → builtin
    ``prefix_sum`` → ``TileFill`` → builtin radix ``sort`` → remap gather →
    ``TileRanges`` → tiled blend. No Python ``build_tiled_layout`` per frame.

    **Untiled path:** ``depths.sort()`` (radix subgraph) + global blend.
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
            "depths": struct.pack(f"<{n}f", *pre["depths"]),
        }
        if self._tiled:
            payload["radii"] = struct.pack(f"<{n}f", *pre["radii"])

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
        depths = dsl.param(shape=(count,), etype=F4, name="depths")

        if self._tiled:
            radii = dsl.param(shape=(count,), etype=F4, name="radii")
            # TileFill expects cursor buffer = exclusive offsets; pass offsets
            # as the 4th arg. TileFill treats it as atomic cursor — must start
            # as exclusive prefix. We need to copy offsets into a buffer that
            # TileFill can atomicAdd on. build_gpu_tiled_blend_graph passes
            # offsets from prefix_sum directly; TileFill must not assume
            # pre-init from host. Fix TileFill to initialize cursor from
            # offsets_in on thread 0 before fill — update tiling_gpu.
            image = build_gpu_tiled_blend_graph(
                width=self.width,
                height=self.height,
                means2d=means2d,
                depths=depths,
                radii=radii,
                conics=conics,
                colors=colors,
                opacities=opacities,
                tile_size=self._tile_size,
                max_tiles_per_gaussian=self._max_tiles_per_gaussian,
            )
            params: dict[str, dsl.View] = {
                "means2d": means2d,
                "conics": conics,
                "colors": colors,
                "opacities": opacities,
                "depths": depths,
                "radii": radii,
            }
        else:
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

        compiled = compile_program(params=params, sinks={"image": image})
        self._program_id = compiled.admit(self._interp)
        self._binding = compiled.binding(self._interp, self._program_id)
        self._count = count
