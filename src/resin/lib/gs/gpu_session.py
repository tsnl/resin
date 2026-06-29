"""Reusable GPU forward-render session for 3DGS."""

import struct

import resin_rt_pybind

from resin import dsl
from resin.core.etype import F4
from resin.lib.gs.reference import PreprocessResult, pad_preprocess_result
from resin.interp import ProgramId
from resin.runtime.compile import ParamBinding
from resin.lib.gs.render import (
    argsort_depths,
    gaussian_blend,
    pack_means2d_flat,
    pack_triplets_flat,
)
from resin.runtime import compile_program


class GpuForwardSession:
    """Compile once per visible-gaussian count, then replay preprocess inputs."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        interp: resin_rt_pybind.Interp | None = None,
        fixed_count: int | None = None,
    ) -> None:
        self.width: int = width
        self.height: int = height
        self._fixed_count: int | None = fixed_count
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
        self._binding.write(
            {
                "depths": struct.pack(f"<{n}f", *pre["depths"]),
                "means2d": struct.pack(f"<{n * 2}f", *pack_means2d_flat(pre["means2d"])),
                "conics": struct.pack(f"<{n * 3}f", *pack_triplets_flat(pre["conics"])),
                "colors": struct.pack(f"<{n * 3}f", *pack_triplets_flat(pre["colors"])),
                "opacities": struct.pack(f"<{n}f", *pre["opacities"]),
            }
        )
        self._interp.run(self._program_id)
        raw = self._binding.read_sink("image")
        return struct.unpack(f"<{self.width * self.height * 3}f", raw)

    def _admit(self, count: int) -> None:
        if self._count != -1:
            # Re-admitting another program shape on the same device currently
            # trips wgpu bind-group validation; use a fresh interpreter instead.
            self._interp = resin_rt_pybind.Interp("wgpu")

        depths = dsl.param(shape=(count,), etype=F4, name="depths")
        order = argsort_depths(depths)

        means2d = dsl.param(shape=(count, 2), etype=F4, name="means2d")
        conics = dsl.param(shape=(count, 3), etype=F4, name="conics")
        colors = dsl.param(shape=(count, 3), etype=F4, name="colors")
        opacities = dsl.param(shape=(count,), etype=F4, name="opacities")

        image = gaussian_blend(
            width=self.width,
            height=self.height,
            means2d=means2d,
            conics=conics,
            colors=colors,
            opacities=opacities,
            order=order,
        )

        compiled = compile_program(
            params={
                "depths": depths,
                "means2d": means2d,
                "conics": conics,
                "colors": colors,
                "opacities": opacities,
            },
            sinks={"image": image},
        )
        self._program_id = compiled.admit(self._interp)
        self._binding = compiled.binding(self._interp, self._program_id)
        self._count = count