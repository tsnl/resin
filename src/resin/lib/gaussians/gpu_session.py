"""Reusable GPU forward-render session for interactive 3DGS viewing."""

import struct

import resin_rt_pybind

from resin import dsl
from resin.core.etype import F4
from resin.interp import ProgramId
from resin.lib.gaussians.blend import gaussian_blend
from resin.lib.gaussians.reference import PreprocessResult, pad_preprocess_result
from resin.runtime import compile_program
from resin.runtime.compile import ParamBinding


class GpuForwardSession:
    """Compile once per gaussian slot count, then replay preprocess inputs.

    Pass ``fixed_count`` (e.g. the full cloud size) so camera motion that culls
    gaussians does not force a program recompile / re-admit.
    """

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
        _values, order = depths.sort()
        _ = _values

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
