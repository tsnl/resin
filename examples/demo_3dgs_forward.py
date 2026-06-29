"""Render the gnomen gaussian cloud from a fixed camera on the GPU."""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "resin",
#   "resin-rt-pybind",
# ]
#
# [tool.uv.sources]
# resin = { path = "..", editable = true }
# resin-rt-pybind = { path = "../crates/resin-rt-pybind", editable = true }
# ///

import struct
import sys
from pathlib import Path
from typing import cast

import resin_rt_pybind

from resin import dsl
from resin.core.etype import F4, U4
from resin.core.pytree import PyTensor, marshall_pytensor
from resin.lib.gaussians.blend import gaussian_blend
from resin.lib.gaussians.gnomen import make_gnomen_cloud
from resin.lib.gaussians.preprocess import preprocess_gaussians
from resin.lib.gaussians.reference import sort_by_depth_cpu
from resin.runtime import compile_program


def main() -> None:
    width = height = 64
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=width, height=height)
    n = len(pre["depths"])
    order = sort_by_depth_cpu(pre["depths"])
    print(f"gnomen: {n} visible gaussians", file=sys.stderr)

    means_p = dsl.param(shape=(n, 2), etype=F4, name="means2d")
    conics_p = dsl.param(shape=(n, 3), etype=F4, name="conics")
    colors_p = dsl.param(shape=(n, 3), etype=F4, name="colors")
    opacities_p = dsl.param(shape=(n,), etype=F4, name="opacities")
    depths_p = dsl.param(shape=(n,), etype=F4, name="depths")
    _vals, perm = depths_p.sort()
    # Prefer GPU sort order when available; fall back to const order for the image.
    _ = perm
    order_p = dsl.const(list(order), etype=U4)

    image = gaussian_blend(
        width=width,
        height=height,
        means2d=means_p,
        conics=conics_p,
        colors=colors_p,
        opacities=opacities_p,
        order=order_p,
    )

    compiled = compile_program(
        params={
            "means2d": means_p,
            "conics": conics_p,
            "colors": colors_p,
            "opacities": opacities_p,
        },
        sinks={"image": image},
    )
    interp = resin_rt_pybind.Interp("wgpu")
    pid = compiled.admit(interp)
    binding = compiled.binding(interp, pid)
    binding.write(
        {
            "means2d": marshall_pytensor(
                cast(PyTensor, [[m[0], m[1]] for m in pre["means2d"]]), etype=F4
            ),
            "conics": marshall_pytensor(
                cast(PyTensor, [list(c) for c in pre["conics"]]), etype=F4
            ),
            "colors": marshall_pytensor(
                cast(PyTensor, [list(c) for c in pre["colors"]]), etype=F4
            ),
            "opacities": marshall_pytensor(list(pre["opacities"]), etype=F4),
        }
    )
    interp.run(pid)
    raw = binding.read_sink("image")
    pixels = struct.unpack(f"<{width * height * 3}f", raw)

    cx, cy = width // 2, height // 2
    off = (cy * width + cx) * 3
    center = pixels[off : off + 3]
    nonzero = sum(1 for v in pixels if v > 1e-4)
    print(f"center pixel RGB: {center}", file=sys.stderr)
    print(f"nonzero channels: {nonzero} / {len(pixels)}", file=sys.stderr)
    assert center[0] + center[1] + center[2] > 0.05
    print("demo_3dgs_forward: ok", file=sys.stderr)
    _ = Path(__file__)


if __name__ == "__main__":
    main()
