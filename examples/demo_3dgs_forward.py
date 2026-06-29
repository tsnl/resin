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

import resin_rt_pybind

from resin import dsl
from resin.core.etype import F4
from resin.gs.gnomen import make_gnomen_cloud
from resin.gs.reference import preprocess_gaussians_cpu
from resin.gs.render import (
    argsort_depths,
    gaussian_blend,
    pack_means2d_flat,
    pack_triplets_flat,
)
from resin.runtime import compile_program


def main() -> None:
    width = height = 64
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians_cpu(cloud, width=width, height=height)
    n = len(pre["depths"])
    print(f"gnomen: {n} visible gaussians", file=sys.stderr)
    print(f"depths: {pre['depths']}", file=sys.stderr)
    print(f"means2d: {pre['means2d']}", file=sys.stderr)

    depths = dsl.param(shape=(n,), etype=F4, name="depths")
    order = argsort_depths(depths)

    means2d = dsl.param(shape=(n, 2), etype=F4, name="means2d")
    conics = dsl.param(shape=(n, 3), etype=F4, name="conics")
    colors = dsl.param(shape=(n, 3), etype=F4, name="colors")
    opacities = dsl.param(shape=(n,), etype=F4, name="opacities")

    image = gaussian_blend(
        width=width,
        height=height,
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
    interp = resin_rt_pybind.Interp("wgpu")
    program_id = compiled.admit(interp)
    binding = compiled.binding(interp, program_id)

    binding.write(
        {
            "depths": struct.pack(f"<{n}f", *pre["depths"]),
            "means2d": struct.pack(f"<{n * 2}f", *pack_means2d_flat(pre["means2d"])),
            "conics": struct.pack(f"<{n * 3}f", *pack_triplets_flat(pre["conics"])),
            "colors": struct.pack(f"<{n * 3}f", *pack_triplets_flat(pre["colors"])),
            "opacities": struct.pack(f"<{n}f", *pre["opacities"]),
        }
    )

    interp.run(program_id)
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


if __name__ == "__main__":
    main()