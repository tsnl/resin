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

import sys
from pathlib import Path

from resin.lib.gs.gnomen import make_gnomen_cloud
from resin.lib.gs.gpu_session import GpuForwardSession
from resin.lib.gs.image_io import save_rgb_f32_png
from resin.lib.gs.reference import preprocess_gaussians_cpu


def main() -> None:
    width = height = 64
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians_cpu(cloud, width=width, height=height)
    n = len(pre["depths"])
    print(f"gnomen: {n} visible gaussians", file=sys.stderr)
    print(f"depths: {pre['depths']}", file=sys.stderr)
    print(f"means2d: {pre['means2d']}", file=sys.stderr)

    pixels = GpuForwardSession(width=width, height=height).render(pre)

    cx, cy = width // 2, height // 2
    off = (cy * width + cx) * 3
    center = pixels[off : off + 3]
    nonzero = sum(1 for v in pixels if v > 1e-4)
    print(f"center pixel RGB: {center}", file=sys.stderr)
    print(f"nonzero channels: {nonzero} / {len(pixels)}", file=sys.stderr)
    assert center[0] + center[1] + center[2] > 0.05

    png_path = save_rgb_f32_png(
        Path(__file__).with_name("gnomen_forward.png"),
        pixels,
        width=width,
        height=height,
    )
    print(f"wrote {png_path}", file=sys.stderr)
    print("demo_3dgs_forward: ok", file=sys.stderr)


if __name__ == "__main__":
    main()