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
"""Render a single 3DGS frame from a camera pose (no window).

Pose JSON can be copied from the interactive viewer by pressing **C**.

Examples::

    uv run examples/demo_3dgs_offline.py -o frame.png
    uv run examples/demo_3dgs_offline.py -o frame.png \\
        --pose '{"x":0,"y":0,"z":0,"yaw":0,"pitch":0,"fov_y_deg":60}'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resin.lib.gaussians.offline import render_frame_to_png


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("gnomen_offline.png"),
        help="output PNG path",
    )
    p.add_argument(
        "--pose",
        type=str,
        default=None,
        help='camera pose JSON object, e.g. \'{"x":0,"y":0,"z":0,"yaw":0,"pitch":0,"fov_y_deg":60}\'',
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--untiled", action="store_true", help="use global-sort blend")
    args = p.parse_args(argv)

    out, visible = render_frame_to_png(
        args.output,
        camera=args.pose,
        width=args.width,
        height=args.height,
        tiled=not args.untiled,
    )
    print(f"wrote {out} ({args.width}x{args.height}, visible={visible})", file=sys.stderr)


if __name__ == "__main__":
    main()
