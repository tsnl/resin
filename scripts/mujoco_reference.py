#!/usr/bin/env python3
"""Dump a MuJoCo CPU reference trajectory for an MJCF model.

Steps the model with MuJoCo's own simulator and writes a plain-text
trajectory that `cargo run --example mujoco_compare` checks resin against:

    line 1:  timestep <dt> bodies <n> <name_1> ... <name_n>
    line k:  x y z qw qx qy qz   (7 numbers per body, all bodies, one step)

Coordinates are MuJoCo's own (z-up); the comparer converts. Bodies are in
MuJoCo order, excluding the world body.

Usage: python3 scripts/mujoco_reference.py model.xml steps > dump.txt
"""

import sys

import mujoco


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_path, steps = sys.argv[1], int(sys.argv[2])

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    names = [model.body(i).name or f"body{i}" for i in range(1, model.nbody)]
    print(f"timestep {model.opt.timestep} bodies {len(names)} " + " ".join(names))

    for _ in range(steps):
        mujoco.mj_step(model, data)
        # After mj_step, xpos/xquat still reflect the pre-step qpos (the
        # forward pass runs before integration); refresh them.
        mujoco.mj_forward(model, data)
        fields = []
        for i in range(1, model.nbody):
            fields += [*data.xpos[i], *data.xquat[i]]  # xquat is (w, x, y, z)
        print(" ".join(f"{v:.9g}" for v in fields))


if __name__ == "__main__":
    main()
