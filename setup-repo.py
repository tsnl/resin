#!/usr/bin/env -S uv run --script

import subprocess as sp
from pathlib import Path

ROOT = Path(__file__).parent

sp.run(
    ["git", "submodule", "update", "--init", "--recursive"],
    check=True,
    cwd=ROOT,
)

sp.run(
    ["make", "lib-native-release"],
    check=True,
    cwd=ROOT / "deps" / "wgpu-native",
)
