# `zfw` -- Zero Framework

A game engine for fun and profit, written in pure Python.

## Quick Start

If you're on NixOS, just use the provided `shell.nix` to get started.

Otherwise, you will need:
-   `uv`
-   Vulkan SDK 1.4.335.1
    -   `slangc` accessible via the `PATH` (usually comes with the Vulkan SDK)
-   `astcenc`

We use `uv` for project management.

```bash
# Pull all submodules and LFS assets
$ git submodule update --init --recursive
$ git submodule foreach git lfs pull
$ git lfs pull

# Create a virtual environment with the right Python version, install all dependencies.
$ uv sync

# Run the example
$ uv run --with zfw -- zfw-replay

# Lint and Typecheck
$ uv run --with zfw -- ruff check .
$ uv run --with zfw -- pyright
```

## Resources

-   Rendering
    -   [Vulkan `VK_KHR_dynamic_rendering_local_read` in 1.4](https://docs.vulkan.org/spec/latest/appendices/legacy.html#_render_pass_objects_superseded_via_dynamic_rendering)
    -   [Dynamic rendering blog post](https://www.khronos.org/blog/streamlining-render-passes)
    -   [Irradiance Caching](https://www.ludicon.com/castano/blog/articles/irradiance-caching-part-1/)

## Project Structure

-  `modules/` Python packages for libraries, apps, and build tools.

## Conventions

All units conform to the International System of Units (SI). We use meters for distance,
seconds for time, radians for angles, and so on.

All coordinate systems are **right-handed**.

| Name        | Dim | Units          | Interpretation                                |
| ----------- | --- | -------------- | --------------------------------------------- |
| World space | 3D  | Meters         | +Z up, +Y forward, +X right                   |
| Clip space  | 3D  | NDC, [-1,+1]^3 | +X right, +Y up, -Z forward                   |
| Image space | 2D  | Pixels         | +Y down (rows), +X right (cols), origin at TL |

The above table means that **cameras look down -Z**, and that the projection matrix maps
the near plane to Z=-1 and the far plane to Z=+1 in clip space.

**Colors are in linear space using sRGB primaries** unless otherwise noted. APIs 
explicitly specify when colors are in sRGB space, and we try to convert to linear space
as early as possible.
