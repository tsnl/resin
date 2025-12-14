# `zfw` -- Zero Framework

A game engine for fun and profit, written in pure Python.

## Quick Start

We use `uv` for project management.

```bash
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

## Project Structure

-  `modules/` Python packages for libraries, apps, and build tools.
