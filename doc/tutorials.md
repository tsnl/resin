# Tutorials

Learn Resin by building complete programs. Begin with [Getting started](getting-started.md)
to build the tools, start the compiler service, and set `RESIN_SERVER`.

| Tutorial | What you build | What you learn |
| --- | --- | --- |
| [Your first Resin program](tutorial/basics.md) | A small console program | Functions, values, structs, references, and control flow |
| [Build a Mandelbrot explorer](tutorial/index.md) | A renderer with PNG output and an interactive viewer | Shared CPU/GPU algorithms, buffers, ownership, errors, and window events |
| [Ray tracing with a fisheye camera](ray-tracing.md) | A PNG of triangles viewed through a Kannala–Brandt lens | Primary rays, ray tracing pipelines, payloads, instancing, and GPU readback |

The first program and the initial Mandelbrot checkpoints run on the CPU. The
Mandelbrot GPU chapters require a compatible Vulkan device; the interactive viewer
also needs a display. The fisheye renderer requires Vulkan ray tracing support and
runs without a window.

Each tutorial links to its complete source. Code excerpts come from those files,
which are checked by the compiler's tests. Use the [language reference](language.md)
and [library reference](library.md) for the detailed contracts behind the examples.
