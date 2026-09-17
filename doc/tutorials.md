# Tutorials

Learn Resin by building complete programs. Begin with [Getting started](getting-started.md)
to build the tools, start the compiler service, and set `RESIN_SERVER`.

| Tutorial | What you build | What you learn |
| --- | --- | --- |
| [Your first Resin program](tutorial/basics.md) | A small console program | Functions, values, structs, references, and control flow |
| [Build a Mandelbrot explorer](tutorial/index.md) | A renderer with PNG output and an interactive viewer | Shared CPU/GPU algorithms, buffers, ownership, errors, and window events |
| [Ray tracing with a fisheye camera](ray-tracing.md) | A PNG of triangles viewed through a Kannala–Brandt lens | Primary rays, ray tracing pipelines, payloads, instancing, and GPU readback |
| [Discard fragments](discard-fragments.md) | An alpha-masked checkerboard over a gradient | Optional fragment outputs, discard, and sampling an alpha buffer |

| [Render a glTF scene with Arris](arris.md) | A headless glTF + HDRI renderer with PNG and EXR output | Raster visibility, secondary ray lighting, reusable GPU resources, and standalone SVGF |

The first program and the Mandelbrot explorer's headless CPU mode need no GPU.
The explorer's GPU mode requires a compatible Vulkan device; its interactive
viewer also needs a display. The fisheye renderer requires Vulkan ray tracing
support and runs without a window. Arris also needs ray tracing pipelines. The discard example also runs without a
window and needs a compatible Vulkan device, with no ray tracing requirement.

Each page shows an example image followed by the entire runnable source file.
The comments are the tutorial: run the file, read it in order, and change it as
you learn. The book includes the source directly, so the explanations stay with
the code checked by the compiler's tests. Use the [language reference](language.md)
and [library reference](library.md) for detailed contracts.
