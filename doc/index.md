# The Resin Manual

Resin is a systems language for programs that run on the CPU and GPU. It exposes
allocation, layout, and dispatch while letting ordinary functions supply both
host code and shader helpers.

Its **compiler service** handles compilation and editor analysis with shared
caches; the **CLI tool** submits source, runs compiled programs locally, and
connects editors through LSP.

**Start here:** [run your first program](getting-started.md), then
[learn the basic language forms](tutorial/basics.md), and
[build a Mandelbrot explorer](tutorial/index.md).

| You want to… | Read… |
| --- | --- |
| Learn by writing complete programs | [First program](tutorial/basics.md), then [Mandelbrot](tutorial/index.md) |
| Check what a language construct guarantees | [Language reference](language.md) |
| Find a library operation and its ownership requirements | [Library reference](library.md) |
| Build, format, debug a diagnostic, or configure an editor | [Tools](tools.md) |
| Work on the compiler | [Development](development.md) and [architecture](architecture.md) |

This manual describes the source revision it is built with. Resin is evolving;
language guarantees and current implementation limitations are identified
separately. Examples and tutorial checkpoints live alongside the compiler tests.
