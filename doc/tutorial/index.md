# Build a Mandelbrot explorer

<!-- MANDELBROT_IMAGE -->

A Mandelbrot renderer is small enough to understand completely and substantial
enough to exercise a systems language: arithmetic, mutable state, owned buffers,
foreign I/O, GPU execution, and an event loop.

This tutorial assumes you can program and know complex arithmetic. Work through
it in order to learn Resin, or use the source links to examine one part. The
[setup](../getting-started.md) gives the compiler-service commands used throughout.
All shell commands run from the repository root with `RESIN_SERVER` set.

| Checkpoint | Result | Resin concepts |
| --- | --- | --- |
| [Theory](theory.md) | A recurrence, escape rule, and test table | Complex arithmetic and finite iteration |
| [One orbit](orbit.md) | Known escape counts | Functions, local mutation, `Ref`, assertions |
| [An image](image.md) | A PNG on disk | Structs, spans, ownership, `Err` and `?` |
| [GPU execution](gpu.md) | The same algorithm on CPU and GPU | Pointer capabilities, shader entry points |
| [Interaction](interactive.md) | Pan, zoom, resize | Resource ownership, event-driven rendering |
| [Application](application.md) | CLI options and headless output | Parsing, error reporting, headless execution |

Chapters one and two have small executable checkpoints in
[`examples/tutorial/`](../../examples/tutorial/). Chapters three through five walk
through the [complete explorer](../../examples/eg011_mandelbrot.resin), including
its allocation and presentation helpers. Its CPU and GPU paths call the same
solver. Snippets are included from the source files that tests compile; the manual
does not maintain a second copy of that application.

Start with [Mandelbrot theory](theory.md). If the syntax is new, first work through
[Your first Resin program](basics.md).
