# Build a Mandelbrot explorer

A Mandelbrot renderer is small enough to understand completely and substantial
enough to exercise a systems language: arithmetic, mutable state, owned buffers,
foreign I/O, GPU execution, and an event loop.

This tutorial assumes you can program and know complex arithmetic. Work through
it in order to learn Resin, or use the source links to examine one part. The
[setup](../getting-started.md) gives the compiler-service commands used throughout.
All shell commands run from the repository root with `RESIN_SERVER` set.

| Checkpoint | Result | Resin concepts |
| --- | --- | --- |
| [One orbit](orbit.md) | Known escape counts | Functions, local mutation, `Ref`, assertions |
| [An image](image.md) | A PNG on disk | Structs, spans, ownership, `Err` and `?` |
| [Sampling](sampling.md) | Smoother edges | Arrays, tuples, a lookup table |
| [GPU execution](gpu.md) | The same algorithm on CPU and GPU | Pointer capabilities, shader entry points |
| [Interaction](interactive.md) | Pan, zoom, resize | Resource ownership, event-driven rendering |
| [Application](application.md) | CLI options and screenshots | Parsing, error reporting, headless execution |

The first three chapters have small executable checkpoints in
[`examples/tutorial/`](../../examples/tutorial/). Chapters four through six walk
through the [complete explorer](../../examples/eg011_mandelbrot.resin), including
its allocation and presentation helpers. Its CPU and GPU paths call the same
solver. Snippets are included from the source files that tests compile; the manual
does not maintain a second copy of that application.

Start with [one orbit](orbit.md).
