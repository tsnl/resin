# Library guides

The [library reference](library.md) documents each importable module. These chapters
explain workflows involving several modules:

- [Strings and console I/O](strings.md): literals, owned strings, formatting, and streams.
- [PNG images](images.md): image ownership, channels, row strides, and pixel buffers.
- [Shaders and graphics](shaders.md): entry points, pipelines, and device requirements.
- [Ray tracing pipelines](ray-tracing-pipelines.md): triangle scenes, stages, payloads, and ray launches.
- [GPU buffers](gpu-buffers.md): ownership, access, uploads, and readback.
- [Typed resource bindings](resource-bindings.md): resource records and shared storage-using helpers.
- [Windows and presentation](windowing.md): event handling, input, resize, and images.

For complete rendering projects, see the [tutorials](tutorials.md), including
the Mandelbrot explorer and a ray tracing pipeline with a fisheye camera.

Library operations are exported free functions, usable with colon syntax. Fallible
operations return ordinary error unions; resource owners clean up on scope exit.
Importing a module does not re-export its dependencies. Import `$/status.resin`
when naming runtime errors explicitly; inferred `Err<_>` callers need not name them.
Modules under `$/internal/` support the public wrappers and are implementation details.
