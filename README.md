# Resin

**CUDA for Vulkan.**

Resin is a low-level systems programming language for host CPUs and GPUs. Write
ordinary functions, share them between application code and shaders, and build
reusable graphics and compute libraries.

Data layout, pointers, memory allocation, and GPU dispatch stay explicit. Resin
gives you the control to implement the algorithms and kernels that higher-level
tools call into.

## Why Resin?

We're building toward an ecosystem of libraries that make graphics and computation
easier to reuse:

- **Graphics middleware.** GPU radix sorting, prefix scans, and other building
  blocks should be libraries you can bring into your renderer or simulation.
- **Computational primitives across CPU and GPU.** BVHs, raycasting, and spatial
  queries using structures such as BSP trees belong in reusable libraries too.
  Share the functions and data types, then choose where the work runs.
- **Portability across backends.** Vulkan today, with Metal and eventually WebGPU
  as future directions. The aim is to carry useful libraries across graphics APIs.
- **A language you enjoy using every day.** Type inference, methods, operator
  overloading, `Err` and `?`, automatic resource cleanup, C interop, a formatter,
  and editor support make room for application code and tools alongside the
  graphics work. Systems programming should feel direct, ergonomic, and fun.

Resin is early and evolving. CPU compilation and Vulkan compute and rendering work
today; the middleware ecosystem above is a goal, and native Metal and WebGPU
backends are not implemented. Host builds target 64-bit Linux, macOS, and Windows
and run without Vulkan or a GPU. GPU programs need a compatible
[Vulkan device](doc/shaders.md#gpu-requirements); macOS GPU compatibility is still limited.

## One language for the whole program

Ordinary helpers can run on the CPU or be called from shaders, within the
[supported shader subset](doc/shaders.md#shaders-and-graphics):

```resin
export { main };
import { "$/string.resin", "$/stdio.resin" };

struct Point { x: f32, y: f32, }

fn squared_distance(a: Point, b: Point) -> f32  {
	let mut dx = a.x - b.x;
	let mut dy = a.y - b.y;
	dx * dx + dy * dy
}

fn main()  {
	let mut a = Point { x = 0, y = 0 };
	let mut b = Point { x = 3, y = 4 };
	let text = fmt("distance squared = {0}\n", (squared_distance(a, b),));
	print(text);
}
```

Shader entry points use `@compute_shader`, `@vertex_shader`, or `@fragment_shader`;
shared helpers need no decorator. The [CPU and GPU benchmarks](doc/benchmarks.md)
exercise the same kernels on both processors. The [particle demo](examples/particles.resin)
combines a million-particle GPU simulation, rendering, and interactive camera controls
in one Resin program.

## Try it

Start with the [setup and first-program guide](doc/getting-started.md). Resin
consists of a compiler service with shared compilation and editor-analysis caches,
and a CLI that submits source and runs the resulting executable locally. The guide
explains dependencies, optional Nix setup, and starting the service.

With a compatible Vulkan GPU and a desktop display on the client, try the particle demo:

```sh
cargo run -- examples/particles.resin
```

Read the [Resin Manual](doc/index.md) for setup, language and library references,
and GPU programming. The [Mandelbrot tutorial](doc/tutorial/index.md) builds from
one orbit to a shared CPU/GPU explorer. Preview the searchable manual locally with
`mdbook serve --open`. [Zed](doc/zed.md) and
[Helix](doc/helix.md) support includes diagnostics, completion,
navigation, and formatting.

The compiler phases are reusable async Rust libraries with immutable source graphs,
completed-result caches, and bounded execution. The HTTP server sequences the phases
explicitly and shares results across independent CLI and editor callers. The client owns
local source capture, formatting, stdio LSP, and execution of verified downloads.
Native build outputs retain independent lifetimes across later builds.
See the [compiler architecture](doc/architecture.md) for APIs and ownership guarantees,
and the [repository tour](doc/repository-tour.md) for the code layout.

## License

Resin's compiler, runtime, standard library, editor integrations, and examples are
licensed under the [Apache License, Version 2.0](LICENSE), except for third-party
material with its own license notices. See [NOTICE](NOTICE) for attribution.

Using Resin to compile your own code does not change that code's license. You can
develop open-source or proprietary programs. When distributing Resin code,
including runtime, standard-library, or compiler support code incorporated into
your program, comply with Apache-2.0 and the applicable third-party licenses.

The bundled [Tree-sitter headers](crates/tree-sitter-resin/src/tree_sitter/) retain
their [MIT license](crates/tree-sitter-resin/src/tree_sitter/LICENSE). Dependencies
retain their respective licenses.
