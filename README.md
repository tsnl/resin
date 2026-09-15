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
- **A language you enjoy using every day.** Type inference, methods, `Result` and
  `?`, automatic resource cleanup, C interop, a formatter, and editor support make
  room for application code and tools alongside the graphics work. Systems
  programming should feel direct, ergonomic, and fun.

Resin is early and evolving. CPU compilation and Vulkan compute and rendering work
today; the middleware ecosystem above is a goal, and native Metal and WebGPU
backends are not implemented. Host builds target 64-bit Linux, macOS, and Windows
and run without Vulkan or a GPU. GPU programs need a compatible
[Vulkan device](doc/guide.md#gpu-requirements); macOS GPU compatibility is still limited.

## One language for the whole program

Ordinary helpers can run on the CPU or be called from shaders, within the
[supported shader subset](doc/guide.md#shaders-and-graphics):

```resin
export { main };
import { "$/string.resin" };

struct Point { x: float32, y: float32 };

def squared_distance(a: Point, b: Point) -> float32 = {
	var dx = a.x - b.x;
	var dy = a.y - b.y;
	dx * dx + dy * dy
};

def main() = {
	var a = Point { x = 0, y = 0 };
	var b = Point { x = 3, y = 4 };
	print(fmt("distance squared = {0}\n", (squared_distance(a, b),)));
};
```

Shader entry points use `@compute_shader`, `@vertex_shader`, or `@fragment_shader`;
shared helpers need no decorator. The [CPU and GPU benchmarks](benchmarks/README.md)
exercise the same kernels on both processors. The [particle demo](examples/particles.resin)
combines a million-particle GPU simulation, rendering, and interactive camera controls
in one Resin program.

## Try it

From a checkout on Linux or macOS, use the development shell:

```sh
nix-shell
cargo run -- examples/eg001.resin
```

With a compatible Vulkan GPU and a desktop display, try the particle demo:

```sh
cargo run -- examples/particles.resin
```

See the [guide](doc/guide.md) for setup, Windows instructions, language features,
building executables, and GPU programming. [Zed support](editors/zed/README.md)
includes diagnostics, completion, navigation, and formatting.

To explore the implementation, start with the [repository tour](TOUR.md) and
[compiler architecture](doc/architecture.md).

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
