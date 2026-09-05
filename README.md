# `resin`

CUDA for graphics. A simple systems programming language for programming heterogeneous systems.

## Development

Initialize the parser submodule with `git submodule update --init`, then run `cargo test --workspace`.
After changing `tree-sitter-resin/grammar.js`, regenerate its parser with
`tree-sitter generate --js-runtime native` (tree-sitter CLI 0.27.0).

Backend tests compile and execute generated C, so development requires a C11 compiler
(`cc`, or the executable named by `CC`). Shader tests use `glslc` (`GLSLC` overrides its path)
and skip if it is absent; `RESIN_REQUIRE_GLSLC=1` makes its absence a failure.

GPU tests skip when no suitable Vulkan device or `glslc` is available. To require the complete
compiler-to-image path, including the existing runtime tests:

```sh
RESIN_REQUIRE_GPU=1 cargo test --workspace --all-features -- --nocapture
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The runtime tests expect `glslc` on `PATH`. The Vulkan loader must be discoverable by the OS;
on NixOS, this may require adding the 64-bit `vulkan-loader` library to `LD_LIBRARY_PATH`.
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` enables installed validation layers.

## Host prototype

```sh
cargo run -- program.resin --output c -o program.c
cargo run -- program.resin -o program
cargo run -- program.resin --output run
```

The default output is an executable, with its destination required via `-o PATH`.
Use `--output ir` to print verified IR. `--cc PATH` selects a compiler without shell parsing.
Compilation replaces the output only after success. A program first evaluates its top-level
definitions, then calls an optional `main = () => { ... };`. Main returns `int` (the process exit
status) or unit. Without main, only top-level initialization runs.

The C backend supports closures, recursion, mutation, records, arrays, pointers, nominal types,
and typed block edges. Closure environments live until process exit; this is not yet a bounded
memory ownership model. Integer arithmetic wraps to its declared width; invalid division, shifts,
and dynamic array indexes terminate with a diagnostic. Backend-unsupported operators are errors.
There is no C FFI, language-level resource API, optimizer, or stable generated ABI yet.

## Shader prototype

Emit one entry function as GLSL or compile it with `glslc` to Vulkan 1.3 SPIR-V:

```sh
cargo run -- examples/shaders/gradient.resin --output glsl -o gradient.comp
cargo run -- examples/shaders/gradient.resin --output spirv -o gradient.spv
cargo run -- examples/shaders/triangle.resin --output spirv --stage vertex -o triangle.vert.spv
cargo run -- examples/shaders/triangle.resin --output spirv --stage fragment -o triangle.frag.spv
```

`--stage` defaults to `compute`. `--entry` selects a function, defaulting to `kernel`, `vertex`,
or `fragment` for its stage. `--glslc PATH` selects the compiler. Emission uses the same verified
IR as the C backend; no parser or language syntax changes are needed.

The optional `gpu` feature adds a small headless runner using the existing Vulkan runtime:

```sh
cargo run --features gpu -- examples/shaders/gradient.resin --output compute -o gradient.png
cargo run --features gpu -- examples/shaders/triangle.resin --output graphics -o triangle.png
```

These produce 256×256 RGBA PNGs. Compute calls `kernel: uint -> uint` once per pixel; its argument
is the linear pixel index, and its result packs red in bits 0–7, green in 8–15, blue in 16–23,
and alpha in 24–31. Graphics draws three vertices: `vertex` takes an `int` vertex index and
returns `{ position: Position, color: Color }`; `fragment` maps `Color` to `Color`. `Position`
has `float32` fields `x, y, z, w`; `Color` has `float32` fields `r, g, b, a`, in those orders.
Nominal wrappers are supported. The image runner uses these fixed entry names and interfaces.

The shader subset supports 32-bit numbers, booleans, records, nominal types, local mutation,
and branches. Shader entries cannot capture or access globals, call helpers, recurse, or use
arrays, spans, or real pointers yet. Integer division, remainder, shifts, and addresses carried
across block edges are explicitly rejected. Top-level initialization is not executed for shaders.
The runner supplies resources and dispatch/draw commands; this is not yet a general host/device
programming API or a standalone graphics executable emitted by the C backend.

## Resources

- No Graphics API by Sebastian Aaltonen <br/>
  <https://www.sebastianaaltonen.com/blog/no-graphics-api>
