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
cargo run -- program.resin
cargo run -- program.resin -o dist/
cargo run -- program.resin --output exe -o program
```

By default, Resin builds in `build/<source-name>-<path-hash>/` under the current working directory
and immediately runs `program` from that directory. `--output run` selects
the same behavior explicitly. Runs inherit the working directory and standard streams, and
Resin returns the program's exit status.

Each source path has a stable build directory containing the generated C, executable, and an
input fingerprint. Unchanged programs skip C compilation and linking. Changes to generated C,
Resin or the selected C compiler, runtime headers/archive, compiler flags, or the environment
invalidate the cache. Failed rebuilds never run the old executable. Calls for the same source
are serialized through building, running, and copying; different sources can run concurrently.
This is a single-module cache, not incremental IR compilation. Delete `build/` to clean it,
including after system SDK/library changes or changes behind a compiler wrapper.

With `-o PATH`, Resin runs first, then copies the executable to PATH, preserving its executable
permissions—even if the program returns a nonzero exit status. An existing directory or a
path ending in `/` receives `<source-name>`; otherwise PATH names the executable. Missing
destination directories are created. Use `--output exe -o PATH` to build and copy without running.
Explicit text, shader, and image output modes keep their existing behavior.

Use `--output ir` to print verified IR. `--cc PATH` selects a compiler without shell parsing.
Compilation replaces the output only after success. A program first evaluates its top-level
definitions, then calls an optional `main = () => { ... };`. Main returns `int` (the process exit
status) or unit. Without main, only top-level initialization runs.

Host executables statically link `resin-runtime`; ordinary host programs do not initialize Vulkan.
Generated C includes `resin_runtime.h` from `crates/resin-runtime/include`, including its child
headers. Cargo builds the runtime archive alongside the compiler. For a relocated installation,
set `RESIN_RUNTIME_INCLUDE` to the include directory and `RESIN_RUNTIME_LIB` to the archive path.
To compile emitted C manually on Linux after `cargo build`:

```sh
cc -std=c11 -I crates/resin-runtime/include program.c target/debug/deps/libresin_runtime.a \
  -ldl -lpthread -lm -lrt -lutil -o program
```

`print` is a polymorphic host builtin returning unit. Like every function, it takes one argument:
the outer tuple contains the format string and a tuple of values, not C-style variadic arguments.

```resin
n = 42;
print("x = {0}\n", (n,));
print("{1}, {0}; literal {{braces}}\n", (n, "hello"));
print("done\n", ());
```

Placeholders are zero-based and may repeat. Newlines are explicit; malformed formats or
out-of-range placeholders terminate with a diagnostic before that call writes any output.
Values may be numbers, booleans, unit, byte strings, pointer addresses, or nominal wrappers of
these. Other aggregates and functions are not printable yet. Arguments are evaluated once,
left-to-right, including unused ones. The builtin must be called directly; a local `print` binding
shadows it normally.

String literals are UTF-8 byte arrays, with `\n`, `\r`, `\t`, `\0`, `\"`, and `\\` escapes.
Their lengths exclude any implicit terminator; embedded NUL bytes are preserved.

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
`print` is host-only; shader examples produce images instead of text.
The runner supplies resources and dispatch/draw commands; this is not yet a general host/device
programming API or a standalone graphics executable emitted by the C backend.

## Resources

- No Graphics API by Sebastian Aaltonen <br/>
  <https://www.sebastianaaltonen.com/blog/no-graphics-api>
