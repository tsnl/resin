# `resin`

CUDA for graphics. A simple systems programming language targeting host CPUs and Vulkan GPUs.

## Development

Enter `nix-shell` for Rustup (using `rust-toolchain.toml`), a C compiler, `glslc`, Vulkan tools,
validation layers, and RenderDoc. Initialize the parser submodule with
`git submodule update --init`, then run `cargo test --workspace`.
Non-interactive commands work too: `nix-shell --run 'cargo test --workspace'`.

For parser development, install `cargo install --locked tree-sitter-cli --version 0.27.0`.
After changing `tree-sitter-resin/grammar.js`, regenerate from that directory with
`tree-sitter generate --js-runtime native`.
Parser changes and generated files belong in the submodule as well as its parent gitlink.

Backend tests compile generated C with a C11 compiler (`CC` or `cc`).
Shader tests use `GLSLC` or `glslc` and skip if absent.
The `gpu` feature enables compiler-to-image integration tests, not a different execution mode:

```sh
RESIN_REQUIRE_GPU=1 RESIN_REQUIRE_GLSLC=1 cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Runtime GPU tests expect `glslc` on `PATH` and a working system Vulkan driver.
The development shell supplies the Vulkan loader through `LD_LIBRARY_PATH`.
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` enables installed validation layers.

## Functions and values

```resin
fibonacci (n: int) -> int = {
    if (n <= 1) { n } else { fibonacci(n - 1) + fibonacci(n - 2) }
};

main () -> () = {
    print("fibonacci(10) = {0}\n", (fibonacci(10),));
};
```

Functions are top-level, explicitly typed, immutable definitions. All signatures are in scope
before any body is checked, so mutual recursion needs no forward declarations. There are no
lambdas, nested function definitions, or captured environments. Ordinary function values can
be stored, passed, and returned on the host.

Every function takes one argument. Empty parameter lists mean unit `()`; multiple parameters
destructure a tuple. Calling `add(1, 2)` is the same as calling `add(pair)` when `pair = (1, 2)`.
Function types use the same arrow: `(int, int) -> int`.

Value bindings use `name = value;`; assignment uses `name := value`.
Nominal types use `Name = Type;`, with explicit wrapping and unwrapping, one layer at a time.
See `examples/` for functions, recursion, records, pointers, and linked lists.

## Build and run

```sh
cargo run -- examples/eg001.resin
cargo run -- examples/eg001.resin -o dist/
cargo run -- examples/eg001.resin --output exe -o fibonacci
cargo run -- examples/eg001.resin --output c -o fibonacci.c
```

Resin builds in `build/<source-name>-<path-hash>/` under cwd and immediately runs the executable.
Runs inherit cwd and standard streams; Resin returns the program's exit status.
Top-level value initialization runs in source order, followed by an optional
`main () -> int` or `main () -> ()`. Without main, only initialization runs.

With `-o PATH`, Resin runs first, then copies the executable, even after a nonzero program exit.
An existing directory or trailing `/` receives the source name; otherwise PATH names the file.
Use `--output exe -o PATH` to build and copy without running.
`--output ir`, `ast`, `cst`, and `check` inspect earlier stages; `check` checks syntax only.

Each source path has a stable directory containing generated C, the executable, and an input
fingerprint. Unchanged programs skip C compilation and linking. Generated C, Resin/compiler
metadata, included C headers, the runtime archive, flags, and environment changes invalidate the cache.
Calls for the same source are serialized through building, running, and copying.
Failed rebuilds never run the old executable. This is a whole-program cache, not incremental IR.
Delete `build/` to clean it, including after linked system library changes or changes hidden
behind a compiler wrapper.

Host executables statically link `resin-runtime`; host-only programs do not initialize Vulkan.
Generated C includes `resin_runtime.h` and its hierarchy from `crates/resin-runtime/include`.
Cargo builds the runtime archive alongside the compiler. For relocated installations, set
`RESIN_RUNTIME_INCLUDE` and `RESIN_RUNTIME_LIB`. To compile emitted C manually on Linux:

```sh
cc -std=c11 -fno-strict-aliasing -I crates/resin-runtime/include fibonacci.c \
  target/debug/deps/libresin_runtime.a -ldl -lpthread -lm -lrt -lutil -o fibonacci
```

`--cc PATH` selects the C compiler without shell parsing.
Integer arithmetic wraps to its declared width; invalid division, shifts, and dynamic array
indexes fail with a diagnostic. There is no optimizer or stable generated ABI yet.

## Printing

`print` is a polymorphic host builtin returning unit. Its one argument is a tuple containing a
format string and a tuple of values, not C-style variadic arguments.

```resin
n = 42;
print("x = {0}\n", (n,));
print("{1}, {0}; literal {{braces}}\n", (n, "hello"));
print("done\n", ());
```

Placeholders are zero-based and may repeat. Newlines are explicit. Invalid formats or indices
fail before the call writes output. Printable values are numbers, booleans, unit, byte strings,
pointer addresses, and nominal wrappers of those. Arguments evaluate once, left-to-right,
including unused ones. A local `print` binding shadows the builtin normally.

Strings are UTF-8 byte arrays with `\n`, `\r`, `\t`, `\0`, `\"`, and `\\` escapes.
There is no implicit NUL terminator.

## Runtime bindings

`lib/runtime.resin` binds GPU, allocation, pipeline, command, and PNG operations directly:

```resin
extern type ResinGpu;
extern "resin_runtime.h" resin_gpu_create (gpu: Ptr (Ptr (ResinGpu))) -> int;
extern "resin_runtime.h" resin_gpu_destroy (gpu: Ptr (ResinGpu)) -> ();

main () -> () = {
    gpu = Ptr (ResinGpu) (ulong (0));
    status = resin_gpu_create(&gpu);
    print("GPU creation status: {0}\n", (status,));
    if (status == 0) { resin_gpu_destroy(gpu) } else { () };
};
```

`include "relative/path.resin";` loads source relative to the including file, once per canonical
path. Includes share one module namespace; cycles are errors.
Foreign headers use the C compiler's include search paths (or an absolute path).

The prototype targets 64-bit hosts. Foreign functions accept scalar/pointer parameters and return a scalar, pointer, or unit.
The wrapper unpacks Resin's single tuple argument into the C call. Opaque `extern type`
declarations name C structs and may only be used behind pointers; aggregates by value,
variadic calls, and C callbacks are not supported yet.

This is an unchecked C boundary: declarations must match the header's ABI, and callers own
pointer validity, lifetimes, buffer lengths, and synchronization. `&place` takes an address;
`pointer.*` dereferences it. Explicit casts allow pointer-to-pointer and pointer-to-`ulong`
roundtrips. There is no borrow checker; addresses of locals must not outlive their storage.
Initialize output slots before passing their addresses: Resin does not infer initialization
effects from foreign calls. C strings need an explicit `\0` and a byte-pointer cast.

## Shaders and graphics

Run either demo like any other Resin program:

```sh
cargo run -- examples/gradient.resin
cargo run -- examples/triangle.resin
```

They write `gradient.png` and `triangle.png` in cwd. Their Resin `main` functions allocate
resources, create pipelines, record dispatch/draw commands, submit, write PNGs, and free resources.
There are no compiler-side graphics/image execution modes.

The only GPU-specific compiler intrinsic is `shader`:

```resin
kernel (index: uint) -> uint = { uint (0xff400000) | (index & uint (0xffff)) };
code = shader(kernel, "compute");
```

It takes a named function and a literal stage (`"compute"`, `"vertex"`, or `"fragment"`).
The result is `{ data: Ptr (ubyte), length: ulong }`: program-lifetime embedded SPIR-V bytes,
passed directly to runtime pipeline creation. The function remains callable normally on the host.
Runtime function aliases and dynamic stage values are not accepted by `shader`.

Resin lowers the entry and its reachable named helpers to GLSL, invokes `glslc`, and embeds the
result in generated C. Shader objects are deduplicated and cached under `build/shaders/`;
included helper changes invalidate them. Copied executables need the Vulkan loader/device,
but neither Resin, source files, nor `glslc` at runtime.

The initial entry interfaces are still intentionally narrow:

- Compute maps `uint -> uint`, one packed RGBA8 pixel per invocation (R in bits 0–7, A in 24–31).
  Workgroups contain 64 invocations. The root address points to `{ count: uint, pixels: ulong }`,
  with `pixels` at byte offset 8; the wrapper bounds-checks against count.
- Vertex takes an `int` vertex index and returns `{ position: Position, color: Color }`.
  Position has `float32` fields `x, y, z, w`; Color has `r, g, b, a`, in those orders.
- Fragment maps Color to Color. Nominal wrappers are supported.

Shader bodies support 32-bit numbers, booleans, records, nominal types, local mutation, branches,
and direct calls to named Resin helpers. Globals, foreign calls, recursion, indirect calls,
arrays, spans, real pointers, integer division/remainder/shifts, and addresses carried across
block edges are rejected. `print` is host-only. General device-pointer kernels and richer stage
interfaces remain future work; resource orchestration is already ordinary Resin code.

For inspection/export, `--output glsl` or `--output spirv -o PATH` still emits one entry.
`--stage` defaults to compute; `--entry` defaults to kernel, vertex, or fragment.
`--glslc PATH` selects the compiler.

## Resources

[No Graphics API — Sebastian Aaltonen](https://www.sebastianaaltonen.com/blog/no-graphics-api)
