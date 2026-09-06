# `resin`

CUDA for graphics. A simple systems programming language targeting host CPUs and Vulkan GPUs.

Currently supported on 64-bit Linux with native Vulkan. macOS/MoltenVK support is on hold.

## Development

Enter `nix-shell` for Rustup (using `rust-toolchain.toml`), a C compiler, CMake, GLFW's native
build dependencies, `glslc`, Vulkan tools, validation layers, and RenderDoc.
Initialize the parser submodule with
`git submodule update --init`, then run `cargo test --workspace`.
Non-interactive commands work too: `nix-shell --run 'cargo test --workspace'`.

Cargo builds and statically links the GLFW source bundled in `glfw-sys`; no GLFW installation
or library search path is needed. Outside Nix, install Rustup, a C compiler, CMake, pkg-config,
`glslc`, the Vulkan loader, and the X11, Wayland, and xkbcommon development packages, including
`wayland-scanner`. Cargo uses `rust-toolchain.toml` to install the project's Rust toolchain.

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
The development shell supplies the Vulkan loader and window-system libraries through
`LD_LIBRARY_PATH`.
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` enables installed validation layers.
Window integration tests use the `gpu` feature and run on a desktop display or Xvfb. Set `RESIN_REQUIRE_WINDOW=1`
to fail instead of skipping when windowing or presentation is unavailable.

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
`pointer + index` and `pointer - index` offset by elements, not bytes. The offset must be an
integer; pointer differences and offsets into opaque foreign types are not supported.
Pointer arithmetic and dereferences are unchecked: keep them within the allocation and aligned.
Initialize output slots before passing their addresses: Resin does not infer initialization
effects from foreign calls. C strings need an explicit `\0` and a byte-pointer cast.

## Loops

`while` works on both the host and GPU:

```resin
n = 1;
sum = 0;
while (n <= 10) {
    sum := sum + n;
    n := n + 1;
};
```

The condition must be boolean and is evaluated before every iteration. The body has its own
scope; its result is discarded, and the loop returns `()`. As with other expression statements,
the trailing semicolon is required unless the loop is the enclosing block's final expression.
The body may run zero times, so initializing a variable only in the body does not make it
definitely initialized afterward. `break` and `continue` are not implemented yet.

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

Shaders receive application data through the root address passed to `resin_gpu_dispatch` or
`resin_gpu_draw`. Add a typed pointer as the second tuple element:

```resin
Params = { count: uint, values: Ptr (float32), scale: float32 };

kernel (index: uint, root: Ptr (Params)) -> () = {
    if (index < root.count) {
        p = root.values + index;
        p.* := p.* * root.scale;
        ()
    } else { () }
};
```

The entry interfaces are:

- Compute takes `(uint, Ptr (T))` and returns `()`. Workgroups contain 64 invocations; the
  index is the global X invocation index. Dispatch only in X (`y = z = 1`) and guard any
  excess invocations in the function, as above.
- Vertex takes an `int` vertex index, optionally paired with `Ptr (T)`, and returns
  `{ position: Position, color: Color }`.
  Position has `float32` fields `x, y, z, w`; Color has `r, g, b, a`, in those orders.
- Fragment takes Color, optionally paired with `Ptr (T)`, and returns Color.
- The original `uint -> uint` compute entry still writes one packed RGBA8 pixel per invocation
  (R in bits 0–7, A in 24–31). Its implicit root is `{ count: uint, pixels: ulong }`, with
  `pixels` at byte offset 8; this wrapper bounds-checks against count.

Device pointers support loads, stores, record fields, element offsets, casts, and passing to
ordinary helpers. Shared storage supports `int`, `uint`, `float32`, `ulong`, pointers, nonempty
records, and nominal wrappers. Scalars align to their size; records align to their largest
member, with member and trailing padding. This matches C and GLSL `std430` without requiring
scalar-block-layout support. Generated C asserts sizes, alignments, and member offsets.
Storage containing booleans, unit, arrays, or other numeric widths is rejected for now.

Use `resin_allocation_host_pointer` to initialize mapped data on the CPU. Store
`resin_allocation_device_pointer` addresses in records consumed by shaders; these are not
interchangeable with host addresses. Pointer types do not enforce the address space or bounds.
Calling the same function on the CPU requires a root containing host pointers instead.

Shader bodies support 32-bit numbers, `ulong`, booleans, records, nominal types, local mutation,
branches, loops, and direct calls to named Resin helpers. Globals, foreign calls, recursion,
indirect calls, arrays, spans, and integer division/remainder/shifts are rejected. Local addresses
may only be used directly for loads, stores, and field access; they cannot be stored, passed,
returned, or carried across control-flow edges. Device addresses can. `print` is host-only.

Invocations must avoid racing on shared buffers. Workgroup-local storage, shader barriers, and
atomics are not exposed yet. For multi-pass algorithms, record separate dispatches: the runtime
inserts memory barriers before dispatches and rendering, including compute-to-vertex reads.
Submission currently waits for completion, making mapped results readable by the host.

For inspection/export, `--output glsl` or `--output spirv -o PATH` still emits one entry.
`--stage` defaults to compute; `--entry` defaults to kernel, vertex, or fragment.
`--glslc PATH` selects the compiler.

## GPU requirements

The runtime uses conventional Vulkan compute and graphics pipelines, with dynamic rendering
and a dynamic viewport/scissor. Graphics currently target one RGBA8 UNORM color attachment,
triangle lists, one sample, and no blending or depth/stencil testing.

A Vulkan 1.3 device must support graphics and compute, buffer device addresses, 64-bit shader
integers, timeline semaphores, synchronization2, dynamic rendering, and maintenance4.
Shader objects, map_memory2, maintenance5, and maintenance6 are not required. Additional shader
and memory features are enabled only when supported.

## Windowing

```sh
nix-shell --run 'cargo run -- examples/window.resin'
nix-shell --run 'cargo run -- examples/particles.resin'
```

The demo renders a triangle until Escape or the close button is pressed. It uses a `while`
event loop and shares its shader functions with the headless PNG demo in `examples/lib/triangle.resin`.
Resizing scales the fixed-size offscreen image. Building this demo requires `glslc`; running the
resulting executable does not. The PNG demos remain headless.

`particles.resin` initializes 64 particles on the host, updates their positions in a compute
shader, and draws them directly from the same buffer. It reuses pipelines and allocations
across frames, uses a fixed 1/60-second simulation step, and closes on Escape or the close button.
It is a small synchronous demo, not a frame-rate-independent simulation. Its shader functions
and shared data definitions live in `examples/lib/particles.resin`.

Windowing is an ordinary runtime API, exposed by `resin_runtime/window.h` and
`lib/runtime.resin`:

- `resin_window_create`, `resin_window_destroy`, and `resin_window_poll_events` manage
  GLFW windows and events. Close state, framebuffer size, resizing, and GLFW key codes
  are available through the corresponding `resin_window_*` functions.
- `resin_gpu_create_for_window` selects a graphics/compute/present-capable GPU for a window.
  The existing GPU constructors stay headless. There is one GPU per window; multiple
  windows can each have their own GPU.
- `resin_gpu_present` blits an already-submitted `ResinImage` to the window, scaling to
  its framebuffer with FIFO presentation. Swapchains are recreated after resize.
  `RESIN_STATUS_INCOMPLETE` means the frame was skipped (minimized, timed out, or out of date):
  poll events and retry.

Create and use windows and their GPUs on the process main thread. Destroy image/pipeline
resources first, then the GPU, then the window. The GPU retains the native window while
it uses its surface. Do not mix the runtime with independently managed GLFW initialization.

GLFW is statically linked into the runtime and generated executables, and initialized only
when creating a window. Headless programs do not need a display. Initialization and window
creation failures print the GLFW error to stderr and return `RESIN_STATUS_WINDOW_UNAVAILABLE`.
Windowed executables need a display, its system libraries, and a suitable Vulkan driver at runtime,
but no GLFW shared library.
Presentation additionally requires `VK_EXT_swapchain_maintenance1` and its instance
dependencies, so presentation fences can safely govern resource reuse and teardown.
This initial path is deliberately synchronous and presents offscreen images; rendering
directly into swapchain images and multiple frames in flight are not implemented.

## Resources

[No Graphics API — Sebastian Aaltonen](https://www.sebastianaaltonen.com/blog/no-graphics-api)
