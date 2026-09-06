# `resin`

CUDA for graphics. A simple systems programming language targeting host CPUs and Vulkan GPUs.

Builds target 64-bit Linux, macOS, and Windows. GPU execution requires a compatible Vulkan
driver; host-only programs do not require Vulkan or a GPU.

New to the implementation? Start with the [guided repository tour](TOUR.md).

## Development

On Linux or macOS, enter `nix-shell` for Rustup (using `rust-toolchain.toml`), a C compiler,
CMake, GLFW's native build dependencies, and `glslc`. On Linux it also supplies Vulkan tools,
validation layers, and RenderDoc.
The parser is included in `tree-sitter-resin/`; run `cargo test --workspace` directly.
Non-interactive commands work too: `nix-shell --run 'cargo test --workspace'`.

Cargo builds and statically links the GLFW source bundled in `glfw-sys`; no GLFW installation
or library search path is needed. Cargo uses `rust-toolchain.toml` to install the project's
Rust toolchain. Outside Nix:

- Linux: install Rustup, a C compiler, CMake, pkg-config, and the X11, Wayland, and xkbcommon
  development packages, including `wayland-scanner`. For GPU programs, add `glslc`, the Vulkan
  loader, and a Vulkan driver.
- macOS: install Xcode Command Line Tools (`xcode-select --install`), Rustup, and CMake
  (`brew install cmake`). `cargo run -- examples/eg001.resin` then builds and runs a host program.
  For GPU programs, install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#mac), which supplies
  `glslc`, the Vulkan loader, and MoltenVK; use its `setup-env.sh` before running Resin. If Cargo
  strips the loader's search path, run `target/debug/resin` directly from that configured shell.
- Windows: install Rustup's **x86_64-pc-windows-msvc** toolchain, Visual Studio's **Desktop
  development with C++** workload (including a Windows SDK), LLVM Clang, and CMake. Open a
  **Developer PowerShell for VS** targeting x64 and put `clang.exe` and `cmake.exe` on PATH.
  Run `cargo run -- examples/eg001.resin`. For GPU programs, install the
  [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) for `glslc` and a Vulkan-capable GPU driver.

Windows emitted C uses the GNU-style `clang` driver with the MSVC ABI, not `cl` or `clang-cl`.
MinGW and cross-compiling Resin programs are not tested. macOS enables Vulkan portability
enumeration and the portability-subset extension when available, but does not relax the runtime's
Vulkan 1.3 feature requirements. Some MoltenVK devices and presentation paths may still report
`unsupported`; macOS build support is not a promise that every GPU demo works.
Native CI builds and tests the compiler, runtime, and language server on all three systems
without opening windows.

For parser development, install `cargo install --locked tree-sitter-cli --version 0.27.0`.
After changing `tree-sitter-resin/grammar.js`, regenerate from that directory with
`tree-sitter generate --js-runtime native`.
Commit grammar changes and generated files together in this repository.

Backend tests compile generated C with `CC`, defaulting to `cc` on Unix and `clang` on Windows MSVC.
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

## Editor support

The [Zed extension](zed-resin/README.md) provides Resin syntax support and uses
[`resin-lsp`](resin-lsp/README.md) for diagnostics, hover, go-to-definition, and
basic completion. Build the server with `nix-shell --run 'cargo build -p resin-lsp'`.

Both the CLI and LSP use a persistent `resin::compiler::Session`: source overlays,
incremental parsing, dependency invalidation, and immutable checked snapshots
live in the compiler library. The LSP hosts that session for ongoing edits; the
CLI uses it for one build/run invocation.

## Functions and values

```resin
export { main };

def fibonacci(n: int) -> int = {
    if (n <= 1) { n } else { fibonacci(n - 1) + fibonacci(n - 2) }
};

def main() = {
    print("fibonacci(10) = {0}\n", (fibonacci(10),));
};
```

Functions use `def` and are top-level, immutable definitions. Parameter types are explicit;
omitting the result annotation means `()`. Non-unit results require `-> T` or explicit inference
with `-> _`; a non-unit tail expression without an annotation is a type error.
Foreign functions can also omit `-> ()` for C `void` results. Function types still spell out
the result, such as `() -> ()`. All function names are in scope before bodies are checked,
so mutual recursion needs no forward declarations. There are no
lambdas, nested function definitions, or captured environments. Ordinary function values can
be stored, passed, and returned on the host.

Every function takes one argument. Empty parameter lists mean unit `()`; multiple parameters
destructure a tuple. Calling `add(1, 2)` is the same as calling `add(pair)` after `var pair = (1, 2);`.
Function types use the same arrow: `(int, int) -> int`.

Files contain only function, foreign, and type declarations, after their export/import clauses.
There are no global variables or executable top-level statements. Values and mutable state
belong inside functions and are passed explicitly to helpers, by value or pointer.
Local value bindings use `var name = value;`, or `var name: Type;` to reserve uninitialized storage.
Assignment still uses `name := value`. `struct Point { x: int, y: int };` creates a nominal type;
`type Position = Point;` is a transparent alias for that same type. Only `struct` creates
a new nominal identity. Construct values with `Point { x = 1, y = 2 }`.
Record initializers keep bare `name = value`
fields, and parameters and record type fields keep bare `name: Type` declarations.
Type formers use angle brackets: `Ptr<int>`, `Span<float32>`, and `Ptr<Ptr<int>>`.
Parenthesized calls and conversions use `fibonacci(n)` and `int(n)`; brace and bracket
arguments use `Name {...}` and `Converter [...]`. These spaces are a convention, not required syntax.
See `examples/` for functions, recursion, records, pointers, and linked lists.

## Type inference

Write `_` to request a concrete type inferred from the surrounding code:

```resin
export { main };

def next(n: int) -> _ = { n + 1 };

def main() = {
    var value: _;
    var pointer: Ptr<_>;
    value := next(41);
    pointer := &value;
    print("value = {0}\n", (pointer.*,));
};
```

Holes can nest inside local annotations, local type ascriptions, and function
return annotations: `Ptr<Ptr<_>>`, `Span<_>`, `(_, Ptr<_>)`, and `(int) -> _`
all use the same inference mechanism. Each `_` is independent. Local constraints
can come from later assignments or uses; numeric literals default to `int` or
`float64` only after those constraints have been considered.

Function results are inferred from bodies in dependency order, checking mutually
recursive groups together. Callers outside a group cannot determine its return
types. A recursive group without enough information is an error, not a generic
function. Parameters, aliases, struct fields, and
foreign signatures remain fully explicit. Unresolved or infinitely recursive
inferred types are errors; `_` is not a wildcard, unit, or a dynamic type.

Omitting a function result annotation still means unit; inference is opt-in.
Types are fully resolved before IR generation, so C and GLSL share the same
inference behavior. Run `cargo run -- examples/inference.resin` for an example.

## Unions and errors

Unions are structural sets of nominal structs: `A | B`, `B | A`, and `A | B | A`
are the same type. Aliases preserve the identities of their targets. A variant's u32
tag identifies its struct throughout a compiled program; it is not its position in a
particular union. Tags are not persistent IDs across separate builds.
`Never` is the empty union. Structs and aliases are declared in source order;
a struct can refer to itself through a pointer, but aliases cannot introduce cycles.

```resin
struct DivideByZero {};
struct NegativeInput { value: int };
type CalculationError = DivideByZero | NegativeInput;

def divide(n: int, d: int) -> Result<int, DivideByZero> = {
    if (d == 0) { err(DivideByZero {}) } else { ok(n / d) }
};

def calculate(n: int) -> Result<int, _> = {
    if (n < 0) { err(NegativeInput { value = n }) }
    else { ok(divide(n, 2)?) }
};
```

`Result<T, E>` is an ordinary value type: it can be stored, passed, returned, and
nested. `ok(value)` and `err(error)` use the surrounding Result type. Errors are
structs or unions of structs. An inferred error slot collects the least union of
errors that can escape, including through recursive calls. With no errors it becomes
`Never`. Nested holes work too: `Result<Result<int, _>, _>`.
Ambiguous success types still need an annotation; `err(Error {})` alone cannot say
what success would contain.

Postfix `?` evaluates a Result once, unwraps success, or immediately returns its
error. The enclosing function must return a Result whose error set includes that
error. Union values and Result errors can widen by value; pointers remain invariant.
Handle failures with exhaustive, duplicate-free matches:

```resin
def describe(result: Result<int, CalculationError>) = {
    match (result) {
        ok(value) => { print("value = {0}\n", (value,)) },
        err(error) => {
            match (error) {
                DivideByZero(zero) => { print("division by zero\n", ()) },
                NegativeInput(negative) => { print("negative: {0}\n", (negative.value,)) },
            }
        },
    }
};
```

Host entry points can return `Result<(), E>` or `Result<int, E>`; an unhandled
error prints its struct name and exits with status 1. Run `cargo run -- examples/errors.resin`
or `cargo run -- examples/errors.resin:failure` to try both paths.
Helpers using Result and match also compile to GLSL. C uses a tag and a union of
payloads; GLSL uses separate payload fields because it has no native union type.
Shared host/device buffer layouts for tagged values are not yet supported.

The standard library's `status(code)` converts native status integers to
`Result<(), RuntimeError>`; `RuntimeError` retains its numeric `code`.
Register resource cleanup with `defer` before using further fallible operations.

### Deferred cleanup

```resin
def work(fail: bool) -> Result<int, _> = {
    var value = allocate()?;
    defer free(Ptr<ubyte>(value));
    value.* := 42;
    fail_if(fail)?;
    ok(value.*)
};
```

The complete [defer example](examples/defer.resin) defines `allocate`, `free`, and
the fallible `fail_if` helper above. Run `cargo run -- examples/defer.resin` for success,
or `cargo run -- examples/defer.resin:failure` to see cleanup before an error exits.

`defer expression;` registers any expression in the current lexical scope and discards
its result. It is a statement in a chain's prefix, never an expression itself: neither
`var x = defer ...` nor `consume(defer ...)` is valid. For example:

```resin
defer release(resource);
defer if (armed) { release(resource) } else { () };
defer { release(first); release(second); };
```

Statement-only chain blocks yield unit, so the block form uses ordinary expression syntax.
Only registrations reached during execution run. They execute in reverse order on
normal scope exit and early return through `?`, inner scopes before outer ones.
A loop body's defers run at the end of each iteration, not at function exit.
Nested defers follow the same rules.

Names resolve where the defer is registered, but the entire expression—including the
callee, arguments, and conditions—is evaluated when cleanup runs.
Later shadowing does not change the referenced binding. Reads must be definitely
initialized on every exit that executes the defer. The block's result or propagated
error is saved before cleanup, so mutations do not change the value being returned.
Returning a pointer does not copy its pointee: do not free memory that escapes.

Deferred expressions cannot use `?`; handle failures locally with `match`. Cleanup is
ordinary code, not automatic ownership management, and works in C and GLSL wherever
the deferred operations are supported. Aborts, traps, and process termination do not
run defers. The standard library's existing `check(code)` exits the process on failure
and therefore bypasses cleanup; use `status(code)?` in new resource-owning callers.
Existing graphics examples still use explicit cleanup and `check`.

## Build and run

```sh
cargo run -- examples/eg001.resin
cargo run -- examples/eg009_imports.resin:independent
cargo run -- examples/eg001.resin -o dist/
cargo run -- examples/eg001.resin --output exe -o fibonacci
cargo run -- examples/eg001.resin --output c -o fibonacci.c
```

Without `-o`, Resin builds in `build/<source-name>-<path-and-entry-hash>/debug/` under cwd and immediately runs the executable.
Ordinary runs compile generated C with `-O0` for fast iteration. Requesting an executable with
`-o` uses `-O3` and the sibling `release/` cache. Both variants are retained, so switching between
them does not force a rebuild. This does not change Cargo's Rust build profile or shader optimization;
`-o` for textual output (such as `--output c`) only saves that output.
Runs inherit cwd and standard streams; Resin returns the program's exit status.
Use `FILE:ENTRY` to select an exported function; omitting `:ENTRY` selects `main`.
A file can export several entry points. Host entries must be Resin functions of type
`() -> int` or `() -> ()`; `main` is just the default name, not special syntax.
Execution begins at the selected function. It must be explicitly exported by the entry file,
including when re-exporting an imported function. Missing or private entries are errors.
There is no module initialization phase.
The selector uses the last colon in the filename, not in its parent directories. For a filename
that itself contains a colon, append `:main` (or another entry) explicitly.

With `-o PATH` (or `--out PATH`), Resin builds and copies the executable without running it.
This also applies when `--output run` is explicitly selected. A successful build and copy
returns status 0, independently of the program's eventual exit status.
An existing directory or trailing separator receives the source name (or `source-entry` for a
non-main entry), with `.exe` on Windows; otherwise PATH names the file exactly. Use an `.exe`
extension for Windows executable filenames.
`--output exe -o PATH` remains an explicit spelling of the same build-only behavior.
`--output ir`, `ast`, `cst`, and `check` inspect earlier stages; `check` checks syntax only.

Each canonical source path and entry name has a stable directory with separate debug and release
artifacts. Each profile contains generated C, the executable, and an input fingerprint. Unchanged
programs skip C compilation and linking. Generated C, Resin/compiler metadata, included C headers,
the runtime archive, flags, and environment changes invalidate the cache.
Calls for the same source and entry are serialized through building, running, and copying.
Failed rebuilds never run the old executable. This is a whole-program cache, not incremental IR.
Delete `build/` to clean it, including after linked system library changes or changes hidden
behind a compiler wrapper.

Host executables statically link `resin-runtime`; host-only programs do not initialize Vulkan.
Generated C includes `resin_runtime.h` and its hierarchy from `resin-runtime/include`.
Cargo builds the runtime archive alongside the compiler. For relocated installations, set
`RESIN_RUNTIME_INCLUDE` and `RESIN_RUNTIME_LIB` (`libresin_runtime.a` on Unix,
`resin_runtime.lib` on Windows MSVC). To compile emitted C manually on Linux:

```sh
cc -std=c11 -fno-strict-aliasing -I resin-runtime/include fibonacci.c \
  target/debug/deps/libresin_runtime.a -ldl -lpthread -lm -lrt -lutil -o fibonacci
```

`--cc PATH` selects the C compiler without shell parsing.
Integer arithmetic wraps to its declared width; invalid division, shifts, and dynamic array
indexes fail with a diagnostic. There is no optimizer or stable generated ABI yet.

## Printing

`print` is a polymorphic host builtin returning unit. Its one argument is a tuple containing a
format string and a tuple of values, not C-style variadic arguments.

```resin
export { main };

def main() -> () = {
    var n = 42;
    print("x = {0}\n", (n,));
    print("{1}, {0}; literal {{braces}}\n", (n, "hello"));
    print("done\n", ());
};
```

Placeholders are zero-based and may repeat. Newlines are explicit. Invalid formats or indices
fail before the call writes output. Printable values are numbers, booleans, unit, byte strings,
pointer addresses, and nominal wrappers of those. Arguments evaluate once, left-to-right,
including unused ones. `print` is a compiler builtin and cannot be redefined or shadowed.

Strings are UTF-8 byte arrays with `\n`, `\r`, `\t`, `\0`, `\"`, and `\\` escapes.
Host byte-array storage includes an extra trailing NUL, so string literals can be passed to C
without writing `"triangle.png\0"`. The terminator is outside the logical array length and is not
printed. Explicit `\0` bytes remain part of the string, including at the end; Resin's length-based
printing preserves them, while C string functions stop at the first NUL. Import paths, foreign
headers, and shader-stage names still use the decoded literal text, without an added terminator.

## Files and the standard library

Each file has its own scope. An optional `export` clause comes first, followed by an optional
`import` clause, then declarations. Each clause may appear only once:

```resin
export { answer };
import { "helpers.resin", "std/status.resin" };

def answer() -> int = { helper() };
```

Imports bring only the dependency's exported names into the file's flat namespace. Without
an export clause (or with `export {}`), everything is private. An exported function can use
its private helpers and types. Imported bindings may be explicitly re-exported; dependencies
are not implicitly re-exported. Only functions and types can be exported.

Two different bindings with the same name are an error, including imports conflicting with
local definitions. Nested scopes can still shadow names. Re-importing the same binding through
multiple paths is harmless. Ordinary paths resolve relative to the importing file; each canonical
file is loaded once. Imports never execute code. Import cycles are errors; mutually recursive
functions within one file remain supported.
`include` has been replaced by `import`.

Syntax keywords (`export`, `import`, `extern`, `type`, `struct`, `def`, `var`, `if`, `else`, `while`, `match`, and `defer`), primitive
type names, `Never`, and `Ptr`/`Span`/`Result` are reserved, including in parameters and field names. Names such
as `if_value` are ordinary identifiers. `print`, `shader`, `ok`, and `err` are unshadowable compiler builtins,
not syntax keywords: definitions and parameters cannot use those names, but record fields can.

`std/` resolves to the standard-library sources in `stdlib/`, independent of the source file or
working directory. Set `RESIN_STDLIB` to relocate that directory when distributing the compiler.
The native Rust crate lives separately at `resin-runtime/`; it has no dependency on the standard
library. Programs use the standard library's exports, which initially stay close to the runtime's C API:

- `std/gpu.resin`: devices, allocations, images, pipelines, and command recording.
- `std/window.resin`: windows and presentation; import `std/gpu.resin` separately for GPU operations.
- `std/image.resin`: PNG reading and writing.
- `std/status.resin`: `status`, `RuntimeError`, `check`, status strings, and `resin_status_incomplete()`.
- `std/graphics.resin`: shared `Position`, `Color`, and `Vertex` types.

The polymorphic `print` and compile-time `shader` operations remain compiler builtins.
Runtime flags and status constants are zero-argument functions, such as `resin_memory_default()`.
Run `cargo run -- examples/eg009_imports.resin` for an explicitly owned counter, or append
`:independent` to run a second entry that uses two independent counters.

```resin
export { main };
import { "std/gpu.resin" };

def main() -> () = {
    var gpu = Ptr<ResinGpu>(ulong(0));
    var status = resin_gpu_create(&gpu);
    print("GPU creation status: {0}\n", (status,));
    if (status == 0) { resin_gpu_destroy(gpu) } else { () };
};
```

## Foreign functions

Standard-library modules declare native operations with `extern`, exporting appropriate bindings
directly or wrapping them in Resin functions. For example:

```resin
export { ResinGpu, resin_gpu_create };

extern type ResinGpu;
extern "resin_runtime.h" def resin_gpu_create(gpu: Ptr<Ptr<ResinGpu>>) -> int;
```

Foreign headers use the C compiler's include search paths (or an absolute path).
Use forward slashes in header paths, including Windows paths such as `C:/SDK/include/api.h`.

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
effects from foreign calls. String literals are NUL-terminated; pass their storage with a
byte-pointer cast, such as `Ptr<ubyte>(&path)` for `var path = "triangle.png";`.

## Loops

`while` works on both the host and GPU:

```resin
export { main };

def main() -> () = {
    var n = 1;
    var sum = 0;
    while (n <= 10) {
        sum := sum + n;
        n := n + 1;
    };
    print("sum = {0}\n", (sum,));
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
export { main };

def kernel(index: uint) -> uint = { uint(0xff400000) | (index & uint(0xffff)) };
def main() -> () = {
    var code = shader(kernel, "compute");
    print("shader size: {0} bytes\n", (code.length,));
};
```

It takes a named function and a literal stage (`"compute"`, `"vertex"`, or `"fragment"`).
The result is `{ data: Ptr<ubyte>, length: ulong }`: program-lifetime embedded SPIR-V bytes,
passed directly to runtime pipeline creation. The function remains callable normally on the host.
Runtime function aliases and dynamic stage values are not accepted by `shader`.

Resin lowers the entry and its reachable named helpers to GLSL, invokes `glslc`, and embeds the
result in generated C. Shader objects are deduplicated and cached under `build/shaders/`;
imported helper changes invalidate them. Copied executables need the Vulkan loader/device,
but neither Resin, source files, nor `glslc` at runtime.

Shaders receive application data through the root address passed to `resin_gpu_dispatch` or
`resin_gpu_draw`. Add a typed pointer as the second tuple element:

```resin
struct Params { count: uint, values: Ptr<float32>, scale: float32 };

def kernel(index: uint, root: Ptr<Params>) -> () = {
    if (index < root.count) {
        var p = root.values + index;
        p.* := p.* * root.scale;
        ()
    } else { () }
};
```

The entry interfaces are:

- Compute takes `(uint, Ptr<T>)` and returns `()`. Workgroups contain 64 invocations; the
  index is the global X invocation index. Dispatch only in X (`y = z = 1`) and guard any
  excess invocations in the function, as above.
- Vertex takes an `int` vertex index, optionally paired with `Ptr<T>`, and returns
  `{ position: Position, color: Color }`.
  Position has `float32` fields `x, y, z, w`; Color has `r, g, b, a`, in those orders.
- Fragment takes Color, optionally paired with `Ptr<T>`, and returns Color.
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
branches, loops, and direct calls to named Resin helpers. Foreign calls, recursion,
indirect calls, arrays, spans, and integer division/remainder/shifts are rejected. Local addresses
may only be used directly for loads, stores, and field access; they cannot be stored, passed,
returned, or carried across control-flow edges. Device addresses can. `print` is host-only.

Invocations must avoid racing on shared buffers. Workgroup-local storage, shader barriers, and
atomics are not exposed yet. For multi-pass algorithms, record separate dispatches: the runtime
inserts memory barriers before dispatches and rendering, including compute-to-vertex reads.
Submission currently waits for completion, making mapped results readable by the host.

For inspection/export, `--output glsl` or `--output spirv -o PATH` still emits one entry.
Use the same `FILE:ENTRY` selector, for example
`cargo run -- examples/gradient.resin:kernel --output glsl`.
The selected function must be exported; `shader(private_helper, "compute")` inside a function
does not require exporting the helper. `--stage` defaults to compute, and an omitted entry
still defaults to `main`. The old `--entry` option has been removed.
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
`std/window.resin`:

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
