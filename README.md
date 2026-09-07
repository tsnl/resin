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
When using Xvfb, set `DISPLAY` to its display and `XDG_SESSION_TYPE=x11` to select GLFW's X11
backend. Unsetting `WAYLAND_DISPLAY` alone is insufficient: GLFW can still connect to a default
Wayland socket. With Xvfb already running, run the full suite without excluding window tests:

```sh
XDG_SESSION_TYPE=x11 RESIN_REQUIRE_GPU=1 RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_WINDOW=1 \
  cargo test --workspace --all-features
```

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
    print(fmt("fibonacci(10) = {0}\n", (fibonacci(10),)));
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

Numeric suffixes are case insensitive and fix the literal's primitive type. Prefix an
integer width with `u` for unsigned values. The formatter writes lowercase suffixes with
an underscore separator; the separator is optional in source except for hexadecimal `b`.

| Suffix | Type | Example |
| --- | --- | --- |
| `b` / `ub` | `sbyte` / `ubyte` (8 bits) | `-128_b`, `255_ub` |
| `h` / `uh` | `short` / `ushort` (16 bits) | `-32768_h`, `65535_uh` |
| `i` / `ui` | `int` / `uint` (32 bits) | `42_i`, `42_ui` |
| `l` / `ul` | `long` / `ulong` (64 bits) | `42_l`, `42_ul` |
| `f` / `d` | `float32` / `float64` | `1.5_f`, `1e3_d` |

These widths are the same on every target. For example, `42UI`, `42uI`, and `42_ui`
all mean the same thing, and format as `42_ui`. Uppercase `42L` now means signed `long`;
use `42_ul` for unsigned `ulong`.

Unsuffixed literals take their type from context, including later assignments and uses.
Integer notation can infer any numeric type; decimal-point and exponent notation infer
floating-point types. Unconstrained integers default to `long`, and floats to `float64`.
A suffixed literal cannot be retyped by an annotation or ascription. Out-of-range literals
are errors, including negative unsigned literals and floating-point overflow. Integer
suffixes require integer notation. Hexadecimal literals accept integer suffixes, such as
`0xffff_ffff_ui` and `0xff_ub`. A signed byte suffix needs its separator (`0x7f_b`);
otherwise `b/B` remains a hex digit. Hex `d/D/f/F` always remain digits.
Use an explicit supported width for otherwise unconstrained shader literals, such as
`var step = 0_i;`: the shader profile supports `ubyte`, `int`, `uint`, `ulong`, and `float32`.

An `if` without `else` has an implicit unit branch, so its body must also yield unit:

```resin
if (count > 0_ul) {
    count := count - 1_ul;
};
```

It behaves like `if (...) { ... } else {}`. Bindings inside its body remain local, and
assignments made only inside that body do not establish definite initialization afterward.

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
    print(fmt("value = {0}\n", (pointer.*,)));
};
```

Holes can nest inside local annotations, local type ascriptions, and function
return annotations: `Ptr<Ptr<_>>`, `Span<_>`, `(_, Ptr<_>)`, and `(int) -> _`
all use the same inference mechanism. Each `_` is independent. Local constraints
can come from later assignments or uses; numeric literals default to `long` or
`float64` only after those constraints have been considered.

Every function uses the same checker, including functions with no explicit holes.
Later uses can constrain unsuffixed local literals; use an annotation or suffix to fix
a local’s type independently of those uses.

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
        ok(value) => { print(fmt("value = {0}\n", (value,))) },
        err(error) => {
            match (error) {
                DivideByZero(zero) => { print("division by zero\n") },
                NegativeInput(negative) => { print(fmt("negative: {0}\n", (negative.value,))) },
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

The standard library's `RuntimeStatus.from_code(code)` converts native status integers to
`Result<(), RuntimeError>`. `RuntimeError` is a union of named errors such as
`InvalidArgument`, `OutOfMemory`, and `IoError`; `UnknownRuntimeError { code }`
preserves unrecognized codes. `RuntimeStatus.code(error)` and `RuntimeStatus.message(error)`
recover the native code and C diagnostic string. Standard-library operations already
return Results, so callers normally use `Gpu.new()?` rather than converting statuses.
Standard-library resources release themselves on scope exit, including early returns
through `?`. Copies retain shared ownership.

### Ownership and methods

`T | None` is an ordinary union containing the builtin singleton `None`. Values of
`T` widen into it directly. Exhaustive `match` handles absence; postfix `optional!`
removes `None` or traps. See [optional values](doc/options.md).

All values use compiler-defined copying. Reading a named value copies it; applying
a function or type consumes the resulting argument value. Infix operators follow
the same rule, and tuple, record, and array constructors consume their initializers
directly into fields or elements. Fresh expression results do not incur an extra
copy or destruction simply because they cross an application boundary.

```resin
struct Resource { handle: Ptr<ubyte> };
impl Resource {
    def drop(self: Ptr<Resource>) = { release_native_handle(self.handle); };
}

// Inside a function:
var shared = Arc<Resource> { handle = acquire_native_handle() };
var alias = shared; // Retains the same allocation; does not copy Resource.
var weak = shared.downgrade();
```

The example assumes native acquire/release declarations for the wrapped library.
`Arc<Resource>(make_resource())` consumes a fresh function result in the same way.
`Ptr<Arc<T>>` points to the handle; `arc.get()` returns a pointer to the pointee.
`weak.upgrade()` returns `Arc<T> | None`, matched with `Arc<T>(owner)` and
`None` arms. See [the shared ownership example](examples/shared.resin).

A copied struct receives its own `drop()`. Native-library authors must therefore
make copies safe or expose an Arc-based interface that avoids copying the inner
owner. There is no static move checking or borrow checking. A wrapper-specific
transfer function can extract its native handle using `pointer.replace(replacement)`
and return a fresh owner while leaving the source disarmed. Raw pointers and spans
still require the programmer to maintain their lifetimes.

Destructors run before fields are released, and scopes clean up in reverse order.
Reference counting and custom destructors are host-only; shaders reject consumption
of managed values while allowing ordinary fields alongside opaque managed slots.
See the [ownership specification](doc/lifetimes.md) for exact rules and
current limitations. The [ownership example](examples/ownership.resin) demonstrates
cleanup on success and early error returns. Do not manually free resources already
owned by a standard-library wrapper.

## Build and run

```sh
cargo run -- examples/eg001.resin
cargo run -- examples/eg009_imports.resin:independent
cargo run -- examples/eg001.resin -o dist/
cargo run -- examples/eg001.resin -o fibonacci
```

Without `-o`, Resin builds in `build/<source-name>-<path-and-entry-hash>/debug/` under cwd and immediately runs the executable.
Ordinary runs compile generated C with `-O0` for fast iteration. Requesting an executable with
`-o` uses `-O3` and the sibling `release/` cache. Both variants are retained, so switching between
them does not force a rebuild. This does not change Cargo's Rust build profile or shader optimization.
Runs inherit cwd and standard streams; Resin returns the program's exit status.
Use `FILE:ENTRY` to select an exported function; omitting `:ENTRY` selects `main`.
A file can export several entry points. Host entries take either `()` or
`(int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>)` for `argc`, `argv`, and `envp`, and return
`int`, `()`, or a Result with either success type;
`main` is just the default name, not special syntax.
Execution begins at the selected function. It must be explicitly exported by the entry file,
including when re-exporting an imported function. Missing or private entries are errors.
There is no module initialization phase.
The selector uses the last colon in the filename, not in its parent directories. For a filename
that itself contains a colon, append `:main` (or another entry) explicitly.

Pass program arguments after `--`: `resin examples/process.resin -- "hello world" --flag`.
Arguments are forwarded literally, including empty strings and leading dashes, without a shell.
The exported entry may use the conventional three-argument form:

```resin
export { main };
import { "std/process.resin" };

def main(argc: int, argv: Ptr<Ptr<ubyte>>, envp: Ptr<Ptr<ubyte>>) -> () = {
    var args = arguments(argc, argv);
    var index = 1_ul;
    while (index < args.length) {
        print(fmt("{0}\n", (argument(args, index),)));
        index := index + 1_ul;
    };
};
```

The runtime deep-copies the argument and environment arrays and strings before entering Resin.
`argv[0]` is the executable invocation name (the cached executable when using `resin FILE`),
`argv[argc]` is null, and `envp` is a null-terminated array of `NAME=value` strings.
`envp` is frozen at startup: later environment mutations do not change its values or lookups.
These process-lifetime views are borrowed and must be treated as read-only; Resin's current
pointer types do not enforce immutability. Unix preserves native bytes, including non-UTF-8;
Windows converts its native wide inputs to UTF-8, replacing unpaired UTF-16 surrogates.

`std/process.resin` provides `arguments(argc, argv)` and `environment(envp)` as pointer spans,
`argument(args, index)` as a checked byte-span view, and `c_string(pointer)` for a valid
NUL-terminated string. `environment_get(envp, name)` takes a NUL-terminated name and returns
`Result<Span<ubyte>, EnvironmentVariableNotFound>`. Lookup is exact and case-sensitive on
all platforms; an empty value succeeds with length zero. It never reads live OS state.
See `examples/process.resin` for looking up a selected variable without dumping the environment.
Program arguments apply only to run mode; `-o` builds the executable to invoke separately.

With `-o PATH` (or `--out PATH`), Resin builds and copies the executable without running it.
A successful build and copy returns status 0, independently of the program's eventual exit status.
An existing directory or trailing separator receives the source name (or `source-entry` for a
non-main entry), with `.exe` on Windows; otherwise PATH names the file exactly. Use an `.exe`
extension for Windows executable filenames.

Compilation follows one pipeline: generate the requested shaders' GLSL, compile it to SPIR-V,
embed those bytes in generated C, then compile and link the executable. Shader stages come from
decorators. To inspect intermediates without running the program, build with `-o PATH` and read
`build/<source-name>-<path-and-entry-hash>/release/program.c` and
`build/shaders/<hash>/shader.glsl` / `shader.spv`. Frontend inspection is available through
`compiler::Session::analyze` and its AST/IR snapshots in the library.

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
Integer arithmetic wraps to its declared width; on the host, invalid division, shifts, and
dynamic array indexes fail with a diagnostic. There is no optimizer or stable generated ABI yet.

## Formatting

The CLI uses the same canonical formatter as `resin-lsp`:

```sh
# Format every .resin file under examples/, including examples/lib/.
nix-shell --run 'cargo run --quiet -- --format examples'

# CI/lint: list files needing formatting without modifying anything.
nix-shell --run 'cargo run --quiet -- --format --check examples'
```

With an installed binary, use `resin --format examples` and
`resin --format --check examples`. `-f` is the short form of `--format`, a boolean
mode flag; `resin examples -f` also works. Formatting accepts multiple files/directories.
Directory traversal selects `.resin` files and does not follow symlinks. Explicit
file arguments are treated as Resin source regardless of extension. Use `--`
before paths beginning with a dash.

`--check` requires `--format`. Formatting cannot be combined with
`-o`/`--out` or compiler/shader options. Running or compiling accepts exactly one
`FILE[:ENTRY]`; formatter paths are literal filenames, including any colons.

Normal mode writes changed files and prints their paths. Check mode prints paths
that would change, returning 0 when all selected files are formatted and 1 on
formatting differences or file/syntax errors. Invalid syntax is reported and left
unchanged; other selected files are still processed. Formatting needs no imports,
entry point, type checking, shader compiler, or GPU execution.

Indentation uses hard tabs. Trailing commas are preserved and, except in singleton tuples
such as `(x,)` and `(int,)`, force multiline
lists; add one to keep long calls or records readable. Comments and literal
contents are preserved, and repeated blank lines collapse to one. Keep one blank
line between example functions and between logical sections inside a function.
The [full formatting rules](resin-lsp/README.md#formatting) also apply to the CLI.
CI checks the examples with `--format --check` on Linux, macOS, and Windows.

## Strings, formatting, and output

String literals have type `Span<ubyte>`. They refer to static UTF-8 bytes, so copying or returning
one does not allocate or introduce an owner. The span's length excludes a trailing NUL; embedded
and explicitly trailing `\0` bytes count toward its length. The escapes are `\n`, `\r`, `\t`,
`\0`, `\"`, and `\\`. Pass `text.data` to C functions that take a NUL-terminated string.
Literal storage may be shared; treat it as read-only.

`fmt(format, arguments)` is a polymorphic host builtin returning `String`, an ordinary nominal
wrapper with a `bytes: Arc<Span<ubyte>>` field. Its allocation contains both the span and its bytes,
plus a trailing NUL. Copying a String retains the allocation; the final owner releases it.
Extracting a raw span or pointer does not retain that owner.
`String.from_str(span)` copies a byte span verbatim into an owned String; braces are ordinary
bytes, and the source need not have a NUL terminator.

```resin
export { main };
import { "std/io.resin" };

def main() -> Result<(), _> = {
    var n = 42;
    var message = fmt("n = {0}\n", (n,));
    Io.stdout().write(message)?;
    Io.stderr().write(fmt("diagnostic: {0}", (message,)))?;
    print("done\n");
    ok(())
};
```

Formats and string arguments accept `Span<ubyte>` or `String`. Other supported arguments are
numbers, booleans, unit, and pointer addresses. The argument tuple is explicit, including the
trailing comma for a single argument. `{0}`, `{1}`, etc. are zero-based and may repeat;
`{{` and `}}` escape braces. Arguments evaluate once in source order, including unused arguments.
Malformed formats and invalid indices terminate with a diagnostic before any formatted output
is written. Formatting itself performs no output.

`Io.stdout()` and `Io.stderr()` return ordinary library `Output` values. Their `write` method
accepts `Span<ubyte> | String`, writes bytes verbatim, flushes, adds no newline, and returns
`Result<(), WriteError>`. The builtin `print(text)` is a stdout shorthand returning unit;
it terminates on an output error. Neither writer interprets braces. Both `fmt` and `print`
are reserved builtins and host-only. No formatting operator or string method is introduced.

Device-backed `Span<ubyte>` values support shader reads and writes using 8-bit storage and
arithmetic extensions. The runtime enables the corresponding Vulkan features when available.
Shader literal spans remain unsupported: GLSL constant arrays cannot supply the device-buffer
addresses used by Resin spans. Pass a span of uploaded bytes in the shader root instead.

## Console input

Import `std/console.resin` for `Console.read_line()`, a line reader implemented in Resin on top of C's
`getchar()`. It grows its buffer as needed and strips LF or CRLF. Write a prompt with `print`
before reading:

```resin
export { main };
import { "std/console.resin" };

def main() -> Result<(), _> = {
    print("Name: ");
    var name = Console.read_line()?;
    print("Hello, ");
    Console.print(name)?;
    print("!\n");
    ok(())
};
```

The result is a shared `InputLine` owner exposing `data: Ptr<ubyte>` and `length: ulong`.
Copies retain its allocation; the final owner frees it. `Console.print(line)` prints the bytes without adding a newline. Empty lines succeed, EOF before any
bytes returns `EndOfInput`, and a final line without a newline succeeds. Read and allocation
failures are also explicit errors. See [the console API](stdlib/README.md#console-input) for
ownership and byte semantics, or run `cargo run -- examples/input.resin`.

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

Syntax keywords (`export`, `import`, `extern`, `type`, `struct`, `impl`, `def`, `var`, `if`,
`else`, `while`, and `match`), primitive type names, `Never`, and
`Ptr`/`Span`/`Arc`/`Weak`/`Result`/`None` are reserved, including in parameters and field names.
Names such as `if_value` are ordinary identifiers. `fmt`, `print`, `ok`, `err`,
`size_of`, `align_of`, and `absurd` are unshadowable compiler builtins, not syntax
keywords: definitions and parameters cannot use those names, but record fields can.

`std/` resolves to the standard-library sources in `stdlib/`, independent of the source file or
working directory. Set `RESIN_STDLIB` to relocate that directory when distributing the compiler.
The native Rust crate lives separately at `resin-runtime/`; it has no dependency on the standard
library. Programs use standard-library wrappers; the integer-status C ABI stays private
to those modules. Public operations are static constructors and instance methods:

- `std/gpu.resin`: devices, allocations, images, pipelines, and command recording.
- `std/window.resin`: windows and input; `Gpu.new_for_window(window)` and `gpu.present(image)` live in `std/gpu.resin`.
- `std/image.resin`: PNG reading and writing.
- `std/status.resin`: `RuntimeStatus` conversion methods and the `RuntimeError` union and its variants.
- `std/graphics.resin`: shared `Position`, `Color`, and `Vertex` types.
- `std/io.resin`: `Io.stdout().write(text)` and `Io.stderr().write(text)`.
- `std/console.resin`: `Console.read_byte()`, `Console.read_line()`, and shared `InputLine` owners with `Console.print(line)`.

The polymorphic `fmt` operation and string-only `print` are compiler builtins; decorated shaders expose `.spirv`.
Runtime flags are static methods, such as `Memory.default()`.
Run `cargo run -- examples/eg009_imports.resin` for an explicitly owned counter, or append
`:independent` to run a second entry that uses two independent counters.

```resin
export { main };
import { "std/gpu.resin" };

def main() -> Result<(), _> = {
    var gpu = Gpu.new()?;
    print("GPU ready\n");
    ok(())
};
```

## Foreign functions

Standard-library modules keep native declarations private and export Resin wrappers that
check statuses before returning out-parameter values. For example:

```resin
export { Gpu };
import { "std/status.resin" };

extern type ResinGpu;
struct GpuOwner { handle: Ptr<ResinGpu> };
type Gpu = Arc<GpuOwner>;
impl GpuOwner {
    def new() -> Result<Gpu, RuntimeError> = {
        var handle = Ptr<ResinGpu>(0_ul);
        RuntimeStatus.from_code(resin_gpu_create(&handle))?;
        ok(Gpu { handle = handle })
    };
    def drop(self: Ptr<GpuOwner>) = { resin_gpu_destroy(self.handle); };
}

extern "resin_runtime.h" def resin_gpu_create(gpu: Ptr<Ptr<ResinGpu>>) -> int;
extern "resin_runtime.h" def resin_gpu_destroy(gpu: Ptr<ResinGpu>);
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
Pointer arithmetic is forbidden. Use array or span indexing, or explicitly convert a pointer
into `ulong`, perform **byte** arithmetic, and convert back when low-level address manipulation
is necessary. Pointer casts and dereferences remain unchecked.

Arrays and spans use `.at(index)` for indexing and return `Ptr<T>`. The index
parameter is `ulong` (unsigned 64-bit); unsuffixed literals infer this type, while
other integer values need an explicit conversion, such as `.at(ulong(i))`:

```resin
var values = [10, 20, 30];
values.at(1).* := 42;
var view = Span<int> { data = Ptr<int>(&values), length = ulong(3) };
var element = view.at(1);
print(fmt("{0}\n", (element.*,)));
```

The original `values(index)` spelling also remains available. `Span<T>` has `data: Ptr<T>`
and `length: ulong` fields. Neither spelling guarantees bounds checking. Host indexing checks
the array or span length and terminates with a diagnostic for negative or out-of-range indices,
before forming an element address. This failure does not unwind automatic cleanup.
Shader array and span indexing is unchecked: callers must keep indices within valid storage;
out-of-range access has undefined behavior. Constructing a span does not validate its pointer,
allocation size, or lifetime.

Initialize output slots before passing their addresses: Resin does not infer initialization
effects from foreign calls. String literals are NUL-terminated; pass their storage with a
span data field, such as `path.data` for `var path = "triangle.png";`.

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
    print(fmt("sum = {0}\n", (sum,)));
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

Shader entry points are ordinary functions with declaration decorators:

```resin
export { main };

@compute_shader
def kernel(index: uint, output: Ptr<uint>) = { output.* := index; };
def main() = {
    var code = kernel.spirv;
    print(fmt("shader size: {0} bytes\n", (code.length,)));
};
```

`@compute_shader`, `@vertex_shader`, and `@fragment_shader` register shader candidates and
validate their stage signatures. Each function accepts one shader decorator. Helpers require
no decorators, and decorated functions remain ordinary host-callable functions. Decorators
currently describe compiler-defined entry points; user-defined compile-time transformers are
not implemented yet.

`kernel.spirv` requests program-lifetime embedded SPIR-V bytes as `Span<ubyte>`. It must name a
decorated function declaration directly, including an imported declaration; runtime function
aliases do not expose `.spirv`. The compiler records artifact requests by declaration identity,
without following function values or analyzing runtime branches. Merely declaring or calling a
decorated function on the host requires no shader compiler. Artifact requests anywhere in the
loaded modules require compilation even when their containing function is not executed.

The Resin pipeline wrappers accept these spans directly:
`gpu.create_compute_pipeline(kernel.spirv)` and
`gpu.create_graphics_pipeline(vertex.spirv, fragment.spirv)`. The private C ABI still uses
pointer/length pairs.

Resin lowers the entry and its reachable named helpers to GLSL, invokes `glslc`, and embeds the
result in generated C. Shader objects are deduplicated and cached under `build/shaders/`;
imported helper changes invalidate them. Copied executables need the Vulkan loader/device,
but neither Resin, source files, nor `glslc` at runtime.

Shaders receive application data through the root address passed to `gpu_dispatch` or
`gpu_draw`. Add a typed pointer as the second tuple element:

```resin
struct Params { count: uint, values: Span<float32>, scale: float32 };

@compute_shader
def kernel(index: uint, root: Ptr<Params>) -> () = {
    if (index < root.count) {
        var p = root.values(index);
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


Device pointers support loads, stores, record fields, explicit casts, and passing to
ordinary helpers. Shared storage supports `int`, `uint`, `float32`, `ulong`, pointers, nonempty
records, arrays, spans, and nominal wrappers. Scalars align to their size; records align to their largest
member, with member and trailing padding. This matches C and GLSL `std430` without requiring
scalar-block-layout support. Generated C asserts sizes, alignments, and member offsets.
Spans occupy 16 bytes (address and length) with alignment 8; arrays retain their element alignment.
Storage containing booleans, unit, or other numeric widths is rejected for now.

Use `buffer.host_pointer()` to initialize mapped data on the CPU. Store
`buffer.device_pointer()` addresses in records consumed by shaders; these are not
interchangeable with host addresses. Pointer types do not enforce the address space or bounds.
Calling the same function on the CPU requires a root containing host pointers instead.

Shader bodies support `ubyte`, 32-bit numbers, `ulong`, booleans, records, nominal types, local mutation,
branches, loops, and direct calls to named Resin helpers. Foreign calls, recursion,
indirect calls, and integer division/remainder/shifts are rejected. Arrays and spans support
unchecked `.at()` indexing. Local addresses
may only be used directly for loads, stores, indexing, and field access; they cannot be stored, passed,
returned, or carried across control-flow edges. Device addresses can. `fmt` and `print` are host-only.

Invocations must avoid racing on shared buffers. Workgroup-local storage, shader barriers, and
atomics are not exposed yet. For multi-pass algorithms, record separate dispatches: the runtime
inserts memory barriers before dispatches and rendering, including compute-to-vertex reads.
Submission currently waits for completion, making mapped results readable by the host.

Build a GPU program with `-o` to inspect its cached GLSL and SPIR-V without executing GPU work,
for example `cargo run -- examples/gradient.resin -o dist/`. Only the host entry selected by
`FILE:ENTRY` needs to be exported; accessing `private_helper.spirv` inside its module does not
require exporting that helper. `--glslc PATH` selects the shader compiler.

## GPU requirements

The runtime uses conventional Vulkan compute and graphics pipelines, with dynamic rendering
and a dynamic viewport/scissor. Graphics currently target one RGBA8 UNORM color attachment,
triangle lists, one sample, and no blending or depth/stencil testing.

A Vulkan 1.3 device must support graphics and compute, buffer device addresses, 64-bit shader
integers, timeline semaphores, synchronization2, dynamic rendering, and maintenance4.
Shader objects, map_memory2, maintenance5, and maintenance6 are not required. Optional memory-priority and pageable-memory features are enabled when supported.
Shader capabilities are limited to the profile Resin emits; externally supplied SPIR-V must
fit that profile too. The emitted byte profile additionally enables supported `storageBuffer8BitAccess` and `shaderInt8` features.

## Windowing

```sh
nix-shell --run 'cargo run -- examples/window.resin'
nix-shell --run 'cargo run -- examples/particles.resin'
```

The demo renders a triangle until Escape or the close button is pressed. It uses a `while`
event loop and defines its decorated shader functions inline, as does the headless triangle demo.
Resizing scales the fixed-size offscreen image. Building this demo requires `glslc`; running the
resulting executable does not. The PNG demos remain headless.

`particles.resin` seeds **1,000,000 particles** with pseudorandom 3D positions and velocities,
then advects them through a Lorenz attractor field on the GPU. A slowly orbiting camera shows
the two swirling lobes. Small sphere billboards use a blue → cyan → yellow → orange → red speed heatmap,
with smooth per-pixel normals, an upper-left light, and specular highlights. Perspective size
and distance fog distinguish near and far particles. Each sphere uses an eight-triangle octagon
(24 million vertices per frame); the silhouette is approximate, and overlaps still follow draw
order because the renderer has no depth buffer. Compute and graphics share a 24 MB particle
buffer, with constant-time vertex indexing and 15,625 compute workgroups per frame. Pipelines
and allocations are reused.

The heatmap anchors are **20, 45, 65, 115, and 220 units per simulation second**. These
approximate the settled cloud's 5th, 25th, 50th, 75th, and 95th speed percentiles, sampled
with 50,000 particles across three seeds at frames 400, 1,000, and 2,000. Colors interpolate
between squared-speed anchors and clamp outside the range, keeping the palette useful
without letting rare fast particles stretch it. The scale stays fixed across frames and
reseeds; initial random velocities fall in the blue end. Lighting and fog modify brightness.

**Left-drag** to orbit horizontally and vertically, and **scroll** to zoom (0.35×–3×).
Either control stops automatic rotation; **Home** resets the view and resumes it. The camera
remains interactive while paused. Press **Space** to pause/resume, **R** to reseed the cloud,
and **Escape** to quit. The initial seed is reproducible; each reseed starts a different cloud.
The 1280×800 offscreen image scales with the window. Each frame advances two fixed 0.005-second simulation steps, so playback
speed depends on rendering throughput. Its decorated shader functions, ordinary helpers, and
shared data definitions live alongside the host code in the same file. Initialization accepts
a host `Span<Particle>`; the shaders use the same allocation's device address.

Input is available through `std/window.resin`. `Window.keys()` names GLFW key codes (`Window.keys().w`,
`Window.keys().space`, `Window.keys().left_shift`); `Window.mouse_buttons()` names the eight mouse buttons.
After polling, `window.key_state(Window.keys().w)` and
`window.mouse_button_state(Window.mouse_buttons().left)` return `ButtonState` records
with `down`, `pressed`, and `released` booleans. A quick tap can set both edge flags;
key repeat does not create another press. `window.key_pressed(key)` queries `down`.

Poll each window once per frame before reading its input. Polling pumps GLFW events for
all windows, then commits that window's snapshot; other windows retain pending input
until their own poll. Edges and scroll reset on the next poll of that window, and repeated
queries read the same snapshot. `window.scroll_delta()` returns accumulated horizontal/vertical
scroll offsets (positive vertical scroll is up). `window.cursor_position()` returns coordinates
in window content units, with a top-left origin and positive y downward, independently of
framebuffer scaling. Both return `Result<(float64, float64), RuntimeError>`.

`window.focused()` reports keyboard focus. `window.capture_cursor(capture)` hides and
captures the cursor for camera controls when `capture` is true, with unbounded virtual
coordinates; set it false to restore normal behavior.
GLFW synthesizes button releases on focus loss. These APIs report physical controls;
they do not decode typed text or implement text composition.

Windowing is an ordinary runtime API, exposed by `resin_runtime/window.h` and
`std/window.resin`:

- `Window.new(width, height, title: String)` returns a shared window owner; use `String.from_str("Resin")` for a literal title. `window.poll_events()` processes
  GLFW events. Close state, framebuffer size, resizing, and GLFW key codes
  are available through the corresponding window methods. Predicates return `bool`;
  fallible operations return Results, including framebuffer size as `(width, height)`.
- `Gpu.new_for_window(window)` selects a graphics/compute/present-capable GPU for a window.
  The existing GPU constructors stay headless. There is one GPU per window; multiple
  windows can each have their own GPU.
- `gpu.present(image)` blits an already-submitted `GpuImage` to the window, scaling to
  its framebuffer with FIFO presentation. Swapchains are recreated after resize.
  Its `Result<bool, RuntimeError>` is `ok(true)` when presented and `ok(false)` when
  skipped (minimized, timed out, or out of date): poll events and retry. Other failures propagate.

Create, use, and release windows and their GPUs on the process main thread. Shared
image/pipeline owners retain their GPU, which retains the window while using its surface.
The final owners release native resources in dependency order. Do not mix the runtime
with independently managed GLFW initialization.

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

### Host byte-array storage

On the host, every `ubyte` array is intentionally a sentinel array, including binary
array literals. Its logical length is N; its physical storage is N+1 bytes with a
trailing zero, alignment 1, and stride N+1 when nested in another array. The sentinel
is outside checked indexing and is preserved by whole-array copies. Empty byte arrays
occupy one zero byte. Embedded zero bytes count toward logical length; C string
functions stop at the first zero. Byte-array storage can be passed to C through an explicit `Ptr<ubyte>` cast;
string literals instead expose their storage as the span's `.data` field.

For packed binary data, use a `Span<ubyte>` over an explicitly allocated N-byte region
and copy only the N logical elements. Copying the entire array representation also
copies its sentinel and is inappropriate for a packed wire format. There is no implicit packed
conversion. `String` uses packed owned bytes rather than the sentinel-array representation.
Byte arrays currently have no shared
host/device storage layout, so shared-layout queries must reject them; they must not
silently report the host representation as a device layout.

### Shared size and alignment

`size_of(T)` and `align_of(T)` return `ulong` constants for the shared host/device
layout of a concrete type. They use the same layout rules as C assertions and GLSL
storage emission. Scalars in the shared profile, padded/nested records, pointers,
spans, and nonempty arrays are supported; unsupported layouts produce a source error.
For an inferred array type, `size_of(array_expression)` queries its type. Expression
operands are checked but not executed, as with C `sizeof`; side effects do not run.
Holes in an explicit type argument are rejected. Byte arrays, empty arrays/records,
booleans, function values, and other types outside the shared profile are rejected.

For example, allocate one record with
`gpu.malloc(size_of(Params), align_of(Params), Memory.default())?`.
For N elements, multiplication remains ordinary `ulong` arithmetic: validate a dynamic
count before multiplying. No unchecked element-count allocation helper is introduced.

### Explicit numeric conversions

`T(value)` converts an already typed numeric value to numeric type T. Integer-to-integer
conversions preserve the mathematical value and trap if it is outside the destination
range, including narrowing and signedness changes. Float-to-integer conversions truncate
toward zero, then range-check; NaNs, infinities, and out-of-range results trap. Checks use
exclusive power-of-two upper bounds, including for 64-bit integer destinations.

Integer-to-float and float narrowing use round-to-nearest, ties-to-even in the normal
floating-point environment. Float32-to-float64 is exact. Float64-to-float32 overflow
produces signed infinity; results below the smallest normal float32 magnitude become
signed zero. NaNs remain NaNs without a payload guarantee. Signed zero is preserved.
C traps abort the process; shader traps stop that invocation and propagate failure
through helper calls. Traps do not unwind automatic cleanup.
A shader trap is not a host-visible Result error, and earlier writes remain visible.

The shared shader profile supports `ubyte`, `int`, `uint`, `ulong`, and `float32` conversions.
Other numeric types remain available on the host and are diagnosed when reached by a
shader. There are no implicit numeric conversions. `float32(1)` still contextually types
a direct unsuffixed literal; suffixes fix source literal types. A conversion of a local
value does not narrow that local's storage merely because of the destination type.
Pointer reinterpretation and nominal record ascription remain separate IR operations.

### Eliminating Never

`absurd(value)` consumes a value of `Never` and has no returning execution path.
Its result type comes from its context; annotate the enclosing result or binding
if that type cannot otherwise be inferred. An inhabited struct or union is rejected.
For example, an infallible Result can be unwrapped without inventing an error value:

```resin
def unwrap(r: Result<int, Never>) -> int = {
    match (r) { ok(n) => { n }, err(impossible) => { absurd(impossible) } }
};
```

The IR explicitly marks elimination as divergent. Its continuation type is only for
checking unreachable code; neither backend constructs a value of that type. C aborts
and shaders stop the invocation if invalid external memory somehow supplies a `Never`.
This defensive trap does not unwind cleanup. Reachable `ok` and `?` paths retain normal
scope destruction. Matches over inhabited variants still require exhaustive, unique arms.

[`T | None`](doc/options.md) supports direct widening, exhaustive matching,
and postfix `!` to exclude `None` or trap.

Inherent methods and associated functions use [`impl` blocks](doc/methods.md).
