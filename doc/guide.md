# Resin guide

Setup, language features, and GPU programming. For the project's goals, see the
[README](../README.md). Run the commands in this guide from the repository root.

- Get started: [development setup](#development), [build and run](#build-and-run),
  [editor support](#editor-support), [formatting](#formatting).
- Learn the language: [functions and values](#functions-and-values),
  [type inference](#type-inference), [unions and errors](#unions-and-errors),
  [ownership and methods](#ownership-and-methods), [loops](#loops).
- Write applications: [strings and output](#strings-formatting-and-output),
  [console input](#console-input), [modules and libraries](#files-and-the-standard-library),
  [foreign functions](#foreign-functions).
- Use the GPU: [shaders and graphics](#shaders-and-graphics),
  [GPU requirements](#gpu-requirements), [windowing](#windowing).
- Look up [representation details](#representation-details) or [resources](#resources).

## Development

Builds target 64-bit Linux, macOS, and Windows. GPU execution requires a compatible Vulkan
driver; host-only programs do not require Vulkan or a GPU.

New to the implementation? Start with the [guided repository tour](../TOUR.md) and the
[compiler architecture](architecture.md), including the phase crates and their public APIs.

The root is both the `resin` CLI package and a Cargo workspace. Reusable libraries
live under `crates/`; the Zed extension has its own workspace under `editors/zed/`.
Root Cargo commands select `resin`, and `--workspace` builds or tests all native
packages. The `resin` client runs downloaded programs, requests builds, formats source,
and serves stdio LSP. The separate `resin-server` application owns semantic compilation
and native tools. Build/run and LSP require an explicit `RESIN_SERVER` HTTP(S) URL;
formatting works locally without a service.

On Linux or macOS, enter `nix-shell` for Rustup (using `rust-toolchain.toml`), a C compiler,
CMake, Ninja, GLFW's native build dependencies, SPIR-V Tools, and `glslc` for handwritten
test fixtures. On Linux it also supplies Vulkan tools, validation layers, and RenderDoc.
The parser is included in `crates/tree-sitter-resin/`; run `cargo test --workspace` directly.
Non-interactive commands work too: `nix-shell --run 'cargo test --workspace'`.

Cargo builds and statically links the GLFW source bundled in `glfw-sys`; no GLFW installation
or library search path is needed. Cargo uses `rust-toolchain.toml` to install the project's
Rust toolchain. Outside Nix:

- Linux: install Rustup, a C compiler, CMake, Ninja, SPIR-V Tools (`spirv-opt`, `spirv-val`),
  `glslc` for tests, pkg-config, and the X11, Wayland, and xkbcommon development packages,
  including `wayland-scanner`. For GPU execution, add the Vulkan
  loader and a Vulkan driver.
- macOS: install Xcode Command Line Tools (`xcode-select --install`), Rustup, CMake, Ninja, and SPIR-V Tools
  (`brew install cmake ninja spirv-tools shaderc`; Shaderc supplies `glslc` for tests).
  Start the compiler service below before requesting a host program.
  For GPU programs, install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#mac), which supplies
  `spirv-opt`, the Vulkan loader, and MoltenVK; use its `setup-env.sh` before running Resin. If Cargo
  strips the loader's search path, run `target/debug/resin` directly from that configured shell.
- Windows: install Rustup's **x86_64-pc-windows-msvc** toolchain, Visual Studio's **Desktop
  development with C++** workload (including a Windows SDK), LLVM Clang, CMake, Ninja, and SPIR-V Tools. Open a
  **Developer PowerShell for VS** targeting x64 and put `clang.exe`, `cmake.exe`, `ninja.exe`, and `spirv-opt.exe` on PATH.
  Start the compiler service below. For GPU programs, install the
  [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) for SPIR-V Tools and a Vulkan-capable GPU driver.

Start a service in one development shell:

```sh
cargo build -p resin -p resin-server -p resin-runtime
cargo run -p resin-server -- --listen 127.0.0.1:7412
```

In another shell, set `RESIN_SERVER=http://127.0.0.1:7412` before running `resin`
or your editor. On Windows PowerShell, use `$env:RESIN_SERVER = "http://127.0.0.1:7412"`.
The [compiler service guide](compiler-service.md) covers remote deployment, native
installation paths, cache settings, header bundles, and pinned dependencies.
The service never starts automatically and the client has no local compiler fallback.

Windows emitted C uses the GNU-style `clang` driver with the MSVC ABI, not `cl` or `clang-cl`.
MinGW and cross-compiling Resin programs are not tested. macOS enables Vulkan portability
enumeration and the portability-subset extension when available, but does not relax the runtime's
Vulkan 1.3 feature requirements. Some MoltenVK devices and presentation paths may still report
`unsupported`; macOS build support is not a promise that every GPU demo works.
Native CI builds and tests the compiler, runtime, and language server without opening windows.
Pull requests and pushes to `main` run on Linux; manually dispatching the `Build` workflow
also checks Windows and macOS. CI caches Rust dependencies and checks Clippy before building
executables. Example formatting, `cargo test --doc`, and a host smoke check then run before
`cargo nextest` executes integration and GPU tests across suites concurrently. A failed check
skips the remaining steps in its job; grammar checks run in parallel with the native job.
Nextest uses the available CPU count for its total worker limit; `--test-threads N`
overrides it. Every suite that takes the GPU test lock belongs to a group with at most
eight tests in flight, also bounded by that total worker limit. CI selects `--profile ci`,
which keeps the GPU limit at two for the hosted runners and their software Vulkan driver.
GPU pointer fixtures build only their requested host entry and prepare their private
native projects before taking that lock, so compilation can overlap device execution.
The lock still serializes GPU access. `cargo test` uses the same fixture
separation; Nextest also overlaps work across test binaries. Each GPU pointer test thread
retains a lazily initialized GPU singleton to establish device availability.
CI omits Rust debug symbols to reduce compile and native-link work while retaining debug
assertions and overflow checks. Local Cargo profile defaults are unchanged.
CI sets `RESIN_TEST_PARTICLE_COUNT=10000` for the particle compute/render test.
Its default remains one million particles for local stress testing; the particle example
also keeps its one-million default. The smaller test spans the same cloud volume and
checks synchronization, bounds, visible output, and sphere lighting.

For parser development, install `cargo install --locked tree-sitter-cli --version 0.27.0`.
After changing `crates/tree-sitter-resin/grammar.js`, regenerate from that directory with
`tree-sitter generate --js-runtime native`.
Commit grammar changes and generated files together in this repository.

The service's native builds require Ninja and a C compiler, selected with `CC` or server `--cc` (default `cc`
on Unix, `clang` on Windows MSVC). Shader builds use the `spirv-opt` binary from
[SPIR-V Tools](https://github.com/KhronosGroup/SPIRV-Tools),
selected with `SPIRV_OPT` or server `--spirv-opt`; `NINJA` selects the build runner. Resin performs
no tool preflight; required commands report errors when executed. Host-only builds never
invoke `spirv-opt`. SPIR-V embedding invokes the running service executable through the platform
`current_exe` API, so it neither searches PATH for Resin nor mixes compiler versions.
Backend tests build generated projects through the same Ninja toolchain.
Generated shaders are validated with `spirv-val` and optimized with `spirv-opt`. Set
`RESIN_REQUIRE_SPIRV_TOOLS=1` to require these tools in tests. Handwritten GLSL fixtures
in runtime tests still use `glslc` from `PATH` and skip if absent, unless GPU tests are required.
The `gpu` feature enables compiler-to-image integration tests, not a different execution mode:

```sh
RESIN_REQUIRE_GPU=1 RESIN_REQUIRE_SPIRV_TOOLS=1 RESIN_REQUIRE_GLSLC=1 cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Runtime GPU tests expect `glslc` on `PATH` and a working system Vulkan driver.
The development shell supplies the Vulkan loader and window-system libraries through
`LD_LIBRARY_PATH`.
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` enables installed validation layers.

Dedicated [CPU and GPU benchmarks](../benchmarks/README.md) live under `benchmarks/`.
Run them separately with `cargo bench --bench cpu` and `cargo bench --bench gpu`
inside the development environment. Both measure compiled Resin workloads, validate
their outputs, and can save raw timings and hardware metadata with `-- --json PATH`.

Window integration tests use the `gpu` feature and run on a desktop display or Xvfb. Set `RESIN_REQUIRE_WINDOW=1`
to fail instead of skipping when windowing or presentation is unavailable.
When using Xvfb, set `DISPLAY` to its display and `XDG_SESSION_TYPE=x11` to select GLFW's X11
backend. Unsetting `WAYLAND_DISPLAY` alone is insufficient: GLFW can still connect to a default
Wayland socket. With Xvfb already running, run the full suite without excluding window tests:

```sh
XDG_SESSION_TYPE=x11 RESIN_REQUIRE_GPU=1 RESIN_REQUIRE_SPIRV_TOOLS=1 RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_WINDOW=1 \
  cargo test --workspace --all-features
```

## Build and run

```sh
cargo run -- examples/eg001.resin
cargo run -- examples/eg009_imports.resin:independent
cargo run -- examples/eg001.resin -o dist/
cargo run -- examples/eg001.resin -o fibonacci
```

Without `-o`, the client requests the service's Debug native profile (`-O0`),
downloads a verified executable into an owned local temporary directory, and runs it.
With `-o`, it requests Release (`-O3`) and atomically publishes the verified download
at the selected local destination. Debug and Release native caches live on the service;
local paths and execution arguments do not enter build requests. These profiles do
not change Cargo's Rust profile or shader optimization.
Runs inherit cwd and standard streams; Resin returns the program's exit status.
Use `FILE:ENTRY` to select an exported function; omitting `:ENTRY` selects `main`.
A file can export several entry points. Host entries take either `()` or
`(int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>)` for `argc`, `argv`, and `envp`, and return
`int`, `()`, or a union of those types with `Err<E>`;
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
import { "$/string.resin", "$/process.resin", "$/span.resin" };

fn main(argc: int, argv: Ptr<Ptr<ubyte>>, envp: Ptr<Ptr<ubyte>>) -> () {
	let args = arguments(argc, argv);
	let mut index = 1_ul;
	while (index < args.length) {
		print(fmt("{0}\n", (argument(args, index):bytes(),)));
		index = index + 1_ul;
	};
}
```

The runtime deep-copies the argument and environment arrays and strings before entering Resin.
`argv[0]` is the executable invocation name (the owned artifact generation when using `resin FILE`),
`argv[argc]` is null, and `envp` is a null-terminated array of `NAME=value` strings.
`envp` is frozen at startup: later environment mutations do not change its values or lookups.
These process-lifetime views are borrowed and must be treated as read-only; Resin's current
pointer types do not enforce immutability. Unix preserves native bytes, including non-UTF-8;
Windows converts its native wide inputs to UTF-8, replacing unpaired UTF-16 surrogates.

`$/process.resin` provides `arguments(argc, argv)` and `environment(envp)` as pointer spans,
`argument(args, index)` as a checked byte-span view, and `c_string(pointer)` for a valid
NUL-terminated string. `environment_get(envp, name)` takes a NUL-terminated name and returns
`(Span<ubyte> | Err<EnvironmentVariableNotFound>)`. Lookup is exact and case-sensitive on
all platforms; an empty value succeeds with length zero. It never reads live OS state.
See `examples/process.resin` for looking up a selected variable without dumping the environment.
Program arguments apply only to run mode; `-o` builds the executable to invoke separately.

`$/argparse.resin` provides a streaming option parser. Its small specification lists
space-separated spellings, with `|` for aliases and a trailing `=` for a required value:

```resin
let parser = argparse(arguments(argc, argv), "--help|-h --count= --output|-o=");
while (true) {
    let option = match (parser:next()?) {
        Argument(option) => { option },
        None => { break; },
    };
    if (option:named("--count")) {
        let count = argument_integer(option.value)?;
        print(fmt("Count: {0}\n", (count,)));
    };
};
```

The parser accepts `--count 12` and `--count=12`, returns canonical names for aliases,
and yields repeated options in command-line order. Unknown options, missing values,
and values attached to flags return `Err<String>`. It supports options only, without
positional arguments or bundled short flags. A separate value may start with `-`,
including negative numbers and filenames. Flags have an empty value view.
`argument_integer` checks unsigned 32-bit range; `argument_number` parses finite
float32 values from bounded bytes. Returned views borrow the process argument snapshot
and the literal specification. Applications handle defaults, ranges, and help text.


With `-o PATH` (or `--output PATH`, also accepted as `--out PATH`), Resin builds and copies the executable without running it.
A successful build and copy returns status 0, independently of the program's eventual exit status.
An existing directory or trailing separator receives the source name (or `source-entry` for a
non-main entry), with `.exe` on Windows; otherwise PATH names the file exactly. Use an `.exe`
extension for Windows executable filenames.

Compilation happens on the selected service: explicit CST → AST → HIR → LIR →
verified LIR → C/SPIR-V passes produce a Ninja project. Ninja prepares embedded
shaders and compiles captured preprocessed C. Compiler libraries remain independently
usable; see the [architecture](architecture.md#calling-the-passes) for their async APIs.
The client checks the returned target, revision, filename, length, and BLAKE3 digest
before replacing any output. Failed or cancelled downloads preserve an existing file.

Native caches and intermediates live under the service working directory's `build/`.
Completed artifact generations retain independent lifetimes during downloads. CPU
cache heads are shared across callers; equivalent logical sources and import graphs
can reuse editor analysis when a separate CLI later builds the same saved bytes.
Header directory contents and native settings participate in native invalidation.
C compilation consumes captured `.i` bytes, so later header changes affect the next
build. The configured compiler installation and linked libraries must stay stable
during a build. See [service ownership and configuration](compiler-service.md).

Host executables statically link `resin-runtime`; host-only programs do not initialize
Vulkan. The service requires a matching runtime archive and header hierarchy. For
relocated service installations, set `RESIN_RUNTIME_INCLUDE` and `RESIN_RUNTIME_LIB`
(`libresin_runtime.a` on Unix, `resin_runtime.lib` on Windows MSVC) in the service
environment. Native tool flags `--cc` and `--spirv-opt` belong to `resin-server`.
Client `-I DIR` / `--include-root DIR` selects local header directories to upload.

Integer arithmetic wraps to its declared width; on the host, invalid division, shifts, and
dynamic array indexes fail with a diagnostic. There is no optimizer or stable generated ABI yet.

## Editor support

The [Zed extension](../editors/zed/README.md) and
[Helix configuration](../editors/helix/README.md) provide Resin syntax support and launch
`resin --lsp DIR` for diagnostics, hover, go-to-definition, completion, and formatting.
Build the client with `nix-shell --run 'cargo build -p resin'`. Launch the editor
with `RESIN_SERVER` set to a running compatible service. The [client/editor guide](../crates/resin-client/README.md)
describes local formatting, buffer ownership, navigation, and `resin.build`.
Initialization negotiates service capabilities before reporting success.

The client captures exact local files and unsaved buffers, parses preambles for
imports/header dependencies, and sends immutable logical inputs. The service owns
semantic passes and shared caches; compiler phases never read user filesystem paths.
Returned managed definitions become client-owned read-only mirror files. Standard
library selection belongs to the service's `--library-root` / `RESIN_LIBRARY_ROOT`,
not an editor `libraryRoot` option. Editor initialization may set `includeRoots`
for local native header directories.

The coordinator bounds requests, coalesces accepted editor states, and rejects
stale results by document epoch/version and dependencies. Superseded analysis and
shutdown cancel owned HTTP work. The service bounds compiler work with `Execution`
(default: available logical CPUs), shares immutable cache heads through compare-and-swap,
and reserves one slot per native build with Ninja `-j 1`. Started synchronous parser
calls may finish before releasing a slot; native cancellation terminates and reaps
owned process trees. Entry-count capacities are not byte limits.

## Formatting

The CLI and the language server in `resin --lsp DIR` use the same canonical formatter:

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
`-o`/`--out` or native include-root options. Running or compiling accepts exactly one
`FILE[:ENTRY]`; formatter paths are literal filenames, including any colons.

Normal mode writes changed files and prints their paths. Check mode prints paths
that would change, returning 0 when all selected files are formatted and 1 on
formatting differences or file/syntax errors. Invalid syntax is reported and left
unchanged; other selected files are still processed. Formatting needs no imports,
entry point, type checking, service connection, shader compiler, or GPU execution.

Indentation uses hard tabs. Trailing commas are preserved and, except in singleton tuples
such as `(x,)` and `(int,)`, force multiline
lists; add one to keep long calls or records readable. Comments and literal
contents are preserved, and repeated blank lines collapse to one. Keep one blank
line between example functions and between logical sections inside a function.
The [full formatting rules](../crates/resin-client/README.md#formatting) also apply to the CLI.
CI checks the examples with `--format --check` on Linux, macOS, and Windows.

## Functions and values

```resin
export { main };
import { "$/string.resin" };

fn fibonacci(n: int) -> int {
	if (n <= 1) {
		n
	} else {
		fibonacci(n - 1) + fibonacci(n - 2)
	}
}

fn main() {
	print(fmt("fibonacci(10) = {0}\n", (fibonacci(10),)));
}
```
Functions use `fn` and are top-level, immutable definitions. Parameter types are explicit;
omitting the result annotation means `()`. Non-unit results require `-> T` or explicit inference
with `-> _`; a non-unit tail expression without an annotation is a type error.
Foreign functions can also omit `-> ()` for C `void` results. Function types still spell out
the result, such as `() -> ()`. All function names are in scope before bodies are checked,
so mutual recursion needs no forward declarations. There are no
lambdas, nested function definitions, or captured environments. Ordinary function values can
be stored, passed, and returned on the host.

Functions take a sequence of arguments enclosed in parentheses. `add(1, 2)` passes two
arguments; `add(pair)` passes one value and requires a function with one parameter.
Function types list their parameters before the arrow: `(int, int) -> int` takes two integers,
while `((int, int)) -> int` takes one tuple. `f()` has no arguments; `f(())` passes one unit
value. A trailing comma, as in `f(value,)`, does not create a tuple. Use `f((value,))` to pass
a singleton tuple. Access tuple members by index: `pair.0`, `pair.1`. Calls evaluate the
callee first, followed by arguments from left to right.

Files contain only function, foreign, type, and constant declarations, after their export/import clauses.
There are no global variables or executable top-level statements. Values and mutable state
belong inside functions and are passed explicitly to helpers, by value or pointer.
Local bindings use `let name = value;`, or `let name: Type;` to reserve uninitialized storage.
Use `let mut name = value;` for direct reassignment. Assignment uses `name = value` and returns unit. `struct Point { x: int, y: int }` creates a nominal type;
`type Position = Point;` is a transparent alias for that same type. Only `struct` creates
a new nominal identity. Construct values with `Point { x = 1, y = 2 }`.
Record initializers keep bare `name = value`
fields, and parameters and struct fields keep bare `name: Type` declarations.
Type formers use angle brackets: `Ptr<int>`, `Span<float32>`, and `Ptr<Ptr<int>>`.
Calls and conversions require parentheses: `fibonacci(n)`, `int(n)`, and `process([1, 2])`.
`process [1, 2]` and `process { value }` are not calls. Nominal record construction retains
its dedicated `Name { field = value }` syntax. Anonymous record types and values
are not supported; use a named `struct` or a tuple.
See `examples/` for functions, recursion, records, pointers, and linked lists.

### Constants and `iota`

`const` declares a compile-time numeric, `bool`, or `str` value at module or local scope.
Constants have no mutable storage: assignment, address-taking, and reference binding are errors.
Module constants can be exported and may refer to later constants; cycles are errors.
Local constants become visible after their specification, and can shadow outer names.

```resin
const answer: int = 40 + 2;
const (
	read: uint = 1 << iota; // 1
	write: uint = 1 << iota; // 2
	_ = iota; // Skip index 2.
	execute: uint = 1 << iota; // 8
);
const reset = iota; // 0, with type long.
```

`iota` starts at zero in each `const` declaration and increments once per specification,
including discarded `_` specifications. Every specification requires an explicit
`= expression`, including later rows in a group and discarded names. Each row supplies
its own expression list and optional type annotation. A specification can bind multiple
names, such as `a, b: uint = iota, iota + 10;`; their counts must match. Resin requires
semicolons and uses `: Type` annotations and lowercase value names.

Constants use Resin's fixed-width types. An annotation or numeric suffix selects a type;
otherwise numeric inference finishes at the declaration, defaulting to `long` or `float64`.
Later uses do not change that type. Arithmetic, comparisons, logical and bitwise operators,
numeric conversions, constant references, and type layout queries are allowed. Function calls
and aggregate initializers are not constant expressions. Integer overflow, zero divisors,
non-finite floating-point results, and shifts outside the operand width are compile errors,
including in unused declarations. Floating-point operations round at their declared precision.

### Type sizes

`sizeof(Type)` returns the native value size in bytes as `ulong`, including padding:

```resin
struct Pair<T> { first: T, second: T }
const pair_bytes = sizeof(Pair<int>); // 8
const real_bytes = sizeof(float64); // 8
fn size<T>() -> ulong {
	sizeof(T)
}
```
Only a type operand is accepted; `sizeof(value)` is an error. Generic queries resolve when
the function is specialized. A `const` initializer must determine its size at declaration.
Sizes follow Resin's 64-bit native representations, including the one-byte placeholder for
unit/empty records, an element placeholder for empty arrays, and union tags. Opaque foreign
types and `Ref<T>` bindings have no queryable value layout.
The existing `size_of` and `align_of` builtins retain their narrower shared CPU/GPU storage
layout contract. `sizeof` does not imply that a type supports GPU storage.

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
`let mut step = 0_i;`: the shader profile supports `ubyte`, `int`, `uint`, `ulong`, and `float32`.

An `if` without `else` has an implicit unit branch, so its body must also yield unit:

```resin
if (count > 0_ul) {
    count = count - 1_ul;
};
```

It behaves like `if (...) { ... } else {}`. Bindings inside its body remain local, and
assignments made only inside that body do not establish definite initialization afterward.

## Type inference

Write `_` to request a concrete type inferred from the surrounding code:

```resin
export { main };
import { "$/string.resin" };

fn next(n: int) -> _ {
	n + 1
}

fn main() {
	let mut value: _;
	let mut pointer: Ptr<_>;
	value = next(41);
	pointer = &value;
	print(fmt("value = {0}\n", (pointer.*,)));
}
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
Types are fully resolved before IR generation, so C and SPIR-V share the same
inference behavior. Run `cargo run -- examples/inference.resin` for an example.

## Unions and errors

Unions are structural sets of value types: `A | B`, `B | A`, and `A | B | A`
are the same type. Aliases preserve the identities of their targets. A variant's u32
tag identifies its member type throughout a compiled program; it is not its position in a
particular union. Tags are not persistent IDs across separate builds.
`Never` is the empty union. Structs and aliases are declared in source order;
a struct can refer to itself through a pointer, but aliases cannot introduce cycles.

```resin
struct DivideByZero {}
struct NegativeInput { value: int }
type CalculationError = DivideByZero | NegativeInput;

fn divide(n: int, d: int) -> (int | Err<DivideByZero>) {
	if (d == 0) {
		Err(DivideByZero {})
	} else {
		(n / d)
	}
}

fn calculate(n: int) -> (int | Err<_>) {
	if (n < 0) {
		Err(NegativeInput { value = n })
	} else {
		(divide(n, 2)?)
	}
}
```

`T | Err<E>` is an ordinary union. Return a plain `T` for success and `Err(error)`
for failure. Error payloads may be any value type, including `str`, owned `String`,
numbers, and user-defined structs. `Err<E>` is itself a value type; `Err(value)`
infers its payload type from the value and context. `Err<Err<int>>` nests wrappers,
whereas nested unions flatten and duplicate members collapse.

An inferred payload `Err<_>` collects the least union of errors that can escape,
including through recursive calls. With no errors it becomes `Err<Never>`.
Success holes remain monomorphic and need a determining value or annotation.

Postfix `?` evaluates its operand once. If the active member is an `Err`, it
returns that wrapper immediately; otherwise it yields the remaining value.
It preserves all non-error members, so `(int | str | Err<E>)?` yields `int | str`.
The enclosing result must include every propagated error. Union and error values
may widen while preserving their ownership transfer; mutable pointers remain invariant.
Handle failures with exhaustive, duplicate-free matches:

```resin
fn describe(result: (int | Err<CalculationError>))  {
    match (result) {
        int(value) => { print(fmt("value = {0}\n", (value,))) },
        Err(error) => {
            match (error) {
                DivideByZero(zero) => { print("division by zero\n") },
                NegativeInput(negative) => { print(fmt("negative: {0}\n", (negative.value,))) },
            }
        },
    }
}
```

Host entry points can return `(() | Err<E>)` or `(int | Err<E>)`; an unhandled
error prints its payload value and exits with status 1. Run `cargo run -- examples/errors.resin`
or `cargo run -- examples/errors.resin:failure` to try both paths.
Helpers using `Err` and `match` also compile to SPIR-V. C uses a tag and a union of
payloads; shader values use a tag and separate payload fields.
Shared host/device buffer layouts for tagged values are not yet supported.

The standard library's `runtime_status_from_code(code)` converts native status integers to
`(() | Err<RuntimeError>)`. `RuntimeError` is a union of named errors such as
`InvalidArgument`, `OutOfMemory`, and `IoError`; `UnknownRuntimeError { code }`
preserves unrecognized codes. `runtime_status_code(error)` and `runtime_status_message(error)`
recover the native code and C diagnostic string. Standard-library operations already
return error unions, so callers normally use `gpu_new()?` rather than converting statuses.
Standard-library resources release themselves on scope exit, including early returns
through `?`. Explicit clones retain shared ownership.

### Ownership and operations

`T | None` is an ordinary union containing the builtin singleton `None`. Values
widen into it directly; postfix `optional!` removes `None` or traps. See
[optional values](options.md).

Named structs move when assigned, passed, returned, or placed in an aggregate.
The compiler rejects subsequent uses of a moved value. Primitive values copy,
as do tuples, arrays, unions, and error wrappers whose contents all copy.
A struct remains noncopyable even when every field copies. `Copy`/`Clone`
interfaces are deferred; library `clone` functions are ordinary overloads.

```resin
struct Item { value: int }
fn read(item: Ref<Item>) -> int {
	item.value
}
fn consume(item: Item) -> int {
	item.value
}
fn example() -> int {
	let item = Item { value = 21 };
	let before = item:read(); // Borrow without moving.
	let moved = item;
	// item is now unavailable.
	before + consume(moved)
}
```

Structs contain fields only. `item:read()` calls the visible free function
`read(item)`; its signature determines ownership just like any other call.
Operations can be overloaded on all their arguments. Export them separately from
the types they use. See [free operations](methods.md).

`let` is immutable by default; `let mut` permits direct assignment, which returns
unit. Moving an immutable owner is allowed. A declared but uninitialized immutable
binding may be initialized once. Parameters and match binders use the same
identifier pattern, with optional `mut`:

```resin
fn increment(mut value: int) -> int {
	value = value + 1;
	value
}
fn increment_optional(value: int | None) -> int {
	match (value) {
		int(mut number) => {
			number = number + 1;
			number
		},
		None => {
			0
		},
	}
}
```

Ownership analysis runs during HIR completion, after inference, including unused
function bodies. It tracks initialization and moved field paths in evaluation
order, intersects reachable branch states, and checks loop backedges. Disjoint
fields remain usable after a partial move; fields of a type with a drop hook
cannot be moved out. LIR specializes operations and emits transfers and cleanup.

References and pointers have unchecked lifetimes and mutable aliasing. A
`Ref<T>` parameter accepts a place or a temporary that survives the full expression.
Reading a noncopyable value through an alias is rejected; use
`pointer:replace(replacement)` to transfer it and leave valid storage behind.
An immutable reference binding can still write its referent. See
[references](references.md) for the distinction between binding mutability and
unchecked alias access.

Shared owners retain allocations explicitly:

```resin
import { "$/shared.resin", "$/status.resin" };
fn example() -> int | Err<OutOfMemory> {
	let owner = arc_ptr_alloc(21_i)?;
	let retained = owner:clone();
	let moved = owner;
	retained:get().* + moved:get().*
}
```

`arc_ptr_alloc(initial)` consumes its initializer into one allocation.
`arc_span_alloc(count, initial)` repeats a copyable initializer; move-only values
cannot be repeated. Allocation checks size arithmetic and reports `OutOfMemory`.
The final owner destroys elements in reverse order. `get` borrows a pointer or
span without retaining the allocation; keep an owner alive while using that view.
`downgrade` produces a weak handle, and `upgrade` returns a retained handle or
`None`. Reference counting protects allocation lifetime, not concurrent access or
raw aliases.

| Ownership | One value | Sequence |
| --- | --- | --- |
| Nonowning | `Ptr<T>` | `Span<T>` |
| Shared host | `ArcPtr<T>` | `ArcSpan<T>` |
| Weak host | `WeakPtr<T>` | `WeakSpan<T>` |
| GPU | `GpuPtr<T>` | `GpuSpan<T>` |

A free `fn drop(value: Ptr<Item>) { ... }` declared alongside `Item` supplies its
cleanup hook. It runs before fields are destroyed. Scope exit and early returns
clean up owners in reverse order; moves suppress cleanup of the source. Custom
destruction and reference counting remain host-only. See [lifetimes](lifetimes.md)
and [the ownership example](../examples/ownership.resin).

Effects, traits, inheritance, and general compile-time execution of functions are
not part of this language revision. Numeric constants and layout queries retain
their existing compile-time behavior.

## Loops

`while` works on both the host and GPU:

```resin
export { main };
import { "$/string.resin" };

fn main() -> () {
	let mut n = 1;
	let mut sum = 0;
	while (n <= 10) {
		sum = sum + n;
		n = n + 1;
	};
	print(fmt("sum = {0}\n", (sum,)));
}
```

`true` and `false` are literals of type `bool`.

The condition must be boolean and is evaluated before every iteration. The body has its own
scope; its result is discarded, and the loop returns `()`. As with other expression statements,
the trailing semicolon is required unless the loop is the enclosing block's final expression.
The body may run zero times, so initializing a variable only in the body does not make it
definitely initialized afterward. `break;` exits the nearest loop; `continue;` starts its next iteration. Both
are valid inside a loop body, including nested branches, and destroy exited
scope owners before transferring control. Loop conditions cannot contain these exits.

## Generic functions and structs

Named type parameters bind one type throughout a definition. Calls infer their
arguments from values and expected results; explicit function arguments use `::<T>`:

```resin
struct Pair<T> { left: T, right: T }
type View<T> = Ptr<Pair<T>>;

fn first<T>(pair: Pair<T>) -> T {
	pair.left
}
fn identity<T>(value: T) -> T {
	value
}

fn example() -> int {
	let pair = Pair<int> { left = 40, right = 2 };
	first(pair) + identity::<int>(2)
}
```
Struct constructors take explicit type arguments. Different applications retain
distinct nominal types, even if their layouts agree; aliases keep their target's
identity. Pointer fields may recurse through the same generic declaration. Local
structs remain field-only and can use enclosing function type parameters.

Free operations declare their own complete type parameter list. Colon calls
pass the receiver first and can infer generic arguments from the whole signature:

```resin
struct Cell<T> { value: T }
fn replace_with<T, U>(cell: Cell<T>, value: U) -> Cell<U> {
	Cell<U> { value = value }
}
fn take<T>(cell: Cell<T>) -> T {
	cell.value
}
fn example() -> int {
	let original = Cell<ulong> { value = 7 };
	let changed = original:replace_with::<ulong, int>(42);
	changed:take()
}
```

Generic bodies retain their visible overload candidates. Once operand types are
concrete, signature substitution selects one applicable operation. Ambiguity is
an error; a failing body is never used to discard a candidate. Caller imports do
not change the definition's candidate set. See [free operations](methods.md).

`_` is a weak inference variable in local annotations, function results, and
explicit applications. It may resolve to a named parameter but never creates
another generic parameter. Unsuffixed literals follow expected types before
falling back to the ordinary integer/float defaults. Template bodies retain their
type relationships; unsupported concrete operations and layouts fail when an
application is required. Specialization substitutes these relations and selects concrete operations; it does
not reopen source inference or use function bodies to deduce signature parameters.

## Operator overloading

Declare free functions with Python-style operator names:

```resin
struct Vec2 { x: int, y: int }
fn __add__(left: Ref<Vec2>, right: Ref<Vec2>) -> Vec2 {
	Vec2 { x = left.x + right.x, y = left.y + right.y }
}
fn __mul__(scale: int, value: Ref<Vec2>) -> Vec2 {
	Vec2 { x = scale * value.x, y = scale * value.y }
}
```

Both operands participate in resolution. These signatures borrow vectors;
value parameters would move them. Operands evaluate once, left to right.
`__add__(left, right)` and `left:__add__(right)` select the same overload as
`left + right`. Export the function to make it visible in another module.
See the [operator mapping](methods.md#operator-overloading) and
[vector example](../examples/operators.resin).

## Strings, formatting, and output

String literals have primitive type `str`, distinct from raw `Span<ubyte>` views and owned
`String` values. They expose `data: Ptr<ubyte>` and `length: ulong` over static UTF-8 bytes,
so copying or returning one does not allocate or introduce an owner. The length excludes a
trailing NUL; embedded and explicitly trailing `\0` bytes count toward its length. The escapes are `\n`, `\r`, `\t`,
`\0`, `\"`, and `\\`. Pass `text.data` to C functions that take a NUL-terminated string.
Literal storage may be shared; treat it as read-only.

Import `$/span.resin` and use `bytes(text)` to explicitly borrow the bytes of a `str`; this preserves its pointer
and length without copying. There is no implicit conversion, and arbitrary byte spans cannot
be converted to `str`. `text:at(index)` returns a byte reference (`Ref<ubyte>`) and uses a `ulong` index.

Import `$/string.resin` for `String`, `fmt`, and `print`.
`fmt(format, arguments)` is an ordinary generic host function returning `String`,
a source wrapper with a `storage: ArcSpan<ubyte>` field. Its allocation owns the bytes and an additional
trailing NUL outside their logical length. Cloning a String retains the allocation; the final owner releases it.
Extracting a raw span or pointer does not retain that owner.
`string_from_str(text)` copies a `str` verbatim into an owned String. For raw bytes, use
`string_from_bytes(span)`; the span need not be UTF-8 or have a NUL terminator. Both constructors
preserve embedded NULs and treat braces as ordinary bytes.

```resin
export { main };
import { "$/io.resin", "$/string.resin" };

fn main() -> (() | Err<_>) {
	let n = 42;
	let message = fmt("n = {0}\n", (n,));
	io_stdout():write(message)?;
	io_stderr():write(fmt("diagnostic: {0}", (message:bytes(),)))?;
	print("done\n");
	(())
}
```

The format accepts `str`, `Span<ubyte>`, or `String`. In the argument tuple,
use `view:bytes()` or `message:bytes()` for span and String values. These methods
provide an explicit structural byte view to the formatting primitive. Literal
`str` arguments work directly. Other supported arguments are numbers, booleans,
unit, and pointer addresses. The argument tuple is explicit, including the
trailing comma for a single argument. `{0}`, `{1}`, etc. are zero-based and may repeat;
`{{` and `}}` escape braces. Arguments evaluate once in source order, including unused arguments.
Malformed formats and invalid indices terminate with a diagnostic before any formatted output
is written. Formatting itself performs no output.

`io_stdout()` and `io_stderr()` return ordinary library `Output` values. Their `write` method
has overloads for `str`, `Ref<Span<ubyte>>`, and `Ref<String>`, writes bytes verbatim, flushes, adds no newline, and returns
`(() | Err<WriteError>)`. The library function `print(text)` is a stdout shorthand returning unit;
it terminates on an output error. Neither writer interprets braces. Both `fmt` and `print`
are ordinary source functions and host-only.

Byte arrays and device-backed `Span<ubyte>` values support shader reads and writes using
8-bit storage and arithmetic extensions. The runtime enables the corresponding Vulkan features when available.
Shader `str` literals remain unsupported: the backend does not yet provide addressable
constant storage for the device addresses used by Resin spans. Pass a span of uploaded bytes in the shader root instead.

## Console input

Import `$/console.resin` for `console_read_line()`, a line reader implemented in Resin on top of C's
`getchar()`. It grows its buffer as needed and strips LF or CRLF. Write a prompt with `print`
before reading:

```resin
export { main };
import { "$/string.resin", "$/console.resin" };

fn main() -> (() | Err<_>) {
	print("Name: ");
	let name = console_read_line()?;
	print("Hello, ");
	console_print(name)?;
	print("!\n");
	(())
}
```

The result is a shared `InputLine` owner; `line:get()` borrows its `Span<ubyte>` view.
Explicit clones retain its allocation; the final owner frees it. `console_print(line)` prints the bytes without adding a newline. Empty lines succeed, EOF before any
bytes returns `EndOfInput`, and a final line without a newline succeeds. Read and allocation
failures are also explicit errors. See [the console API](../resin/README.md#console-input) for
ownership and byte semantics, or run `cargo run -- examples/input.resin`.

## Files and the standard library

Each file has its own scope. Optional `export`, grouped `extern`, and `import`
clauses appear in that order before declarations. Each clause may appear only once:

```resin
export { answer };
import { "helpers.resin", "$/status.resin" };

fn answer() -> int  { helper() }
```
Imports bring only the dependency's exported names into the file's flat namespace. Without
an export clause (or with `export {}`), everything is private. An exported function can use
its private helpers and types. Imported bindings may be explicitly re-exported; dependencies
are not implicitly re-exported. Functions, types, and constants can be exported.

Different functions with the same name form an overload set, including imported
functions. Calls must select exactly one signature. Conflicting types, constants,
and nonfunction bindings remain errors. Nested scopes can still shadow names. Re-importing the same binding through
multiple paths is harmless. Import paths without a leading `$` resolve relative to the
importing file; each canonical file is loaded once. Imports never execute code. Import cycles are errors; mutually recursive
functions within one file remain supported.
`include` has been replaced by `import`.

Syntax keywords (`export`, `import`, `extern`, `type`, `struct`, `fn`, `let`, `mut`, `const`, `sizeof`, `if`,
`else`, `while`, and `match`), primitive type names, `Never`, and
`Ptr`, `Err`, `None`, and the opaque compiler handle types are reserved,
including in parameters and field names. Wrapper names such as `Span`, `ArcPtr`,
and `GpuSpan` are ordinary source type names.
Names such as `if_value` are ordinary identifiers. `size_of`, `align_of`, `iota`, and `absurd` are unshadowable compiler builtins, not syntax
keywords: definitions and parameters cannot use those names, but record fields can.

Imports beginning with `$/` resolve from the service's frozen [`resin/`](../resin/) library root,
independent of the source file or working directory. For example, `$/gpu.resin`
loads `resin/gpu.resin`; a future `$/math/matrix.resin` would load `resin/math/matrix.resin`.
Set `RESIN_LIBRARY_ROOT` or `--library-root` on the service to choose that snapshot.
Pinned packages are also service-owned; see [managed dependencies](compiler-service.md#pinned-dependencies).
Other imports resolve relative to the importing file. Files have explicit imports and exports;
directories need no manifest or special entry file.
The native Rust crate lives separately at `crates/resin-runtime/`; it has no dependency on the standard
library. Programs use standard-library wrappers; the integer-status C ABI stays private
to those modules. Public operations are exported free functions, including constructors and colon-call operations:

- `$/gpu.resin`: devices, allocations, images, pipelines, and command recording.
- `$/window.resin`: windows and input; `gpu_new_for_window(window)` and `gpu:present(image)` live in `$/gpu.resin`.
- `$/image.resin`: PNG reading and writing.
- `$/status.resin`: Native status conversion functions and the `RuntimeError` union and its variants.
- `$/graphics.resin`: shared `Position`, `Color`, and `Vertex` types.
- `$/io.resin`: `io_stdout():write(text)` and `io_stderr():write(text)`.
- `$/span.resin`: borrowed `Span<T>` and `bytes(text)` for literal byte views.
- `$/shared.resin`: `arc_ptr_alloc::<T>(initial)?`, `arc_span_alloc::<T>(count, initial)?`, and weak owners.
- `$/string.resin`: owned `String`, `string_from_str`, `string_from_bytes`, `fmt`, and `print`.
- `$/console.resin`: `console_read_byte()`, `console_read_line()`, and shared `InputLine` owners with `console_print(line)`.

Pass decorated shader declarations directly to GPU pipeline creation. Compiled shader
representations belong to code generation and the runtime; functions expose no bytecode property.
GPU memory modes are exported `int` constants: `memory_default`, `memory_gpu`, and
`memory_readback`. Pass them directly, for example `gpu:alloc_in::<uint>(count, memory_readback)`.
Run `cargo run -- examples/eg009_imports.resin` for an explicitly owned counter, or append
`:independent` to run a second entry that uses two independent counters.

```resin
export { main };
import { "$/gpu.resin", "$/string.resin" };

fn main() -> (() | Err<_>) {
	let gpu = gpu_new()?;
	print("GPU ready\n");
	(())
}
```

## Foreign functions

Standard-library modules keep native declarations private and export Resin wrappers that
check statuses before returning out-parameter values. For example:

```resin
export { Gpu, gpu_new };
extern {
	"resin_runtime.h": {
		fn resin_gpu_create(gpu: Ptr<Ptr<ResinGpu>>) -> int;
		fn resin_gpu_destroy(gpu: Ptr<ResinGpu>);
	},
};
import { "$/status.resin", "$/shared.resin" };

extern type ResinGpu;
struct GpuOwner { handle: Ptr<ResinGpu> }
fn drop(owner: Ptr<GpuOwner>) {
	if (ulong(owner.handle) != 0_ul) {
		resin_gpu_destroy(owner.handle);
	};
}
struct Gpu { owner: ArcPtr<GpuOwner> }
fn gpu_new() -> Gpu | Err<RuntimeError> {
	let owner = arc_ptr_alloc(GpuOwner { handle = Ptr<ResinGpu>(0_ul) })?;
	runtime_status_from_code(resin_gpu_create(&owner:get().handle))?;
	Gpu { owner = owner }
}
```
Declare foreign functions in a top-level `extern` block after any `export` clause
and before any `import` clause. Each header names a group of ordinary `fn`
signatures. Groups use comma separators with an optional trailing comma; the block
ends with `};`. The block and individual groups may be empty. Every declared header
remains a native dependency even when none of its functions is called, including
empty groups. The old `extern "header.h" fn ...;` spelling
is no longer accepted.

Header groups do not introduce a scope: their functions have the module's usual
export rules, and signatures can use types from its imports and declarations.
Opaque foreign types remain standalone `extern type Name;` declarations.

Foreign headers resolve through ordered client `-I` / `--include-root` directories,
the declaring file's directory, advertised managed header roots, then target system
headers. Local absolute spellings must resolve on the client. Selected directories are
uploaded as immutable bundles; their transitive includes must remain within those
bundles or configured service toolchain roots. See [C header directories](compiler-service.md#c-header-directories).
Use forward slashes in header paths, including Windows paths such as `C:/SDK/include/api.h`.

The prototype targets 64-bit hosts. Foreign functions accept scalar/pointer parameters and return a scalar, pointer, or unit.
The wrapper forwards each Resin parameter as a separate C argument. Opaque `extern type`
declarations name C structs and may only be used behind pointers; aggregates by value,
variadic calls, and C callbacks are not supported yet.

This is an unchecked C boundary: declarations must match the header's ABI, and callers own
pointer validity, lifetimes, buffer lengths, and synchronization. `&place` takes an address;
`pointer.*` dereferences it. On the host, explicit casts allow pointer-to-pointer and
pointer-to-`ulong` roundtrips. There is no borrow checker; addresses of locals must not outlive their storage.
Pointer arithmetic is forbidden. Use array or span indexing, or explicitly convert a pointer
into `ulong`, perform **byte** arithmetic, and convert back when low-level address manipulation
is necessary on the host. Shaders reject pointer casts; use typed pointers and indexing.
Host pointer casts and all raw pointer dereferences remain unchecked.

Arrays, spans, and `str` use `:at(index)` for indexing and return `Ref<T>` (`Ref<ubyte>` for `str`).
The index parameter is `ulong` (unsigned 64-bit); unsuffixed literals infer this type, while
other integer values need an explicit conversion, such as `:at(ulong(i))`:

```resin
let values = [10_i, 20, 30];
values:at(1) = 42;
let view = Span<int> { data = &values:at(0), length = 3_ul };
let element = view:at(1);
print(fmt("{0}\n", (element,)));
```
Use `let element: Ref<int> = view:at(1);` to retain an alias instead of copying the
value. Reference parameters and results expose the same place semantics in user
functions; see [references](references.md) for binding, lifetime, and migration rules.

Arrays retain the original `values(index)` spelling; source spans use `:at(index)`. `Span<T>` has `data: Ptr<T>`
and `length: ulong` fields. Host indexing checks
the array or span length and terminates with a diagnostic for negative or out-of-range indices,
before forming an element address. This failure does not unwind automatic cleanup.
Shader array and span indexing is unchecked: callers must keep indices within valid storage;
out-of-range access has undefined behavior. Constructing a span does not validate its pointer,
allocation size, or lifetime.

Initialize output slots before passing their addresses: Resin does not infer initialization
effects from foreign calls. String literals are NUL-terminated; pass their storage with a
span data field, such as `path.data` for `let mut path = "triangle.png";`.

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
import { "$/gpu.resin" };

@compute_shader
fn kernel(index: ulong, output: Ptr<ulong>) {
	output.* = index;
}
fn main() -> (() | Err<_>) {
	let gpu = gpu_new()?;
	let pipeline = gpu:create_compute_pipeline(kernel)?;
	(())
}
```

`@compute_shader`, `@vertex_shader`, and `@fragment_shader` register shader candidates and
validate their stage signatures. Each function accepts one shader decorator. Helpers require
no decorators, and decorated functions remain ordinary host-callable functions. Decorators
currently describe compiler-defined entry points; user-defined compile-time transformers are
not implemented yet.

Create typed pipelines from shader declarations with
`gpu:create_compute_pipeline(kernel)` and
`gpu:create_graphics_pipeline(vertex, fragment)`. Creation requests their embedded
shader representation automatically and preserves the shader stage and root type.
Creation currently requires direct declarations, including imported declarations; runtime
function aliases are not accepted yet. A reachable pipeline creation site requests its shaders
even if its branch is not executed. Merely declaring a shader or calling it on the host
requires no shader optimizer. There is no `.spirv` property; inspect artifacts in the build
cache instead. The current Vulkan implementation's private C ABI uses pointer/length pairs.

Resin lowers the entry and its reachable named helpers directly to SPIR-V. The toolchain runs
`spirv-opt -O --target-env=vulkan1.3` and embeds the optimized binary in generated C headers.
Shader objects are deduplicated and retained with their generated project;
imported helper changes invalidate them. Copied executables need the Vulkan loader/device,
but neither Resin, source files, nor `spirv-opt` at runtime.

Pass a typed pipeline and its host arguments to
`commands:dispatch(pipeline, arguments, x, y, z)` or
`commands:draw(pipeline, arguments, count)`. The compiler checks the arguments against
the pipeline and projects GPU views internally. Shader entries keep a typed pointer
as their second parameter:

```resin
struct Params { values: Span<float32>, scale: float32 }

@compute_shader
fn kernel(index: ulong, root: Ptr<Params>) -> ()  {
    if (index < root.values.length) {
        let mut p: Ref<float32> = root.values:at(index);
        p = p * root.scale;
        ()
    } else { () }
}
```
The entry interfaces are:

- Compute takes `(ulong, Ptr<T>)` and returns `()`. Call `gpu:compute_workgroup_size()`
  to get the `ulong` number of invocations per workgroup for that `Gpu`. The runtime
  selects it from the device's reported default subgroup size, bounded by its workgroup
  limits, and specializes each compute pipeline to match. The value stays fixed for
  that GPU's lifetime. With `let mut workgroup_size = gpu:compute_workgroup_size();`, compute
  dispatch groups with `uint((count + workgroup_size - 1_ul) / workgroup_size)`.
  The index is the global X invocation index. Dispatch only in X (`y = z = 1`) and guard
  any excess invocations in the function, as above.
- Vertex takes an `int` vertex index, optionally paired with `Ptr<T>`, and returns
  `Vertex` with `position: Position` and `color: Color` fields.
  Position has `float32` fields `x, y, z, w`; Color has `r, g, b, a`, in those orders.
- Fragment takes Color, optionally paired with `Ptr<T>`, and returns Color.

Allocate typed GPU storage with `gpu:create(value)?` (an inferred `GpuPtr<T>`) or
`gpu:alloc::<T>(count)?`. A host launch record replaces shader `Ptr<T>`
and `Span<T>` fields with `GpuPtr<T>` and `GpuSpan<T>` values:

```resin
let values = gpu:alloc::<float32>(1024)?;
let mut index = 0_ul;
while (index < values.length) {
    values:at(index):store(1.0_f);
    index = index + 1_ul;
};
struct HostParams { values: GpuSpan<float32>, scale: float32 }
let pipeline = gpu:create_compute_pipeline(kernel)?;
let commands = gpu:start_command_recording()?;
commands:dispatch(pipeline, HostParams { values = values, scale = 2.0_f }, 16, 1, 1)?;
commands:submit()?;
```

Projection checks the shader root layout, translates owning views internally, and
retains every referenced allocation. Use `commands:draw(pipeline, None, count)` for graphics
shaders without a root. Successful recording retains arguments and allocations
through synchronous submission or cancellation. They must belong to the recording's
GPU. See [GPU buffers](gpu-buffers.md).

Device pointers support loads, stores, record fields, typed indexing, and passing to
ordinary helpers. Pointer reinterpretation and pointer/integer conversions are host-only. Shared storage supports `ubyte`, `int`, `uint`, `long`, `float32`, `ulong`, pointers, nonempty
records, arrays, spans, and nominal wrappers. Scalars align to their size; records align to their largest
member, with member and trailing padding. This matches C and Vulkan's base alignment rules without requiring
scalar-block-layout support. Generated C asserts sizes, alignments, and member offsets.
Spans occupy 16 bytes (address and length) with alignment 8; arrays retain their element alignment.
Storage containing booleans, unit, or other numeric widths is rejected for now.

Host `GpuPtr` and `GpuSpan` operations retain their allocation, including indexing,
and slicing. `load`, `store`, and `replace` access elements on the host.
`:read_only()` and `:write_only()` narrow per-view
access permissions. Host accesses check bounds, alignment, mapping, permissions,
and pending recorded GPU use. `:copy_from(Span<T>)` uploads a bounded host span into the beginning of a writable,
host-visible GPU span; its source length must fit the destination.
`:copy_to(Span<T>)` copies into caller-owned host
memory. GPU views cannot be converted to raw `Ptr` values; the compiler's shader
projection is the host-to-device address conversion boundary. GPU buffer elements
must have a shared layout without pointers, spans, owners, or drop hooks.

Shader bodies support `ubyte`, 32-bit numbers, `long`, `ulong`, booleans, records, nominal types, local mutation,
branches, loops, and direct calls to named Resin helpers. Foreign calls, recursion,
indirect calls, and integer division/remainder/shifts are rejected. Arrays and spans support
unchecked `:at()` indexing. Local addresses
may only be used directly for loads, stores, indexing, and field access; they cannot be stored, passed,
returned, or carried across control-flow edges. Device addresses can. `fmt` and `print` are host-only.

Invocations must avoid racing on shared buffers. Workgroup-local storage, shader barriers, and
atomics are not exposed yet. For multi-pass algorithms, record separate dispatches: the runtime
inserts memory barriers before dispatches and rendering, including compute-to-vertex reads.
Submission currently waits for completion, making mapped results readable by the host.

Build a GPU program with `-o` to inspect its unoptimized and optimized SPIR-V without executing GPU work,
for example `cargo run -- examples/gradient.resin -o dist/`. Only the host entry selected by
`FILE:ENTRY` needs to be exported; passing a private shader declaration to pipeline creation
inside its module does not require exporting that shader. `resin-server --spirv-opt PATH` selects the service shader optimizer.

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
Resizing scales the fixed-size offscreen image. Building this demo requires `spirv-opt`; running the
resulting executable does not. The PNG demos remain headless.

`particles.resin` seeds **1,000,000 particles** with pseudorandom 3D positions and velocities,
then advects them through a Lorenz attractor field on the GPU. A slowly orbiting camera shows
the two swirling lobes. Small sphere billboards use a blue → cyan → yellow → orange → red speed heatmap,
with smooth per-pixel normals, an upper-left light, and specular highlights. Perspective size
and distance fog distinguish near and far particles. Each sphere uses an eight-triangle octagon
(24 million vertices per frame); the silhouette is approximate, and overlaps still follow draw
order because the renderer has no depth buffer. Compute and graphics share a 24 MB particle
buffer, with constant-time vertex indexing and compute dispatch sized for the selected GPU. Pipelines
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

Input is available through `$/window.resin`. Exported `int` constants name GLFW key codes
(`key_w`, `key_space`, `key_left_shift`, `key_escape`) and the eight mouse buttons
(`mouse_button_left`, `mouse_button_right`, `mouse_button_middle`, `mouse_button_4` through `mouse_button_8`).
After polling, `window:key_state(key_w)` and
`window:mouse_button_state(mouse_button_left)` return `ButtonState` records
with `down`, `pressed`, and `released` booleans. A quick tap can set both edge flags;
key repeat does not create another press. `window:key_pressed(key)` queries `down`.

Poll each window once per frame before reading its input. Polling pumps GLFW events for
all windows, then commits that window's snapshot; other windows retain pending input
until their own poll. Edges and scroll reset on the next poll of that window, and repeated
queries read the same snapshot. `window:scroll_delta()` returns accumulated horizontal/vertical
scroll offsets (positive vertical scroll is up). `window:cursor_position()` returns coordinates
in window content units, with a top-left origin and positive y downward, independently of
framebuffer scaling. Both return `((float64, float64) | Err<RuntimeError>)`.

`window:focused()` reports keyboard focus. `window:capture_cursor(capture)` hides and
captures the cursor for camera controls when `capture` is true, with unbounded virtual
coordinates; set it false to restore normal behavior.
GLFW synthesizes button releases on focus loss. These APIs report physical controls;
they do not decode typed text or implement text composition.

Windowing is an ordinary runtime API, exposed by `resin_runtime/window.h` and
`$/window.resin`:

- `window_new(width, height, title: String)` returns a shared window owner; use `string_from_str("Resin")` for a literal title. `window:poll_events()` processes
  GLFW events. Close state, framebuffer size, resizing, and GLFW key codes
  are available through the corresponding window methods. Predicates return `bool`;
  fallible operations return error unions, including framebuffer size as `(width, height)`.
- `gpu_new_for_window(window)` selects a graphics/compute/present-capable GPU for a window.
  The existing GPU constructors stay headless. There is one GPU per window; multiple
  windows can each have their own GPU.
- `gpu:present(image)` blits an already-submitted `GpuImage` to the window, scaling to
  its framebuffer with FIFO presentation. Swapchains are recreated after resize.
  Its `(bool | Err<RuntimeError>)` is `(true)` when presented and `(false)` when
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

### Mandelbrot explorer

Run `resin examples/eg011_mandelbrot.resin` with the compiler service running.
The default GPU mode solves pixels in a compute shader; `--cpu` runs the same solver
by calling that exact `compute` entry in a CPU loop. Each invocation handles one
explicit 8×8 tile, clipped at image edges. Tile origins are prepared on the host;
large dispatches use batches of tile descriptors within Vulkan's workgroup limit.
Orbit iteration, palette evaluation, and color blending are separate functions,
using `Complex<float32>` from `$/math.resin`.

Both interactive paths produce the same RGBA8 GPU buffer: CPU mode uploads its
completed bytes once with `:copy_from`, and GPU mode writes directly. A fullscreen
triangle presents that buffer through the same pipeline. The example groups pipeline
setup, uploads, dispatch, presentation, and readback in its GPU helpers section.
Headless CPU mode writes its host bytes directly, without GPU setup.
Pass example options after Resin's `--` separator:

```sh
resin examples/eg011_mandelbrot.resin -- --help
resin examples/eg011_mandelbrot.resin -- --cpu
resin examples/eg011_mandelbrot.resin -- --output mandelbrot.png --width 1920 --height 1080
resin examples/eg011_mandelbrot.resin -- --cpu --output detail.png --real -0.7435 --imag 0.1314 --span 0.005 --iterations 1024
```

`--output PATH` writes a PNG once and exits without creating a window. GPU output needs
a Vulkan device; CPU output needs neither a display nor a Vulkan device. Interactive
mode uses Vulkan for presentation with either solver. `--screenshot PATH` sets the
interactive screenshot filename, defaulting to `mandelbrot.png`; saving replaces that file.
`--width`/`--height` accept 1–8192, `--iterations` accepts 32–4096, and `--samples`
accepts 1–16. `--real`, `--imag`, and `--span` set the initial view.

The image matches the window's framebuffer resolution and preserves the complex plane's
aspect ratio when resized. Moving uses one sample per pixel; when input stops, a second
pass blends the selected number of subpixel colors (default four; `--samples 1`
disables refinement). Samples use prefixes of a fixed 16-point Halton lookup table,
with radical inverses in bases 2 and 3.
Headless output and screenshots use the selected sample count immediately.
The default budget is 256 iterations
per sample, with early escape and shortcuts for the main cardioid and period-two bulb.
The image is recomputed only after a change or for that refinement pass.
Zoom stops at a vertical span of `framebuffer_height * 1e-6` to leave room for subpixel
offsets with float32 coordinates near the Mandelbrot set.

- **Arrows / WASD:** pan.
- **Scroll / + / −:** zoom around the center.
- **[ / ]:** halve or double the iteration budget (32–4096).
- **R / Home:** reset the view and restore 256 iterations.
- **P:** save the current view as a PNG screenshot at framebuffer resolution.
- **Escape:** close.

Black pixels have not escaped within the chosen budget; this does not prove membership.
Run the exported `test` entry for known orbits, the strict escape-radius boundary,
pixel coordinates, and zoom limits: `resin examples/eg011_mandelbrot.resin:test`.

`window:wait_events(seconds)?` waits for input or a timeout and commits the same input
snapshot as `poll_events`. Use either operation once per frame; the timeout must be finite
and positive. The explorer uses it to avoid spinning while idle or minimized.

## Representation details

### Byte-array storage

An ordinary `ubyte` array contains exactly its N declared bytes, with alignment 1 and
stride N when nested in another array. Host and device layouts agree. Whole-array copies
copy those elements without an extra sentinel; embedded zeros remain ordinary data.
`size_of([1_ub, 2_ub])` is 2, and `size_of([[1_ub, 2_ub], [3_ub, 4_ub]])` is 4.
Empty arrays reserve a C storage placeholder for portability; it is not an accessible
array element, and empty arrays have no shared host/device layout.

For C calls, explicitly cast byte-array storage to `Ptr<ubyte>` and pass its logical
length. Raw arrays and spans do not promise NUL termination. Use a `str` literal's `.data`
or an owned `String`'s `:get().data` when a C function requires a terminator. `string_from_bytes(span)`
copies raw bytes and appends that terminator outside the logical length.

### Shared size and alignment

`size_of(T)` and `align_of(T)` return `ulong` constants for the shared host/device
layout of a concrete type. They use the same layout rules as C assertions and SPIR-V
storage emission. Scalars in the shared profile, padded/nested records, pointers,
spans, and nonempty arrays are supported; unsupported layouts produce a source error.
For an inferred array type, `size_of(array_expression)` queries its type. Expression
operands are checked but not executed, as with C `sizeof`; side effects do not run.
Holes in an explicit type argument are rejected. Empty arrays/records,
booleans, function values, and other types outside the shared profile are rejected.

Use `gpu:create(value)?` to allocate and initialize one GPU element, with its type
inferred from the value or result context. `gpu:alloc::<T>(count)?` allocates
uninitialized elements and checks the multiplication of count by element size.
The byte allocator `gpu:alloc_in::<ubyte>(bytes, memory)?` returns `GpuSpan<ubyte>`.

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
A shader trap is not a host-visible Err value, and earlier writes remain visible.

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
For example, an infallible error union can be unwrapped without inventing an error value:

```resin
fn unwrap(r: (int | Err<Never>)) -> int {
	match (r) {
		int(n) => {
			n
		},
		Err(impossible) => {
			absurd(impossible)
		}
	}
}
```

The IR explicitly marks elimination as divergent. Its continuation type is only for
checking unreachable code; neither backend constructs a value of that type. C aborts
and shaders stop the invocation if invalid external memory somehow supplies a `Never`.
This defensive trap does not unwind cleanup. Reachable success and `?` paths retain normal
scope destruction. Matches over inhabited variants still require exhaustive, unique arms.

[`T | None`](options.md) supports direct widening, exhaustive matching,
and postfix `!` to exclude `None` or trap.

Operations are [visible free functions](methods.md); struct bodies contain only fields.

## Resources

[No Graphics API — Sebastian Aaltonen](https://www.sebastianaaltonen.com/blog/no-graphics-api)

### Early returns

`return value;` leaves the current function. `return;` returns unit. The value is
preserved before locals and unfinished operands are destroyed in reverse order.
Only paths that continue participate in definite-initialization checks.

### Assertions

`assert(condition);` evaluates a `bool` once and returns unit. False traps with
`assertion failed` on the host, or stops the current shader invocation through
the shader failure path. Assertions remain enabled in optimized builds.

In a `match`, `Variant(_)` ignores the payload. A final `_ => { ... }` arm
covers all remaining variants. Duplicate, unreachable, and non-final wildcard
arms are rejected.

Error payloads can be any value type, including `str`, numbers, tuples, and owned
`String` values. Inferred error sets collect their union; mutable pointers remain
invariant and errors are still owned and destroyed normally.

### Value representations

Import `repr` from `$/string.resin` to obtain an owned `String` describing any
host value. Records show their names and fields, arrays and tuples show their
elements, and unions show their active payload. Strings are quoted and escaped;
pointers show addresses and opaque handles show their type. `fmt` accepts these
values too, while direct string arguments retain their verbatim text behavior.
Unhandled entry-point errors include this representation before cleanup.

A struct may provide `repr_bytes(self: Ptr<Self>)` returning the primitive
`(Ptr<ubyte>, ulong)` byte view. Use the actual struct name in
place of `Self`. The view must remain readable while its receiver is alive;
the hook borrows its receiver and must not invalidate it. `String` uses this
hook so formatting and nested representations show its text. Representation
is a host operation and limits nested output to 128 levels.

`Err<E>` is a builtin wrapper for an error payload of type `E`. Construct it with
`Err(value)` or an explicit payload type such as `Err<int>(7)`. Wrappers copy and
destroy their payload normally, have distinct type identities in unions, and
may be nested. `repr(Err("message"))` produces `Err("message")`.

Fallible functions return ordinary unions such as `int | Err<str>`. Return a
plain `int` on success and `Err("message")` on failure. Postfix `?` returns any
`Err` member immediately, after cleaning up the exited scopes; its value type
is the union of all remaining members. The enclosing function must admit every
propagated error. An `Err<_>` result hole collects the least union of error
payloads, using `Never` if none occur. Mutable pointers remain invariant.

Handle failures with `match (operation()) { int(value) => { ... }, Err(error) =>
{ ... } }`. `Err(error)` covers all error wrapper members and binds the union
of their payloads; `Err(_)` discards those payloads. A final `_ => { ... }` can
handle any remaining members. Explicit `Err<E>(value)` type patterns bind the
wrapper itself, like other explicit type patterns.
