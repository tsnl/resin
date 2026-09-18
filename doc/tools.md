# Running Resin and editor tools

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
`(i32, Ptr<Ptr<u8>>, Ptr<Ptr<u8>>)` for `argc`, `argv`, and `envp`, and return
`i32`, `()`, or a union of those types with `Err<E>`;
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
import { "$/string.resin", "$/stdio.resin", "$/process.resin", "$/span.resin" };

fn main(argc: i32, argv: Ptr<Ptr<u8>>, envp: Ptr<Ptr<u8>>) -> () {
	let args = arguments(argc, argv);
	let mut index: u64 = 1;
	while (index < args.length) {
		let item = argument(args, index);
		let text = fmt("{0}\n", (item:bytes(),));
		print(text);
		index = index + u64(1);
	};
}
```

The runtime deep-copies the argument and environment arrays and strings before entering Resin.
`argv[0]` is the executable invocation name (the owned artifact generation when using `resin FILE`),
`argv[argc]` is null, and `envp` is a null-terminated array of `NAME=value` strings.
`envp` is frozen at startup: later environment mutations do not change its values or lookups.
These process-lifetime views are borrowed and expose read-only pointers. Unix preserves
native bytes, including non-UTF-8; Windows converts its native wide inputs to UTF-8, replacing unpaired UTF-16 surrogates.

`$/process.resin` provides `arguments(argc, argv)` and `environment(envp)` as pointer spans,
`argument(args, index)` as a checked byte-span view, and `c_string(pointer)` for a valid
NUL-terminated string. `environment_get(env, name)` takes the span returned by
`environment(envp)` and a bounded byte span for the name, such as
`bytes("RESIN_GREETING")`, and returns
`(Span<u8> | Err<EnvironmentVariableNotFound>)`. Lookup is exact and case-sensitive on
all platforms; an empty value succeeds with length zero. It never reads live OS state.
See `examples/process.resin` for looking up a selected variable without dumping the environment.
Program arguments apply only to run mode; `-o` builds the executable to invoke separately.

`$/argparse.resin` provides a streaming option parser. Its small specification lists
space-separated spellings, with `|` for aliases and a trailing `=` for a required value:

```resin
let args = arguments(argc, argv);
let parser = argparse(args, "--help|-h --count= --output|-o=");
while (true) {
    let option = match (parser:next()?) {
        Argument(option) => { option },
        None => { break; },
    };
    if (option:named("--count")) {
        let count = argument_integer(option.value)?;
        let text = fmt("Count: {0}\n", (count,));
        print(text);
    };
};
```

The parser accepts `--count 12` and `--count=12`, returns canonical names for aliases,
and yields repeated options in command-line order. Unknown options, missing values,
and values attached to flags return `Err<String>`. It supports options only, without
positional arguments or bundled short flags. A separate value may start with `-`,
including negative numbers and filenames. Flags have an empty value view.
`argument_integer` checks unsigned 32-bit range; `argument_number` parses finite
f32 values from bounded bytes. Returned views borrow the process argument snapshot
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

The [Zed extension](zed.md) and
[Helix configuration](helix.md) provide Resin syntax support and launch
`resin --lsp DIR` for diagnostics, hover, go-to-definition, completion, and formatting.
Build the client with `cargo build -p resin`. Launch the editor
with `RESIN_SERVER` set to a running compatible service. The [client/editor guide](editor-client.md)
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
cargo run --quiet -- --format examples

# CI/lint: list files needing formatting without modifying anything.
cargo run --quiet -- --format --check examples
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
such as `(x,)` and `(i32,)`, force multiline
lists; add one to keep long calls or records readable. Comments and literal
contents are preserved, and repeated blank lines collapse to one. Keep one blank
line between example functions and between logical sections inside a function.
The [full formatting rules](editor-client.md#formatting) also apply to the CLI.
CI checks the examples with `--format --check` on Linux, macOS, and Windows.
