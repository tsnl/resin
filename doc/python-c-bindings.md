# Python and C bindings proposal

Status: scoped design, not implemented. Based on `main` at `638b322c`.

Generate an installable Python package from a Resin module graph. Python calls a
generated C ABI through `ctypes`; C consumers use the same library and a generated
header. Building uses the configured Resin server. Installing, importing, and
calling the finished package require no Resin compiler, server, or source files.

The first delivery format is a wheel selected by `-o ...whl`, with its filename
validated. Source imports that compile on demand are outside this first version.
The examples below describe proposed behavior.

## Build and import contract

For a module exporting a concrete `add(i32, i32) -> i32` function:

```sh
resin arithmetic.resin -o arithmetic-0.1.0-py3-none-linux_x86_64.whl
python -m pip install ./arithmetic-0.1.0-py3-none-linux_x86_64.whl
```

```python
import arithmetic

assert arithmetic.add(20, 22) == 42
```

Here the server must support the requested Linux x86-64 build and packaging
target. A wheel build selects a library's exports; it does not require or invoke
`main`. Reject `FILE:ENTRY` for wheel output initially, because entry selection
and library selection have different meanings. Preserve the existing release
build and atomic publication behavior of `-o`, including preservation of a prior
output on failure or cancellation. Never install or upload a wheel automatically.

The root module supplies the package's top-level exports. Selected additional
modules become Python submodules, preserving their individual export boundaries.
Dependencies needed only for implementation remain private. Offer explicit
module selection for packaging a collection, including the standard library;
do not turn every transitive dependency into public Python API. No new project
manifest is required. Module selection and generic specializations are explicit
build inputs; their CLI spelling remains to be designed.

### Wheel filename validation

Parse the complete basename using the wheel grammar:

```text
distribution-version[-build]-python-abi-platform.whl
```

Require a canonical escaped distribution name, a normalized Python package
version, and a valid optional build tag. Report malformed names before source
acquisition. `arithmetic.whl` is incomplete; show the expected filename in the
diagnostic. Do not silently rename the requested output. These rules follow the
[wheel filename specification](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-name-convention),
[name normalization](https://packaging.python.org/en/latest/specifications/name-normalization/),
and [version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/).

For the initial implementation, accept one `py3-none-<platform>` tag set. Reject
`any`: the payload contains native code. Reject CPython extension ABI tags such
as `cp314` and `abi3`, and initially reject compressed/multiple platform tags.
Python-version requirements belong in `Requires-Python`. This is a Resin producer
restriction, not a claim that other wheel tag combinations are invalid. Wheels
can be platform-specific without using the CPython extension ABI.
See [compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/).

After capability negotiation, validate the platform against the **server's
artifact target**, including architecture and supported platform baseline, not
the client's machine. A valid filename does not enable cross-compilation.
Advertise supported wheel targets explicitly. Do not accept a `manylinux`,
`musllinux`, macOS deployment target, or universal binary claim merely because
the user wrote it in a filename. Packaging must establish that claim for every
bundled binary and dependency; unsupported claims fail before compilation where
possible, and failed final dependency checks prevent publication.

Use the parsed distribution/version/build as explicit package metadata. Derive
the default Python import name from the escaped distribution name; require a
valid, non-keyword identifier, with an explicit import-name override for other
valid distribution names. Detect module paths that collide after escaping,
Python keywords, and names reserved by the generated loader. Keep the original
Resin names in metadata and diagnostics.

The archive contains `.py`, `.pyi`, `py.typed`, native libraries, a binding schema,
and the generated C header. Emit matching `.dist-info/METADATA`, `WHEEL`, and
`RECORD`, with file hashes and sizes, and `Root-Is-Purelib: false`. Filename,
metadata, and native target must agree. Include required redistribution notices.
See [wheel contents](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-contents)
and [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/).

The client performs early syntax validation; the server independently validates
the structured package request. Use a standards-aware parser, not an extension
check or a loose regular expression. Test accepted and rejected names against
PyPA's [wheel filename parser](https://packaging.pypa.io/en/stable/utils.html#packaging.utils.parse_wheel_filename),
with additional tests for Resin's narrower producer policy. The compiler client
need not invoke Python to parse a filename.

## What exists and what is missing

| Area | Current implementation | Required change |
| --- | --- | --- |
| Public declarations | [HIR module construction](../crates/resin-hir/src/lower/mod.rs) privately collects each file's exports; completed `Module.entries` contains only the entry file's unambiguous functions. | Preserve completed per-module exports, overload candidates, types, aliases, constants, and source names. |
| Specialization | [LIR entries](../crates/resin-lir/src/lib.rs) carry a function, concrete type arguments, and host/shader profile. | Select a finite library interface and materialize its callable/lifecycle bridges as roots. |
| C emission | [C lowering](../crates/resin-codegen/src/c/lower/mod.rs) requires a host entry and emits `main`; function symbols use internal indices such as `r_fn0`. | Emit library projects with namespaced public adapters and hidden implementation symbols. |
| Native builds | [Toolchain](../crates/resin-toolchain/src/lib.rs) builds generated Ninja projects with executable-oriented rules. | Add shared-library linking, target export visibility, dependency staging, and immutable library artifacts. |
| Runtime | [Runtime crate](../crates/resin-runtime/Cargo.toml) already builds static and dynamic libraries; its headers describe low-level runtime operations. | Define embedding ownership and runtime-instance policy; runtime headers alone do not expose arbitrary Resin modules. |
| Transport | [Protocol](../crates/resin-protocol/src/lib.rs) describes a single executable or SPIR-V artifact and a single entry. | Add a negotiated library/wheel request and artifact kind, including package metadata and selected API. |
| Client | [`-o` request handling](../crates/resin-client/src/cli/request.rs) validates output paths. | Select wheel output, validate its name and negotiated target, then atomically publish the verified download. |

Generate from completed compiler data. Parsing emitted C, scraping documentation,
or exposing numeric function/type IDs would discard language information and
couple users to incidental compiler ordering.

## One generated foreign interface

The translation remains syntax → AST → HIR → LIR → verified LIR → C/SPIR-V.
HIR retains the public module catalog. Library selection resolves declarations
and explicit type applications before LIR construction. LIR retains concrete
public identities, source names, signatures, and ownership operations, and
verification covers generated call/copy/drop bridges as well as source bodies.

Code generation lowers those completed exports to a private ABI description and
emits the C adapters, header, binding schema, Python modules, and type stubs from
that one description. It consumes verified LIR without reopening HIR scopes.
Keep the new public contracts in their owning crates' `lib.rs`; a separate
binding compiler or reusable orchestration service is unnecessary initially.

The server sequences these passes, native compilation, and packaging explicitly.
The toolchain owns shared-library flags and dependency inspection; compiler
passes do not run external tools. Package assembly belongs to the application
layer and consumes completed artifacts. The client remains independent of the
compiler and runtime crates.

Use a versioned C ABI with fixed-width scalars, explicit pointer/length pairs,
opaque typed handles, and output parameters. An illustrative scalar adapter is:

```c
/* Actual symbol prefixes include the package identity and ABI version. */
uint32_t arithmetic_v1_add(int32_t left, int32_t right, int32_t *out_value);
```

The status reports adapter failures; zero means the output is initialized.
Specify each failure's output and ownership behavior. Language `Err` values are
ordinary typed results, separate from adapter status. Never publish internal
module-wide union tags; assign public variant tags and translate them explicitly.
Symbols derive from package/module/export identity and concrete signature, with
collision detection, rather than LIR indices. Check an ABI version and schema
fingerprint before Python binds callable symbols. Initially require a matching
header/wrapper/library build; stable naming alone does not promise compatibility
across arbitrary library versions.

Pass aggregates by pointer, converting between the public representation and
Resin storage. Generate size/alignment/offset checks for exposed C records. Use
opaque handles where layout or lifecycle is complex. This avoids relying on
aggregate return conventions; `ctypes` also does not guarantee unions passed by
value. Always set `argtypes` and `restype` explicitly.
See the [ctypes reference](https://docs.python.org/3/library/ctypes.html).

Export only public adapter symbols. Internal `r_fn*` functions and generated
lifecycle callbacks must not interpose across separately loaded packages. C
consumers receive the same header, library, runtime dependencies, and ABI checks;
they do not need Python. Define C SDK output packaging separately; the wheel
filename requirement does not settle the C output CLI.

## Meaning of “all modules”

The generator should accept any user or library module, honoring its explicit
exports. Every selected export must receive a binding or a precise diagnostic.
Never quietly discard overloads, unsupported types, or generic declarations.
Produce an export coverage report during the build, including deliberate
exclusions; an unsupported requested export makes the build fail.

An ahead-of-time wheel contains finitely many generic specializations. It cannot
support every future `T`, tuple shape, or shader root without further compilation.
Collect concrete nongeneric exports automatically and accept explicit type
applications for generics. Allow a build policy to add documented standard
specializations, such as floating-point math, but do not silently instantiate a
Cartesian product of types. Reuse LIR's canonical keys and instance limits.
Type-only exports also need constructor/accessor/lifecycle roots even when no
exported function mentions them.

Python generic selection chooses an already compiled specialization and reports
available choices when one is missing. Overloads get explicit signature-specific
entry points; a convenience dispatcher may choose among unambiguous mappings.
Do not recreate Resin inference or resolve ambiguity by attempting bodies.
Python integers do not by themselves distinguish all Resin integer widths, nor
does Python provide an expected result type for return-directed overloads.

| Resin API | C/Python mapping and scope |
| --- | --- |
| Integers, floats, booleans, unit | Fixed-width values; range-check Python integers; define boolean encoding; unit returns Python `None`. |
| Constants and aliases | Preserve exported names and evaluated values; aliases retain their public spelling and expand to their completed target. |
| Records, tuples, arrays | Generated constructors/accessors or plain value records where recursively suitable. Hide private support types behind named opaque values. Handle zero-sized and recursive types explicitly. |
| Unions, `None`, `Err<E>` | Tagged result objects preserve every variant and error payload. Offer explicit error-unwrapping convenience; do not erase ordinary union semantics. |
| Owned/custom-drop values | Typed handles with generated lifecycle operations and explicit ownership transfer. |
| `Ptr`, `Ref`, `RefMut`, borrowed fields/spans | Explicit foreign views with a stated lifetime and aliasing contract. General automatic safe borrowing is unavailable. |
| Literal `str` | Preserve static, NUL-terminated lifetime; Python temporary bytes are not a valid replacement. See below. |
| Source `String` and byte containers | Owned byte-preserving wrappers; explicit encoding/decoding conveniences and copy operations. |
| Generic operations | Only selected concrete applications; `fmt` needs selected argument tuple shapes. |
| Function values and callbacks | Deferred adapter work. Resin function-pointer representation is not automatically the public C calling convention. |
| Shader declarations and typed pipelines | Build-time shader selection and generated host bridges; Python receives a compiled shader/pipeline identity, not a native callable shader pointer. |

Python classes represent records and resources. Preserve independently exported
operations as module functions; do not infer an operation namespace from the
first parameter or accidentally expose private destruction hooks.

## Ownership, strings, and borrowed memory

A handle owns one Resin value and records its package/type identity. Generated
`close()` and context-manager operations release it exactly once. Copy only when
Resin permits copying; shared-owner copies retain ownership, and consuming a
move-only value invalidates the Python handle. Validate all arguments before
transferring any ownership. Define the native transfer point so failures and
Python conversion exceptions cannot double-drop inputs or leak outputs.
Reject cross-package handles initially, even when types have the same spelling.
Modules built together share their type identity and can exchange values.

Finalizers are a fallback, not the only cleanup path. Keep the native library
alive while any value or destruction callback depends on it. Pin loaded libraries
for the process in the first version. Resources with thread-affine destruction,
particularly windows, require explicit close on the owning thread; an arbitrary
Python finalizer thread must not destroy them.

Raw Resin pointers/references do not encode escape behavior or the owner of a
returned view. Even retaining every Python argument until a call returns cannot
protect a pointer saved by Resin for later. Apply this recursively to aggregates
containing borrowed fields. Provide automatic copying for values whose validity
is established, owner-retaining views for explicitly described operations, and
an explicitly unsafe raw interface for the rest. Safe buffer conveniences need
binding contracts stating no-escape or the retained owner, plus mutability,
contiguity, element format, and length rules. Do not promise zero-copy NumPy or
arbitrary Python buffer support in the first milestone.

For primitive `str` inputs, the conservative option is explicit interning into
immutable process-lifetime storage, with a visible `Literal` wrapper. Deduplicate
repeated literals and document that distinct interned text remains allocated.
Use owned String/byte APIs for dynamic text. Preserve exact byte lengths and
embedded NULs where the source operation permits them; decoding to Python text
is explicit. Recognize source-container adapters by resolved declaration
identity or an explicit binding contract, never just the spelling `String` or
`Span`. A transient Python string must never become an escaping literal view.

## Embedding and native dependencies

Current [host failure paths](../crates/resin-runtime/src/host.rs) call
`std::process::exit`, and emitted C also uses `abort`. Typed error unions can be
returned across the ABI, but existing traps still terminate the embedding process.
The initial contract must say this explicitly. Turning traps into recoverable
Python exceptions would require a separate compiler/runtime design that preserves
cleanup through all call paths; a C shim cannot supply that guarantee.

Executable startup currently registers global `resin_cleanup`. A library must not
run that cleanup when an individual module or Python value is collected. Keep
normal managed-value destruction separate from process-lifetime allocations.
Process/argument adapters should accept explicitly captured argv/environment
snapshots with retained ownership. Do not invoke the executable startup routine
on import or repeatedly read live process state behind the caller's back.

`ctypes.CDLL` releases the GIL during native calls. Establish synchronization in
the native embedding contract, covering close-versus-call races and aliases;
Python's GIL is not a resource lock. Start with serialized resource access where
needed and preserve main-thread requirements for window operations. Callbacks
need lifetime retention, exception handling, and reentrancy rules before support
is enabled. See [ctypes loading and callbacks](https://docs.python.org/3/library/ctypes.html#loading-shared-libraries).

Bundle or declare the native runtime and transitive redistributable libraries,
using paths relative to the installed package. Import must not depend on a build
directory, Nix store path, current directory, or Resin installation. Audit Linux
loader paths, macOS install names, and Windows DLL lookup. CPU package import
must work without a Vulkan SDK, GPU, or display. GPU and window operations retain
their runtime hardware/platform requirements.

Settle runtime sharing before resource packages ship. Recommend one compatible
runtime instance for interacting libraries, delivered as an explicitly versioned
runtime dependency if necessary. Independently bundling static runtimes can
duplicate allocation registries and GLFW initialization. A matching filename or
ABI version alone does not prove that the OS loaded a single instance. Test two
independently built packages together, including their destruction callbacks;
cross-package value exchange remains a separate future contract.

## Delivery sequence

| Milestone | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 1. Public API catalog | Completed module exports and an exhaustive coverage report; concrete and generic selection. | Reexports, duplicate names, constants, aliases, types, overloads, and unsupported exports are accounted for deterministically. |
| 2. Callable C library | Shared output without `main`; scalar adapters, symbol visibility, schema/version check, C header. | A standalone C program calls multiple Resin functions; malformed inputs fail according to the ABI; two libraries do not interpose internal symbols. |
| 3. Installable scalar wheel | `-o ...whl`, name/tag validation, negotiated package request, native packaging, generated `ctypes` loader and stubs. | Offline install in a clean environment; import/call without sources or server; incorrect name/target fails; failed publication preserves an existing wheel. |
| 4. Values and ownership | Aggregates, unions, constructors, selected generics, copy/move/drop bridges, owned strings. | Mixed-width layouts, empty/nested values, error payloads, finalization, repeated close, move invalidation, and partial conversion failure behave correctly. |
| 5. Borrowed library APIs | Explicit buffer/lifetime contracts; standard-library adapters and finite specialization sets. | Span/owner lifetime, retained views, mutability, encoding, process snapshots, and formatting shapes are tested; unsupported signatures remain visible. |
| 6. GPU and window modules | Compiled shader identities, pipeline bridges, resource lifetime and threading policy. | Packaged compute and rendering examples run; import succeeds without GPU/display; explicit execution failures and main-thread cleanup are covered. |

Milestones 1–3 are the first usable release. Milestones 4–6 expand coverage toward
all public standard-library modules and user modules with the same supported
contracts. A useful intermediate target is `math`, `graphics` value types, and
`status`; then `ownership`, `shared`, `string`, `span`, `stdio`, `process`,
`argparse`, and `image`; finally the GPU/window resource APIs. This ordering does
not imply that every generic application in those modules is precompiled.

Canonical API selection, ABI/schema versions, package metadata, runtime identity,
target baseline, and native dependency contents participate in the appropriate
cache keys. A wheel filename or local output path is not sufficient cache
identity. Retain native artifact generations until packaging completes and retain
the completed wheel through download consumption. Package files deterministically
where practical and verify the download before atomic publication.

Run automatic integration checks on Linux. Keep Windows and macOS shared-library,
loader, and wheel checks available as manual jobs, and require all three platforms
for release validation under the repository's existing CI policy. Test wheel
installation outside the development shell, including relocation and absent build
paths. Run trap and shutdown tests in subprocesses. Reuse PyPA tooling to check
the wheel format and metadata instead of testing only Resin's own reader.

The largest work is retaining a complete public API, defining ownership at the
foreign boundary, and making runtime resources embeddable. Writing Python wrapper
text is comparatively small. Keep callback support, import-time compilation,
arbitrary generic specialization at runtime, cross-package value exchange, and
recoverable traps as separately scoped extensions.
