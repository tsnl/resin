# Resin editor support plan

This is the historical implementation plan. The current layout uses the root
`resin` executable with `--lsp <directory>`, the `resin-lsp` protocol library,
the `resin-compiler` library, and the filesystem adapter in `resin-source`.
See [current setup](README.md).

Scope: a Zed extension plus a reusable language server providing diagnostics,
hover, go-to-definition, and basic completion. Development targets 64-bit Linux.

Stack: `feature/compiler-session` → `feature/resin-lsp` →
`feature/zed-extension`, based on `main` at `70a05ac`.

Tracking issue: [tsnl/resin#75](https://github.com/tsnl/resin/issues/75).

The grammar migration is complete in commit `2a39afd`. Editor implementation is
on the stack above; see the [extension setup](README.md) and
[compiler/server architecture](../../crates/resin-lsp/README.md).

The current compiler API accepts immutable `Source` values from
`resin_source::prelude::*` and a concrete `resin_source::Loader` that resolves imports.
A reusable `resin_compiler::Compiler` owns syntax and compilation caches;
`compile` returns an immutable `Compilation`. Every call resolves imports before
reusing a result. Native generation passes verified LIR to codegen; the toolchain builds its Ninja project.
`resin_source::Loader` owns filesystem identities and standard-library lookup.
The language server owns open buffers, URI/version bookkeeping, frontend revisions,
and its background worker. Buffer changes supply new immutable source versions
and schedule compilation. Semantic caches cover each entry's import closure;
per-function incremental checking remains follow-up work. The CLI retains its
build/run/exit behavior.
Module analysis follows the declarations-only grammar and needs no exported
entry; the CLI selects an exported function with `FILE:ENTRY`. Function/type/local
declarations use `def`/`type`/`var`, and omitted function returns default to unit.

Earlier automated validation completed on Linux: native protocol tests, compiler
tests, query capture tests, workspace tests with GPU and shader checks required,
Clippy in both workspaces, and the WASI build. Coverage includes the current
`def`/`var`/`type` syntax and omitted unit result annotations. Earlier Zed 1.17.2
smoke tests under Xvfb covered imported/standard-library navigation and unsaved
editor features; the most recent editor smoke test used grammar pin `79e4a26`.

## Repository layout

```text
editors/zed/
  extension.toml
  Cargo.toml
  src/lib.rs
  languages/resin/
    config.toml
    highlights.scm
    brackets.scm
    indents.scm
    outline.scm
    overrides.scm
    textobjects.scm
  README.md
crates/resin-lsp/
  Cargo.toml
  src/
crates/resin-compiler/src/lib.rs
crates/resin-source/src/lib.rs
tests/lsp.rs                     # complete executable protocol tests
crates/tree-sitter-resin/          # grammar tracked directly in this repository
```

`editors/zed/` is an ordinary directory tracked in the Resin repository. Give its
Rust package an independent Cargo workspace and exclude it from the native root
workspace: its WebAssembly build should only depend on Zed's extension API.
No separate extension repository is needed: Zed's registry supports a repository
subdirectory through `path = "editors/zed"` in its registry entry. See the
[publishing guide](https://zed.dev/docs/extensions/publishing/publishing-guide).
Keep `resin-lsp` as a native workspace library depending on `resin-compiler` and
`resin-source`. Compiler caches and phase products contain no Zed or LSP protocol
types; open buffers and protocol revisions belong to the language server.

Use the grammar in `https://github.com/tsnl/resin` with
`path = "crates/tree-sitter-resin"` and pin a Resin commit containing the required parser.
Put Zed queries in the extension. Commit grammar fixes and generated files in
Resin, then update the extension pin to that commit. Ordinary worktrees contain
their own grammar files without submodule initialization.

## 1. Deliver basic Zed language support

Register `.resin` files, line and block comments, bracket pairs, and indentation.
Write queries for highlighting, matching brackets, outlines, syntax scopes, and
function/comment text objects. Cover functions and parameters, value and nominal
type definitions, foreign declarations, imports/exports, builtins, control flow,
record fields, pointer operations, literals, and comments.

Validate queries against the actual grammar and representative examples and
standard-library files. Syntax features must remain usable when the LSP is absent
or a document contains incomplete code.

## 2. Expose compiler analysis suitable for editing

The AST retains byte spans, and source locations retain immutable `Source`
handles. Source scopes and definition origins belong to HIR construction; the
compiler's `Compilation` exposes diagnostics and queries over those phase products.

Use `resin_source::Loader` to resolve references from their importing source.
The CLI loads files; the server registers authoritative buffer text through
`source_from_text` and removes it with `remove_source` when a document closes. Preserve relative imports, `$/std/`, `RESIN_STDLIB`,
canonical filesystem identities, explicit exports, re-exports, duplicate-import
handling, and cycle detection. File-backed buffers need not exist on disk.
`resin_source::Loader` maps logical source identities back to exact OS paths, including
paths that have identical lossy display names.

Preserve structured diagnostics: source identity, byte range, severity, message,
and related locations for import chains or conflicting definitions. Keep CLI
formatting as a presentation layer over those diagnostics.

Expose definition identities, lexical scope ranges, resolved occurrences, and
available types through an analysis result. Extend or extract the compiler's
existing resolution logic so editor answers obey the same shadowing, declaration
order, value/type namespaces, and export rules. Preserve nominal type identity
and original declaration locations through re-exports.

Keep tolerant Tree-sitter parsing separate from semantic analysis. Collect syntax
errors from ERROR and missing nodes; retain usable declarations in valid regions.
Use compiler-derived types and resolution where available. In incomplete regions,
offer only completions and navigation justified by the current syntax and scopes;
omit uncertain type information. Never reuse stale byte ranges as current results.

Initially preserve the compiler's first-error behavior for semantic checking of an
entry and its dependencies. Multiple syntax errors can be reported immediately;
full semantic error recovery is a later improvement.

## 3. Build the native `resin-lsp` server

Use `lsp-server`, `lsp-types`, and serde-based message conversion. Run over stdio;
reserve stdout for protocol messages and send logs to stderr. Implement lifecycle,
capability negotiation, unsupported-request responses, cancellation, and clean exit.

Maintain document URI, version, open-buffer sources, and a line index in the
server. Submit immutable sources to the compiler and retain its completed results.
Use full-document synchronization with incremental Tree-sitter reparsing; handle
open/change/save/close notifications and clear obsolete diagnostics. Convert byte spans to UTF-16 LSP positions, including
non-BMP characters, CRLF, and end-of-file ranges.

Treat each open file as an analysis entry with its transitive imports. The shared
loader supplies registered buffers for both entries and imports. Edits and watched-file
changes schedule analysis; each compiler call resolves imports again, so missing
files and retargeted symlinks are discovered without compiler notifications.
Advertise/register only supported capabilities. Aggregate results by source URI so
one entry cannot clear another entry's active diagnostics.

Run compilation on immutable sources in a worker, coalesce pending edits, and
discard results whose frontend revision is obsolete. Keep protocol
handling responsive while analysis is running.

Analysis uses parsing, resolution, typing, and LIR verification without invoking
C compilation, shader compilation, GPU operations, or user programs. Native builds
pass verified LIR to codegen and build the resulting Ninja project through the toolchain. The root executable also links
the runtime for its build/run modes; the compiler and LSP libraries have no runtime
dependency.

## 4. Implement the agreed editor features

| Feature | First-version behavior |
| --- | --- |
| Diagnostics | Syntax, import/export, name resolution, and compiler type errors at their actual source ranges; related import/declaration locations where relevant. |
| Hover | Resin signatures and available inferred/declared types for values, parameters, functions, and types; concise help for compiler builtins. |
| Go-to-definition | Lexical bindings, parameters, types, exported imports/re-exports, and standard-library definitions; import strings navigate to their source files. |
| Basic completion | Prefix-filtered visible names, parameters, exported imports, keywords, primitive types, `Ptr`/`Span`, and compiler builtins; context-appropriate value/type suggestions. |

Completion replaces the current identifier range and respects shadowing and
private module scope. It must work while typing an incomplete identifier or call.
For malformed regions, conservative partial results are acceptable. Automatic
imports and inferred record-field completion are outside this first version.

## 5. Connect Zed and validate the complete workflow

Register the language server in `editors/zed/extension.toml`. The Rust adapter
launches a configured executable or finds `resin` through Zed's worktree PATH
API, passing `--lsp <directory>`. Preserve the worktree environment, including `RESIN_STDLIB` and Nix library
paths. Provide an actionable installation message when the executable is missing.

Document building/installing the native server inside `nix-shell`, installing
`editors/zed/` as a dev extension, specifying a binary path, and checking Zed/LSP logs.
Use the Zed extension API and WASI target supported by the Zed version being tested;
current development docs specify `wasm32-wasip2` and a WASI SDK for grammar builds.

Validation:

- Compile every query and check expected captures on representative Resin syntax.
- Test compiler analysis parity for nested shadowing, nominal types, private
  imports, re-exports, diamond imports, duplicate bindings, and cycles.
- Exercise the real LSP process over stdio: initialization, unsaved edits, each
  feature, dependency updates, diagnostic clearing, and shutdown.
- Cover Unicode/CRLF positions, unsaved new files, malformed syntax, sequential
  document versions, and rejection of stale analysis results.
- Run `nix-shell --run 'cargo test --workspace --all-features'` and applicable
  formatting/lint checks; build the Zed extension separately for its WASI target.
- Smoke-test in Zed using `examples/eg009_imports.resin`, a standard-library import,
  and nested scopes. Confirm that unsaved changes affect all four LSP features.
  Zed 1.17.2 is available as `zeditor`; use an isolated Xvfb display for validation
  in this environment.

References, rename, signature help, automatic server downloads, and
extension-registry publication remain follow-up work. Whole-document formatting
is implemented through `resin_cst::format_source`; the AST printer remains an
S-expression renderer for inspecting that phase.

## References

- [Zed language extensions](https://zed.dev/docs/extensions/languages)
- [Zed extension development](https://zed.dev/docs/extensions/developing-extensions)
- [Zed worktree environment and executable lookup](https://docs.rs/zed_extension_api/latest/zed_extension_api/struct.Worktree.html)
- [lsp-server](https://docs.rs/lsp-server/latest/lsp_server/)
- [lsp-types](https://docs.rs/lsp-types/latest/lsp_types/)
- [LSP position encoding](https://github.com/microsoft/language-server-protocol/blob/gh-pages/_specifications/lsp/3.17/types/position.md)
- [LSP document synchronization](https://github.com/microsoft/language-server-protocol/blob/gh-pages/_specifications/lsp/3.17/textDocument/didChange.md)
