# Resin editor support plan

Scope: a Zed extension plus a reusable language server providing diagnostics,
hover, go-to-definition, and basic completion. Development targets 64-bit Linux.

Branch: `feature/zed-lsp`, created from `main` at `00bfa9e`.

Tracking issue: [tsnl/resin#75](https://github.com/tsnl/resin/issues/75).

The grammar migration is complete in commit `2a39afd`; editor implementation is pending.

## Repository layout

```text
zed-resin/
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
resin-lsp/
  Cargo.toml
  src/
  tests/
src/analysis/
tree-sitter-resin/          # grammar tracked directly in this repository
```

`zed-resin/` is an ordinary directory tracked in the Resin repository. Give its
Rust package an independent Cargo workspace and exclude it from the native root
workspace: its WebAssembly build should only depend on Zed's extension API.
No separate extension repository is needed: Zed's registry supports a repository
subdirectory through `path = "zed-resin"` in its registry entry. See the
[publishing guide](https://zed.dev/docs/extensions/publishing/publishing-guide).
Add `resin-lsp` as a native workspace member depending on the Resin library.
Keep shared compiler analysis in that library, with no Zed or LSP protocol types.

Use the grammar in `https://github.com/tsnl/resin` with
`path = "tree-sitter-resin"` and pin a Resin commit containing the required parser.
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

The AST already retains byte spans. `ast::load`, however, reads and canonicalizes
files directly, and `SourceError` retains only a path and formatted message.
Semantic scopes and definition origins currently live inside IR generation.

Introduce a source-provider interface with a filesystem implementation for the
CLI and an in-memory overlay for editor buffers. Preserve relative imports,
`std/`, `RESIN_STDLIB`, canonical identity for existing files, explicit exports,
re-exports, duplicate-import handling, and cycle detection. Support file-backed
buffers that have not yet been saved without requiring them to exist on disk.

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

Maintain document text, URI, version, and a line index. Start with full-document
synchronization and reparsing; handle open/change/save/close notifications and
clear obsolete diagnostics. Convert byte spans to UTF-16 LSP positions, including
non-BMP characters, CRLF, and end-of-file ranges.

Treat each open file as an analysis entry with its transitive imports. Unsaved
overlays apply to imported files as well as the entry. Track reverse dependencies
so changes invalidate affected open entries. Handle watched-file changes for disk
dependencies and advertise/register only supported capabilities. Aggregate results
by source URI so one entry cannot clear another entry's active diagnostics.

Run analysis on immutable snapshots with a worker, coalesce pending edits, and
discard results whose document or dependency versions changed. Keep protocol
handling responsive while analysis is running.

Analysis uses parsing, resolution, and typing without invoking C compilation,
shader compilation, GPU operations, or user programs. The initial native package
still builds the existing Resin library and its runtime dependency in `shell.nix`;
removing that build dependency can be a separate change.

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

Register the language server in `zed-resin/extension.toml`. The Rust adapter
launches a configured executable or finds `resin-lsp` through Zed's worktree PATH
API. Preserve the worktree environment, including `RESIN_STDLIB` and Nix library
paths. Provide an actionable installation message when the executable is missing.

Document building/installing the native server inside `nix-shell`, installing
`zed-resin/` as a dev extension, specifying a binary path, and checking Zed/LSP logs.
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
  Zed is currently unavailable on PATH in this environment, so GUI verification
  needs an available Zed installation; report that limitation if it remains.

References, rename, signature help, formatting, automatic server downloads, and
extension-registry publication are follow-up work. Resin's current AST printer
emits S-expressions and cannot serve as a source formatter.

## References

- [Zed language extensions](https://zed.dev/docs/extensions/languages)
- [Zed extension development](https://zed.dev/docs/extensions/developing-extensions)
- [Zed worktree environment and executable lookup](https://docs.rs/zed_extension_api/latest/zed_extension_api/struct.Worktree.html)
- [lsp-server](https://docs.rs/lsp-server/latest/lsp_server/)
- [lsp-types](https://docs.rs/lsp-types/latest/lsp_types/)
- [LSP position encoding](https://github.com/microsoft/language-server-protocol/blob/gh-pages/_specifications/lsp/3.17/types/position.md)
- [LSP document synchronization](https://github.com/microsoft/language-server-protocol/blob/gh-pages/_specifications/lsp/3.17/textDocument/didChange.md)
