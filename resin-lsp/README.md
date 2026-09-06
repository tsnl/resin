# Resin language server

`resin-lsp` provides diagnostics, hover, go-to-definition, basic completion, and formatting
over stdio. It uses the Resin compiler's persistent `compiler::Session`.
Semantic editor requests run parsing, resolution, typing, and IR verification; they do not
compile C/GLSL, initialize a GPU, or run the program. Analysis accepts library
modules without an exported entry function; runtime bindings belong inside
functions, following the compiler's declarations-only module rules. Functions use
`def`, local bindings use `var`, nominal records use `struct`, and aliases use `type`. Omitted function
result annotations default to `()` and appear as unit in hover/completion.
Explicit `_` holes in local annotations and function results are solved by the
compiler before lowering; hover and completion use those concrete types, including
inferred results from imported modules. Incomplete-source recovery remains
best-effort and may show `?` when a complete program would infer a type.
Result error sets appear in hover as nominal names joined by `|` (or `Never` when empty).
Match-arm bindings are scoped to their arm and expose the selected payload's fields.
Deferred expressions retain lexical name resolution and inferred local types; navigation and
completion use the bindings visible at registration, not later declarations.

## Build and run

From the Resin repository root:

```sh
nix-shell --run 'cargo build -p resin-lsp'
nix-shell --run 'cargo install --path resin-lsp --locked'
```

Configure your editor to launch `resin-lsp` (or `resin-lsp --stdio`). The server
uses stdout exclusively for the protocol and stderr for logs. See
[the Zed extension](../zed-resin/README.md) for a complete editor setup.

Standard-library lookup, in descending precedence:

1. `resin-lsp --stdlib /absolute/path/to/stdlib`
2. Initialization options: `{ "stdlibPath": "/absolute/path/to/stdlib" }`
3. `RESIN_STDLIB`
4. The repository's `stdlib/` path recorded when the Resin library was built.

Relative overrides resolve against the server's working directory. Prefer an
absolute path, especially when using the server in another checkout. Keep the
Nix shell environment when launching Zed/the server; it supplies native libraries.

## Formatting

The server advertises `documentFormattingProvider` and handles
`textDocument/formatting`. Use your editor's **Format Document** command or enable
format-on-save. The editor synchronizes the buffer, requests formatting, and
applies the returned text edit through its normal undo/save workflow. The server
does not write the file. Rebuild/reinstall the server and restart it in your editor
to pick up formatting support; see the [Zed setup](../zed-resin/README.md#formatting).

Formatting uses `resin::formatting::format_source` on the latest accepted open
buffer, independently of background semantic analysis. Unresolved names, imports,
and type errors do not prevent formatting. Syntax errors return no edits, as does
formatting a document that is not open. An already formatted buffer returns an
empty edit list. A changed buffer returns one replacement trimmed to exclude its
unchanged prefix/suffix, with UTF-16 ranges that respect Unicode and CRLF boundaries.

Resin has a fixed source style:

- One hard tab per indentation level, regardless of LSP `tabSize`/`insertSpaces`.
  Your editor controls how wide tabs appear.
- Trailing commas are preserved and force one item per line in parameter lists,
  calls, tuples, arrays, records, struct fields, imports, and exports. Without a trailing comma,
  lists collapse unless comments or nested multiline constructs require breaks.
- Nonempty blocks and match arm lists use multiple lines. Operators, declarations, and separators
  receive consistent spacing; no automatic line-length wrapping is performed.
- Runs of blank lines collapse to at most one between items. Padding inside
  delimiters and at file boundaries is removed. Nonempty output ends with one LF;
  empty/whitespace-only input becomes empty. Whitespace outside comments and
  literals is normalized to tabs, spaces, and LF.
- Comments and literal spellings are preserved, including whitespace within block
  comments. Same-line trailing comments stay attached to their preceding token.

For example, `var xs = [1,2,3,];` becomes:

```resin
var xs = [
	1,
	2,
	3,
];
```

Only whole-document formatting is supported; range/on-type formatting and a CLI
format command are not implemented.

## Stateful compiler core

The compiler owns source state, incremental Tree-sitter parses, cached ASTs,
import dependencies, and immutable checked snapshots. The CLI and the language
server both use this API. The server adds URI/version bookkeeping, UTF-16 position
conversion, client file-watch notifications, and a background worker.

```rust
use resin::compiler::Session;
use std::path::Path;

fn main() -> std::io::Result<()> {
    let entry = Path::new("/tmp/example.resin");
    let mut compiler = Session::default();
    compiler.set_overlay(entry, "def main () -> int = { var value = 1; value };".into())?;
    let before = compiler.analyze(entry)?;
    assert!(before.module().is_ok());

    compiler.set_overlay(entry, "def main () -> int = { var value = missing; value };".into())?;
    let after = compiler.analyze(entry)?;
    assert!(!after.diagnostics.is_empty());
    assert!(before.module().is_ok()); // Retained readers keep their old snapshot.
    Ok(())
}
```

Hosts call `set_overlay`/`remove_overlay` for buffers and `file_changed` after disk
changes, creations, or deletions. An open overlay takes precedence over disk.
`analyze` reuses an unchanged entry's snapshot; edits invalidate entries that
transitively depend on the changed file. Missing imports also register dependencies
so creating a file can recover an error. `set_stdlib` invalidates checked entries.
`retain_entries` accepts canonical entry paths to release results no longer needed.

Parsing is incremental per file. Semantic checking currently reruns the affected
entry's complete import closure, while unrelated checked entries remain cached.
There is no per-function query engine or shared compiler daemon yet. The core
consumes change events rather than owning an OS watcher. The CLI still builds,
runs, and exits once; a watch command can host the same session later.

## Protocol behavior and limits

- Local `file://` documents, including new files and unsaved imported buffers.
- Full-document synchronization and UTF-16 positions, including CRLF and non-BMP
  characters. Older document versions are ignored.
- Each open file is an analysis entry. Diagnostics from its dependencies are
  published at the actual source URI and aggregated across entries.
- Multiple syntax errors; the first semantic error for each entry/import closure.
  Strict compiler diagnostics remain authoritative. A separate editor pass walks
  AST expression/type holes and missing-field nodes, continuing through later
  statements, functions, and imported buffers. It shares compiler type rules and
  records unknown bindings as `?`; it never emits IR or treats unknown as unit.
  Strict source loading and compilation still reject incomplete programs.
- Navigation includes locals, parameters, nominal types, explicit exports,
  re-exports, standard-library names, and import strings.
- Completion includes visible names, keywords, builtin types, and intrinsics,
  with identifier replacement ranges. Typing `.` offers fields from the receiver's
  record type, including nominal records, pointers, and nested access. Field
  suggestions use the cached recovered AST and semantic snapshot without inserting
  synthetic identifiers. Unrecoverable declarations, unresolved imports, and
  unknown receiver types can still prevent suggestions. Automatic imports are
  not implemented.
- The client is asked to watch `**/*.resin` if it supports dynamic registration.
  A client without file notifications needs a server restart after external
  changes to closed dependencies.
- Queued changes are coalesced; obsolete results are discarded. Requests can be
  cancelled while queued. An edit invalidating a queued request returns
  `ContentModified`; a running compiler pass finishes before the next pass.

## Validation

```sh
nix-shell --run 'cargo test -p resin-lsp'
nix-shell --run 'cargo test -p resin --lib --test analysis --test zed_queries'
nix-shell --run 'cargo test -p resin --test formatting'
```

The stdio integration tests launch the real server and exercise lifecycle,
unsaved Unicode buffers, semantic features and formatting, dependency overlays, watched changes,
diagnostic clearing, out-of-order versions, and rapid edits.
Formatter tests check expected layouts, comment/token/tree preservation, and
idempotence, including the examples and standard-library source corpus.
