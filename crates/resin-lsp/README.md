# Resin language server

The `resin --lsp <directory>` mode provides diagnostics, hover, go-to-definition, basic completion, and formatting
over stdio. It reuses `resin_compiler::Compiler` with immutable source versions.
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

## Build and run

From the Resin repository root:

```sh
nix-shell --run 'cargo build -p resin'
nix-shell --run 'cargo install --path . --locked'
```

Configure your editor to launch `resin --lsp /path/to/project`. The root `resin`
package supplies the sole executable; `resin-lsp` is its protocol library. The server
uses stdout exclusively for the protocol and stderr for logs. See
[the Zed extension](../../editors/zed/README.md) for a complete editor setup.

Library-root lookup for `$/` imports, in descending precedence:

1. Initialization options: `{ "libraryRoot": "/absolute/path/to/checkout/resin" }`
2. `RESIN_LIBRARY_ROOT`
3. The repository's `resin/` path recorded when `resin-source` was built.

Relative initialization overrides resolve against the selected project directory.
`RESIN_LIBRARY_ROOT` resolves against the invoking process's working directory. An
absolute override is useful when using the server in another checkout. Keep the
Nix shell environment when launching Zed/the server; it supplies native libraries.

## Formatting

The server advertises `documentFormattingProvider` and handles
`textDocument/formatting`. Use your editor's **Format Document** command or enable
format-on-save. The editor synchronizes the buffer, requests formatting, and
applies the returned text edit through its normal undo/save workflow. The server
does not write the file. Rebuild/reinstall the server and restart it in your editor
to pick up formatting support; see the [Zed setup](../../editors/zed/README.md#formatting).

Formatting uses `resin_cst::format_source` on the latest accepted open
buffer, independently of background semantic analysis. Unresolved names, imports,
and type errors do not prevent formatting. Syntax errors return no edits, as does
formatting a document that is not open. An already formatted buffer returns an
empty edit list. A changed buffer returns one replacement trimmed to exclude its
unchanged prefix/suffix, with UTF-16 ranges that respect Unicode and CRLF boundaries.

Resin has a fixed source style:

- One hard tab per indentation level, regardless of LSP `tabSize`/`insertSpaces`.
  Your editor controls how wide tabs appear.
- Trailing commas are preserved and force one item per line in parameter lists,
  calls, tuples, arrays, records, struct fields, imports, and exports, except singleton
  tuples such as `(x,)` and `(int,)`. Singleton tuples and lists without trailing commas
  collapse unless comments or nested multiline constructs require breaks.
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

The CLI shares this formatter: `resin --format examples` formats files recursively,
and `resin --format --check examples` checks them without writing. See the
[CLI formatting guide](../../README.md#formatting) for exit codes and file selection.
Only whole-document LSP formatting is supported; range/on-type formatting is not
implemented.

## Immutable sources and compiler caches

`resin_source::Source` is immutable named text with a stable logical
identity. Cloning shares a version; `with_text` creates a new version of the same
source. Names are diagnostic labels and need not be filesystem paths or unique.
`resin_compiler::Compiler` retains syntax and compilation caches. Its `compile`
method receives an entry source and a concrete `resin_source::Loader`, resolves the import
graph, and returns an immutable `Compilation` with diagnostics and editor queries.

Explicit import bindings also support sources held entirely in memory. This complete
example changes an imported module while keeping its entry unchanged:

```rust
use resin_source::prelude::*;
fn main() {
    let library = Source::new(
        "library",
        "export { answer }; def answer() -> int = { 42 };",
    );
    let entry = Source::new(
        "example",
        r#"import { "library" }; def main() -> int = { answer() };"#,
    );
    let mut loader = resin_source::Loader::new(resin_source::library_root());
    loader.set_import(&entry, "library", library.clone()).unwrap();
    let mut compiler = resin_compiler::Compiler::new();
    let before = compiler.compile(entry.clone(), &mut loader);
    assert!(before.module().is_ok());

    loader.set_import(
        &entry,
        "library",
        library.with_text("export { answer }; def answer() -> int = { missing };"),
    ).unwrap();
    let after = compiler.compile(entry, &mut loader);
    assert!(!after.diagnostics().is_empty());
    assert!(before.module().is_ok()); // Retained results keep their original sources.
}
```

`resin_source::Loader` supplies filesystem loading, canonical path identities,
relative imports, and `$/` resolution. `load_file` reads disk contents;
`source_from_text` registers authoritative supplied text for a file's imports, and
`remove_source` restores disk loading when a buffer closes. Unchanged text reuses its
source version. The CLI loads its entry through this loader. Codegen and the native
toolchain use `compilation.verified()` to build an executable from the retained
result, without reading the source again. Imports starting with `$/` select
the configured library root: `$/std/` selects its standard library, and sibling
libraries use paths such as `$/math/`. A plain `std/` is an ordinary relative
directory. Other references beginning with `$` report an unknown namespace.

The language server owns open buffers, document versions, URI/path mappings, and
frontend revisions. Open/change notifications register supplied text with the loader;
close notifications remove it. Saves and file-watch notifications schedule another
analysis. The compiler receives immutable sources and the concrete loader, while
protocol changes and scheduling remain server operations.

Each compile call resolves imports before checking its caches. An unchanged graph
reuses its compilation. Missing files that appear, changed file contents, and
retargeted import symlinks are observed on the next call without invalidation calls.
Tree-sitter reparsing is incremental per source; semantic checking reruns the
changed entry's complete import closure. There is no per-function query engine or
shared compiler daemon. The CLI builds, runs, and exits once; the server retains a
compiler in its background worker and discards results from obsolete frontend revisions.

## Protocol behavior and limits

- Local `file://` documents, including new files and unsaved imported buffers.
- Full-document synchronization and UTF-16 positions, including CRLF and non-BMP
  characters. Older document versions are ignored.
- Each open file is an analysis entry. Diagnostics from its dependencies are
  published at the actual source URI and aggregated across entries.
- Syntax and semantic diagnostics from the compiler's normal frontend. HIR checking
  traverses expression/type holes and missing-field nodes, preserving facts from
  healthy statements, functions, and imported buffers. Unknown bindings appear as
  `?`; incomplete programs retain editor facts without publishing executable IR.
- Navigation includes locals, parameters, nominal types, explicit exports,
  re-exports, standard-library names, and import strings.
- Completion includes visible names, keywords, builtin types, and intrinsics,
  with identifier replacement ranges. Typing `.` offers fields from the receiver's
  record type, including nominal records, pointers, and nested access. Field
  suggestions use the cached recovered AST and semantic analysis without inserting
  synthetic identifiers. Unrecoverable declarations, unresolved imports, and
  unknown receiver types can still prevent suggestions. Automatic imports are
  not implemented.
- The client is asked to watch `**/*.resin` if it supports dynamic registration.
  Without file notifications, external changes to closed dependencies are
  discovered when a later edit or save schedules analysis.
- Queued changes are coalesced; obsolete results are discarded. Requests can be
  cancelled while queued. An edit invalidating a queued request returns
  `ContentModified`; a running compiler pass finishes before the next pass.

## Validation

```sh
nix-shell --run 'cargo test -p resin-lsp'
nix-shell --run 'cargo test -p resin-compiler -p resin-source'
nix-shell --run 'cargo test -p resin --test lsp --test analysis --test zed_queries'
nix-shell --run 'cargo test -p resin --test formatting'
```

The stdio integration tests launch the real server and exercise lifecycle,
unsaved Unicode buffers, semantic features and formatting, dependency overlays, watched changes,
diagnostic clearing, out-of-order versions, and rapid edits.
Formatter tests check expected layouts, comment/token/tree preservation, and
idempotence, including the examples and standard-library source corpus.
