# Resin language server

`resin-lsp` provides diagnostics, hover, go-to-definition, and basic completion
over stdio. It uses the Resin compiler's persistent `compiler::Session`.
Editor requests run parsing, resolution, typing, and IR verification; they do not
compile C/GLSL, initialize a GPU, or run the program. Analysis accepts library
modules without an exported entry function; runtime bindings belong inside
functions, following the compiler's declarations-only module rules. Functions use
`def`, local bindings use `var`, and nominal types use `type`. Omitted function
result annotations default to `()` and appear as unit in hover/completion.

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
  Types and reference origins come from the compiler. In incomplete regions,
  syntax scopes provide partial completion/navigation without invented types.
- Navigation includes locals, parameters, nominal types, explicit exports,
  re-exports, standard-library names, and import strings.
- Completion includes visible names, keywords, builtin types, and intrinsics,
  with identifier replacement ranges. Typing `.` offers fields from the receiver's
  record type, including nominal records, pointers, and nested access. An isolated
  snapshot repairs the unfinished field for type checking; earlier errors can
  still prevent field suggestions. Automatic imports are not implemented.
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
```

The stdio integration tests launch the real server and exercise lifecycle,
unsaved Unicode buffers, all four features, dependency overlays, watched changes,
diagnostic clearing, out-of-order versions, and rapid edits.
