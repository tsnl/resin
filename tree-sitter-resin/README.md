# Tree-sitter grammar for Resin

The grammar, generated parser, and Rust bindings are maintained in this directory
as part of the [Resin repository](https://github.com/tsnl/resin).

From the repository root, enter `nix-shell` and run `cargo test -p tree-sitter-resin`
to test the Rust bindings and parser. The root Cargo workspace includes this crate.

Install the matching CLI inside the shell:

```sh
cargo install --locked tree-sitter-cli --version 0.27.0
```

After editing `grammar.js`, run these commands from this directory inside the shell:

```sh
tree-sitter generate --js-runtime native
tree-sitter test
```

Commit the grammar and generated files together in Resin. Zed extensions can load
this grammar from `https://github.com/tsnl/resin` with `path = "tree-sitter-resin"`
and a pinned Resin commit containing the parser.

The initial contents were imported from `tsnl/tree-sitter-resin` at commit
`8f167ef57f5c894aba3f05f1bfae1d856d95ad6b`.
