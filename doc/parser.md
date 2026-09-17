# Maintaining the parser

The grammar, generated parser, and Rust bindings are maintained in this directory
as part of the [Resin repository](https://github.com/tsnl/resin).

After [setting up the development tools](development.md), run
`cargo test -p tree-sitter-resin` from the repository root to test the Rust bindings
and parser. The root Cargo workspace includes this crate.

Install the matching CLI:

```sh
cargo install --locked tree-sitter-cli --version 0.27.0
```

From `crates/tree-sitter-resin/`, install the locked JavaScript tooling and
check `grammar.js`:

```sh
npm ci --ignore-scripts
npm run check
```

`check` runs strict TypeScript checking of the grammar's JSDoc and Tree-sitter DSL
types, ESLint, and Prettier's formatting check. The individual commands are
`npm run typecheck`, `npm run lint`, and `npm run format:check`; use `npm run format`
to apply formatting. Rule callbacks receive their parameter and result types from
the DSL; standalone helpers declare their parameters and results with JSDoc.
Unknown rule references and implicit `any` types fail the type check.

CI runs all three checks in the **Grammar checks** job alongside the native build
job. All checks must pass. Installation skips lifecycle scripts
because these checks need only JavaScript tooling and the DSL declarations, not
the native Node bindings or the npm CLI binary.

After editing `grammar.js`, also regenerate and test the parser using the Cargo-installed CLI:

```sh
tree-sitter generate --js-runtime native
tree-sitter test
```

Commit the grammar and generated files together in Resin. Zed extensions can load
this grammar from `https://github.com/tsnl/resin` with `path = "crates/tree-sitter-resin"`
and a pinned Resin commit containing the parser.

The initial contents were imported from `tsnl/tree-sitter-resin` at commit
`8f167ef57f5c894aba3f05f1bfae1d856d95ad6b`.

## License

The Resin grammar and bindings are licensed under
[Apache-2.0](../crates/tree-sitter-resin/LICENSE). See [NOTICE](../crates/tree-sitter-resin/NOTICE) for attribution. The bundled
Tree-sitter headers in `src/tree_sitter/` retain their
[MIT license](../crates/tree-sitter-resin/src/tree_sitter/LICENSE).
