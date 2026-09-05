# `resin`

CUDA for graphics. A simple systems programming language for programming heterogeneous systems.

## Development

Initialize the parser submodule with `git submodule update --init`, then run `cargo test --workspace`.
After changing `tree-sitter-resin/grammar.js`, regenerate its parser with
`tree-sitter generate --js-runtime native` (tree-sitter CLI 0.27.0).

GPU tests skip when no suitable Vulkan device or `glslc` is available. Hardware CI should run
`RESIN_REQUIRE_GPU=1 cargo test -p resin-runtime --tests -- --nocapture` so missing prerequisites
fail the run. Backend tests compile and execute generated C, so development also requires a C11
compiler (`cc`, or the executable named by `CC`).

## Host prototype

```sh
cargo run -- program.resin --output c -o program.c
cargo run -- program.resin --output exe -o program
cargo run -- program.resin --output run
```

The default output is still verified IR. `--cc PATH` selects a compiler without shell parsing.
Compilation replaces the output only after success. A program first evaluates its top-level
definitions, then calls an optional `main = () => { ... };`. Main returns `int` (the process exit
status) or unit. Without main, only top-level initialization runs.

The C backend supports closures, recursion, mutation, records, arrays, pointers, nominal types,
and typed block edges. Closure environments live until process exit; this is not yet a bounded
memory ownership model. Integer arithmetic wraps to its declared width; invalid division, shifts,
and dynamic array indexes terminate with a diagnostic. Backend-unsupported operators are errors.
There is no C FFI, runtime resource API, optimizer, or stable generated ABI yet.

## Resources

- No Graphics API by Sebastian Aaltonen <br/>
  <https://www.sebastianaaltonen.com/blog/no-graphics-api>
