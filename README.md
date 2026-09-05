# `resin`

CUDA for graphics. A simple systems programming language for programming heterogeneous systems.

## Development

Initialize the parser submodule with `git submodule update --init`, then run `cargo test --workspace`.
After changing `tree-sitter-resin/grammar.js`, regenerate its parser with
`tree-sitter generate --js-runtime native` (tree-sitter CLI 0.27.0).

GPU tests skip when no suitable Vulkan device or `glslc` is available. Hardware CI should run
`RESIN_REQUIRE_GPU=1 cargo test -p resin-runtime --tests -- --nocapture` so missing prerequisites
fail the run. The compiler currently generates verified IR; an executable backend is still to come.

## Resources

- No Graphics API by Sebastian Aaltonen <br/>
  <https://www.sebastianaaltonen.com/blog/no-graphics-api>
