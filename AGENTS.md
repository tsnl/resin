# `AGENTS.md`

Resin is a simple systems programming language targeting both host (CPU) and device (GPU). 
Think CUDA, but lowering to Vulkan and exposing fixed-function rendering functionality.

## Development Practices

- We use `tree-sitter` to generate a parser. The repo `tree-sitter-resin` is checked out as a 
  submodule in the root of the repo. Be careful to update both this and that repo if needed when you
  make changes.
