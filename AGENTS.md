# resin — project goals

Resin will be published as an **educational project, distributed as code
only**. The source is the documentation: a curious reader with basic Rust
should be able to understand how a tensor compiler works — tracing, autodiff,
lowering, optimization, code generation — just by reading it.

Everything follows from that:

**Simplicity outranks everything.** Features, performance, generality, and
cleverness all lose to readability. A slower, plainer implementation that a
layperson can follow beats a faster or more general one they cannot.

**Be relentless about cutting.**
- Delete before adding. Every line, type, and dependency must earn its place.
- No speculative abstraction: build for what exists, not what might.
- No unreachable code paths, no "just in case" error taxonomies, no config
  surface without a user.
- If a change makes the code harder to read, it is wrong — regardless of what
  it gains.

**Write for the reader.**
- Prefer plain constructs over clever ones; avoid generics and macros unless
  they remove more complexity than they add.
- Comments state invariants and *why* — never narrate what the code does.
- Keep the pipeline legible: `dsl` (trace + grad) → `ir` (kernels over
  buffers and views, optimization passes) → `jit` (cpu / wgpu backends).
- Tests double as worked examples; make them read like ones.
- Small files, small functions, one idea per module.

When in doubt: cut it, inline it, or simplify it.
