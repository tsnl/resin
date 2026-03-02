# TODO

## Phase 1: Make It Work (Current)

-   Model a graph-based language for static graphs, feed-forward, no loops/mutation/tail-recursion.
-   Create an `Interpreter` that allows execution of this graph.

## Phase 2: Make It Good

-   Extend existing graph with a `RecurNode` for tail calls, enabling loops, tail recursion, mutation, etc.
-   Extend existing type-system to support richer constructs, e.g. tuples, structs, etc. Broadcasting and other semantics should also generalize over these accessors, not just array indexing.
-   Implement custom syntactic frontend: pure functional programming language. See `examples/` for more.
