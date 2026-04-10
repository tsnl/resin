# TODO

## Phase 1: Make It Work (Current)

-   Model a graph-based language for static graphs, feed-forward, no loops/mutation/tail-recursion.
-   Create an `Interpreter` that allows execution of this graph.

Also
-   For better compiler optimization, need to rewrite `graph.py` once more, with `View` used to read/write
    nodes instead of being its own node type.

> [!NOTE]
>
> Support for GPU-side loops adds a LOT of complexity.
>
> Keep it simple, stupid. Feed-forward only.
