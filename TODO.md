# `TODO`

-   `draw_3d.py`: finish renderer implementation.

-   `draw_2d.py`, `draw_3d.py`: make `draw()` more stateless, i.e. all draw input as 
    args. Lets us guarantee we call `clear()` ahead of time.

-   `gpu.py`: merge `GpuEzBuffer` into `GpuBuffer`. Remove persistent staging buffer.

-   `draw_2d.py`: clean up text rendering API.
    -   Remove dedicated text quad batch. Wrap `add_quad()` instead.
    -   Fix multiple text quad batches.
    -   Create descriptor sets at init time, not lazily.
    -   Defer glyph cache updates to start of `draw()`, then call `add_quad()` on frozen
        glyph cache.
