# `TODO`

-   [ ] `draw_3d.py`: finish renderer implementation.
    -   [x] Render flat-shaded geometry with fixed light source.
    -   [x] Basic IBL, but no convoluted cubemaps yet.
    -   [ ] Cubemap convolution for diffuse irradiance.
    -   [ ] Prefiltered environment map for specular reflections.

-   `draw_2d.py`, `draw_3d.py`: make `draw()` more stateless, i.e. all draw input as 
    args. No need to call `clear()`.

-   `gpu.py`: merge `GpuEzBuffer` into `GpuBuffer`. Remove persistent staging buffer.

-   `draw_2d.py`: clean up text rendering API.
    -   Remove dedicated text quad batch. Wrap `add_quad()` instead.
    -   Fix multiple text quad batches.
    -   Create descriptor sets at init time, not lazily.
    -   Defer glyph cache updates to start of `draw()`, then call `add_quad()` on frozen
        glyph cache.
