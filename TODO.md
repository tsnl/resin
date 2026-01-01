# `TODO`

-   [x] (ABANDONED) `draw_3d.py`: finish renderer implementation.
    -   [x] Render flat-shaded geometry with fixed light source.
    -   [x] Basic IBL, but no convoluted cubemaps yet.
    -   [ ] Cubemap convolution for diffuse irradiance.
    -   [ ] Prefiltered environment map for specular reflections.

-   [ ] Implement `draw_3d_v2.py`, hardware-accelerated ray-tracer.
    
    This lets us use the same renderer for offline cubemap generation and real-time
    ray-tracing. We can still sample irradiance probes, specular probes at runtime, 
    using RT for primary rays. We can also scale the renderer quality more smoothly,
    performing more bounces or more samples per pixel as needed.
    -   [ ] Basic ray-tracer (primary rays only), flat-shaded at most.

-   `draw_2d.py`, `draw_3d.py`: make `draw()` more stateless, i.e. all draw input as 
    args. No need to call `clear()`.
    -   [x] `draw_2d.py`
    -   [ ] `draw_3d.py` (rewrite WIP)

-   `gpu.py`: merge `GpuEzBuffer` into `GpuBuffer`. Remove persistent staging buffer.
