//! Record layouts shared by the fixed-function hardware nodes (see
//! `docs/hw-nodes.md`). Both the DSL (shape construction) and the IR/backends
//! (validation, kernel emission) address these packed f32 records by channel.

/// Channels per `trace_rays` hit record: `(t, u, v, prim, hit)`.
pub const TRACE_HIT_WIDTH: usize = 5;

/// `trace_rays` hit-record channel indices.
pub mod trace_channel {
    /// Ray parameter of the closest hit (`0` on miss).
    pub const T: usize = 0;
    /// Barycentric coordinate of vertex 1.
    pub const U: usize = 1;
    /// Barycentric coordinate of vertex 2 (`w = 1 - u - v` for vertex 0).
    pub const V: usize = 2;
    /// Triangle index as an exact integer-valued f32 (`0` on miss).
    pub const PRIM: usize = 3;
    /// `1.0` on hit, `0.0` on miss.
    pub const HIT: usize = 4;
}

/// Channels per `rasterize` visibility-buffer pixel: `(prim, hit, u, v)`.
pub const RASTER_PIXEL_WIDTH: usize = 4;

/// `rasterize` pixel channel indices.
pub mod raster_channel {
    /// Triangle index as an exact integer-valued f32 (`0` for background).
    pub const PRIM: usize = 0;
    /// `1.0` where a triangle covers the pixel center, `0.0` for background.
    pub const HIT: usize = 1;
    /// Perspective-correct barycentric coordinate of vertex 1.
    pub const U: usize = 2;
    /// Perspective-correct barycentric coordinate of vertex 2.
    pub const V: usize = 3;
}
