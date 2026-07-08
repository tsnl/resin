//! Renderers composed from resin graphs and the fixed-function hardware
//! nodes (`docs/hw-nodes.md`), in the spirit of "composition over custom
//! kernels": the nodes resolve *visibility*, everything else — ray
//! generation, interpolation, materials, lighting, accumulation — is
//! ordinary differentiable graph code.
//!
//! - [`pathtrace`]: Monte-Carlo path tracer over `trace_rays`.
//! - [`raster`]: deferred shading over the `rasterize` visibility buffer.
//! - [`scene`] / [`camera`]: shared host-side fixtures and camera math.

pub mod camera;
pub mod pathtrace;
pub mod ppm;
pub mod raster;
pub mod scene;

pub use camera::Camera;
pub use scene::{cornell_box, Scene};
