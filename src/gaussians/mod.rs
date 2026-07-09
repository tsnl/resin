//! 3D Gaussian Splatting as a *library of graph functions* over the resin
//! DSL — zero domain-specific kernels.
//!
//! The renderer is ordinary data flow built from general tensor ops:
//! project → cull (validity masks, not compaction) → depth `argsort` →
//! `gather_rows` → per-pixel alpha from conics → transmittance
//! (`cumprod_exclusive`, i.e. a scan) → weighted reduction. Because every
//! stage is composed from differentiable nodes, `grad_wrt` works through the
//! whole image formation with no renderer-specific adjoint code.
//!
//! This is the *untiled* formulation: it materializes `O(N·H·W)` alpha and
//! transmittance tensors, which is fine for small scenes and golden tests.
//! Tiling (per-tile gaussian lists via a tile-key sort + segment offsets)
//! restores locality with the same combinators.

mod camera;
mod cloud;
mod linalg;
mod preprocess;
mod reference;
mod render;

pub use camera::Camera;
pub use cloud::{gnomen_cloud, CloudData, GaussianCloud};
pub use preprocess::{preprocess, Preprocessed};
pub use reference::render_reference;
pub use render::render;
