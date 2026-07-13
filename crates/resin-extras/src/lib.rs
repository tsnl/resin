//! Optional extras for resin: host dataset loaders and HTML notebooks.
//!
//! Kept out of the core `resin` crate so the compiler pipeline stays small and
//! free of download / plotting dependencies.

pub mod dataset;
pub mod notebook;
pub mod sampler;
