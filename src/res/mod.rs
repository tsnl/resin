//! `res` manages resources that are resident on the GPU or another peripheral.

mod generic;
mod geometry;
mod heap;
mod texture;

pub use geometry::{
    Geometry, GeometryArgs, GeometryDrawArgs, GeometryInfo, GeometryManager, GeometryManagerArgs,
};
pub use texture::{Texture, TextureManager};
