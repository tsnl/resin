//! GLSL lowering and printing.
pub(super) mod lower;
pub(super) mod print;

pub(super) use crate::{
    GlslBlock as Block, GlslEdge as Edge, GlslEdgeValue as EdgeValue, GlslExit as Exit,
    GlslFunction as Function, GlslModule as Module,
};
