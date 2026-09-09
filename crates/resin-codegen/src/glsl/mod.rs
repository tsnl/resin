//! GLSL lowering and printing.
pub(super) mod lower;
pub(super) mod print;

//
// Target language
//

/// GLSL translation unit and control flow.
#[derive(Debug, Clone)]
pub(super) struct GlslModule {
    extensions: String,
    declarations: String,
    globals: String,
    functions: Vec<GlslFunction>,
    entry: String,
}

#[derive(Debug, Clone)]
pub(super) struct GlslFunction {
    signature: String,
    locals: String,
    entry: usize,
    blocks: Vec<GlslBlock>,
    default_result: String,
}

#[derive(Debug, Clone)]
pub(super) struct GlslBlock {
    label: usize,
    statements: String,
    exit: GlslExit,
}

#[derive(Debug, Clone)]
pub(super) enum GlslExit {
    Return(String),
    Jump(GlslEdge),
    Branch {
        condition: String,
        then: GlslEdge,
        els: GlslEdge,
    },
    Unreachable,
}

#[derive(Debug, Clone)]
pub(super) struct GlslEdge {
    target: usize,
    values: Vec<GlslEdgeValue>,
}

#[derive(Debug, Clone)]
pub(super) struct GlslEdgeValue {
    slot: usize,
    ty: String,
    value: String,
}
