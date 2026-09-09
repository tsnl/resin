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
    statements: Vec<GlslStatement>,
}

#[derive(Debug, Clone)]
pub(super) enum GlslStatement {
    Text {
        source: String,
    },
    If {
        condition: String,
        then: Vec<GlslStatement>,
        els: Vec<GlslStatement>,
    },
    Loop {
        body: Vec<GlslStatement>,
    },
    Break,
    Return {
        value: String,
    },
}
