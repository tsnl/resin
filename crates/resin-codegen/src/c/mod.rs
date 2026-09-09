//! C11 lowering and printing.
pub(super) mod lower;
pub(super) mod print;

//
// Target language
//

/// C11 translation unit and control flow.
#[derive(Debug, Clone)]
pub(super) struct CModule {
    includes: Vec<String>,
    assertions: Vec<(String, String)>,
    declarations: String,
    local_includes: Vec<String>,
    prototypes: Vec<String>,
    lifecycle: String,
    functions: Vec<CFunction>,
    entry: CFunction,
}

#[derive(Debug, Clone)]
pub(super) struct CFunction {
    signature: String,
    body: CBody,
}

#[derive(Debug, Clone)]
pub(super) enum CBody {
    Inline(String),
    Blocks {
        locals: String,
        entry: usize,
        blocks: Vec<CBlock>,
    },
}

#[derive(Debug, Clone)]
pub(super) struct CBlock {
    label: usize,
    statements: String,
    exit: CExit,
}

#[derive(Debug, Clone)]
pub(super) enum CExit {
    Return(String),
    Jump(CEdge),
    Branch {
        condition: String,
        then: CEdge,
        els: CEdge,
    },
    Unreachable,
}

#[derive(Debug, Clone)]
pub(super) struct CEdge {
    target: usize,
    values: Vec<CEdgeValue>,
}

#[derive(Debug, Clone)]
pub(super) struct CEdgeValue {
    ty: String,
    value: String,
    live: Option<String>,
}
