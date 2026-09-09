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
    Structured {
        locals: String,
        statements: Vec<CStatement>,
    },
}

#[derive(Debug, Clone)]
pub(super) enum CStatement {
    Text {
        source: String,
    },
    If {
        condition: String,
        then: Vec<CStatement>,
        els: Vec<CStatement>,
    },
    Loop {
        body: Vec<CStatement>,
    },
    Break,
    Return {
        value: String,
    },
}
