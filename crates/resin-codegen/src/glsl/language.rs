//! Shader source after device restrictions and symbolic addresses are resolved.
//! Leaf strings contain GLSL syntax. Control flow remains explicit data until printing.

#[derive(Debug, Clone)]
pub struct Module {
    pub extensions: String,
    pub declarations: String,
    pub globals: String,
    pub functions: Vec<Function>,
    pub entry: String,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub signature: String,
    pub locals: String,
    pub entry: usize,
    pub blocks: Vec<Block>,
    pub default_result: String,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub label: usize,
    pub statements: String,
    pub exit: Exit,
}

#[derive(Debug, Clone)]
pub enum Exit {
    Return(String),
    Jump(Edge),
    Branch {
        condition: String,
        then: Edge,
        els: Edge,
    },
    Unreachable,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub target: usize,
    pub values: Vec<EdgeValue>,
}

#[derive(Debug, Clone)]
pub struct EdgeValue {
    pub slot: usize,
    pub ty: String,
    pub value: String,
}
