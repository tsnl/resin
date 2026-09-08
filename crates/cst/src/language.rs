//! The concrete syntax language is Tree-sitter’s tree paired with its source text.
use tree_sitter::Tree;

pub struct Document {
    pub(super) text: String,
    pub(super) tree: Tree,
}
impl Document {
    pub fn source(&self) -> &str {
        &self.text
    }
    pub fn tree(&self) -> &Tree {
        &self.tree
    }
}
