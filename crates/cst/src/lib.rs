//! Concrete syntax and syntax-only operations.
use resin_common::{source, types};
mod language;
mod query;
pub use language::Document;
pub mod lower;
pub mod print;
pub use lower::{contains, span};
pub use tree_sitter::{Node, Tree};

/// Construct the parser for Resin's concrete syntax language.
pub fn parser() -> tree_sitter::Parser {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .expect("Resin grammar");
    parser
}
