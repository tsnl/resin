//! Concrete syntax: source text, an incremental parse tree, and syntax-only queries.
//!
//! [`Document`] keeps text and its Tree-sitter tree together. Parsing, recovery,
//! and formatting require no name resolution or type checking.
//!
//! Implementation modules are deliberately private:
//! ```compile_fail
//! use resin_cst::lower;
//! ```

use resin_common::prelude::*;
mod lower;
mod print;
mod query;

pub use tree_sitter::{Node, Tree};

/// One source revision and the concrete syntax tree parsed from it.
pub struct Document {
    text: String,
    tree: Tree,
}

impl Document {
    /// Parse text, reusing an earlier revision's tree when supplied.
    pub fn reparse(text: String, previous: Option<&Self>) -> Self {
        lower::reparse(text, previous)
    }

    pub fn source(&self) -> &str {
        &self.text
    }

    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Append missing delimiters for editor recovery; original offsets stay valid.
    pub fn recovery(&self) -> Option<Self> {
        lower::recovery(self)
    }

    /// Read a node from this document's tree.
    pub fn node_text(&self, node: Node<'_>) -> &str {
        &self.text[node.byte_range()]
    }

    /// Find an identifier, literal, or comment touching a UTF-8 byte offset.
    pub fn token(&self, offset: usize) -> Option<Node<'_>> {
        query::token(self, offset)
    }

    /// Whether an identifier can refer to a declaration in its lexical scope.
    pub fn reference(&self, node: Node<'_>) -> bool {
        query::reference(node)
    }

    /// Whether completions at a UTF-8 byte offset should suggest types.
    pub fn type_context(&self, offset: usize) -> bool {
        query::type_context(self, offset)
    }
}

/// Construct a Tree-sitter parser configured for Resin.
pub fn parser() -> tree_sitter::Parser {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .expect("Resin grammar");
    parser
}

pub fn span(node: Node<'_>) -> Span {
    Span {
        start: node.start_byte(),
        end: node.end_byte(),
    }
}

/// Include both boundaries when testing an editor cursor against a source span.
pub fn contains(span: Span, offset: usize) -> bool {
    span.start <= offset && offset <= span.end
}

/// Format complete syntax, preserving comments; return `None` for malformed input.
pub fn format_source(source: &str) -> Option<String> {
    print::format_source(source)
}
