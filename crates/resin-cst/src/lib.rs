//! Concrete syntax: source text, an incremental parse tree, and syntax-only queries.
//!
//! [`Document`] keeps text and its Tree-sitter tree together. Parsing, recovery,
//! and formatting require no name resolution or type checking.
//!
//! Implementation modules are deliberately private:
//! ```compile_fail
//! use resin_cst::lower;
//! ```

use resin_source::prelude::*;
use std::sync::Arc;
mod lower;
mod preamble;
mod print;
mod query;

pub use tree_sitter::{Node, Tree};

/// One source revision and the concrete syntax tree parsed from it.
pub struct Document {
    text: String,
    tree: Tree,
}

/// Syntax-only dependencies, including valid entries recovered from incomplete input.
/// Nonempty diagnostics mean the preamble cannot describe a complete input graph.
#[derive(Debug, Clone, Default)]
pub struct Preamble {
    pub imports: Vec<Spanned<Arc<str>>>,
    pub headers: Vec<Spanned<Arc<str>>>,
    pub diagnostics: Vec<Spanned<String>>,
}

/// Build a CST document, reusing an earlier tree when supplied.
pub fn build_cst(text: impl Into<String>, previous: Option<&Document>) -> Document {
    lower::reparse(text.into(), previous)
}

impl Document {
    pub fn source(&self) -> &str {
        &self.text
    }

    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Read imports and foreign headers without requiring valid function bodies.
    pub fn preamble(&self) -> Preamble {
        preamble::read(self)
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

/// Decode a complete quoted Resin string, rejecting malformed or incomplete text.
pub fn decode_string(text: &str) -> Option<String> {
    let text = text.strip_prefix('"')?.strip_suffix('"')?;
    let mut chars = text.chars();
    let mut result = String::new();
    while let Some(ch) = chars.next() {
        result.push(match ch {
            '\\' => match chars.next()? {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '0' => '\0',
                '"' => '"',
                '\\' => '\\',
                _ => return None,
            },
            '"' | '\r' | '\n' => return None,
            _ => ch,
        });
    }
    Some(result)
}
