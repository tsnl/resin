//! Concrete syntax: source text, an incremental parse tree, and syntax-only queries.
//!
//! [`Document`] keeps text and its Tree-sitter tree together. Parsing, recovery,
//! and formatting require no name resolution or type checking.
//!
//! Implementation modules are deliberately private:
//! ```compile_fail
//! use resin_cst::lower;
//! ```

use resin_executor::{Cancellation, Execution};
use resin_source::prelude::*;
use std::sync::Arc;
mod documentation;
mod lower;
mod preamble;
mod print;
mod query;

pub use tree_sitter::{Node, Tree};

/// One source revision and the concrete syntax tree parsed from it.
/// Clones share immutable text and Tree-sitter storage; reparsing edits a private tree copy.
#[derive(Clone)]
pub struct Document {
    text: Arc<str>,
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

/// Syntax documentation for one module. Spans belong to this exact source.
#[derive(Debug, Clone, Default)]
pub struct Documentation {
    pub module: String,
    pub declarations: Vec<DeclarationDocumentation>,
    pub diagnostics: Vec<Spanned<String>>,
}

/// Includes undocumented and private declarations for editor lookup.
#[derive(Debug, Clone)]
pub struct DeclarationDocumentation {
    pub name: Spanned<String>,
    pub span: Span,
    pub signature: String,
    pub markdown: String,
    pub exported: bool,
    /// Containing struct for fields; other declarations have no field owner.
    pub field_owner: Option<Span>,
}

/// Parse on a bounded worker, reusing an earlier tree without changing its document.
/// Cancellation discards the result; an already running Tree-sitter call may finish first.
pub async fn build_cst(
    text: impl Into<String> + Send,
    previous: Option<&Document>,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<Document, resin_executor::Error> {
    cancellation.check()?;
    let text = text.into();
    let previous = previous.cloned();
    execution
        .run(cancellation, move |_| {
            lower::reparse(text, previous.as_ref())
        })
        .await
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

    /// Read Markdown and declaration signatures without resolving names.
    /// Misplaced doc comments produce diagnostics rather than being discarded.
    pub fn documentation(&self) -> Documentation {
        documentation::read(self)
    }

    /// Recover incomplete editor syntax without shifting original source offsets.
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
