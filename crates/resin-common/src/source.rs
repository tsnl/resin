//! Source identities, spans, diagnostics, and access shared by all phases.
use std::{fmt, path::PathBuf, sync::Arc};

pub type Ident = Spanned<Arc<str>>;

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub val: T,
    pub span: Span,
}
impl<T> Spanned<T> {
    pub fn new(val: T, span: Span) -> Self {
        Self { val, span }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl<T> std::ops::Deref for Spanned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.val
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceLocation {
    pub path: PathBuf,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct SourceNote {
    pub location: SourceLocation,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct SourceError {
    pub path: PathBuf,
    pub span: Option<Span>,
    /// Human-readable CLI rendering, including the import chain.
    pub message: Box<str>,
    /// The diagnostic itself, without path prefixes or an import chain.
    pub diagnostic: String,
    pub related: Vec<SourceNote>,
}

impl SourceError {
    pub fn new(path: PathBuf, span: Option<Span>, diagnostic: String) -> Self {
        Self {
            message: format!("{}: {diagnostic}", path.display()).into(),
            path,
            span,
            diagnostic,
            related: Vec::new(),
        }
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for SourceError {}
