use std::fmt;

use crate::Span;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Syntax,
    Scope,
    Type,
    Io,
    Internal,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self {
            Self::Syntax => "syntax",
            Self::Scope => "scope",
            Self::Type => "type",
            Self::Io => "io",
            Self::Internal => "internal",
        };
        f.write_str(kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorNote {
    pub message: String,
    pub span: Span,
}

impl ErrorNote {
    pub fn new<M: Into<String>>(message: M, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<ErrorNote>,
}

impl Error {
    pub fn new<M: Into<String>>(kind: ErrorKind, message: M) -> Self {
        Self {
            kind,
            message: message.into(),
            span: None,
            notes: Vec::new(),
        }
    }

    pub fn new_at<M: Into<String>>(kind: ErrorKind, message: M, span: Span) -> Self {
        Self {
            kind,
            message: message.into(),
            span: Some(span),
            notes: Vec::new(),
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_note<M: Into<String>>(mut self, message: M, span: Span) -> Self {
        self.notes.push(ErrorNote::new(message, span));
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} error: {}", self.kind, self.message)?;
        if let Some(span) = &self.span {
            write!(f, " at {}", span)?;
        }
        for note in &self.notes {
            write!(f, "\n  note: {} at {}", note.message, note.span)?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Source};

    #[test]
    fn test_display_without_span() {
        let error = Error::new(ErrorKind::Syntax, "something went wrong");
        assert_eq!(error.to_string(), "syntax error: something went wrong");
    }

    #[test]
    fn test_display_with_span_and_note() {
        let source = Source::new("test", "abc", &Config::default());
        let error = Error::new_at(
            ErrorKind::Scope,
            "symbol is undefined",
            Span::new(&source, 0, 0),
        )
        .with_note("symbol was introduced here", Span::new(&source, 1, 1));
        assert_eq!(
            error.to_string(),
            "scope error: symbol is undefined at test:1:1\n  note: symbol was introduced here at test:1:2"
        );
    }
}
