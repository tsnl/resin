use crate::Span;
use std::fmt::{Debug, Display};

pub type Result<T> = std::result::Result<T, Error>;

pub enum Error {
    SyntaxError(SyntaxError),
    MalformedSyntacticConstructError(MalformedSyntacticConstructError),
    UndefinedSymbolError(UndefinedSymbolError),
    IncorrectSymbolKindError(IncorrectSymbolKindError),
    TypeJoinError(TypeJoinError),
    TypeError(TypeError),
}
impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::SyntaxError(e) => f.write_fmt(format_args!("SyntaxError: {e}")),
            Error::MalformedSyntacticConstructError(e) => {
                f.write_fmt(format_args!("SyntaxUsageError: {e}"))
            }
            Error::UndefinedSymbolError(e) => {
                f.write_fmt(format_args!("UndefinedSymbolError: {e}"))
            }
            Error::IncorrectSymbolKindError(e) => {
                f.write_fmt(format_args!("IncorrectSymbolKindError: {e}"))
            }
            Error::TypeJoinError(e) => f.write_fmt(format_args!("TypeJoinError: {e}")),
            Error::TypeError(e) => f.write_fmt(format_args!("TypeError: {e}")),
        }
    }
}

pub struct SyntaxError {
    pub title: String,
    pub expected: Option<Vec<String>>,
    pub loc: Loc,
}
impl Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.title)?;
        if let Some(expected) = &self.expected {
            f.write_fmt(format_args!("\n- expected: {}", expected.join(", ")))?;
        }
        f.write_fmt(format_args!("\n- see: {}", self.loc))?;
        Ok(())
    }
}

pub struct MalformedSyntacticConstructError {
    pub title: String,
    pub loc: Loc,
}
impl Display for MalformedSyntacticConstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.title)?;
        f.write_fmt(format_args!("\n- see: {}", self.loc))?;
        Ok(())
    }
}

pub struct UndefinedSymbolError {
    pub name: String,
    pub loc: Loc,
}
impl Display for UndefinedSymbolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Symbol used but not defined: ")?;
        f.write_str(&self.name)?;
        f.write_str("\n- see: ")?;
        f.write_fmt(format_args!("{}", self.loc))?;
        Ok(())
    }
}

pub struct IncorrectSymbolKindError {
    pub name: String,
    pub loc: Loc,
}
impl Display for IncorrectSymbolKindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Symbol used in incorrect context: ")?;
        f.write_str(&self.name)?;
        f.write_str("\n- see: ")?;
        f.write_fmt(format_args!("{}", self.loc))?;
        Ok(())
    }
}

pub struct TypeJoinError {
    pub title: String,
    pub src: String,
    pub dst: String,
    pub loc: Loc,
}
impl Display for TypeJoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.title)?;

        f.write_str("\n- src type: ")?;
        f.write_str(&self.src)?;

        f.write_str("\n- dst type: ")?;
        f.write_str(&self.dst)?;

        f.write_str("\n- see: ")?;
        f.write_fmt(format_args!("{}", self.loc))?;
        Ok(())
    }
}

pub struct TypeError {
    pub title: String,
    pub loc: Loc,
}
impl Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.title)?;
        f.write_fmt(format_args!("\n- see: {}", self.loc))?;
        Ok(())
    }
}

#[derive(Clone)]
pub enum Loc {
    File(Span),
    Builtin(String),
}
impl Display for Loc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Loc::File(span) => write!(f, "{}", span),
            Loc::Builtin(caption) => write!(f, "{}", caption),
        }
    }
}
impl Debug for Loc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Loc({:?})", format!("{self}")))
    }
}
impl From<Span> for Loc {
    fn from(span: Span) -> Self {
        Loc::File(span)
    }
}
