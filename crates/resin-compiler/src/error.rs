use crate::{Error, ast, codegen, lir_verifier, toolchain};
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<codegen::Error> for Error {
    fn from(e: codegen::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<toolchain::Error> for Error {
    fn from(e: toolchain::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<ast::SourceError> for Error {
    fn from(e: ast::SourceError) -> Self {
        Self(e.to_string())
    }
}
impl From<lir_verifier::VerifyError> for Error {
    fn from(e: lir_verifier::VerifyError) -> Self {
        Self(e.to_string())
    }
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Self(e)
    }
}
