use crate::Error;
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
impl From<resin_codegen::Error> for Error {
    fn from(e: resin_codegen::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<resin_platform_toolchain::Error> for Error {
    fn from(e: resin_platform_toolchain::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<resin_ast::SourceError> for Error {
    fn from(e: resin_ast::SourceError) -> Self {
        Self(e.to_string())
    }
}
impl From<resin_lir_verifier::VerifyError> for Error {
    fn from(e: resin_lir_verifier::VerifyError) -> Self {
        Self(e.to_string())
    }
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Self(e)
    }
}
