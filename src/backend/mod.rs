mod build;
mod shaders;
pub use build::{Executable, Target, compile, generate};
pub use shaders::build_shaders;

pub mod c;
pub mod glsl;
mod layout;

use std::fmt;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<crate::ir::VerifyError> for Error {
    fn from(error: crate::ir::VerifyError) -> Self {
        Self(error.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<crate::ast::SourceError> for Error {
    fn from(error: crate::ast::SourceError) -> Self {
        Self(error.to_string())
    }
}
