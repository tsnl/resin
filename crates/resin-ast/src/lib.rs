//! Abstract syntax tree for Resin.
use resin_common::source;
use resin_cst as cst;

mod language;
pub use language::*;
mod load;
pub mod lower;
pub mod print;
pub use load::{
    FileSystem, SourceError, SourceLocation, SourceNote, SourceProvider, load, load_with,
    resolve_import, stdlib_path,
};
pub use load::{Loaded, load_parsed};

use lower::AstGen;
pub use lower::{AstError, AstErrorKind};
