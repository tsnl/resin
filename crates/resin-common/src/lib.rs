//! Shared source identities, concrete types, and small utilities.
//!
//! These namespaces list the vocabulary available to every compiler phase.
//! Implementations have no dependencies on syntax, lowering, or native tools.
//!
//! Implementation modules are deliberately private:
//! ```compile_fail
//! use resin_common::resolved::TypeTable;
//! ```

#[path = "diagnostic.rs"]
mod diagnostics;
#[path = "source.rs"]
mod locations;
#[path = "types/mod.rs"]
mod resolved;
mod temporary;

/// Shared vocabulary for compiler phases and their callers.
///
/// Import this privately with `use resin_common::prelude::*;`. Phase crates use
/// these types in their interfaces without re-exporting them. Operations such as
/// layout and literal parsing keep their qualified module paths.
pub mod prelude {
    pub use crate::diagnostic::{GenerateError, GenerateErrorKind};
    pub use crate::source::{Ident, SourceError, SourceLocation, SourceNote, Span, Spanned};
    pub use crate::types::shader::{ShaderEntry, Stage};
    pub use crate::types::{
        ArrayValue, BuiltinCall, Case, Conv, Converted, FieldAccess, Foreign, FunctionId,
        Intrinsic, LocalId, RecordField, RecordFieldValue, RecordValue, StaticAddressValue, Ty,
        TypeDef, TypeError, TypeErrorKind, TypeId, TypeTable, TyperContext, Value,
    };
    pub use crate::{TempDir, define_id};
}

/// Byte spans, source identities, and diagnostics with source locations.
pub mod source {
    pub use crate::locations::{Ident, SourceError, SourceLocation, SourceNote, Span, Spanned};
}

/// Shared language diagnostics; phase-local construction state stays private.
pub mod diagnostic {
    pub use crate::diagnostics::{GenerateError, GenerateErrorKind};
}

/// Concrete type data and the rules shared by construction and verification.
pub mod types {
    pub(crate) use crate::resolved::definitions;
    pub use crate::resolved::{
        ArrayValue, BuiltinCall, Case, Conv, Converted, FieldAccess, Foreign, FunctionId,
        Intrinsic, LocalId, RecordField, RecordFieldValue, RecordValue, StaticAddressValue, Ty,
        TypeDef, TypeError, TypeErrorKind, TypeId, TypeTable, TyperContext, Value, check_layout,
        check_references, definition_body,
    };

    pub mod check {
        pub use crate::resolved::check::{
            BuiltinCall, BuiltinRule, Conv, Converted, ExplicitConversion, FieldAccess, TypeError,
            TypeErrorKind, TyperContext, ascription,
        };
    }

    pub mod layout {
        pub use crate::resolved::layout::{Error, Layout, layout};
    }

    pub mod literal {
        pub use crate::resolved::literal::{format, split, unsuffixed_type};
    }

    pub mod print {
        pub use crate::resolved::print::format_type;
    }

    pub mod shader {
        pub use crate::resolved::shader::{Interface, ShaderEntry, Stage, validate};
    }
}

/// A unique directory removed when its owner is dropped.
pub struct TempDir(std::path::PathBuf);

impl TempDir {
    pub fn new(parent: &std::path::Path) -> std::io::Result<Self> {
        temporary::create(parent).map(Self)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Define distinct index types without giving them interchangeable integer identities.
#[macro_export]
macro_rules! define_id {
    (
        $(
            $(#[$attr:meta])*
            $visibility:vis struct $name:ident(usize);
        )+
    ) => {
        $(
            $(#[$attr])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            $visibility struct $name(usize);

            impl $name {
                pub const fn from_index(index: usize) -> Self {
                    Self(index)
                }

                pub const fn index(self) -> usize {
                    self.0
                }
            }
        )+
    };
}

/// Helpers shared by several compiler languages.
pub mod util {
    pub use crate::define_id;
}
