//! Editor queries delegated to HIR's opaque analysis interface.
pub use crate::compiler::{Diagnostic, Snapshot as Analysis, Sources, normalize_path};
pub use crate::hir::analysis::{Completion, Definition, DefinitionKind, Hover};
use crate::source::SourceLocation;
use std::path::Path;

impl crate::hir::analysis::Documents for Analysis {
    fn get(&self, path: &Path) -> Option<&crate::cst::Document> {
        self.documents.get(path).map(|document| &document.syntax)
    }
}
impl Analysis {
    pub fn definition(&self, path: &Path, offset: usize) -> Option<SourceLocation> {
        self.semantics.definition(self, path, offset)
    }
    pub fn hover(&self, path: &Path, offset: usize) -> Option<Hover> {
        self.semantics.hover(self, path, offset)
    }
    pub fn completions(&self, path: &Path, offset: usize) -> Vec<Completion> {
        self.semantics.completions(self, path, offset)
    }
}
