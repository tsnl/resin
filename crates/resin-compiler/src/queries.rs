use crate::{compilation::Data, cst, hir};
use std::path::Path;
impl hir::Documents for Data {
    fn get(&self, path: &Path) -> Option<&cst::Document> {
        self.documents.get(path).map(|document| &document.syntax)
    }
}
