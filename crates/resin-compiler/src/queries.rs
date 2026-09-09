use crate::compilation::Data;
use std::path::Path;
impl resin_hir::Documents for Data {
    fn get(&self, path: &Path) -> Option<&resin_cst::Document> {
        self.documents.get(path).map(|document| &document.syntax)
    }
}
