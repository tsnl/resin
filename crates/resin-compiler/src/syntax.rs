//! Cached products of the CST and AST passes, retained together by the driver.
use crate::{ast, cst};
pub(crate) struct Document {
    pub syntax: cst::Document,
    pub file: ast::SourceFile,
    pub errors: Vec<(ast::Span, String)>,
}
impl Document {
    pub fn reparse(text: String, previous: Option<&Self>) -> Self {
        let syntax = cst::Document::reparse(text, previous.map(|d| &d.syntax));
        let ast::Parsed { file, errors } = ast::recover(&syntax);
        Self {
            syntax,
            file,
            errors,
        }
    }
}
impl std::ops::Deref for Document {
    type Target = cst::Document;
    fn deref(&self) -> &Self::Target {
        &self.syntax
    }
}
