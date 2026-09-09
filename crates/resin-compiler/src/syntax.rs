//! Cached products of the CST and AST passes, retained together by the driver.
pub(crate) struct Document {
    pub syntax: resin_cst::Document,
    pub file: resin_ast::SourceFile,
    pub errors: Vec<(resin_ast::Span, String)>,
}
impl Document {
    pub fn reparse(text: String, previous: Option<&Self>) -> Self {
        let syntax = resin_cst::Document::reparse(text, previous.map(|d| &d.syntax));
        let resin_ast::Parsed { file, errors } = resin_ast::recover(&syntax);
        Self {
            syntax,
            file,
            errors,
        }
    }
}
impl std::ops::Deref for Document {
    type Target = resin_cst::Document;
    fn deref(&self) -> &Self::Target {
        &self.syntax
    }
}
