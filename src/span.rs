use std::fmt::{Debug, Display, Write};

use super::Source;

#[derive(Clone, PartialEq, Eq)]
pub struct Span {
    pub source: Source,
    pub beg_offset: u32,
    pub end_offset: u32,
}
impl Span {
    pub fn dummy() -> Self {
        let source = Source::new("dummy", "", &crate::Config::default());
        Self {
            source,
            beg_offset: 0,
            end_offset: 0,
        }
    }

    pub fn new(source: &Source, beg_offset: usize, end_offset: usize) -> Self {
        let source = source.clone();
        let beg_offset = beg_offset.try_into().unwrap();
        let end_offset = end_offset.try_into().unwrap();
        Self {
            source,
            beg_offset,
            end_offset,
        }
    }
}
impl Debug for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Span(")?;
        f.write_char('"')?;
        Display::fmt(&self, f)?;
        f.write_char('"')?;
        f.write_str(")")?;
        Ok(())
    }
}
impl Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Format this span as a "terminal hyperlink", which is a special string that becomes clickable in terminals
        // that support it.
        // - https://github.com/zed-industries/zed/blob/c241eadb/crates/terminal/src/terminal_hyperlinks.rs
        // - https://learn.microsoft.com/en-us/visualstudio/msbuild/msbuild-diagnostic-format-for-tasks?view=vs-2022

        // Unfortunately, many editors don't support multi-line spans. So we just print the start position in this case.

        let (beg_line, beg_column) = self.source.line_column(self.beg_offset);
        let (end_line, end_column) = self.source.line_column(self.end_offset);

        if beg_line == end_line && beg_column != end_column {
            f.write_fmt(format_args!(
                "{}:{}:{}-{}",
                self.source.name(),
                beg_line,
                beg_column,
                end_column
            ))
        } else {
            f.write_fmt(format_args!(
                "{}:{}:{}",
                self.source.name(),
                beg_line,
                beg_column
            ))
        }
    }
}
