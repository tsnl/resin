use lsp_types::{Position, Range};
use resin_source::prelude::*;

pub struct Text {
    source: String,
    lines: Vec<usize>,
}

impl Text {
    pub fn new(source: &str) -> Self {
        let mut lines = vec![0];
        lines.extend(
            source
                .bytes()
                .enumerate()
                .filter_map(|(i, c)| (c == b'\n').then_some(i + 1)),
        );
        Self {
            source: source.into(),
            lines,
        }
    }

    pub fn offset(&self, position: Position) -> Option<usize> {
        let start = *self.lines.get(position.line as usize)?;
        let end = self
            .lines
            .get(position.line as usize + 1)
            .copied()
            .unwrap_or(self.source.len());
        let line = self.source[start..end].trim_end_matches(['\r', '\n']);
        let mut column = 0;
        for (byte, c) in line.char_indices() {
            if column == position.character {
                return Some(start + byte);
            }
            column += c.len_utf16() as u32;
            if column > position.character {
                return None;
            }
        }
        Some(start + line.len()) // LSP clamps columns past the end of a line.
    }

    pub fn position(&self, offset: usize) -> Position {
        let mut offset = offset.min(self.source.len());
        while !self.source.is_char_boundary(offset) {
            offset -= 1;
        }
        let line = self
            .lines
            .partition_point(|start| *start <= offset)
            .saturating_sub(1);
        let end = self
            .lines
            .get(line + 1)
            .copied()
            .unwrap_or(self.source.len());
        let content = self.source[self.lines[line]..end].trim_end_matches(['\r', '\n']);
        let offset = offset.min(self.lines[line] + content.len());
        Position::new(
            line as u32,
            self.source[self.lines[line]..offset].encode_utf16().count() as u32,
        )
    }

    pub fn range(&self, span: Span) -> Range {
        Range::new(
            self.position(span.start),
            self.position(span.end.max(span.start)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_crlf_and_eof_positions_roundtrip() {
        let source = "a😀é\r\nlast\n";
        let text = Text::new(source);
        assert_eq!(text.offset(Position::new(0, 1)), Some(1));
        assert_eq!(text.offset(Position::new(0, 2)), None);
        assert_eq!(text.offset(Position::new(0, 3)), Some(5));
        assert_eq!(text.offset(Position::new(0, 100)), Some(7));
        assert_eq!(text.position(9), Position::new(1, 0));
        assert_eq!(text.position(8), Position::new(0, 4));
        assert_eq!(text.offset(Position::new(2, 0)), Some(source.len()));
        assert_eq!(text.offset(Position::new(3, 0)), None);
        for (byte, c) in source
            .char_indices()
            .filter(|(_, c)| !matches!(c, '\r' | '\n'))
        {
            assert_eq!(text.offset(text.position(byte)), Some(byte), "{c}");
        }
    }
}
