use std::fmt::Debug;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::Config;

#[derive(Clone)]
pub struct Source(Arc<SourceInner>);

impl Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Source({:#?})", self.0.name)
    }
}

impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Source {}

impl Source {
    pub fn new<S: Into<String>, T: Into<Box<str>>>(name: S, text: T, config: &Config) -> Self {
        let name: String = name.into();
        let text: Box<str> = text.into();
        let offset_to_line_col_map = OffsetToLineColMap::new(&text, config.tab_spaces);
        let inner = SourceInner {
            name,
            text,
            offset_to_line_col_map,
        };
        Self(Arc::new(inner))
    }
    pub fn new_file<P: AsRef<Path>>(path: P, config: &Config) -> Self {
        let path: &Path = path.as_ref();
        let name: String = path.as_os_str().to_string_lossy().into();
        let text = fs::read_to_string(path).unwrap();
        Self::new(name, text, config)
    }
    pub fn name(&self) -> &str {
        &self.0.name
    }
    pub fn text(&self) -> &str {
        &self.0.text
    }
    pub fn line_column(&self, offset: u32) -> (u16, u16) {
        self.0.offset_to_line_col_map.lookup(offset)
    }
}

#[derive(Debug)]
struct SourceInner {
    name: String,
    text: Box<str>,
    offset_to_line_col_map: OffsetToLineColMap,
}

#[derive(Debug)]
pub struct OffsetToLineColMap {
    marker_vec: Vec<(u32, (u16, u16))>,
}
impl OffsetToLineColMap {
    fn new(s: &str, tab_spaces: u16) -> Self {
        let mut marker_vec = vec![(0, (1, 1))];
        let mut line: u16 = 1;
        let mut column: u16 = 1;

        for (offset, byte) in s.as_bytes().iter().cloned().enumerate().skip(1) {
            if byte == b'\n' {
                line += 1;
                column = 1;
                marker_vec.push(((offset + 1) as u32, (line, column)));
            } else if byte == b'\t' {
                column = column.next_multiple_of(tab_spaces) + 1;
                marker_vec.push(((offset + 1) as u32, (line, column)));
            } else {
                column += 1;
                // No need to push a marker for every byte
            }
        }

        Self { marker_vec }
    }
    fn lookup(&self, offset: u32) -> (u16, u16) {
        match self
            .marker_vec
            .binary_search_by_key(&offset, |(offset, _)| *offset)
        {
            Ok(index) => {
                let (_, (line, col)) = self.marker_vec[index];
                (line, col)
            }
            Err(0) => (1, 1),
            Err(index) => {
                let (marker_offset, (marker_line, marker_col)) = self.marker_vec[index - 1];
                let delta = offset - marker_offset;
                (marker_line, marker_col + delta as u16)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset_to_line_col_map() {
        let text = "the quick\nbrown fox jumps\nover the lazy\tdog\n";
        let map = OffsetToLineColMap::new(text, 4);
        assert_eq!(map.lookup(text.find('t').unwrap() as u32), (1, 1));
        assert_eq!(map.lookup(text.find('b').unwrap() as u32), (2, 1));
        assert_eq!(map.lookup(text.find('v').unwrap() as u32), (3, 2));
        assert_eq!(map.lookup(text.find('d').unwrap() as u32), (3, 17));
    }
}
