use std::path::{Path, PathBuf};

pub(super) fn parse(source: &str, input: &Path) -> Vec<PathBuf> {
    let source = source.replace("\\\r\n", "").replace("\\\n", "");
    let mut paths = Vec::new();
    let mut word = String::new();
    let mut chars = source
        .strip_prefix("resin:")
        .unwrap_or(&source)
        .chars()
        .peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if matches!(chars.peek(), Some(' ' | '\t' | '#' | '\\')) => {
                word.push(chars.next().unwrap());
            }
            '$' if chars.peek() == Some(&'$') => {
                chars.next();
                word.push('$');
            }
            c if c.is_whitespace() => {
                if !word.is_empty() {
                    paths.push(PathBuf::from(std::mem::take(&mut word)));
                }
            }
            c => word.push(c),
        }
    }
    if !word.is_empty() {
        paths.push(PathBuf::from(word));
    }
    paths.retain(|path| path != input);
    paths.sort();
    paths.dedup();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dependency_paths_preserve_escaped_characters() {
        let paths = parse(
            "resin: input.c \\\n            /path\\ with\\ spaces/header.h /hash\\#tag.h /dollar$$.h /back\\\\slash.h\n",
            Path::new("input.c"),
        );
        assert_eq!(
            paths,
            [
                "/back\\slash.h",
                "/dollar$.h",
                "/hash#tag.h",
                "/path with spaces/header.h"
            ]
            .map(PathBuf::from)
        );
    }

    #[test]
    fn windows_dependencies_preserve_drive_letters_and_separators() {
        let paths = parse(
            "resin: input.c \\\r\n C:\\sdk\\header.h C:/Program\\ Files/sdk.h\r\n",
            Path::new("input.c"),
        );
        let mut expected = [r"C:\sdk\header.h", "C:/Program Files/sdk.h"].map(PathBuf::from);
        expected.sort();
        assert_eq!(paths, expected);
    }
}
