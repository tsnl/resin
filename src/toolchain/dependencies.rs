use std::path::{Path, PathBuf};

pub(super) fn parse(source: &str, input: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut word = String::new();
    let mut chars = source
        .strip_prefix("resin:")
        .unwrap_or(source)
        .chars()
        .peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('\n') => {}
                Some(c) => word.push(c),
                None => word.push('\\'),
            },
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
}
