use std::{ffi::OsStr, os::unix::ffi::OsStrExt, path::PathBuf};

#[derive(Clone, Debug)]
pub struct Source {
    pub path: PathBuf,
    pub entry: String,
}

pub fn parse(value: &OsStr) -> Result<Source, String> {
    let mut path = PathBuf::from(value);
    let filename = path.file_name().ok_or("expected FILE[:ENTRY]")?.as_bytes();
    let entry = if let Some(colon) = filename.iter().rposition(|&byte| byte == b':') {
        let name = &filename[colon + 1..];
        if colon == 0 || !valid_name(name) {
            return Err("expected FILE[:ENTRY], with a nonempty function name after ':'".into());
        }
        let entry = String::from_utf8(name.to_vec()).unwrap();
        let filename = OsStr::from_bytes(&filename[..colon]).to_os_string();
        path.set_file_name(filename);
        entry
    } else {
        "main".into()
    };
    Ok(Source { path, entry })
}

fn valid_name(name: &[u8]) -> bool {
    name.first()
        .is_some_and(|byte| byte.is_ascii_lowercase() || *byte == b'_')
        && name
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selectors_only_apply_to_the_filename() {
        for (input, path, entry) in [
            ("file.resin", "file.resin", "main"),
            (
                "dir:with:colons/file.resin",
                "dir:with:colons/file.resin",
                "main",
            ),
            (
                "dir:with:colons/file.resin:demo",
                "dir:with:colons/file.resin",
                "demo",
            ),
            (
                "file:with:colons.resin:main",
                "file:with:colons.resin",
                "main",
            ),
            ("file.resin:_test2", "file.resin", "_test2"),
        ] {
            let source = parse(OsStr::new(input)).unwrap();
            assert_eq!(source.path, PathBuf::from(path));
            assert_eq!(source.entry, entry);
        }
        for input in [
            "",
            ":main",
            "file.resin:",
            "file.resin:1",
            "file.resin:demo-name",
        ] {
            assert!(parse(OsStr::new(input)).is_err(), "{input}");
        }
    }

    #[test]
    fn non_utf8_paths_are_preserved() {
        let source = parse(OsStr::from_bytes(b"dir/file\xff.resin:demo")).unwrap();
        assert_eq!(source.path.as_os_str().as_bytes(), b"dir/file\xff.resin");
        assert_eq!(source.entry, "demo");
    }
}
