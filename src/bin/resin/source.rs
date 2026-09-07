use std::{ffi::OsStr, path::PathBuf};

use resin::compiler::Input;

pub fn parse(value: &OsStr) -> Result<Input, String> {
    let mut path = PathBuf::from(value);
    let filename = path
        .file_name()
        .ok_or("expected FILE[:ENTRY]")?
        .as_encoded_bytes();
    let entry = if let Some(colon) = filename.iter().rposition(|&byte| byte == b':') {
        let name = &filename[colon + 1..];
        if colon == 0 || !valid_name(name) {
            return Err("expected FILE[:ENTRY], with a nonempty function name after ':'".into());
        }
        let entry = String::from_utf8(name.to_vec()).unwrap();
        // These bytes came from this OsStr and are split immediately before ASCII ':'.
        let filename =
            unsafe { OsStr::from_encoded_bytes_unchecked(&filename[..colon]) }.to_os_string();
        path.set_file_name(filename);
        entry
    } else {
        "main".into()
    };
    Ok(Input { path, entry })
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
    #[cfg(unix)]
    fn non_utf8_paths_are_preserved() {
        use std::os::unix::ffi::OsStrExt;
        let source = parse(OsStr::from_bytes(b"dir/file\xff.resin:demo")).unwrap();
        assert_eq!(source.path.as_os_str().as_bytes(), b"dir/file\xff.resin");
        assert_eq!(source.entry, "demo");
    }

    #[test]
    #[cfg(windows)]
    fn windows_prefixes_are_not_entry_selectors() {
        for path in [
            r"C:\sources\file.resin",
            r"C:file.resin",
            r"\\server\share\file.resin",
            r"\\?\C:\file.resin",
        ] {
            assert_eq!(parse(OsStr::new(path)).unwrap().entry, "main");
            let source = parse(OsStr::new(&format!("{path}:demo"))).unwrap();
            assert_eq!(source.path, PathBuf::from(path));
            assert_eq!(source.entry, "demo");
        }
    }

    #[test]
    #[cfg(windows)]
    fn unpaired_surrogates_are_preserved() {
        use std::{
            ffi::OsString,
            os::windows::ffi::{OsStrExt, OsStringExt},
        };

        let mut path: Vec<_> = r"C:\file".encode_utf16().collect();
        path.push(0xd800);
        path.extend(".resin".encode_utf16());
        let mut selected = path.clone();
        selected.extend(":demo".encode_utf16());
        let source = parse(&OsString::from_wide(&selected)).unwrap();
        assert_eq!(
            source.path.as_os_str().encode_wide().collect::<Vec<_>>(),
            path
        );
        assert_eq!(source.entry, "demo");
    }
}
