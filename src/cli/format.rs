//! File-oriented front end for the same formatter used by the language server.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub(super) fn run(paths: &[PathBuf], check: bool) -> super::Result<i32> {
    let mut files = BTreeMap::new();
    let mut failed = false;
    for path in paths {
        if let Err(error) = collect(path, &mut files, true) {
            eprintln!("{}: {error}", path.display());
            failed = true;
        }
    }
    for path in files.values() {
        match format_file(path, check) {
            Ok(true) => {
                println!("{}", path.display());
                failed |= check;
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("{}: {error}", path.display());
                failed = true;
            }
        }
    }
    Ok(i32::from(failed))
}

fn collect(
    path: &Path,
    files: &mut BTreeMap<PathBuf, PathBuf>,
    explicit: bool,
) -> super::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_symlink() {
        if explicit {
            return Err("symlinks are not followed; pass the target path instead".into());
        }
    } else if metadata.is_dir() {
        let mut entries = fs::read_dir(path)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort();
        for entry in entries {
            collect(&entry, files, false)?;
        }
    } else if metadata.is_file() && (explicit || path.extension().is_some_and(|ext| ext == "resin"))
    {
        files
            .entry(fs::canonicalize(path)?)
            .or_insert_with(|| path.to_owned());
    } else if explicit {
        return Err("expected a regular file or directory".into());
    }
    Ok(())
}

fn format_file(path: &Path, check: bool) -> super::Result<bool> {
    let source = fs::read_to_string(path)?;
    let formatted =
        crate::cst::format_source(&source).ok_or("syntax errors; file left unchanged")?;
    if source == formatted {
        return Ok(false);
    }
    if !check {
        // Replace only after writing succeeds; retain source permissions.
        let metadata = fs::metadata(path)?;
        if metadata.permissions().readonly() {
            return Err("file is read-only".into());
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temp = resin_common::TempDir::new(parent)?;
        let output = temp.path().join("formatted");
        fs::write(&output, formatted)?;
        fs::set_permissions(&output, metadata.permissions())?;
        fs::rename(&output, path)?;
    }
    Ok(true)
}
