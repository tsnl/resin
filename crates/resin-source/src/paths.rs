use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

/// Canonicalize existing ancestors too, so a new file under a symlinked directory
/// has the same identity when imported and when opened by the editor.
pub(super) fn normalize(path: &Path) -> io::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut result = PathBuf::new();
    for part in absolute.components() {
        match part {
            Component::Prefix(_) | Component::RootDir => result.push(part.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            part => {
                result.push(part.as_os_str());
                match fs::canonicalize(&result) {
                    Ok(canonical) => result = canonical,
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
            }
        }
    }
    Ok(result)
}
