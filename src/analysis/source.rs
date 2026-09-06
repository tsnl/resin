use crate::ast::SourceProvider;
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Component, Path, PathBuf},
};

/// Open buffers override disk sources, including files that have never been saved.
#[derive(Clone, Default)]
pub struct Sources {
    pub overlays: BTreeMap<PathBuf, String>,
}

impl SourceProvider for Sources {
    fn resolve(&self, path: &Path) -> io::Result<PathBuf> {
        normalize_path(path)
    }
    fn read(&self, path: &Path) -> io::Result<String> {
        self.overlays
            .get(path)
            .cloned()
            .map_or_else(|| fs::read_to_string(path), Ok)
    }
}

/// Canonicalize existing ancestors too, so a new file under a symlinked directory
/// has the same identity when imported and when opened by the editor.
pub fn normalize_path(path: &Path) -> io::Result<PathBuf> {
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

pub(crate) struct Snapshot<'a>(pub &'a BTreeMap<PathBuf, std::sync::Arc<super::syntax::Document>>);
impl SourceProvider for Snapshot<'_> {
    fn parsed(&self, path: &Path) -> Option<Result<crate::ast::SourceFile, crate::ast::AstError>> {
        self.0.get(path).map(|d| d.file.clone())
    }
    fn resolve(&self, path: &Path) -> io::Result<PathBuf> {
        normalize_path(path)
    }
    fn read(&self, path: &Path) -> io::Result<String> {
        self.0.get(path).map(|d| d.text.clone()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("cannot read {}", path.display()),
            )
        })
    }
}

/// The editor's tolerant ASTs, with the exact original text and source paths.
pub(crate) struct RecoverySnapshot<'a>(
    pub &'a BTreeMap<PathBuf, std::sync::Arc<super::syntax::Document>>,
);
impl SourceProvider for RecoverySnapshot<'_> {
    fn parsed(&self, path: &Path) -> Option<Result<crate::ast::SourceFile, crate::ast::AstError>> {
        self.0.get(path).map(|d| Ok(d.recovered_file.clone()))
    }
    fn resolve(&self, path: &Path) -> io::Result<PathBuf> {
        normalize_path(path)
    }
    fn read(&self, path: &Path) -> io::Result<String> {
        Snapshot(self.0).read(path)
    }
}
