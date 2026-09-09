use super::{Result, source::Input};
use resin_source::normalize_path;
use std::{
    io,
    path::{Component, Path, PathBuf},
};

pub struct Options {
    pub directory: PathBuf,
    pub profile: resin_toolchain::CProfile,
    pub tools: resin_toolchain::Toolchain,
    pub temporary: PathBuf,
}

/// Paths resolve once against the captured directory before compilation starts.
pub struct Request {
    pub input: Input,
    pub destination: Option<PathBuf>,
    pub options: Options,
}

impl Request {
    pub fn new(mut input: Input, destination: Option<PathBuf>, options: Options) -> Result<Self> {
        let source = normalize_path(&options.directory.join(&input.path))?;
        let destination = destination
            .map(|path| validate_destination(&input, &source, &options.directory, &path))
            .transpose()?;
        input.path = source;
        Ok(Self {
            input,
            destination,
            options,
        })
    }
}

fn validate_destination(
    input: &Input,
    source: &Path,
    directory: &Path,
    path: &Path,
) -> Result<PathBuf> {
    validate_destination_ancestors(directory, path)?;
    let path = executable_destination(input, directory, path)?;
    validate_destination_ancestors(Path::new(""), &path)?;
    if source == normalize_path(&path)? {
        return Err("output would overwrite the source file".into());
    }
    Ok(std::path::absolute(path)?)
}

// Joining a complete path to a verbatim Windows directory can erase `file/..`.
// Check each raw prefix before joining; a nonexistent child can also hide a file.
fn validate_destination_ancestors(directory: &Path, path: &Path) -> io::Result<()> {
    let mut prefix = directory.to_path_buf();
    let components: Vec<_> = path.components().collect();
    for part in &components[..components.len().saturating_sub(1)] {
        prefix.push(part.as_os_str());
        if !matches!(part, Component::Prefix(_)) {
            validate_directory(&prefix)?;
        }
    }
    Ok(())
}

fn validate_directory(path: &Path) -> io::Result<()> {
    match std::fs::metadata(path) {
        Ok(metadata) if !metadata.is_dir() => Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            format!("output ancestor is not a directory: {}", path.display()),
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            if std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_symlink()) {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("output ancestor is a dangling symlink: {}", path.display()),
                ));
            }
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn executable_destination(input: &Input, directory: &Path, path: &Path) -> Result<PathBuf> {
    let trailing_separator = path
        .as_os_str()
        .as_encoded_bytes()
        .last()
        .is_some_and(|&b| b == b'/' || b == std::path::MAIN_SEPARATOR as u8);
    let path = directory.join(path);
    if !path.is_dir() && !trailing_separator {
        return Ok(path);
    }
    let mut name = input
        .path
        .file_stem()
        .ok_or("source file needs a name")?
        .to_os_string();
    if input.entry != "main" {
        name.push(format!("-{}", input.entry));
    }
    name.push(std::env::consts::EXE_SUFFIX);
    Ok(path.join(name))
}

#[cfg(test)]
mod tests;
