use super::{Result, source::Input};
use resin_source::normalize_path;
use std::{
    io,
    path::{Path, PathBuf},
};

pub struct Options {
    pub profile: resin_toolchain::CProfile,
    pub tools: resin_toolchain::Toolchain,
    pub temporary: PathBuf,
}

/// Validated command-line inputs, build options, and executable destination.
pub struct Request {
    pub input: Input,
    pub destination: Option<PathBuf>,
    pub options: Options,
}

impl Request {
    pub fn new(mut input: Input, destination: Option<PathBuf>, options: Options) -> Result<Self> {
        let source = normalize_path(&input.path)?;
        let destination = destination
            .map(|path| validate_destination(&input, &source, path))
            .transpose()?;
        input.path = source;
        Ok(Self {
            input,
            destination,
            options,
        })
    }
}

fn validate_destination(input: &Input, source: &Path, path: PathBuf) -> Result<PathBuf> {
    let path = executable_destination(input, path)?;
    validate_destination_ancestors(&path)?;
    if source == normalize_path(&path)? {
        return Err("output would overwrite the source file".into());
    }
    Ok(std::path::absolute(path)?)
}

// Check each prefix before normalization: Windows may report NotFound for a
// child of a regular file. Checking only the final parent misses that case.
fn validate_destination_ancestors(path: &Path) -> io::Result<()> {
    // Preserve raw components until they have been checked. In particular,
    // Windows absolute-path resolution can collapse `file/..` prematurely.
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut prefix = PathBuf::new();
    let components: Vec<_> = absolute.components().collect();
    for part in &components[..components.len().saturating_sub(1)] {
        prefix.push(part.as_os_str());
        match std::fs::metadata(&prefix) {
            Ok(metadata) if !metadata.is_dir() => {
                return Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    format!("output ancestor is not a directory: {}", prefix.display()),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if std::fs::symlink_metadata(&prefix).is_ok_and(|metadata| metadata.is_symlink()) {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!(
                            "output ancestor is a dangling symlink: {}",
                            prefix.display()
                        ),
                    ));
                }
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn executable_destination(input: &Input, path: PathBuf) -> Result<PathBuf> {
    let trailing_separator = path
        .as_os_str()
        .as_encoded_bytes()
        .last()
        .is_some_and(|&b| b == b'/' || b == std::path::MAIN_SEPARATOR as u8);
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
