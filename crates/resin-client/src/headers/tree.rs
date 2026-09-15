//! Capture complete directory contents with portable paths and contained symlinks.
use base64::{Engine, engine::general_purpose::STANDARD};
use resin_executor::Cancellation;
use resin_protocol::{BundleFile, HeaderBundle};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    fs,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

pub(super) enum Error {
    Cancelled,
    Input { message: String },
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Input {
        message: message.into(),
    }
}

pub(super) fn bundle(
    root: &Path,
    allowed: &[PathBuf],
    contents: &mut BTreeMap<PathBuf, Arc<[u8]>>,
    cancellation: &Cancellation,
) -> Result<HeaderBundle, Error> {
    let mut files = BTreeMap::new();
    walk(
        root,
        "",
        allowed,
        &mut BTreeSet::new(),
        &mut files,
        contents,
        cancellation,
    )?;
    let mut digest = blake3::Hasher::new();
    digest.update(b"resin-header-bundle-v1\0");
    let files = files
        .into_iter()
        .map(|(path, bytes)| {
            digest.update(&(path.len() as u64).to_le_bytes());
            digest.update(path.as_bytes());
            digest.update(&(bytes.len() as u64).to_le_bytes());
            digest.update(&bytes);
            BundleFile {
                path,
                contents_base64: STANDARD.encode(&bytes),
            }
        })
        .collect();
    Ok(HeaderBundle {
        id: digest.finalize().to_hex().to_string(),
        files,
    })
}

fn walk(
    path: &Path,
    relative: &str,
    allowed: &[PathBuf],
    ancestors: &mut BTreeSet<PathBuf>,
    files: &mut BTreeMap<String, Arc<[u8]>>,
    contents: &mut BTreeMap<PathBuf, Arc<[u8]>>,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    cancellation.check().map_err(|_| Error::Cancelled)?;
    let canonical = fs::canonicalize(path).map_err(|error| unreadable(relative, error))?;
    if !allowed.iter().any(|root| canonical.starts_with(root)) {
        return Err(invalid(format!(
            "bundled path {relative:?} follows a symlink outside the selected roots; supply an explicit containing root"
        )));
    }
    let metadata = fs::metadata(&canonical).map_err(|error| unreadable(relative, error))?;
    if metadata.is_dir() {
        if !ancestors.insert(canonical.clone()) {
            return Err(invalid(format!(
                "bundled directory {relative:?} contains a symlink cycle"
            )));
        }
        let children = fs::read_dir(&canonical).map_err(|error| unreadable(relative, error))?;
        let mut children = children
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| unreadable(relative, error))?;
        children.sort_by_key(|entry| entry.file_name());
        let mut names = BTreeMap::new();
        for child in children {
            let component = component(&child.file_name())?;
            if let Some(previous) = names.insert(component.to_ascii_lowercase(), component.clone())
            {
                return Err(invalid(format!(
                    "header bundle names {previous:?} and {component:?} collide ignoring ASCII case"
                )));
            }
            let child_relative = if relative.is_empty() {
                component
            } else {
                format!("{relative}/{component}")
            };
            walk(
                &child.path(),
                &child_relative,
                allowed,
                ancestors,
                files,
                contents,
                cancellation,
            )?;
        }
        ancestors.remove(&canonical);
    } else if metadata.is_file() {
        let bytes = match contents.get(&canonical) {
            Some(bytes) => bytes.clone(),
            None => {
                let bytes: Arc<[u8]> = fs::read(&canonical)
                    .map_err(|error| unreadable(relative, error))?
                    .into();
                cancellation.check().map_err(|_| Error::Cancelled)?;
                contents.insert(canonical, bytes.clone());
                bytes
            }
        };
        files.insert(relative.to_owned(), bytes);
    } else {
        return Err(invalid(format!(
            "bundled path {relative:?} is not a regular file or directory"
        )));
    }
    Ok(())
}

fn unreadable(path: &str, error: std::io::Error) -> Error {
    invalid(format!(
        "could not capture bundled path {path:?}: {}",
        crate::inputs::io_reason(error.kind())
    ))
}

pub(super) fn relative(path: &Path, root: &Path) -> Result<String, Error> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| invalid("header is outside its selected bundle"))?;
    relative
        .components()
        .map(|part| match part {
            Component::Normal(name) => component(name),
            _ => Err(invalid("bundle paths must be normalized relative paths")),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join("/"))
}

pub(super) fn portable(path: &str) -> bool {
    !path.is_empty()
        && path
            .split('/')
            .all(|part| component(OsStr::new(part)).is_ok())
}

fn component(name: &OsStr) -> Result<String, Error> {
    let Some(name) = name.to_str() else {
        return Err(invalid("header bundle filenames must be valid UTF-8"));
    };
    let forbidden = name.is_empty()
        || matches!(name, "." | "..")
        || name.ends_with(['.', ' '])
        || name
            .chars()
            .any(|character| character.is_control() || "\\/<>:\"|?*".contains(character));
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ["COM", "LPT"].iter().any(|prefix| {
            stem.strip_prefix(prefix).is_some_and(|suffix| {
                suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'9')
            })
        });
    if forbidden || reserved {
        return Err(invalid(format!(
            "header bundle filename {name:?} is not portable"
        )));
    }
    Ok(name.to_owned())
}
