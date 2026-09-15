//! Map acquisition paths to stable module names without retaining local paths in keys.
use crate::{LoadError, Source, SourceId, Text, paths};
use resin_executor::Cancellation;
use std::{
    collections::BTreeMap,
    fmt::Write,
    io,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

pub(super) fn sources(
    sources: Vec<(Source, Option<PathBuf>)>,
    root: &Path,
    library: &Path,
    cancellation: &Cancellation,
) -> Result<BTreeMap<SourceId, Source>, LoadError> {
    let roots = if sources.iter().any(|(_, path)| path.is_some()) {
        Some((paths::normalize(root)?, paths::normalize(library)?))
    } else {
        None
    };
    let mut mapped = BTreeMap::new();
    for (source, path) in sources {
        cancellation.check()?;
        let id = source.id();
        let source = match (path, &roots) {
            (Some(path), Some((root, library))) => renamed(&source, name(&path, root, library)?),
            _ => source,
        };
        if let Some(previous) = mapped.insert(id, source.clone())
            && previous != source
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "logical source mapping received different versions of one source",
            )
            .into());
        }
    }
    Ok(mapped)
}

fn renamed(source: &Source, name: String) -> Source {
    let name: Arc<str> = name.into();
    Source(Arc::new(Text {
        id: SourceId::new(name.clone()),
        name,
        digest: *source.content_hash(),
        text: source.0.text.clone(),
    }))
}

fn name(path: &Path, root: &Path, library: &Path) -> io::Result<String> {
    if let Ok(relative) = path.strip_prefix(library) {
        return Ok(format!("$/{}", encode(relative)));
    }
    Ok(encode(&relative(path, root)?))
}

fn relative(path: &Path, root: &Path) -> io::Result<PathBuf> {
    let path: Vec<_> = path.components().collect();
    let root: Vec<_> = root.components().collect();
    let shared = path
        .iter()
        .zip(&root)
        .take_while(|(left, right)| left == right)
        .count();
    if shared == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "source and logical root must use the same filesystem prefix",
        ));
    }
    let mut relative = PathBuf::new();
    for _ in &root[shared..] {
        relative.push("..");
    }
    for component in &path[shared..] {
        relative.push(component.as_os_str());
    }
    Ok(relative)
}

fn encode(path: &Path) -> String {
    path.components()
        .map(|component| match component {
            Component::ParentDir => "..".into(),
            Component::Normal(name) => encode_component(name.as_encoded_bytes()),
            _ => unreachable!("logical names are relative normalized paths"),
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_component(mut bytes: &[u8]) -> String {
    let mut text = String::new();
    while !bytes.is_empty() {
        let (valid, invalid) = match std::str::from_utf8(bytes) {
            Ok(_) => (bytes.len(), 0),
            Err(error) => (
                error.valid_up_to(),
                error
                    .error_len()
                    .unwrap_or(bytes.len() - error.valid_up_to()),
            ),
        };
        for character in std::str::from_utf8(&bytes[..valid])
            .expect("valid prefix")
            .chars()
        {
            if matches!(character, '%' | '$' | '\\') || character.is_control() {
                for byte in character.encode_utf8(&mut [0; 4]).bytes() {
                    escape(byte, &mut text);
                }
            } else {
                text.push(character);
            }
        }
        for &byte in &bytes[valid..valid + invalid] {
            escape(byte, &mut text);
        }
        bytes = &bytes[valid + invalid..];
    }
    text
}

fn escape(byte: u8, text: &mut String) {
    write!(text, "%{byte:02X}").expect("formatting into String");
}
