//! Validate uploaded directory snapshots and bind source-scoped native includes.
use crate::{http::failure, managed::Managed};
use base64::Engine;
use resin_codegen::{NativeHeaders, NativeInclude};
use resin_protocol::{ErrorCode, Failure, HeaderInputs, HeaderTarget, IncludeRoot};
use resin_source::{Source, SourceError};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub(crate) fn prepare(
    inputs: &HeaderInputs,
    sources: &BTreeMap<String, Source>,
    syntax: &BTreeMap<Source, Arc<resin_cst::Document>>,
    managed: &Managed,
) -> Result<(NativeHeaders, Vec<SourceError>), Failure> {
    let mut headers = NativeHeaders::default();
    let mut bundles = BTreeSet::new();
    for bundle in &inputs.bundles {
        if !bundles.insert(bundle.id.clone()) {
            return Err(invalid("duplicate header bundle"));
        }
        let mut files = BTreeMap::new();
        for file in &bundle.files {
            relative(&file.path, false)?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&file.contents_base64)
                .map_err(|_| invalid("invalid base64 header contents"))?;
            if files.insert(file.path.clone(), bytes).is_some() {
                return Err(invalid("duplicate file in header bundle"));
            }
        }
        let mut digest = blake3::Hasher::new();
        digest.update(b"resin-header-bundle-v1\0");
        for (path, bytes) in &files {
            for bytes in [path.as_bytes(), bytes.as_slice()] {
                digest.update(&(bytes.len() as u64).to_le_bytes());
                digest.update(bytes);
            }
        }
        if digest.finalize().to_hex().as_str() != bundle.id {
            return Err(invalid("header bundle digest does not match its contents"));
        }
        for (path, bytes) in files {
            headers.files.insert(
                format!("native/user/{}/{path}", bundle.id).into(),
                bytes.into(),
            );
        }
    }
    for (root, files) in &managed.headers {
        for (path, bytes) in files {
            relative(path, false)?;
            headers
                .files
                .insert(managed_path(root, path).into(), bytes.clone());
        }
    }
    headers.runtime = Some(managed_include("runtime", "resin_runtime.h", managed)?);
    for root in &inputs.include_roots {
        match root {
            IncludeRoot::Uploaded { bundle, directory } => {
                relative(directory, true)?;
                if !bundles.contains(bundle) {
                    return Err(invalid("include root references an unknown bundle"));
                }
                headers
                    .include_directories
                    .push(join(&format!("native/user/{bundle}"), directory).into());
            }
            IncludeRoot::Managed { root, directory } => {
                relative(directory, true)?;
                if root == "system" && directory.is_empty() {
                    continue;
                }
                if !managed.headers.contains_key(root) {
                    return Err(invalid("unknown managed include root"));
                }
                headers
                    .include_directories
                    .push(managed_path(root, directory).trim_end_matches('/').into());
            }
        }
    }
    // Runtime and package transitive angle includes follow explicit uploaded roots.
    for root in &managed.header_order {
        headers
            .include_directories
            .push(managed_path(root, "").trim_end_matches('/').into());
    }
    let mut bindings = BTreeMap::new();
    for binding in &inputs.bindings {
        let Some(source) = sources
            .get(&binding.source)
            .filter(|source| !source.name().starts_with("$/"))
        else {
            return Err(invalid("header binding source is not an uploaded source"));
        };
        if !syntax[source]
            .preamble()
            .headers
            .iter()
            .any(|header| header.val.as_ref() == binding.spelling)
        {
            return Err(invalid(
                "header binding is absent from its declaring source",
            ));
        }
        let key = resin_lir::ForeignHeader {
            source: source.id(),
            spelling: binding.spelling.clone().into(),
        };
        let target = match &binding.target {
            HeaderTarget::Uploaded { bundle, path } => {
                relative(path, false)?;
                let path: Arc<str> = format!("native/user/{bundle}/{path}").into();
                if !bundles.contains(bundle) || !headers.files.contains_key(&path) {
                    return Err(invalid("header binding references a missing uploaded file"));
                }
                NativeInclude::Staged { path }
            }
            HeaderTarget::Managed { root, path } => {
                if absolute(&binding.spelling) {
                    return Err(invalid(
                        "absolute client header paths cannot select server files",
                    ));
                }
                managed_include(root, path, managed)?
            }
        };
        if bindings.insert(key, target).is_some() {
            return Err(invalid("duplicate source header binding"));
        }
    }
    let mut errors = Vec::new();
    for source in sources.values() {
        for header in syntax[source].preamble().headers {
            let key = resin_lir::ForeignHeader {
                source: source.id(),
                spelling: header.val.clone(),
            };
            if source.name().starts_with("$/") {
                let target = managed
                    .header_order
                    .iter()
                    .find(|root| managed.headers[*root].contains_key(header.val.as_ref()))
                    .map(|root| managed_include(root, &header.val, managed))
                    .unwrap_or_else(|| managed_include("system", &header.val, managed))?;
                bindings.insert(key, target);
            } else if !bindings.contains_key(&key) {
                errors.push(SourceError::new(
                    source.clone(),
                    Some(header.span),
                    format!("no native header binding supplied for {}", header.val),
                ));
            }
        }
    }
    headers.bindings = bindings;
    let mut aliases = BTreeMap::new();
    for path in headers
        .files
        .keys()
        .chain(headers.include_directories.iter())
    {
        for end in path
            .match_indices('/')
            .map(|(index, _)| index)
            .chain(std::iter::once(path.len()))
        {
            let prefix = &path[..end];
            if aliases
                .insert(prefix.to_ascii_lowercase(), prefix.to_owned())
                .is_some_and(|old| old != prefix)
            {
                return Err(invalid("header paths collide under ASCII case folding"));
            }
        }
    }
    for path in headers.files.keys() {
        let mut prefix = path.as_ref();
        while let Some((parent, _)) = prefix.rsplit_once('/') {
            if headers.files.contains_key(parent) {
                return Err(invalid("header file conflicts with a directory"));
            }
            prefix = parent;
        }
    }
    Ok((headers, errors))
}

fn managed_include(root: &str, path: &str, managed: &Managed) -> Result<NativeInclude, Failure> {
    relative(path, false)?;
    if root == "system" {
        if !resin_types::Foreign::valid_header(path) {
            return Err(invalid("invalid system header spelling"));
        }
        return Ok(NativeInclude::System {
            spelling: path.into(),
        });
    }
    if !managed
        .headers
        .get(root)
        .is_some_and(|files| files.contains_key(path))
    {
        return Err(invalid("managed header was not advertised by this server"));
    }
    Ok(NativeInclude::Staged {
        path: managed_path(root, path).into(),
    })
}
fn managed_path(root: &str, path: &str) -> String {
    join(
        &format!("native/managed/{}", blake3::hash(root.as_bytes()).to_hex()),
        path,
    )
}
fn join(root: &str, path: &str) -> String {
    if path.is_empty() {
        root.into()
    } else {
        format!("{root}/{path}")
    }
}
fn absolute(path: &str) -> bool {
    path.starts_with('/') || path.starts_with('\\') || path.as_bytes().get(1) == Some(&b':')
}
pub(crate) fn relative(path: &str, empty: bool) -> Result<(), Failure> {
    if empty && path.is_empty() {
        return Ok(());
    }
    if path.is_empty()
        || path
            .chars()
            .any(|c| c.is_control() || "\\:\"<>|?*".contains(c))
        || path.split('/').any(|part| {
            part.is_empty()
                || part == "."
                || part == ".."
                || part.ends_with([' ', '.'])
                || device_name(part)
        })
    {
        return Err(invalid(
            "header paths must be contained portable relative paths",
        ));
    }
    Ok(())
}
fn invalid(message: impl Into<String>) -> Failure {
    failure(ErrorCode::InvalidRequest, message)
}

fn device_name(part: &str) -> bool {
    let base = part
        .split('.')
        .next()
        .unwrap_or(part)
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    matches!(
        base.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || ["COM", "LPT"].iter().any(|prefix| {
        base.strip_prefix(prefix).is_some_and(|number| {
            matches!(
                number,
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    })
}
