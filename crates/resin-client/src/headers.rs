//! Resolve declared headers locally and upload complete immutable directory bundles.
use crate::inputs::{CaptureOptions, io_reason};
use resin_executor::{Cancellation, Execution};
use resin_protocol::{
    Diagnostic, HeaderBinding, HeaderInputs, HeaderTarget, IncludeRoot, Severity,
};
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

#[cfg(test)]
mod tests;
mod tree;

pub(super) struct Declaration {
    pub source: String,
    pub path: PathBuf,
    pub spelling: String,
    pub span: resin_protocol::Span,
}

enum Resolved {
    Local { path: PathBuf, include_parent: bool },
    Managed { root: String, path: String },
}

pub(super) async fn capture(
    declarations: Vec<Declaration>,
    options: &CaptureOptions,
    execution: &Execution,
    cancellation: &Cancellation,
) -> crate::Result<(HeaderInputs, Vec<Diagnostic>)> {
    let options = options.clone();
    execution
        .run(cancellation, move |cancellation| {
            collect(declarations, &options, cancellation)
        })
        .await?
}

fn collect(
    mut declarations: Vec<Declaration>,
    options: &CaptureOptions,
    cancellation: &Cancellation,
) -> crate::Result<(HeaderInputs, Vec<Diagnostic>)> {
    declarations.sort_by(|left, right| {
        (&left.source, &left.spelling).cmp(&(&right.source, &right.spelling))
    });
    declarations
        .dedup_by(|left, right| left.source == right.source && left.spelling == right.spelling);
    let explicit = explicit_roots(&options.include_roots)?;
    let mut directories = explicit.clone();
    let mut selected_directories = explicit.clone();
    let mut resolved = Vec::new();
    let mut diagnostics = Vec::new();
    for declaration in declarations {
        cancellation.check()?;
        match resolve(&declaration, &explicit, options) {
            Ok(Resolved::Local {
                path,
                include_parent,
            }) => {
                let parent = path
                    .parent()
                    .expect("absolute local header parent")
                    .to_owned();
                if !selected_directories
                    .iter()
                    .any(|root| parent.starts_with(root))
                {
                    selected_directories.push(parent.clone());
                }
                if include_parent && !directories.contains(&parent) {
                    directories.push(parent);
                }
                resolved.push((
                    declaration,
                    Resolved::Local {
                        path,
                        include_parent,
                    },
                ));
            }
            Ok(target) => resolved.push((declaration, target)),
            Err(message) => diagnostics.push(diagnostic(Some(&declaration), message)),
        }
    }
    let roots = coalesced(&selected_directories);
    let mut contents = BTreeMap::new();
    let mut bundles = BTreeMap::new();
    let mut failed = BTreeMap::new();
    for root in &roots {
        match tree::bundle(root, &roots, &mut contents, cancellation) {
            Ok(bundle) => {
                bundles.insert(root.clone(), bundle);
            }
            Err(tree::Error::Cancelled) => return Err(resin_executor::Error::Cancelled.into()),
            Err(tree::Error::Input { message }) => {
                failed.insert(root.clone(), message);
            }
        }
    }
    let mut bindings = Vec::new();
    for (declaration, target) in resolved {
        let target = match target {
            Resolved::Managed { root, path } => HeaderTarget::Managed { root, path },
            Resolved::Local { path, .. } => {
                let root = containing(&roots, &path);
                if let Some(message) = failed.get(root) {
                    diagnostics.push(diagnostic(Some(&declaration), message.clone()));
                    continue;
                }
                let relative = match tree::relative(&path, root) {
                    Ok(path) => path,
                    Err(tree::Error::Input { message }) => {
                        diagnostics.push(diagnostic(Some(&declaration), message));
                        continue;
                    }
                    Err(tree::Error::Cancelled) => {
                        return Err(resin_executor::Error::Cancelled.into());
                    }
                };
                HeaderTarget::Uploaded {
                    bundle: bundles[root].id.clone(),
                    path: relative,
                }
            }
        };
        bindings.push(HeaderBinding {
            source: declaration.source,
            spelling: declaration.spelling,
            target,
        });
    }
    // An explicit root can contain relevant transitive inputs without a direct binding.
    for (root, message) in failed {
        if explicit.iter().any(|path| path.starts_with(&root)) {
            diagnostics.push(diagnostic(None, message));
        }
    }
    let mut include_roots = Vec::new();
    for directory in directories {
        let root = containing(&roots, &directory);
        let Some(bundle) = bundles.get(root) else {
            continue;
        };
        let directory = match tree::relative(&directory, root) {
            Ok(directory) => directory,
            Err(tree::Error::Input { message }) => {
                diagnostics.push(diagnostic(None, message));
                continue;
            }
            Err(tree::Error::Cancelled) => return Err(resin_executor::Error::Cancelled.into()),
        };
        let root = IncludeRoot::Uploaded {
            bundle: bundle.id.clone(),
            directory,
        };
        if !include_roots.contains(&root) {
            include_roots.push(root);
        }
    }
    for root in &options.capabilities.header_roots {
        include_roots.push(IncludeRoot::Managed {
            root: root.id.clone(),
            directory: String::new(),
        });
    }
    let bundles: BTreeMap<_, _> = bundles
        .into_values()
        .map(|bundle| (bundle.id.clone(), bundle))
        .collect();
    Ok((
        HeaderInputs {
            bundles: bundles.into_values().collect(),
            bindings,
            include_roots,
        },
        diagnostics,
    ))
}

fn explicit_roots(paths: &[PathBuf]) -> crate::Result<Vec<PathBuf>> {
    let mut roots = Vec::new();
    for path in paths {
        let path = fs::canonicalize(path)?;
        if !path.is_dir() {
            return Err("explicit header include roots must be directories".into());
        }
        if !roots.contains(&path) {
            roots.push(path);
        }
    }
    Ok(roots)
}

fn coalesced(directories: &[PathBuf]) -> Vec<PathBuf> {
    let mut directories = directories.to_vec();
    directories.sort_by(|left, right| {
        left.components()
            .count()
            .cmp(&right.components().count())
            .then_with(|| left.cmp(right))
    });
    let mut roots: Vec<PathBuf> = Vec::new();
    for directory in directories {
        if !roots.iter().any(|root| directory.starts_with(root)) {
            roots.push(directory);
        }
    }
    roots
}

fn containing<'a>(roots: &'a [PathBuf], path: &Path) -> &'a PathBuf {
    roots
        .iter()
        .find(|root| path.starts_with(root))
        .expect("selected header directory belongs to a retained root")
}

fn resolve(
    declaration: &Declaration,
    explicit: &[PathBuf],
    options: &CaptureOptions,
) -> Result<Resolved, String> {
    let spelling = &declaration.spelling;
    let path = Path::new(spelling);
    if absolute_spelling(spelling) {
        if !path.is_absolute() {
            return Err("absolute header spelling is not a local platform path".into());
        }
        return local(path)?
            .map(|path| Resolved::Local {
                path,
                include_parent: true,
            })
            .ok_or_else(|| "absolute header was not found locally".into());
    }
    for root in explicit {
        if let Some(path) = local(&root.join(path))? {
            if !explicit.iter().any(|root| path.starts_with(root)) {
                return Err(
                    "header resolved outside the explicit include roots; supply a containing root"
                        .into(),
                );
            }
            return Ok(Resolved::Local {
                path,
                include_parent: false,
            });
        }
    }
    let parent = declaration.path.parent().expect("registered source parent");
    if let Some(resolved) = local(&parent.join(path))? {
        if !resolved.starts_with(parent)
            && !explicit.iter().any(|root| resolved.starts_with(root))
            && traverses_directory_symlink(parent, path)?
        {
            return Err("header directory symlink resolves outside the selected roots; supply a containing root".into());
        }
        return Ok(Resolved::Local {
            path: resolved,
            include_parent: true,
        });
    }
    if !tree::portable(spelling) {
        return Err(
            "header must resolve locally or name a portable relative managed header".into(),
        );
    }
    for root in &options.capabilities.header_roots {
        if root.id != "system" && root.headers.contains(spelling) {
            return Ok(Resolved::Managed {
                root: root.id.clone(),
                path: spelling.clone(),
            });
        }
    }
    if options
        .capabilities
        .header_roots
        .iter()
        .any(|root| root.id == "system")
    {
        return Ok(Resolved::Managed {
            root: "system".into(),
            path: spelling.clone(),
        });
    }
    Err(format!(
        "header {spelling:?} was not found locally or in advertised managed roots"
    ))
}

fn traverses_directory_symlink(root: &Path, relative: &Path) -> Result<bool, String> {
    let mut current = root.to_owned();
    for component in relative.parent().unwrap_or(Path::new("")).components() {
        current.push(component);
        let metadata =
            fs::symlink_metadata(&current).map_err(|error| io_reason(error.kind()).to_owned())?;
        if metadata.file_type().is_symlink() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn local(path: &Path) -> Result<Option<PathBuf>, String> {
    let Some(name) = path.file_name() else {
        return Err("header spelling must name a file".into());
    };
    let parent = resin_source::normalize_path(path.parent().unwrap_or(Path::new(".")))
        .map_err(|error| io_reason(error.kind()).to_owned())?;
    let path = parent.join(name);
    match fs::metadata(&path) {
        Ok(metadata) if metadata.is_file() => Ok(Some(path)),
        Ok(_) => Err("header spelling identifies a directory or special file".into()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io_reason(error.kind()).to_owned()),
    }
}

fn absolute_spelling(spelling: &str) -> bool {
    Path::new(spelling).is_absolute()
        || spelling.starts_with(['/', '\\'])
        || spelling.as_bytes().get(1) == Some(&b':')
}

fn diagnostic(declaration: Option<&Declaration>, message: String) -> Diagnostic {
    Diagnostic {
        code: None,
        notes: Vec::new(),
        help: None,
        severity: Severity::Error,
        message,
        span: declaration.map(|declaration| declaration.span.clone()),
        related: Vec::new(),
    }
}
