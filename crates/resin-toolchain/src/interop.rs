//! Validate external C declarations from captured headers without compiling C objects.
use crate::{
    Error, ForeignAnalysis, ForeignInputs, ForeignScalar, Settings, files, headers, platform,
    process,
};
use resin_executor::{Cancellation, Execution};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{fs, io::AsyncWriteExt};

mod clang;

pub(super) async fn analyze(
    inputs: Arc<ForeignInputs>,
    temporary: &Path,
    settings: &Settings,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<ForeignAnalysis, Error> {
    let source_inputs = inputs.clone();
    let source = execution
        .run(cancellation, move |_| translation_unit(&source_inputs))
        .await??;
    let permit = execution.acquire(cancellation).await?;
    fs::create_dir_all(temporary).await?;
    let temporary = fs::canonicalize(temporary).await?;
    let directory = files::temporary(&temporary).await?;
    stage(&inputs, directory.path(), cancellation).await?;
    fs::write(directory.path().join("foreign.c"), source).await?;
    let mut clang_settings = settings.clone();
    clang_settings.cc = settings.clang.clone();
    let roots = headers::allowed_roots(&clang_settings, directory.path(), cancellation).await?;
    let flags = flags(&inputs, directory.path());
    preprocess(&clang_settings, directory.path(), &flags, cancellation).await?;
    let depfile = directory.path().join("foreign.d");
    headers::validate(&depfile, directory.path(), &roots, cancellation).await?;
    let includes = headers::dependencies(&fs::read(depfile).await?)?
        .into_iter()
        .filter(|path| path != Path::new("foreign.c"))
        .map(|path| logical_path(&path, directory.path()))
        .collect::<Vec<_>>();
    let library = library(settings)?;
    let captured = fs::read(directory.path().join("foreign.i")).await?;
    let parse_root = directory.path().to_path_buf();
    let functions = inputs.functions.clone();
    drop(permit);
    let (declarations, diagnostics) = execution
        .run(cancellation, move |cancellation| {
            clang::inspect(&library, &captured, &functions, &parse_root, cancellation)
        })
        .await??;
    cancellation.check()?;
    Ok(ForeignAnalysis {
        declarations: declarations.into(),
        includes: includes.into(),
        diagnostics: diagnostics.into(),
    })
}

fn flags(inputs: &ForeignInputs, directory: &Path) -> Vec<OsString> {
    let mut flags = vec![
        "-std=c11".into(),
        "-Werror=implicit-function-declaration".into(),
    ];
    flags.extend(platform::C_FLAGS.iter().map(OsString::from));
    flags.extend(platform::PREPROCESSING_FLAGS.iter().map(OsString::from));
    flags.push(format!("-ffile-prefix-map={}=/resin/foreign", directory.display()).into());
    for root in &inputs.include_directories {
        flags.extend(["-I".into(), directory.join(root).into_os_string()]);
    }
    flags
}

async fn preprocess(
    settings: &Settings,
    directory: &Path,
    flags: &[OsString],
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let mut command = settings.command(&settings.clang)?;
    command.current_dir(directory).args(flags).args([
        "-E",
        "-MD",
        "-MT",
        "resin-input",
        "-MF",
        "foreign.d",
        "foreign.c",
    ]);
    let mut captured = fs::File::create(directory.join("foreign.i")).await?;
    process::preprocess(command, &mut captured, cancellation).await?;
    captured.flush().await?;
    drop(captured.into_std().await);
    Ok(())
}

pub(super) fn library_filename() -> OsString {
    #[cfg(windows)]
    return OsString::from("libclang.dll");
    #[cfg(not(windows))]
    return libloading::library_filename("clang");
}

pub(super) fn library(settings: &Settings) -> Result<PathBuf, Error> {
    settings.libclang.clone().ok_or_else(|| Error::new(
        "cannot locate libclang beside the selected Clang installation; set LIBCLANG_PATH to its library file or directory".into()
    ))
}

fn translation_unit(inputs: &ForeignInputs) -> Result<String, Error> {
    let mut text = String::new();
    for include in &inputs.includes {
        relative(include)?;
        text.push_str(&format!("#include \"{include}\"\n"));
    }
    for function in &inputs.functions {
        if !identifier(&function.name) {
            return Err(Error::new(
                "foreign function names must be C identifiers".into(),
            ));
        }
        for parameter in &function.params {
            if *parameter == ForeignScalar::Void {
                return Err(Error::new(
                    "a foreign parameter cannot have void type".into(),
                ));
            }
            validate_scalar(parameter)?;
        }
        validate_scalar(&function.result)?;
    }
    let mut aliases = BTreeMap::new();
    for path in inputs
        .files
        .keys()
        .map(AsRef::as_ref)
        .chain(inputs.include_directories.iter().map(String::as_str))
    {
        relative(path)?;
        for end in path
            .match_indices('/')
            .map(|(index, _)| index)
            .chain(std::iter::once(path.len()))
        {
            let prefix = &path[..end];
            if aliases
                .insert(prefix.to_ascii_lowercase(), prefix)
                .is_some_and(|old| old != prefix)
            {
                return Err(Error::new(
                    "foreign paths collide under ASCII case folding".into(),
                ));
            }
        }
    }
    Ok(text)
}

fn validate_scalar(ty: &ForeignScalar) -> Result<(), Error> {
    match ty {
        ForeignScalar::Void
        | ForeignScalar::Bool
        | ForeignScalar::Pointer
        | ForeignScalar::Integer {
            bits: 8 | 16 | 32 | 64,
            ..
        }
        | ForeignScalar::Float { bits: 32 | 64 } => Ok(()),
        _ => Err(Error::new(
            "unsupported scalar width at the C interoperability boundary".into(),
        )),
    }
}

fn identifier(text: &str) -> bool {
    let mut bytes = text.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

fn relative(text: &str) -> Result<(), Error> {
    if text.split('/').any(|part| {
        part.is_empty()
            || part == "."
            || part == ".."
            || part.ends_with([' ', '.'])
            || device_name(part)
    }) || text
        .chars()
        .any(|ch| ch.is_control() || "\\:\"<>|?*".contains(ch))
    {
        return Err(Error::new(
            "foreign header paths must be portable relative paths".into(),
        ));
    }
    if matches!(
        text.to_ascii_lowercase().as_str(),
        "foreign.c" | "foreign.i" | "foreign.d"
    ) {
        return Err(Error::new(
            "foreign header path collides with a header-analysis input or output".into(),
        ));
    }
    Ok(())
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

async fn stage(
    inputs: &ForeignInputs,
    directory: &Path,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let mut directories = BTreeSet::new();
    for (path, bytes) in &inputs.files {
        cancellation.check()?;
        let mut parent = directory.to_path_buf();
        let path = Path::new(path.as_ref());
        for component in path.parent().into_iter().flat_map(Path::components) {
            parent.push(component);
            if directories.insert(parent.clone()) {
                fs::create_dir(&parent).await?;
            }
        }
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(directory.join(path))
            .await?;
        file.write_all(bytes).await?;
        file.flush().await?;
    }
    Ok(())
}

fn logical_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}
