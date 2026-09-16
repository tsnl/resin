//! Capture and compile C call adapters; Resin computation never enters this translation.
use crate::{
    Error, ForeignFunction, ForeignInputs, ForeignObject, ForeignScalar, Settings, files, headers,
    platform, process,
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

pub(super) async fn compile(
    inputs: Arc<ForeignInputs>,
    temporary: &Path,
    settings: &Settings,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<ForeignObject, Error> {
    let source_inputs = inputs.clone();
    let source = execution
        .run(cancellation, move |_| wrapper(&source_inputs))
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
    let library = library(settings, cancellation).await?;
    let captured = fs::read(directory.path().join("foreign.i")).await?;
    let parse_root = directory.path().to_path_buf();
    let functions = inputs.functions.clone();
    drop(permit);
    let (declarations, diagnostics) = execution
        .run(cancellation, move |cancellation| {
            clang::inspect(&library, &captured, &functions, &parse_root, cancellation)
        })
        .await??;
    let _permit = execution.acquire(cancellation).await?;
    let mut command = settings.command(&settings.clang)?;
    command
        .current_dir(directory.path())
        .args(platform::C_FLAGS)
        .args([
            "-std=c11",
            "-O2",
            "-Wno-gnu-line-marker",
            "-c",
            "foreign.i",
            "-o",
            "foreign.o",
        ]);
    #[cfg(not(windows))]
    command.arg("-fPIC");
    process::run(command, "C interoperability compilation", cancellation).await?;
    let bytes = fs::read(directory.path().join("foreign.o")).await?;
    cancellation.check()?;
    Ok(ForeignObject {
        bytes: bytes.into(),
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

async fn library(settings: &Settings, cancellation: &Cancellation) -> Result<PathBuf, Error> {
    #[cfg(windows)]
    let filename = OsString::from("libclang.dll");
    #[cfg(not(windows))]
    let filename = libloading::library_filename("clang");
    if let Some(path) = &settings.libclang {
        return Ok(if fs::metadata(path).await?.is_dir() {
            path.join(filename)
        } else {
            path.clone()
        });
    }
    let mut command = settings.command(&settings.clang)?;
    command.arg("-print-resource-dir");
    let (output, _) = process::capture(command, "Clang resource discovery", cancellation).await?;
    let resource = PathBuf::from(
        String::from_utf8(output)
            .map_err(|error| Error::new(error.to_string()))?
            .trim(),
    );
    let adjacent = settings
        .clang
        .parent()
        .unwrap_or(Path::new("."))
        .join(&filename);
    if fs::try_exists(&adjacent).await? {
        return Ok(adjacent);
    }
    let path = resource
        .ancestors()
        .nth(2)
        .unwrap_or(Path::new("."))
        .join(filename);
    if fs::try_exists(&path).await? {
        return Ok(path);
    }
    Err(Error::new("cannot locate libclang beside the selected Clang installation; set LIBCLANG_PATH to its library file or directory".into()))
}

fn wrapper(inputs: &ForeignInputs) -> Result<String, Error> {
    let mut names = BTreeSet::new();
    let mut text = String::new();
    for include in &inputs.includes {
        relative(include)?;
        text.push_str(&format!("#include \"{include}\"\n"));
    }
    for function in &inputs.functions {
        if !identifier(&function.symbol)
            || !identifier(&function.name)
            || !names.insert(&function.symbol)
        {
            return Err(Error::new(
                "foreign function names must be C identifiers and wrapper symbols must be unique"
                    .into(),
            ));
        }
        text.push_str(&function_wrapper(function)?);
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

fn function_wrapper(function: &ForeignFunction) -> Result<String, Error> {
    let params = function
        .params
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            if *ty == ForeignScalar::Void {
                return Err(Error::new(
                    "a foreign parameter cannot have void type".into(),
                ));
            }
            Ok(format!("{} a{index}", scalar(ty)?))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let params = if params.is_empty() {
        "void".into()
    } else {
        params.join(", ")
    };
    let args = (0..function.params.len())
        .map(|index| format!("a{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let result = scalar(&function.result)?;
    let body = if function.result == ForeignScalar::Void {
        format!("(void){}({args});", function.name)
    } else {
        format!("return ({result}){}({args});", function.name)
    };
    Ok(format!(
        "{result} {}({params}) {{ {body} }}\n",
        function.symbol
    ))
}

fn scalar(ty: &ForeignScalar) -> Result<&'static str, Error> {
    Ok(match ty {
        ForeignScalar::Void => "void",
        ForeignScalar::Bool => "_Bool",
        ForeignScalar::Pointer => "void *",
        ForeignScalar::Integer {
            bits: 8,
            signed: true,
        } => "signed char",
        ForeignScalar::Integer {
            bits: 8,
            signed: false,
        } => "unsigned char",
        ForeignScalar::Integer {
            bits: 16,
            signed: true,
        } => "short",
        ForeignScalar::Integer {
            bits: 16,
            signed: false,
        } => "unsigned short",
        ForeignScalar::Integer {
            bits: 32,
            signed: true,
        } => "int",
        ForeignScalar::Integer {
            bits: 32,
            signed: false,
        } => "unsigned int",
        ForeignScalar::Integer {
            bits: 64,
            signed: true,
        } => "long long",
        ForeignScalar::Integer {
            bits: 64,
            signed: false,
        } => "unsigned long long",
        ForeignScalar::Float { bits: 32 } => "float",
        ForeignScalar::Float { bits: 64 } => "double",
        _ => {
            return Err(Error::new(
                "unsupported scalar width at the C interoperability boundary".into(),
            ));
        }
    })
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
        "foreign.c" | "foreign.i" | "foreign.d" | "foreign.o"
    ) {
        return Err(Error::new(
            "foreign header path collides with an adapter output".into(),
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
