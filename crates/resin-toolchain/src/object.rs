//! Link completed object bytes without running a C frontend or incremental build graph.
use crate::{Error, Executable, files, platform, process, settings::Settings};
use resin_executor::Cancellation;
use std::{path::Path, sync::Arc};
use tokio::fs;

pub(super) async fn link(
    object: Arc<[u8]>,
    temporary: &Path,
    settings: &Settings,
    cancellation: &Cancellation,
) -> Result<Executable, Error> {
    link_native(
        crate::NativeLink {
            objects: vec![object],
            runtime: false,
        },
        temporary,
        settings,
        cancellation,
    )
    .await
}

pub(super) async fn link_native(
    inputs: crate::NativeLink,
    temporary: &Path,
    settings: &Settings,
    cancellation: &Cancellation,
) -> Result<Executable, Error> {
    cancellation.check()?;
    if inputs.objects.is_empty() {
        return Err(Error::new(
            "native linking requires at least one object".into(),
        ));
    }
    fs::create_dir_all(temporary).await?;
    let temporary = fs::canonicalize(temporary).await?;
    let directory = files::temporary(&temporary).await?;
    let executable = directory
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    let mut command = settings.command(&settings.cc)?;
    command.args(platform::C_FLAGS);
    for (index, object) in inputs.objects.iter().enumerate() {
        cancellation.check()?;
        let extension = if cfg!(windows) { "obj" } else { "o" };
        let input = directory.path().join(format!("input{index}.{extension}"));
        fs::write(&input, object).await?;
        command.arg(input);
    }
    command.arg("-o").arg(&executable);
    if inputs.runtime {
        command
            .arg(&settings.runtime_library)
            .args(platform::LIBRARIES);
    } else {
        #[cfg(not(windows))]
        command.arg("-lm");
    }
    command.args(platform::LINK_FLAGS);
    process::run(command, "native object linking", cancellation).await?;
    cancellation.check()?;
    if !fs::metadata(&executable).await?.is_file() {
        return Err(Error::new(
            "native linker did not produce an executable file".into(),
        ));
    }
    Ok(Executable {
        executable,
        _directory: Arc::new(directory),
    })
}
