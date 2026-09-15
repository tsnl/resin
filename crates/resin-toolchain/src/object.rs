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
    cancellation.check()?;
    fs::create_dir_all(temporary).await?;
    let temporary = fs::canonicalize(temporary).await?;
    let directory = files::temporary(&temporary).await?;
    let input = directory.path().join(if cfg!(windows) {
        "input.obj"
    } else {
        "input.o"
    });
    let executable = directory
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    fs::write(&input, object).await?;
    cancellation.check()?;
    let mut command = settings.command(&settings.cc)?;
    command
        .args(platform::C_FLAGS)
        .arg(&input)
        .arg("-o")
        .arg(&executable)
        .args(platform::LINK_FLAGS);
    #[cfg(not(windows))]
    command.arg("-lm");
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
