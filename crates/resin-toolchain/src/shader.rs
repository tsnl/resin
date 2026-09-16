//! Optimize captured SPIR-V bytes as one independently owned native operation.
use crate::{Error, files, process, settings::Settings};
use resin_executor::Cancellation;
use std::{path::Path, sync::Arc};
use tokio::fs;

pub(super) async fn optimize(
    bytes: Arc<[u8]>,
    temporary: &Path,
    settings: &Settings,
    cancellation: &Cancellation,
) -> Result<Arc<[u8]>, Error> {
    cancellation.check()?;
    fs::create_dir_all(temporary).await?;
    let parent = fs::canonicalize(temporary).await?;
    let directory = files::temporary(&parent).await?;
    let input = directory.path().join("input.spv");
    let output = directory.path().join("optimized.spv");
    fs::write(&input, bytes).await?;
    let mut command = settings.command(&settings.spirv_opt)?;
    command
        .args(["--target-env=vulkan1.3", "-O"])
        .arg(&input)
        .arg("-o")
        .arg(&output);
    process::run(command, "SPIR-V optimization", cancellation).await?;
    cancellation.check()?;
    Ok(fs::read(output).await?.into())
}
