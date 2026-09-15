//! Build current disk inputs through HTTP, independently of dirty editor roots.
use super::worker::WorkerConfig;
use crate::inputs::{self, CaptureOptions};
use lsp_types::Uri;
use resin_executor::{Cancellation, Execution};
use resin_source::Loader;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BuildRequest {
    pub uri: Uri,
    #[serde(default = "default_entry")]
    pub entry: String,
    pub destination: PathBuf,
    #[serde(default)]
    pub profile: BuildProfile,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum BuildProfile {
    Debug,
    #[default]
    Release,
}

fn default_entry() -> String {
    "main".into()
}

pub(crate) async fn run(
    request: BuildRequest,
    config: &WorkerConfig,
    execution: &Execution,
    cancellation: &Cancellation,
) -> super::Result<Value> {
    let (path, destination) = execution
        .run(cancellation, {
            let request = request.clone();
            move |_| validate_paths(&request)
        })
        .await??;
    for attempt in 0..2 {
        let mut loader = Loader::new(PathBuf::new());
        let source = loader
            .load_file_async(&path, execution, cancellation)
            .await?;
        let options = CaptureOptions {
            include_roots: config.include_roots.clone(),
            capabilities: config.client.capabilities(),
        };
        let captured =
            inputs::capture(source, &mut loader, &options, execution, cancellation).await?;
        if captured
            .origins
            .values()
            .any(|source| source.path == destination)
        {
            return Err("build output would overwrite a source in the import graph".into());
        }
        let profile = match request.profile {
            BuildProfile::Debug => resin_protocol::BuildProfile::Debug,
            BuildProfile::Release => resin_protocol::BuildProfile::Release,
        };
        let contract = crate::interp::build_contract(&config.client, &request.entry, profile)?;
        match config
            .client
            .build(
                &captured.inputs,
                None,
                0,
                contract,
                &destination,
                cancellation,
            )
            .await
        {
            Err(error)
                if attempt == 0
                    && error.failure().is_some_and(|failure| {
                        failure.code == resin_protocol::ErrorCode::ManagedSnapshotUnavailable
                    }) =>
            {
                continue;
            }
            result => {
                result.map_err(|error| crate::interp::display_error(error, &captured))?;
            }
        }
        let output = url::Url::from_file_path(&destination)
            .map_err(|_| "cannot represent output as a file URI")?;
        return Ok(json!({"outputUri": output.as_str()}));
    }
    unreachable!("at most one managed snapshot retry")
}

fn validate_paths(request: &BuildRequest) -> super::Result<(PathBuf, PathBuf)> {
    let path = url::Url::parse(request.uri.as_str())?
        .to_file_path()
        .map_err(|_| "build URI must identify a local file")?;
    let path = resin_source::normalize_path(&path)?;
    if request.entry.is_empty() || request.destination.as_os_str().is_empty() {
        return Err("build entry and destination must be nonempty".into());
    }
    let destination = path
        .parent()
        .expect("absolute source parent")
        .join(&request.destination);
    if destination.is_dir() || destination.file_name().is_none() {
        return Err("build destination must be an explicit output filename".into());
    }
    let destination = resin_source::normalize_path(&destination)?;
    if destination == path {
        return Err("build output would overwrite the source file".into());
    }
    Ok((path, destination))
}
