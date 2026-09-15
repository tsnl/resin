//! Download a server-built host executable, then run it with local process inputs.
use crate::{
    Client,
    inputs::{self, CaptureOptions},
};
use resin_executor::{Cancellation, Execution};
use resin_protocol::{BuildContract, BuildOptions, EntryProfile, EntryTarget};
use std::{ffi::OsString, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub(crate) async fn run(request: &crate::cli::request::Request, args: &[OsString]) -> Result<i32> {
    let client = connect().await?;
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let result = run_request(&client, request, args, &execution, &cancellation).await;
    cancellation.cancel();
    execution.wait_idle().await;
    result
}

pub(crate) async fn connect() -> Result<Client> {
    let url = std::env::var("RESIN_SERVER")
        .map_err(|_| "RESIN_SERVER must contain an absolute http:// or https:// server URL")?;
    Ok(Client::connect(&url).await?)
}

async fn run_request(
    client: &Client,
    request: &crate::cli::request::Request,
    args: &[OsString],
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<i32> {
    let temporary = tempfile::tempdir_in(&request.options.temporary)?;
    let destination = request.destination.clone().unwrap_or_else(|| {
        temporary
            .path()
            .join(format!("program{}", std::env::consts::EXE_SUFFIX))
    });
    let mut built = None;
    for attempt in 0..2 {
        let contract = build_contract(client, &request.input.entry, request.options.profile)?;
        let mut loader = resin_source::Loader::new(PathBuf::new());
        let source = loader
            .load_file_async(&request.input.path, execution, cancellation)
            .await?;
        let options = CaptureOptions {
            include_roots: request.options.include_roots.clone(),
            capabilities: client.capabilities(),
        };
        let captured =
            inputs::capture(source, &mut loader, &options, execution, cancellation).await?;
        if captured
            .origins
            .values()
            .any(|source| source.path == destination)
        {
            return Err("output would overwrite a source in the import graph".into());
        }
        match client
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
                built = Some(result.map_err(|error| display_error(error, &captured))?);
                break;
            }
        }
    }
    let artifact = built.expect("full capture retry returns a result");
    if request.destination.is_some() {
        return Ok(0);
    }
    let status = tokio::process::Command::new(artifact.path())
        .args(args)
        .current_dir(&request.options.directory)
        .kill_on_drop(true)
        .status()
        .await?;
    Ok(status.code().unwrap_or(1))
}

pub(crate) fn build_contract(
    client: &Client,
    entry: &str,
    profile: resin_protocol::BuildProfile,
) -> Result<BuildContract> {
    let capabilities = client.capabilities();
    let target = capabilities
        .targets
        .iter()
        .find(|target| {
            target.os == std::env::consts::OS
                && target.architecture == std::env::consts::ARCH
                && target.abi == host_abi()
        })
        .cloned()
        .ok_or("server does not advertise a compatible local host target")?;
    if !capabilities.entry_profiles.contains(&EntryProfile::Host) {
        return Err("server does not support host builds".into());
    }
    Ok(BuildContract {
        target,
        entry: EntryTarget {
            export: entry.into(),
            profile: EntryProfile::Host,
        },
        profile,
        options: BuildOptions {
            max_instances_per_function: 16_384,
        },
    })
}

fn host_abi() -> &'static str {
    if cfg!(target_env = "msvc") {
        "msvc"
    } else if cfg!(target_env = "musl") {
        "musl"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "gnu"
    }
}

pub(crate) fn display_error(
    error: crate::Error,
    captured: &inputs::CapturedInputs,
) -> Box<dyn std::error::Error + Send + Sync> {
    let Some(failure) = error.failure() else {
        return Box::new(error);
    };
    if failure.diagnostics.is_empty() {
        return Box::new(error);
    }
    failure
        .diagnostics
        .iter()
        .map(|diagnostic| {
            let location = diagnostic
                .span
                .as_ref()
                .map(|span| {
                    let local = captured.origins.get(&span.source);
                    let name = local
                        .map(|local| local.path.display().to_string())
                        .unwrap_or_else(|| span.source.clone());
                    let text = local.map(|local| local.source.text()).or_else(|| {
                        failure
                            .managed_sources
                            .iter()
                            .find(|source| source.name == span.source)
                            .map(|source| source.text.as_str())
                    });
                    match text.and_then(|text| text.get(..usize::try_from(span.start).ok()?)) {
                        Some(prefix) => format!(
                            "{name}:{}:{}",
                            prefix.bytes().filter(|byte| *byte == b'\n').count() + 1,
                            prefix.rsplit('\n').next().unwrap().chars().count() + 1
                        ),
                        None => name,
                    }
                })
                .unwrap_or_else(|| captured.inputs.entry.clone());
            format!("{location}: {}", diagnostic.message)
        })
        .collect::<Vec<_>>()
        .join("\n")
        .into()
}
