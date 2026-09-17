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
            let (location, excerpt) =
                diagnostic_context(diagnostic.span.as_ref(), captured, failure);
            let code = diagnostic
                .code
                .as_ref()
                .map(|code| format!("[{code}]"))
                .unwrap_or_default();
            let mut message = format!("{location}: error{code}: {}", diagnostic.message);
            if let Some(excerpt) = excerpt {
                message.push_str(&format!("\n{excerpt}"));
            }
            for note in &diagnostic.related {
                let (location, _) = diagnostic_context(Some(&note.span), captured, failure);
                message.push_str(&format!("\n  note: {location}: {}", note.message));
            }
            for note in &diagnostic.notes {
                message.push_str(&format!("\n  note: {note}"));
            }
            if let Some(help) = &diagnostic.help {
                message.push_str(&format!("\n  help: {help}"));
            }
            message
        })
        .collect::<Vec<_>>()
        .join("\n")
        .into()
}

fn diagnostic_context(
    span: Option<&resin_protocol::Span>,
    captured: &inputs::CapturedInputs,
    failure: &resin_protocol::Failure,
) -> (String, Option<String>) {
    let Some(span) = span else {
        return (captured.inputs.entry.clone(), None);
    };
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
    let Some((line, column, excerpt)) =
        text.and_then(|text| source_excerpt(text, span.start, span.end))
    else {
        return (name, None);
    };
    (format!("{name}:{line}:{column}"), Some(excerpt))
}

fn source_excerpt(text: &str, start: u64, end: u64) -> Option<(usize, usize, String)> {
    let start = usize::try_from(start).ok()?;
    let end = usize::try_from(end).ok()?;
    let prefix = text.get(..start)?;
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let line_start = prefix.rfind('\n').map_or(0, |index| index + 1);
    let line_end = text
        .get(start..)?
        .find('\n')
        .map_or(text.len(), |index| start + index);
    let before = text.get(line_start..start)?;
    let column = before.chars().count() + 1;
    let source = text.get(line_start..line_end)?.replace('\t', "    ");
    let padding = before.replace('\t', "    ").chars().count();
    let width = text
        .get(start..end.min(line_end))?
        .replace('\t', "    ")
        .chars()
        .count()
        .max(1);
    Some((
        line,
        column,
        format!(
            "  | {source}\n  | {}{}",
            " ".repeat(padding),
            "^".repeat(width)
        ),
    ))
}

#[cfg(test)]
mod diagnostic_tests {
    use super::source_excerpt;

    #[test]
    fn excerpts_preserve_unicode_columns_and_expand_tabs() {
        assert_eq!(
            source_excerpt("first\n\tα + β", 12, 14),
            Some((2, 6, "  |     α + β\n  |         ^".into()))
        );
        assert!(source_excerpt("λ", 1, 2).is_none());
        assert!(source_excerpt("x", 5, 6).is_none());
    }
}
