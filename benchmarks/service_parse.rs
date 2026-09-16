//! Measure full CST parsing and preamble extraction with all inputs already in memory.
use clap::Parser;
use futures::{StreamExt, TryStreamExt, stream};
use resin_executor::{Cancellation, Execution};
use std::{error::Error, fs, num::NonZeroUsize, path::Path, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
type Files = Vec<(String, String)>;

#[derive(Parser)]
#[command(about = "Measure full CST parsing and preamble extraction; print JSON results")]
struct Config {
    /// Read every regular .resin file recursively instead of generating the fixture.
    #[arg(long)]
    directory: Option<PathBuf>,
    /// Number of timed samples; must be positive.
    #[arg(long, default_value = "30")]
    samples: NonZeroUsize,
    /// Number of untimed warmup iterations.
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    // Cargo passes this to custom benchmark harnesses.
    #[arg(long, hide = true)]
    bench: bool,
}

fn fixture() -> Files {
    let mut files = Vec::new();
    for index in 0..16 {
        let mut text = format!("export {{ value_{index} }}; fn value_{index}() -> int  {{ 42 }}\n");
        for helper in 0..64 {
            text.push_str(&format!(
                "fn helper_{helper}(value: int) -> int  {{ value + {helper} }}\n"
            ));
        }
        files.push((format!("file{index}.resin"), text));
    }
    let imports = (0..16)
        .map(|index| format!("\"file{index}.resin\""))
        .collect::<Vec<_>>()
        .join(", ");
    let calls = (0..16)
        .map(|index| format!("value_{index}()"))
        .collect::<Vec<_>>()
        .join(" + ");
    files.push((
        "main.resin".into(),
        format!("export {{ main }}; import {{ {imports} }}; fn main() -> int  {{ {calls} }}\n"),
    ));
    files
}

fn read_sources(directory: &Path, files: &mut Files) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_dir() {
            read_sources(&path, files)?;
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "resin") {
            files.push((path.display().to_string(), fs::read_to_string(path)?));
        }
    }
    Ok(())
}

async fn parse(files: &Files, execution: &Execution) -> Result<usize> {
    let cancellation = Cancellation::new();
    let counts: Vec<usize> = stream::iter(files)
        .map(|(_, text)| {
            let cancellation = &cancellation;
            async move {
                let document =
                    resin_cst::build_cst(text.to_owned(), None, execution, cancellation).await?;
                execution
                    .run(cancellation, move |_| {
                        let preamble = std::hint::black_box(document.preamble());
                        preamble.imports.len() + preamble.headers.len() + preamble.diagnostics.len()
                    })
                    .await
            }
        })
        .buffer_unordered(execution.jobs())
        .try_collect()
        .await?;
    Ok(std::hint::black_box(counts.into_iter().sum()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse();
    let mut files = if let Some(directory) = &config.directory {
        let mut files = Vec::new();
        read_sources(directory, &mut files)?;
        files
    } else {
        fixture()
    };
    files.sort_by(|left, right| left.0.cmp(&right.0));
    if files.is_empty() {
        return Err("fixture directory contains no regular .resin files".into());
    }
    let execution = Execution::default();
    for _ in 0..config.warmup {
        parse(&files, &execution).await?;
    }
    let mut samples_ms = Vec::with_capacity(config.samples.get());
    let mut preamble_items = 0;
    for _ in 0..config.samples.get() {
        let started = Instant::now();
        preamble_items = parse(&files, &execution).await?;
        samples_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    execution.wait_idle().await;
    let mut sorted = samples_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    let median = (sorted[(count - 1) / 2] + sorted[count / 2]) / 2.0;
    let p95 = sorted[count - count / 20 - 1];
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "fixture": config.directory.as_ref().map_or_else(
                || "generated-17-files-1041-declarations".to_owned(),
                |directory| directory.display().to_string()),
            "file_count": files.len(),
            "source_bytes": files.iter().map(|(_, text)| text.len()).sum::<usize>(),
            "os": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "jobs": execution.jobs(),
            "warmup_iterations": config.warmup,
            "iterations": samples_ms.len(),
            "preamble_items": preamble_items,
            "samples_ms": samples_ms,
            "median_ms": median,
            "p95_ms": p95,
            "percentile_method": "nearest rank",
            "measurement": "Full CST parse with no predecessor and preamble extraction; all files available in one bounded parallel batch; includes text copies and scheduling; excludes filesystem reads, import resolution, header acquisition, networking, and server work"
        }))?
    );
    Ok(())
}
