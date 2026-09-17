//! Large compiler and cache workloads, kept out of the ordinary regression suite.
use clap::Parser;
use serde_json::{Value, json};
use std::{num::NonZeroUsize, process::Command, time::Instant};

#[path = "compiler/cache.rs"]
mod cache;
#[allow(dead_code)]
#[path = "../tests/support/mod.rs"]
mod support;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Measure compiler control flow and cache workloads; print JSON results")]
struct Config {
    /// Timed samples per workload.
    #[arg(long, default_value = "5")]
    samples: NonZeroUsize,
    /// Untimed warmup iterations per workload.
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Sequential conditional/error-propagation pairs in each backend fixture.
    #[arg(long, default_value = "512")]
    branches: NonZeroUsize,
    // Cargo passes this to custom benchmark harnesses.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() -> Result<()> {
    let config = Config::parse();
    let optimizer = support::shaders::optimizer().ok_or("spirv-opt is required")?;
    let host = support::control::host_source(config.branches.get());
    let shader = support::control::shader_source(config.branches.get());
    let mut c_samples = Vec::new();
    let mut spirv_samples = Vec::new();
    let mut cache_samples = Vec::new();
    for iteration in 0..config.warmup + config.samples.get() {
        let c = measure_host(&host, (config.branches.get() % 2) as i32)?;
        let spirv = measure_shader(&shader, &optimizer)?;
        let caches =
            support::frontend::block_on(cache::measure()).map_err(|error| error.to_string())?;
        if iteration >= config.warmup {
            c_samples.push(c);
            spirv_samples.push(spirv);
            cache_samples.push(caches);
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "os": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "jobs": support::frontend::execution().jobs(),
            "debug_assertions": cfg!(debug_assertions),
            "warmup_iterations": config.warmup,
            "iterations": config.samples.get(),
            "branches": config.branches.get(),
            "c": c_samples,
            "spirv": spirv_samples,
            "cache": cache_samples,
            "measurement": "Wall-clock milliseconds. Frontend includes parsing, HIR, LIR and verification. Codegen includes temporary project creation and verification. Native builds include Ninja, validation and optimized C/SPIR-V compilation. Host execution is separate. Cache RSS is process-wide, not an allocation delta."
        }))?
    );
    Ok(())
}

fn generate(source: &str, entry: Option<&str>) -> Result<(support::project::Project, Value)> {
    let started = Instant::now();
    let module = support::module(source);
    let frontend_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let project = support::project::Project::new(&module, entry)?;
    let codegen_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok((
        project,
        json!({ "frontend_ms": frontend_ms, "codegen_ms": codegen_ms }),
    ))
}

fn measure_host(source: &str, expected_exit: i32) -> Result<Value> {
    let (project, mut sample) = generate(source, Some("main"))?;
    let c = std::fs::read_to_string(project.generated.c_source().unwrap())?;
    assert!(
        c.lines()
            .all(|line| line.len() - line.trim_start().len() < 32)
    );
    let started = Instant::now();
    let executable = project.build_executable();
    sample["native_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    let started = Instant::now();
    let output = Command::new(executable.path()).output()?;
    sample["execute_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(output.status.code(), Some(expected_exit), "{output:?}");
    Ok(sample)
}

fn measure_shader(source: &str, optimizer: &std::ffi::OsStr) -> Result<Value> {
    let (project, mut sample) = generate(source, None)?;
    let bytes = std::fs::read(project.generated.shaders()[0].unoptimized_spirv())?;
    assert_eq!(support::shaders::instructions(&bytes, 246).count(), 0);
    // Each sample must invoke the optimizer, without a prior Ninja cache hit.
    let directory = tempfile::TempDir::new()?;
    let mut environment = resin_toolchain::Environment::capture()?;
    environment.directory = directory.path().into();
    environment.executable = env!("CARGO_BIN_EXE_resin").into();
    let tools = environment.toolchain(None, Some(optimizer));
    let started = Instant::now();
    let _built = project.build(&tools)?;
    sample["native_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    Ok(sample)
}
