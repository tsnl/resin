//! Shared setup and reporting for the separate CPU and GPU benchmark executables.

use clap::Parser;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[path = "cpu.rs"]
mod cpu;
#[path = "gpu.rs"]
mod gpu;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Cpu,
    Gpu,
}

impl Target {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Measure compiled Resin workloads; setup and validation are outside timings")]
pub struct Config {
    /// Select named workloads; defaults to all. May be repeated.
    #[arg(long, value_parser = ["branch_heavy", "arithmetic", "stream"])]
    workload: Vec<String>,
    /// Timed samples per workload.
    #[arg(long, default_value_t = 30, value_parser = positive_usize)]
    samples: usize,
    /// Untimed warmup runs per workload.
    #[arg(long, default_value_t = 8, value_parser = positive_usize)]
    warmup: usize,
    /// Override lane count (maximum portable one-dimensional GPU dispatch).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=4_194_240))]
    size: Option<u32>,
    /// Save metadata, summary statistics, and individual timings as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Identify this measurement, e.g. a branch name.
    #[arg(long, default_value = "")]
    label: String,
    /// List available workloads without compiling or measuring.
    #[arg(long)]
    list: bool,
    /// Vulkan physical-device index (GPU benchmark only).
    #[arg(long)]
    device: Option<u32>,
    /// List Vulkan devices without measuring (GPU benchmark only).
    #[arg(long)]
    list_devices: bool,
    // Cargo passes this to custom benchmark harnesses.
    #[arg(long, hide = true)]
    bench: bool,
}

fn positive_usize(text: &str) -> std::result::Result<usize, String> {
    let value: usize = text.parse().map_err(|_| "expected a positive integer")?;
    if value == 0 || value > u32::MAX as usize / 2 {
        return Err("expected an integer between 1 and 2147483647".into());
    }
    Ok(value)
}

struct Workload {
    name: &'static str,
    count: u32,
    iterations: u32,
}

const WORKLOADS: [Workload; 3] = [
    Workload {
        name: "branch_heavy",
        count: 262_144,
        iterations: 128,
    },
    Workload {
        name: "arithmetic",
        count: 262_144,
        iterations: 128,
    },
    Workload {
        name: "stream",
        count: 4_000_000,
        iterations: 1,
    },
];

pub fn main(target: Target) {
    if let Err(error) = run(target, &Config::parse()) {
        eprintln!("{} benchmark failed: {error}", target.name());
        std::process::exit(1);
    }
}

pub fn run(target: Target, config: &Config) -> Result<Value> {
    if target == Target::Cpu && (config.device.is_some() || config.list_devices) {
        return Err("device selection is available in the GPU benchmark only".into());
    }
    if config.list {
        for workload in &WORKLOADS {
            println!(
                "{}: {} lanes, {} iterations",
                workload.name, workload.count, workload.iterations
            );
        }
        return Ok(Value::Null);
    }
    if config.list_devices {
        gpu::list_devices()?;
        return Ok(Value::Null);
    }
    measure(target, config)
}

fn measure(target: Target, config: &Config) -> Result<Value> {
    let mut device = if target == Target::Gpu {
        Some(gpu::Device::open(config.device)?)
    } else {
        None
    };
    let mut report = metadata(target, config);
    if let Some(device) = &device {
        report["gpu"] = device.description();
    }
    let mut measurements = Vec::new();
    println!(
        "{} benchmark: {} samples, {} warmups",
        target.name(),
        config.samples,
        config.warmup
    );
    println!(
        "{:<16} {:>10} {:>12} {:>12} {:>12}",
        "workload", "lanes", "median ms", "p10 ms", "p90 ms"
    );
    for workload in selected(config) {
        let samples = match &mut device {
            Some(device) => device.run(workload, config)?,
            None => cpu::run(workload, config)?,
        };
        let measurement = summarize(workload, config, &samples)?;
        println!(
            "{:<16} {:>10} {:>12.6} {:>12.6} {:>12.6}",
            workload.name,
            config.size.unwrap_or(workload.count),
            measurement["median_seconds"].as_f64().unwrap() * 1000.0,
            measurement["p10_seconds"].as_f64().unwrap() * 1000.0,
            measurement["p90_seconds"].as_f64().unwrap() * 1000.0
        );
        measurements.push(measurement);
    }
    report["measurements"] = json!(measurements);
    if let Some(path) = &config.json {
        fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
        eprintln!("Saved {}", path.display());
    }
    Ok(report)
}

fn selected(config: &Config) -> impl Iterator<Item = &Workload> {
    WORKLOADS.iter().filter(|workload| {
        config.workload.is_empty() || config.workload.iter().any(|name| name == workload.name)
    })
}

fn summarize(workload: &Workload, config: &Config, samples: &[f64]) -> Result<Value> {
    if samples.len() != config.samples || samples.iter().any(|t| !t.is_finite() || *t <= 0.0) {
        return Err(format!(
            "{} returned invalid timings; increase --size if below timer resolution",
            workload.name
        )
        .into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(json!({
        "workload": workload.name,
        "count": config.size.unwrap_or(workload.count),
        "iterations": workload.iterations,
        "warmup": config.warmup,
        "samples_seconds": samples,
        "median_seconds": percentile(&sorted, 0.5),
        "p10_seconds": percentile(&sorted, 0.1),
        "p90_seconds": percentile(&sorted, 0.9),
        "validation": "passed",
    }))
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let index = (sorted.len() - 1) as f64 * fraction;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (index - low as f64)
}

fn metadata(target: Target, config: &Config) -> Value {
    let cc = std::env::var("CC").unwrap_or_else(|_| resin_toolchain::DEFAULT_C_COMPILER.into());
    let spirv_opt = std::env::var("SPIRV_OPT").unwrap_or_else(|_| "spirv-opt".into());
    json!({
        "schema_version": 1,
        "target": target.name(),
        "label": config.label,
        "unix_time_seconds": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        "git_commit": command_text("git", &["rev-parse", "HEAD"]),
        "git_dirty": command_text("git", &["status", "--porcelain"]).map(|s| !s.is_empty()),
        "os": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "cpu": cpu_name(),
        "logical_cpus": std::thread::available_parallelism().map(usize::from).ok(),
        "rustc": command_text("rustc", &["--version"]),
        "native_profile": "release (-O3 C, -O SPIR-V, Vulkan 1.3)",
        "input_seed": 2891336453_u32,
        "cc": cc,
        "spirv_opt": spirv_opt,
        "native_compiler_version": match target {
            Target::Cpu => command_text(&cc, &["--version"]),
            Target::Gpu => command_text(&spirv_opt, &["--version"]),
        },
        "timing": if target == Target::Cpu { "monotonic clock around one single-threaded CPU workload" }
                  else { "Vulkan timestamps around one dispatch, including runtime GPU barriers" },
    })
}

fn command_text(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn cpu_name() -> Option<String> {
    if cfg!(target_os = "linux") {
        fs::read_to_string("/proc/cpuinfo")
            .ok()?
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == "model name").then(|| value.trim().to_owned())
            })
    } else if cfg!(target_os = "macos") {
        command_text("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else {
        std::env::var("PROCESSOR_IDENTIFIER").ok()
    }
}

//
// Compile the actual Resin sources with the regular native toolchain
//

struct Built {
    generated: resin_codegen::GeneratedProject,
    artifacts: resin_toolchain::BuiltProject,
}

fn build(
    workload: &Workload,
    source: &Path,
    entry: Option<&str>,
    extra_files: &[(&str, &str)],
) -> Result<Built> {
    let mut loader = resin_source::Loader::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("resin"));
    let source = loader.load_file(source)?;
    let compilation = resin_compiler::Compiler::new().compile(source, &mut loader);
    let directory = tempfile::TempDir::new()?;
    let generated = resin_codegen::generate(compilation.verified()?, entry, directory.path())?;
    for (name, contents) in extra_files {
        fs::write(directory.path().join(name), contents)?;
    }
    if extra_files.iter().any(|(name, _)| *name == "benchmark.h") {
        let path = generated
            .c_source()
            .ok_or("CPU benchmark needs generated C")?;
        let text = fs::read_to_string(path)?;
        fs::write(
            path,
            format!("#ifndef _POSIX_C_SOURCE\n#define _POSIX_C_SOURCE 200809L\n#endif\n{text}"),
        )?;
    }
    if !extra_files.is_empty() {
        let graph = fs::read_to_string(generated.build_file())?;
        fs::write(
            generated.build_file(),
            graph.replace(
                "include toolchain.ninja\n",
                "include toolchain.ninja\ncflags = $cflags -I .\n",
            ),
        )?;
    }
    let mut environment = resin_toolchain::Environment::capture()?;
    // Cargo builds the matching Resin executable, used by Ninja's embedding step.
    environment.executable = env!("CARGO_BIN_EXE_resin").into();
    environment.directory = Path::new(env!("CARGO_MANIFEST_DIR")).into();
    let artifacts = environment.toolchain(None, None).build(
        directory.path(),
        &format!("benchmark-{}", workload.name),
        entry.unwrap_or("benchmark-shader"),
        resin_toolchain::CProfile::Release,
    )?;
    Ok(Built {
        generated,
        artifacts,
    })
}

fn workload_path(workload: &Workload) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/workloads")
        .join(format!("{}.resin", workload.name))
}

//
// Independent scalar references: fixed-width arithmetic is explicitly wrapping
//

fn input(index: u32) -> u32 {
    index.wrapping_mul(747796405).wrapping_add(2891336453)
}

fn expected(workload: &Workload, lane: u32) -> u32 {
    if workload.name == "stream" {
        return input(lane).wrapping_mul(1664525).wrapping_add(lane);
    }
    let mut x = input(lane);
    for n in 0..workload.iterations {
        x = if workload.name == "branch_heavy" && (x ^ lane) & 256 != 0 {
            (x ^ x.wrapping_mul(127))
                .wrapping_mul(22695477)
                .wrapping_add(lane)
        } else {
            (x ^ x.wrapping_mul(8191))
                .wrapping_mul(1664525)
                .wrapping_add(n)
        };
        x = (x ^ x.wrapping_mul(65521)).wrapping_add(1013904223);
    }
    x
}

fn expected_checksum(workload: &Workload, count: u32) -> u64 {
    (0..count).fold(14695981039346656037, |hash, lane| {
        (hash ^ u64::from(expected(workload, lane))).wrapping_mul(1099511628211)
    })
}
