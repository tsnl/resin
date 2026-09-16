//! Shared setup and reporting for the separate CPU and GPU benchmark executables.

use clap::Parser;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

mod cpu;
mod gpu;
mod inputs;

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
    executable: Option<resin_toolchain::Executable>,
    shaders: Vec<Arc<[u8]>>,
}

fn build(
    _workload: &Workload,
    source: &Path,
    entry: Option<&str>,
    extra_files: &[(&str, &str)],
) -> Result<Built> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let execution = resin_executor::Execution::default();
            let cancellation = resin_executor::Cancellation::new();
            let mut loader =
                resin_source::Loader::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("resin"));
            let source = loader
                .load_file_async(source, &execution, &cancellation)
                .await?;
            let inputs = inputs::capture(source, &mut loader, &execution, &cancellation).await?;
            let output = resin_hir::Hir::build(Arc::new(inputs), &execution, &cancellation).await?;
            let (name, profile) = match entry {
                Some(entry) => (entry, resin_lir::Profile::Host),
                None => ("kernel", resin_lir::Profile::Shader),
            };
            let hir = output.hir().map_err(|error| error.to_string())?;
            let request = resin_lir::Entry::exported(hir, name, profile)
                .map_err(|error| error.to_string())?;
            let lir = resin_lir::build_lir(
                hir.clone(),
                vec![request],
                resin_lir::LoweringOptions::default(),
                &execution,
                &cancellation,
            )
            .await?;
            let lir = resin_lir::VerifiedModule::build(lir, &execution, &cancellation).await?;
            let lir = Arc::new(lir);
            let temporary = tempfile::tempdir()?;
            let mut environment = resin_toolchain::Environment::capture()?;
            environment.executable = env!("CARGO_BIN_EXE_resin").into();
            environment.directory = Path::new(env!("CARGO_MANIFEST_DIR")).into();
            let tools = environment.toolchain(None, None);
            let mut shaders = Vec::new();
            let mut native = resin_codegen::NativeInputs::default();
            for &function in lir.view().module().shaders.keys() {
                let bytes =
                    resin_codegen::generate_spirv(lir.clone(), function, &execution, &cancellation)
                        .await?;
                let bytes = tools
                    .optimize_shader(bytes, temporary.path(), &execution, &cancellation)
                    .await?;
                native.shaders.insert(function, bytes.clone());
                shaders.push(bytes);
            }
            let executable = if let Some(entry) = entry {
                let mut foreign = foreign_inputs(lir.view().module(), extra_files)?;
                for (index, function) in lir.view().module().functions.iter().enumerate() {
                    let Some(declaration) = &function.foreign else {
                        continue;
                    };
                    let symbol = format!("benchmark_foreign_{index}");
                    native.foreign.insert(
                        resin_types::FunctionId::from_index(index),
                        symbol.clone().into(),
                    );
                    foreign.functions.push(resin_toolchain::ForeignFunction {
                        symbol,
                        name: function
                            .name
                            .as_ref()
                            .ok_or("unnamed foreign function")?
                            .to_string(),
                        params: declaration
                            .params
                            .iter()
                            .map(foreign_scalar)
                            .collect::<Result<_>>()?,
                        result: foreign_scalar(&function.result)?,
                    });
                }
                let mut objects = Vec::new();
                if !foreign.functions.is_empty() {
                    objects.push(
                        tools
                            .compile_foreign(
                                Arc::new(foreign),
                                temporary.path(),
                                &execution,
                                &cancellation,
                            )
                            .await?
                            .bytes(),
                    );
                }
                let object = resin_codegen::generate_native(
                    lir,
                    entry.into(),
                    resin_codegen::NativeOptimization::Speed,
                    Arc::new(native),
                    &execution,
                    &cancellation,
                )
                .await?;
                objects.insert(0, object.shared_bytes());
                Some(
                    tools
                        .link_native(
                            resin_toolchain::NativeLink {
                                objects,
                                runtime: true,
                            },
                            &std::env::temp_dir(),
                            &execution,
                            &cancellation,
                        )
                        .await?,
                )
            } else {
                None
            };
            Ok(Built {
                executable,
                shaders,
            })
        })
}

fn foreign_scalar(ty: &resin_types::Ty) -> Result<resin_toolchain::ForeignScalar> {
    use resin_toolchain::ForeignScalar;
    use resin_types::Ty;
    Ok(match ty {
        Ty::Unit => ForeignScalar::Void,
        Ty::Bool => ForeignScalar::Bool,
        Ty::Pointer { .. } => ForeignScalar::Pointer,
        Ty::Int8 | Ty::UInt8 => ForeignScalar::Integer {
            bits: 8,
            signed: matches!(ty, Ty::Int8),
        },
        Ty::Int16 | Ty::UInt16 => ForeignScalar::Integer {
            bits: 16,
            signed: matches!(ty, Ty::Int16),
        },
        Ty::Int32 | Ty::UInt32 => ForeignScalar::Integer {
            bits: 32,
            signed: matches!(ty, Ty::Int32),
        },
        Ty::Int64 | Ty::UInt64 => ForeignScalar::Integer {
            bits: 64,
            signed: matches!(ty, Ty::Int64),
        },
        Ty::Float32 => ForeignScalar::Float { bits: 32 },
        Ty::Float64 => ForeignScalar::Float { bits: 64 },
        _ => return Err(format!("unsupported benchmark foreign type {ty:?}").into()),
    })
}

fn foreign_inputs(
    module: &resin_lir::Module,
    extra: &[(&str, &str)],
) -> Result<resin_toolchain::ForeignInputs> {
    let mut inputs = resin_toolchain::ForeignInputs::default();
    let root = Path::new(resin_runtime::INCLUDE_DIR);
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let name = format!(
                    "runtime/{}",
                    path.strip_prefix(root)?
                        .to_string_lossy()
                        .replace('\\', "/")
                );
                inputs.files.insert(name.into(), fs::read(path)?.into());
            }
        }
    }
    for (name, contents) in extra {
        inputs
            .files
            .insert(format!("bundle/{name}").into(), contents.as_bytes().into());
    }
    inputs.include_directories = vec!["runtime".into(), "bundle".into()];
    // Feature-selection macros in fixture headers must precede libc includes.
    inputs
        .includes
        .extend(extra.iter().map(|(name, _)| format!("bundle/{name}")));
    for header in &module.foreign_headers {
        let name = [
            &format!("runtime/{}", header.spelling),
            &format!("bundle/{}", header.spelling),
        ]
        .into_iter()
        .find(|name| inputs.files.contains_key(name.as_str()))
        .cloned()
        .unwrap_or_else(|| header.spelling.to_string());
        if !inputs.includes.contains(&name) {
            inputs.includes.push(name);
        }
    }
    Ok(inputs)
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
