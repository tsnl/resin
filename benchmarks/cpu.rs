use std::{fs, process::Command};

use super::{Config, Result, Workload};

pub fn run(workload: &Workload, config: &Config) -> Result<Vec<f64>> {
    let source = tempfile::tempdir()?;
    let wrapper = source.path().join("host.resin");
    let workload_path = super::workload_path(workload);
    let import = workload_path.to_str().ok_or("workload path is not UTF-8")?;
    let import = import
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    fs::write(
        &wrapper,
        include_str!("support/host.resin").replace("@WORKLOAD@", &import),
    )?;
    let built = super::build(
        workload,
        &wrapper,
        Some("main"),
        &[("benchmark.h", include_str!("support/benchmark.h"))],
    )?;
    let program = built.generated.program().ok_or("missing CPU executable")?;
    let executable = built
        .artifacts
        .executable(program.strip_prefix(built.generated.directory())?)?;
    let count = config.size.unwrap_or(workload.count);
    let output = Command::new(executable.path())
        .args([
            count.to_string(),
            workload.iterations.to_string(),
            config.samples.to_string(),
            config.warmup.to_string(),
            "2891336453".to_string(),
        ])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "CPU benchmark {} failed ({}): {}",
            workload.name,
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    read_samples(
        workload,
        config,
        count,
        std::str::from_utf8(&output.stdout)?,
    )
}

fn read_samples(
    workload: &Workload,
    config: &Config,
    count: u32,
    output: &str,
) -> Result<Vec<f64>> {
    let expected = super::expected_checksum(workload, count);
    let mut samples = Vec::with_capacity(config.samples);
    let mut warmups = 0;
    for line in output.lines() {
        let fields: Vec<_> = line.split(',').collect();
        if fields.len() != 3 {
            return Err(format!("invalid CPU benchmark output: {line}").into());
        }
        let elapsed = fields[1].parse::<f64>()?;
        let checksum = fields[2].parse::<u64>()?;
        if !elapsed.is_finite() || elapsed <= 0.0 {
            return Err(format!("CPU benchmark timer returned {elapsed}").into());
        }
        if checksum != expected {
            return Err(format!(
                "CPU benchmark {} checksum mismatch: expected {expected:016x}, got {checksum:016x}",
                workload.name,
            )
            .into());
        }
        match fields[0] {
            "warmup" if samples.is_empty() => warmups += 1,
            "sample" if warmups == config.warmup => samples.push(elapsed),
            _ => return Err(format!("unexpected CPU benchmark sample: {line}").into()),
        }
    }
    if warmups != config.warmup || samples.len() != config.samples {
        return Err(format!(
            "CPU benchmark returned {} warmups and {} samples; expected {} and {}",
            warmups,
            samples.len(),
            config.warmup,
            config.samples,
        )
        .into());
    }
    Ok(samples)
}
