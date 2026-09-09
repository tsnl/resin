//! Small correctness runs of the real benchmark harness; no performance thresholds.
use clap::Parser;

#[allow(dead_code)]
#[path = "../benchmarks/suite.rs"]
mod suite;

#[cfg(feature = "gpu")]
#[path = "support/shaders.rs"]
mod shaders;

#[cfg(feature = "gpu")]
fn config() -> suite::Config {
    suite::Config::try_parse_from([
        "benchmark",
        "--size",
        "65",
        "--samples",
        "2",
        "--warmup",
        "1",
    ])
    .unwrap()
}

fn check_report(report: serde_json::Value, target: &str) {
    assert_eq!(report["target"], target);
    let measurements = report["measurements"].as_array().unwrap();
    assert_eq!(measurements.len(), 3);
    for measurement in measurements {
        assert_eq!(measurement["count"], 65);
        assert_eq!(measurement["validation"], "passed");
        assert_eq!(measurement["samples_seconds"].as_array().unwrap().len(), 2);
    }
}

#[test]
fn cpu_workloads_match_reference_in_optimized_generated_code() {
    let directory = tempfile::TempDir::new().unwrap();
    let output = directory.path().join("cpu.json");
    let mut args = vec![
        "benchmark",
        "--size",
        "65",
        "--samples",
        "2",
        "--warmup",
        "1",
        "--json",
    ];
    args.push(output.to_str().unwrap());
    let config = suite::Config::try_parse_from(args).unwrap();
    let report = suite::run(suite::Target::Cpu, &config).unwrap();
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    for (original, saved) in report["measurements"]
        .as_array()
        .unwrap()
        .iter()
        .zip(saved["measurements"].as_array().unwrap())
    {
        assert_eq!(original["workload"], saved["workload"]);
        for (original, saved) in original["samples_seconds"]
            .as_array()
            .unwrap()
            .iter()
            .zip(saved["samples_seconds"].as_array().unwrap())
        {
            let (original, saved) = (original.as_f64().unwrap(), saved.as_f64().unwrap());
            assert!((original - saved).abs() <= f64::EPSILON * original);
        }
    }
    check_report(saved, "cpu");
    check_report(report, "cpu");
}

#[test]
#[cfg(feature = "gpu")]
fn gpu_workloads_match_reference_with_a_partial_workgroup() {
    if shaders::optimizer().is_none() {
        return;
    }
    let _lock = resin_runtime::testing::lock_gpu();
    match resin_runtime::ResinGpu::create() {
        Ok(gpu) => drop(gpu),
        Err(
            resin_runtime::ResinStatus::Unsupported | resin_runtime::ResinStatus::VulkanUnavailable,
        ) => {
            assert!(
                std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "a suitable Vulkan device is required"
            );
            eprintln!("skipping GPU benchmark correctness check: no suitable Vulkan device");
            return;
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    }
    check_report(suite::run(suite::Target::Gpu, &config()).unwrap(), "gpu");
}

#[test]
fn benchmark_options_reject_empty_work_and_zero_samples() {
    for args in [
        vec!["bench", "--size", "0"],
        vec!["bench", "--samples", "0"],
        vec!["bench", "--workload", "unknown"],
    ] {
        assert!(suite::Config::try_parse_from(args).is_err());
    }
}
