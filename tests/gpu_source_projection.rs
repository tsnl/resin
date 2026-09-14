#![cfg(feature = "gpu")]

#[path = "support/pipeline.rs"]
mod pipeline;
#[path = "support/project.rs"]
mod project;
#[path = "support/shaders.rs"]
mod shaders;

use resin_runtime::{ResinGpu, ResinStatus, testing::lock_gpu};
use std::{fs, process::Output};
use tempfile::TempDir;

fn run(source: &str) -> Option<Output> {
    run_with_gpu_library(source, None)
}

fn run_with_gpu_library(source: &str, library: Option<&str>) -> Option<Output> {
    let _lock = lock_gpu();
    match ResinGpu::create() {
        Ok(gpu) => drop(gpu),
        Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
            assert!(
                std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "a suitable Vulkan device is required"
            );
            return None;
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    }
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("main.resin");
    fs::write(&path, source).unwrap();
    if let Some(library) = library {
        fs::write(directory.path().join("gpu.resin"), library).unwrap();
    }
    let module = pipeline::generate_program(&pipeline::load(&path).unwrap())
        .unwrap_or_else(|error| panic!("{source}\n{error}"));
    Some(project::Project::new(&module, Some("main")).unwrap().run())
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn source_sequences_project_offsets_and_retain_resources_through_submit() {
    let Some(output) = run(r#"
        export { main };
        import { "$/gpu.resin", "$/span.resin" };
        struct Root { values: Span<uint>, scalar: Ptr<uint> };
        @compute_shader def kernel(index: ulong, root: Ptr<Root>) = {
            if (index < root.values.length) { root.values.at(index).* := root.values.at(index).* + 10_ui; };
            if (index == 0_ul) { root.scalar.* := 42_ui; };
        };
        def main() -> Result<int, _> = {
            var gpu = Gpu.new()?;
            var values = gpu.alloc::<uint>(5_ul)?;
            var i = 0_ul;
            while (i < 5_ul) { values.at(i).store(uint(i)); i := i + 1_ul; };
            var scalar = gpu.create(0_ui)?;
            var commands = gpu.start_command_recording()?;
            {
                var pipeline = gpu.create_compute_pipeline(kernel)?;
                commands.dispatch(pipeline, { values = values.slice(2_ul, 2_ul), scalar = scalar }, 2_ui, 1_ui, 1_ui)?;
            };
            commands.submit()?;
            ok(if (values.at(1_ul).load() == 1_ui && values.at(2_ul).load() == 12_ui && values.at(3_ul).load() == 13_ui && values.at(4_ul).load() == 4_ui && scalar.load() == 42_ui) { 0_i } else { 1_i })
        };
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn source_pipeline_contract_retagging_cannot_change_the_shader_root() {
    let Some(output) = run(r#"
        export { main };
        import { "$/gpu.resin" };
        struct Root { value: uint };
        struct Other { value: uint };
        @compute_shader def kernel(index: ulong, root: Ptr<Root>) = {};
        def main() -> Result<int, _> = {
            var gpu = Gpu.new()?;
            var pipeline = gpu.create_compute_pipeline(kernel)?;
            var forged = GpuComputePipeline<Other, GpuPipelineOwner> { contract = pipeline.contract };
            var commands = gpu.start_command_recording()?;
            commands.dispatch(forged, { value = 0_ui }, 1_ui, 1_ui, 1_ui)?;
            ok(0_i)
        };
    "#) else {
        return;
    };
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("pipeline contract"));
}
