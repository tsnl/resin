#[allow(dead_code)]
mod support;

use std::{path::PathBuf, process::Command};

fn checkpoint(name: &str) -> (support::project::Project, resin_toolchain::Executable) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/tutorial")
        .join(name);
    let module =
        support::pipeline::host_entry(&path, "main").unwrap_or_else(|error| panic!("{error}"));
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let executable = project.build_executable();
    (project, executable)
}

#[test]
fn orbit_checkpoint_checks_escape_counts_and_the_boundary() {
    let (_project, executable) = checkpoint("01_orbit.resin");
    let output = Command::new(executable.path()).output().unwrap();
    assert!(output.status.success(), "{output:?}");
}

#[test]
fn image_checkpoints_write_opaque_pngs_without_a_gpu() {
    let directory = tempfile::TempDir::new().unwrap();
    let mut images = Vec::new();
    for (program, image_name) in [
        ("02_image.resin", "mandelbrot-1.png"),
        ("03_sampling.resin", "mandelbrot-16.png"),
    ] {
        let (_project, executable) = checkpoint(program);
        let output = Command::new(executable.path())
            .current_dir(directory.path())
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_DISPLAY")
            .env("VK_DRIVER_FILES", directory.path().join("no-driver.json"))
            .env("VK_ICD_FILENAMES", directory.path().join("no-driver.json"))
            .output()
            .unwrap();
        assert!(output.status.success(), "{program}: {output:?}");
        let image = resin_runtime::image_read_png(directory.path().join(image_name), 4).unwrap();
        assert_eq!((image.width, image.height), (320, 240));
        assert!(image.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
        assert!(
            image
                .pixels
                .chunks_exact(4)
                .any(|pixel| pixel[..3] != [0, 0, 0])
        );
        images.push(image);
    }
    assert_ne!(images[0].pixels, images[1].pixels);
}

#[test]
fn tutorial_reference_rejections_remain_language_errors() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("doc/tutorial/gpu.md");
    let chapter = std::fs::read_to_string(path).unwrap();
    let mut count = 0;
    for snippet in chapter.split("```resin,compile_fail\n").skip(1) {
        let source = snippet.split("```").next().unwrap();
        let error = support::pipeline::source_module(source).unwrap_err();
        assert!(
            error.to_string().contains("cannot take the address")
                || error
                    .to_string()
                    .contains("TypeMismatch { expected: Pointer { pointee: Int32 }"),
            "{error}"
        );
        count += 1;
    }
    assert_eq!(
        count, 2,
        "keep both documented reference restrictions checked"
    );
}

#[test]
fn introductory_checkpoints_teach_values_functions_and_ufcs() {
    for (program, expected) in [
        ("basics/hello.resin", "Hello, Resin!\n"),
        ("basics/functions.resin", "answer = 42, doubled = 84\n"),
        ("basics/counter.resin", "counter = 42, sum = 55: ready\n"),
    ] {
        let (_project, executable) = checkpoint(program);
        let output = Command::new(executable.path()).output().unwrap();
        assert!(output.status.success(), "{program}: {output:?}");
        assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
    }
}

#[test]
fn type_reference_examples_compile() {
    for name in ["generics.md", "overloads.md", "inference.md"] {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("doc")
            .join(name);
        let chapter = std::fs::read_to_string(path).unwrap();
        let examples: Vec<_> = chapter.split("```resin\n").skip(1).collect();
        assert!(!examples.is_empty(), "{name} has no examples");
        for (index, snippet) in examples.iter().enumerate() {
            let source = snippet.split("```").next().unwrap();
            support::pipeline::source_module(source)
                .unwrap_or_else(|error| panic!("{name}, example {}: {error}", index + 1));
        }
    }
}
