#[allow(dead_code)]
mod support;

use resin_hir::Hir;
use resin_source::Loader;
use resin_toolchain::{CProfile, Environment};
use std::fs;
use tempfile::TempDir;

fn build(
    compilation: &Hir,
    environment: &Environment,
    profile: CProfile,
) -> resin_toolchain::Executable {
    let lir =
        support::pipeline::verified_lir(compilation, "main", resin_lir::Profile::Host).unwrap();
    support::frontend::block_on(async {
        let cancellation = resin_executor::Cancellation::new();
        let object = resin_codegen::generate_native(
            std::sync::Arc::new(lir),
            "main".into(),
            match profile {
                CProfile::Debug => resin_codegen::NativeOptimization::None,
                CProfile::Release => resin_codegen::NativeOptimization::Speed,
            },
            std::sync::Arc::new(resin_codegen::NativeInputs::default()),
            support::frontend::execution(),
            &cancellation,
        )
        .await
        .unwrap();
        environment
            .toolchain(None, None)
            .link_native(
                resin_toolchain::NativeLink {
                    objects: vec![object.shared_bytes()],
                    runtime: true,
                },
                &std::env::temp_dir(),
                support::frontend::execution(),
                &cancellation,
            )
            .await
            .unwrap()
    })
}

#[test]
fn compilation_uses_supplied_source_versions_and_explicit_profiles() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut environment = Environment::capture().unwrap();
    environment.directory = temp.path().into();
    // Unused shader compilers remain optional for decorated host-callable functions.
    environment
        .variables
        .insert("SPIRV_OPT".into(), "/missing/spirv-opt".into());
    let path = temp.path().join("main.resin");
    fs::write(&path, "export { main }; def main() -> int = { 1 };").unwrap();
    let mut loader =
        Loader::new(environment.path("RESIN_LIBRARY_ROOT", resin_source::library_root()));
    let mut previous = None;
    for (profile, destination, code) in [
        (CProfile::Debug, Some(temp.path().join("copied")), 42),
        (CProfile::Release, None, 43),
    ] {
        let source = loader.source_from_text(&path, format!("export {{ main }}; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = {{ var i = uint(invocation); output.* := i; }}; def main() -> int = {{ var output = 0_ui; kernel({code}_ul, &output); if (output == {code}_ui) {{ {code} }} else {{ 0 }} }};")).unwrap();
        let compilation =
            support::frontend::analyze(source.clone(), &mut loader, previous.as_ref());
        let artifact = build(&compilation, &environment, profile);
        assert_eq!(support::frontend::run(&artifact).unwrap(), code);
        assert!(
            compilation.same(&support::frontend::analyze(
                source,
                &mut loader,
                Some(&compilation)
            )),
            "unchanged sources should reuse the completed compilation"
        );
        previous = Some(compilation);
        if let Some(path) = destination {
            support::frontend::copy(&artifact, &path).unwrap();
            assert!(path.is_file());
            fs::remove_file(&path).unwrap();
        }
    }
    assert!(!environment.directory.join("build/shaders").exists());
    assert_eq!(
        fs::read_to_string(path).unwrap(),
        "export { main }; def main() -> int = { 1 };"
    );
}

#[test]
fn retained_compilations_build_their_own_source_version_after_later_edits() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut environment = Environment::capture().unwrap();
    environment.directory = temp.path().into();
    let mut loader = Loader::new(resin_source::library_root());
    let path = temp.path().join("main.resin");
    fs::write(&path, "export { main }; def main() -> int = { 41 };").unwrap();
    let first = support::frontend::analyze(loader.load_file(&path).unwrap(), &mut loader, None);
    fs::write(&path, "export { main }; def main() -> int = { 42 };").unwrap();
    let second =
        support::frontend::analyze(loader.load_file(&path).unwrap(), &mut loader, Some(&first));
    assert_eq!(first.source().id(), second.source().id());
    assert_ne!(first.source(), second.source());
    fs::remove_file(path).unwrap();
    for (compilation, code) in [(second, 42), (first, 41)] {
        let artifact = build(&compilation, &environment, CProfile::Debug);
        assert_eq!(support::frontend::run(&artifact).unwrap(), code);
    }
}

#[test]
#[cfg(unix)]
fn compiler_processes_use_the_supplied_environment_and_working_directory() {
    use std::os::unix::fs::PermissionsExt;
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let wrapper = temp.path().join("compiler");
    fs::write(
        &wrapper,
        "#!/bin/sh\nprintf '%s:%s' \"$RESIN_TOOL_SETTING\" \"$PWD\" >&2\nexit 47\n",
    )
    .unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).unwrap();
    let mut environment = Environment::capture().unwrap();
    environment.directory = temp.path().into();
    environment
        .variables
        .insert("RESIN_TOOL_SETTING".into(), "chosen".into());
    let settings = environment.toolchain(Some(wrapper.as_os_str()), Some(wrapper.as_os_str()));
    environment
        .variables
        .insert("RESIN_TOOL_SETTING".into(), "later".into());
    let project = temp.path().join("generated");
    fs::create_dir(&project).unwrap();
    for tool in ["cc", "spirv_opt"] {
        fs::write(project.join("build.ninja"), format!("include toolchain.ninja\nrule probe\n  command = ${tool}\nbuild output: probe\ndefault output\n")).unwrap();
        let error = support::frontend::build(&settings, &project, "probe", tool, CProfile::Debug)
            .unwrap_err();
        assert!(error.to_string().contains("chosen:"), "{error}");
        assert!(error.to_string().contains(".ninja-work"), "{error}");
        assert!(!error.to_string().contains("later:"), "{error}");
    }
}
