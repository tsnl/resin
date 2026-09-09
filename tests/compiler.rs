use resin_compiler::{Compilation, Compiler};
use resin_source::Loader;
use resin_toolchain::{CProfile, Environment};
use std::{fs, sync::Arc};
use tempfile::TempDir;

fn build(
    compilation: &Compilation,
    environment: &Environment,
    profile: CProfile,
) -> resin_toolchain::Executable {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let project = resin_codegen::generate(
        compilation.verified().unwrap(),
        Some("main"),
        directory.path(),
    )
    .unwrap();
    let built = environment
        .toolchain(None, None)
        .build(project.directory(), project.name(), "main", profile)
        .unwrap();
    built
        .executable(project.program().unwrap().file_name().unwrap())
        .unwrap()
}

#[test]
fn compilation_uses_supplied_source_versions_and_explicit_profiles() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut environment = Environment::capture().unwrap();
    environment.directory = temp.path().into();
    // Unused shader compilers remain optional for decorated host-callable functions.
    environment
        .variables
        .insert("GLSLC".into(), "/missing/glslc".into());
    let path = temp.path().join("main.resin");
    fs::write(&path, "export { main }; def main() -> int = { 1 };").unwrap();
    let mut loader = Loader::new(environment.path("RESIN_STDLIB", resin_source::stdlib_path()));
    let mut compiler = Compiler::new();
    for (profile, destination, code, directory) in [
        (
            CProfile::Debug,
            Some(temp.path().join("copied")),
            42,
            "debug",
        ),
        (CProfile::Release, None, 43, "release"),
    ] {
        let source = loader.source_from_text(&path, format!("export {{ main }}; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = {{ var i = uint(invocation); output.* := i; }}; def main() -> int = {{ var output = 0_ui; kernel({code}_ul, &output); if (output == {code}_ui) {{ {code} }} else {{ 0 }} }};")).unwrap();
        let compilation = compiler.compile(source.clone(), &mut loader);
        let artifact = build(&compilation, &environment, profile);
        assert_eq!(
            artifact.path().parent().unwrap().file_name().unwrap(),
            directory
        );
        assert_eq!(artifact.run().unwrap(), code);
        assert!(
            Arc::ptr_eq(&compilation, &compiler.compile(source, &mut loader)),
            "unchanged sources should reuse the completed compilation"
        );
        if let Some(path) = destination {
            artifact.copy_to(&path).unwrap();
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
    let mut loader = Loader::new(resin_source::stdlib_path());
    let mut compiler = Compiler::new();
    let path = temp.path().join("main.resin");
    fs::write(&path, "export { main }; def main() -> int = { 41 };").unwrap();
    let first = compiler.compile(loader.load_file(&path).unwrap(), &mut loader);
    fs::write(&path, "export { main }; def main() -> int = { 42 };").unwrap();
    let second = compiler.compile(loader.load_file(&path).unwrap(), &mut loader);
    assert_eq!(first.entry().id(), second.entry().id());
    assert_ne!(first.entry(), second.entry());
    fs::remove_file(path).unwrap();
    for (compilation, code) in [(second, 42), (first, 41)] {
        let artifact = build(&compilation, &environment, CProfile::Debug);
        assert_eq!(artifact.run().unwrap(), code);
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
    for tool in ["cc", "glslc"] {
        fs::write(project.join("build.ninja"), format!("include toolchain.ninja\nrule probe\n  command = ${tool}\nbuild output: probe\ndefault output\n")).unwrap();
        let error = settings
            .build(&project, "probe", tool, CProfile::Debug)
            .unwrap_err();
        assert!(error.to_string().contains("chosen:"), "{error}");
        assert!(error.to_string().contains(".ninja-work"), "{error}");
        assert!(!error.to_string().contains("later:"), "{error}");
    }
}
