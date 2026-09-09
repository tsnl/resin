use resin_common::TempDir;
use resin_compiler::{Input, Options, Request, Session};
use resin_platform_toolchain::CProfile;
use resin_platform_toolchain::Environment;
use std::{fs, path::Path, sync::Arc};

fn options(environment: &Environment, profile: CProfile) -> Options {
    Options {
        profile,
        tools: environment.toolchain(None, None),
    }
}

fn input(path: &Path) -> Input {
    Input {
        path: path.into(),
        entry: "main".into(),
    }
}

#[test]
fn requests_reject_source_overwrites_before_compilation() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let environment = Environment::capture().unwrap();
    let source = temp
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    fs::write(&source, "this need not parse").unwrap();
    let alias = temp.path().join(".").join(source.file_name().unwrap());
    for destination in [&source, &alias] {
        let result = Request::new(
            input(&source),
            Some(destination.into()),
            options(&environment, CProfile::Debug),
        );
        let error = result
            .err()
            .expect("source overwrite must fail at construction");
        assert!(
            error.to_string().contains("overwrite the source"),
            "{error}"
        );
    }
    let result = Request::new(
        input(&source),
        Some(temp.path().into()),
        options(&environment, CProfile::Debug),
    );
    assert!(
        result.is_err(),
        "the generated filename also needs validation"
    );
    let unsaved = temp.path().join("unsaved.resin");
    assert!(
        Request::new(
            input(&unsaved),
            Some(unsaved.clone()),
            options(&environment, CProfile::Debug)
        )
        .is_err()
    );
    assert_eq!(fs::read_to_string(source).unwrap(), "this need not parse");
}

#[test]
fn requests_resolve_executable_directories_and_preserve_file_destinations() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let environment = Environment::capture().unwrap();
    let input = Input {
        path: temp.path().join("example.resin"),
        entry: "demo".into(),
    };
    for directory in [temp.path().to_path_buf(), temp.path().join("new/")] {
        let request = Request::new(
            input.clone(),
            Some(directory.clone()),
            options(&environment, CProfile::Release),
        )
        .unwrap();
        assert_eq!(
            request.destination(),
            Some(
                directory
                    .join(format!("example-demo{}", std::env::consts::EXE_SUFFIX))
                    .as_path()
            )
        );
    }
    let output = temp.path().join("custom-program");
    let request = Request::new(
        input,
        Some(output.clone()),
        options(&environment, CProfile::Release),
    )
    .unwrap();
    assert_eq!(request.destination(), Some(output.as_path()));
    assert!(
        !temp.path().join("new").exists(),
        "construction must not build anything"
    );
}

#[test]
fn session_compilation_uses_overlays_and_explicit_profiles() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let mut environment = Environment::capture().unwrap();
    environment.directory = temp.path().into();
    // An unused shader compiler is optional, even for annotated host functions.
    environment
        .variables
        .insert("GLSLC".into(), "/missing/glslc".into());
    let source = temp.path().join("main.resin");
    fs::write(&source, "export { main }; def main() -> int = { 1 };").unwrap();
    let mut session = Session::new(environment.path("RESIN_STDLIB", resin_compiler::stdlib_path()));
    for (profile, destination, code, directory) in [
        (
            CProfile::Debug,
            Some(temp.path().join("copied")),
            42,
            "debug",
        ),
        (CProfile::Release, None, 43, "release"),
    ] {
        session.set_overlay(&source, format!("export {{ main }}; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = {{ var i = uint(invocation); output.* := i; }}; def main() -> int = {{ var output = 0_ui; kernel({code}_ul, &output); if (output == {code}_ui) {{ {code} }} else {{ 0 }} }};")).unwrap();
        let snapshot = session.analyze(&source).unwrap();
        let request = Request::new(
            input(&source),
            destination.clone(),
            options(&environment, profile),
        )
        .unwrap();
        let artifact = session.compile(&request).unwrap();
        assert_eq!(
            artifact.path().parent().unwrap().file_name().unwrap(),
            directory
        );
        assert_eq!(artifact.run().unwrap(), code);
        assert!(
            Arc::ptr_eq(&snapshot, &session.analyze(&source).unwrap()),
            "compilation should reuse the session snapshot"
        );
        if let Some(path) = destination {
            assert!(path.is_file());
            fs::remove_file(&source).unwrap();
        }
    }
    assert!(!environment.directory.join("build/shaders").exists());
}

#[test]
#[cfg(unix)]
fn compiler_processes_use_the_supplied_environment_and_working_directory() {
    use std::os::unix::fs::PermissionsExt;
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
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
    let expected = format!(
        "chosen:{}",
        fs::canonicalize(temp.path()).unwrap().display()
    );
    let c = settings
        .compile_c("", &temp.path().join("output"))
        .unwrap_err();
    let shader = settings
        .compile_glsl("", resin_codegen::Stage::Compute)
        .unwrap_err();
    for error in [c, shader] {
        assert!(error.to_string().contains(&expected), "{error}");
    }
}

#[test]
fn requests_validate_existing_output_ancestors() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let environment = Environment::capture().unwrap();
    let file = temp.path().join("file");
    fs::write(&file, "preserve").unwrap();
    for suffix in ["program", "missing/program", "../program"] {
        assert!(
            Request::new(
                input(&temp.path().join("source.resin")),
                Some(file.join(suffix)),
                options(&environment, CProfile::Debug)
            )
            .is_err()
        );
    }
    assert_eq!(fs::read_to_string(file).unwrap(), "preserve");
    assert!(
        Request::new(
            input(&temp.path().join("source.resin")),
            Some(temp.path().join("missing/nested/program")),
            options(&environment, CProfile::Debug)
        )
        .is_ok()
    );
    assert!(!temp.path().join("missing").exists());
}

#[cfg(unix)]
#[test]
fn output_ancestor_validation_follows_symlinks() {
    use std::os::unix::fs::symlink;
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let environment = Environment::capture().unwrap();
    fs::write(temp.path().join("file"), "preserve").unwrap();
    symlink("file", temp.path().join("file-link")).unwrap();
    symlink("missing", temp.path().join("dangling-link")).unwrap();
    assert!(
        Request::new(
            input(&temp.path().join("source.resin")),
            Some(temp.path().join("dangling-link/program")),
            options(&environment, CProfile::Debug)
        )
        .is_err()
    );
    symlink(".", temp.path().join("directory-link")).unwrap();
    assert!(
        Request::new(
            input(&temp.path().join("source.resin")),
            Some(temp.path().join("file-link/program")),
            options(&environment, CProfile::Debug)
        )
        .is_err()
    );
    assert!(
        Request::new(
            input(&temp.path().join("source.resin")),
            Some(temp.path().join("directory-link/missing/program")),
            options(&environment, CProfile::Debug)
        )
        .is_ok()
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("file")).unwrap(),
        "preserve"
    );
    assert!(!temp.path().join("missing").exists());
}
