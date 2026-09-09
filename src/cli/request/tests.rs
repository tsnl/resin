use super::Options;
use super::{Input, Request};
use resin_toolchain::CProfile;
use resin_toolchain::Environment;
use std::{fs, path::Path};
use tempfile::TempDir;

fn options(environment: &Environment, profile: CProfile) -> Options {
    Options {
        profile,
        temporary: environment.temporary.clone(),
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
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
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
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
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
            request.destination.as_deref(),
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
    assert_eq!(request.destination.as_deref(), Some(output.as_path()));
    assert!(
        !temp.path().join("new").exists(),
        "construction must not build anything"
    );
}

#[test]
fn requests_validate_existing_output_ancestors() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
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
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
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
