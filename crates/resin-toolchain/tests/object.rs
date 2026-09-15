use resin_executor::{Cancellation, Execution};
use resin_toolchain::{Environment, Toolchain};
use std::{ffi::OsStr, fs, path::Path, process::Command, sync::Arc};
use tempfile::TempDir;

fn tools(directory: &Path) -> Toolchain {
    let mut environment = Environment::capture().unwrap();
    environment.directory = directory.into();
    for name in [
        "NINJA",
        "SPIRV_OPT",
        "RESIN_RUNTIME_LIB",
        "RESIN_RUNTIME_INCLUDE",
    ] {
        environment
            .variables
            .insert(name.into(), "missing-and-unused".into());
    }
    environment.toolchain(None, None)
}

fn object(directory: &Path) -> Arc<[u8]> {
    let source = directory.join("fixture.c");
    let object = directory.join("fixture.o");
    fs::write(&source, "int main(void) { return 42; }\n").unwrap();
    let mut command = Command::new(
        std::env::var_os("CC").unwrap_or_else(|| resin_toolchain::DEFAULT_C_COMPILER.into()),
    );
    command.args([
        OsStr::new("-c"),
        source.as_os_str(),
        OsStr::new("-o"),
        object.as_os_str(),
    ]);
    #[cfg(target_env = "msvc")]
    command.arg("-fms-runtime-lib=dll");
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::read(object).unwrap().into()
}

#[tokio::test]
async fn object_linking_returns_a_retained_executable_without_other_native_tools() {
    let temporary = TempDir::new().unwrap();
    let tools = tools(temporary.path());
    let bytes = object(temporary.path());
    let generations = temporary.path().join("objects with spaces");
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let first = tools
        .link_object(bytes.clone(), &generations, &execution, &cancellation)
        .await
        .unwrap();
    let retained = first.clone();
    let first_path = first.path().to_owned();
    let second = tools
        .link_object(bytes, &generations, &execution, &cancellation)
        .await
        .unwrap();
    assert_ne!(first.path(), second.path());
    assert_eq!(first.run(&execution, &cancellation).await.unwrap(), 42);
    drop(first);
    assert!(first_path.is_file());
    assert_eq!(retained.run(&execution, &cancellation).await.unwrap(), 42);
    drop(retained);
    assert!(!first_path.parent().unwrap().exists());
    assert_eq!(second.run(&execution, &cancellation).await.unwrap(), 42);
    drop(second);
    assert_eq!(fs::read_dir(generations).unwrap().count(), 0);
}

#[tokio::test]
async fn cancelled_object_link_creates_no_generation() {
    let temporary = TempDir::new().unwrap();
    let generations = temporary.path().join("generations");
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let error = tools(temporary.path())
        .link_object(
            Arc::from(&b"unused"[..]),
            &generations,
            &Execution::default(),
            &cancellation,
        )
        .await
        .unwrap_err();
    assert!(error.is_cancelled());
    assert!(!generations.exists());
}

#[tokio::test]
async fn invalid_object_link_reports_the_linker_error_and_removes_its_generation() {
    let temporary = TempDir::new().unwrap();
    let generations = temporary.path().join("generations");
    let error = tools(temporary.path())
        .link_object(
            Arc::from(&b"this is not an object"[..]),
            &generations,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(!error.is_cancelled());
    assert!(
        error.to_string().contains("native object linking failed"),
        "{error}"
    );
    assert_eq!(fs::read_dir(generations).unwrap().count(), 0);
}
