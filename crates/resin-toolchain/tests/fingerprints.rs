//! Content validation follows the native operation, without reading unrelated installations.
use resin_executor::{Cancellation, Execution};
use resin_toolchain::{Environment, NativeOperation, Toolchain};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

async fn identities(tools: &Toolchain) -> Vec<String> {
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let mut identities = vec![];
    for operation in [
        NativeOperation::Foreign,
        NativeOperation::Shader,
        NativeOperation::Link { runtime: false },
        NativeOperation::Link { runtime: true },
    ] {
        identities.push(
            tools
                .fingerprint(operation, &execution, &cancellation)
                .await
                .unwrap(),
        );
    }
    identities
}

fn change(path: &Path) {
    let modified = fs::metadata(path).unwrap().modified().unwrap();
    let mut bytes = fs::read(path).unwrap();
    bytes[0] ^= 1;
    fs::write(path, bytes).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(modified)
        .unwrap();
}

#[tokio::test]
async fn content_changes_invalidate_only_operations_that_use_the_changed_input() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    for name in ["headers", "libraries", "clanglib"] {
        fs::create_dir(root.join(name)).unwrap();
    }
    let library_name = if cfg!(windows) {
        "libclang.dll"
    } else if cfg!(target_os = "macos") {
        "libclang.dylib"
    } else {
        "libclang.so"
    };
    let library = PathBuf::from("clanglib").join(library_name);
    for name in [
        Path::new("cc"),
        Path::new("clang"),
        Path::new("spirv"),
        Path::new("ninja"),
        Path::new("resin"),
        Path::new("runtime.a"),
        Path::new("headers/api.h"),
        Path::new("libraries/sdk.a"),
        Path::new("clanglib/unused.a"),
        library.as_path(),
    ] {
        fs::write(root.join(name), b"original").unwrap();
    }
    let environment = Environment {
        variables: BTreeMap::from([
            ("CLANG".into(), root.join("clang").into_os_string()),
            ("SPIRV_OPT".into(), root.join("spirv").into_os_string()),
            ("NINJA".into(), root.join("ninja").into_os_string()),
            (
                "LIBCLANG_PATH".into(),
                root.join("clanglib").into_os_string(),
            ),
            (
                "RESIN_RUNTIME_LIB".into(),
                root.join("runtime.a").into_os_string(),
            ),
            (
                "RESIN_RUNTIME_INCLUDE".into(),
                root.join("headers").into_os_string(),
            ),
            (
                "LIBRARY_PATH".into(),
                root.join("libraries").into_os_string(),
            ),
        ]),
        directory: root.into(),
        executable: root.join("resin"),
        temporary: root.into(),
    };
    let tools = environment.toolchain(Some(root.join("cc").as_os_str()), None);
    let mut before = identities(&tools).await;
    for (path, changed) in [
        (Path::new("clanglib/unused.a"), [false, false, false, false]),
        (Path::new("ninja"), [false, false, false, false]),
        (Path::new("resin"), [false, false, false, false]),
        (Path::new("cc"), [false, false, true, true]),
        (Path::new("clang"), [true, false, false, false]),
        (library.as_path(), [true, false, false, false]),
        (Path::new("spirv"), [false, true, false, false]),
        (Path::new("runtime.a"), [false, false, false, true]),
        (Path::new("headers/api.h"), [true, false, false, false]),
        (Path::new("libraries/sdk.a"), [false, false, true, true]),
    ] {
        change(&root.join(path));
        let after = identities(&tools).await;
        for (index, expected) in changed.into_iter().enumerate() {
            assert_eq!(
                before[index] != after[index],
                expected,
                "{} operation {index}",
                path.display()
            );
        }
        before = after;
    }
}

#[tokio::test]
async fn linking_and_shader_keys_do_not_require_an_installed_c_frontend() {
    let directory = tempfile::tempdir().unwrap();
    let mut environment = Environment::capture().unwrap();
    environment.variables.insert(
        "CLANG".into(),
        directory.path().join("missing-clang").into_os_string(),
    );
    environment.variables.insert(
        "LIBCLANG_PATH".into(),
        directory.path().join("missing-libclang").into_os_string(),
    );
    let tools = environment.toolchain(None, None);
    for operation in [
        NativeOperation::Shader,
        NativeOperation::Link { runtime: false },
    ] {
        tools
            .fingerprint(operation, &Execution::default(), &Cancellation::new())
            .await
            .unwrap();
    }
}
