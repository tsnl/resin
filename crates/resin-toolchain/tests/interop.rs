use resin_executor::{Cancellation, Execution};
use resin_toolchain::{Environment, ForeignFunction, ForeignInputs, ForeignScalar, Toolchain};
use std::{collections::BTreeMap, fs, path::Path, sync::Arc};
use tempfile::TempDir;

fn tools(root: &Path) -> Toolchain {
    let mut environment = Environment::capture().unwrap();
    environment.directory = root.into();
    environment
        .variables
        .insert("NINJA".into(), "missing-and-unused".into());
    environment.toolchain(None, None)
}

fn int() -> ForeignScalar {
    ForeignScalar::Integer {
        bits: 32,
        signed: true,
    }
}

fn inputs() -> ForeignInputs {
    ForeignInputs {
        files: BTreeMap::from([
            ("bundle/api.h".into(), Arc::from(&b"#include <detail/value.h>\nstatic inline int plus(int n) { return n + VALUE; }\n#define answer() plus(40)\n"[..])),
            ("bundle/detail/value.h".into(), Arc::from(&b"#define VALUE 2\n"[..])),
        ]),
        includes: vec!["bundle/api.h".into()],
        include_directories: vec!["bundle".into()],
        functions: vec![
            ForeignFunction { symbol: "main".into(), name: "answer".into(), params: vec![], result: int() },
            ForeignFunction { symbol: "resin_plus".into(), name: "plus".into(), params: vec![int()], result: int() },
        ],
    }
}

#[tokio::test]
async fn inline_and_macro_adapters_retain_owned_metadata_and_link_without_ninja() {
    let root = TempDir::new().unwrap();
    let temporary = root.path().join("foreign");
    let tools = tools(root.path());
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let object = tools
        .compile_foreign(Arc::new(inputs()), &temporary, &execution, &cancellation)
        .await
        .unwrap();
    assert_eq!(fs::read_dir(&temporary).unwrap().count(), 0);
    assert!(
        object.declarations()[0].c_signature.is_none(),
        "a function-like macro has no function declaration"
    );
    assert_eq!(
        object.declarations()[1].c_signature.as_deref(),
        Some("int (int)")
    );
    assert!(object.declarations()[1].inline);
    assert!(
        object
            .includes()
            .iter()
            .any(|path| path.ends_with("detail/value.h"))
    );
    let retained = object.clone();
    drop(object);
    let executable = tools
        .link_object(retained.bytes(), root.path(), &execution, &cancellation)
        .await
        .unwrap();
    drop(retained);
    assert_eq!(executable.run(&execution, &cancellation).await.unwrap(), 42);
}

#[tokio::test]
async fn header_diagnostics_fail_before_an_object_is_published() {
    let root = TempDir::new().unwrap();
    let mut inputs = inputs();
    inputs.functions[0].name = "missing_function".into();
    let error = tools(root.path())
        .compile_foreign(
            Arc::new(inputs),
            root.path(),
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("missing_function"), "{error}");
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn changed_header_bytes_produce_independent_adapters() {
    let root = TempDir::new().unwrap();
    let tools = tools(root.path());
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let old = tools
        .compile_foreign(Arc::new(inputs()), root.path(), &execution, &cancellation)
        .await
        .unwrap();
    let mut changed = inputs();
    changed.files.insert(
        "bundle/detail/value.h".into(),
        Arc::from(&b"#define VALUE 3\n"[..]),
    );
    let new = tools
        .compile_foreign(Arc::new(changed), root.path(), &execution, &cancellation)
        .await
        .unwrap();
    assert_ne!(old.bytes(), new.bytes());
    for (object, expected) in [(old, 42), (new, 43)] {
        let executable = tools
            .link_object(object.bytes(), root.path(), &execution, &cancellation)
            .await
            .unwrap();
        assert_eq!(
            executable.run(&execution, &cancellation).await.unwrap(),
            expected
        );
    }
}

#[tokio::test]
async fn malformed_paths_and_cancelled_requests_create_no_outputs() {
    let root = TempDir::new().unwrap();
    let tools = tools(root.path());
    let execution = Execution::default();
    let mut invalid = inputs();
    invalid
        .files
        .insert("../escape.h".into(), Arc::from(&b""[..]));
    let error = tools
        .compile_foreign(
            Arc::new(invalid),
            root.path(),
            &execution,
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("relative paths"));
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let error = tools
        .compile_foreign(Arc::new(inputs()), root.path(), &execution, &cancellation)
        .await
        .unwrap_err();
    assert!(error.is_cancelled());
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn portable_path_aliases_and_adapter_outputs_are_rejected_before_staging() {
    let root = TempDir::new().unwrap();
    let tools = tools(root.path());
    let execution = Execution::default();
    for names in [
        vec!["../escape.h"],
        vec!["native/../escape.h"],
        vec!["native/C:/escape.h"],
        vec!["native/NUL.h"],
        vec!["native/folder/CoM9.data"],
        vec!["foreign.c"],
        vec!["foreign.i"],
        vec!["FOREIGN.C"],
        vec!["left.h", "LEFT.h"],
        vec!["A/x.h", "a/y.h"],
    ] {
        let files = names
            .iter()
            .map(|name| (Arc::from(*name), Arc::from(&b""[..])))
            .collect();
        let invalid = ForeignInputs {
            files,
            ..Default::default()
        };
        let error = tools
            .compile_foreign(
                Arc::new(invalid),
                root.path(),
                &execution,
                &Cancellation::new(),
            )
            .await;
        assert!(error.is_err(), "accepted {names:?}");
        assert_eq!(
            fs::read_dir(root.path()).unwrap().count(),
            0,
            "staged {names:?}"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[tokio::test]
async fn header_macros_cannot_change_the_adapters_platform_calling_convention() {
    let root = TempDir::new().unwrap();
    let convention = if cfg!(windows) { "sysv_abi" } else { "ms_abi" };
    let header = format!(
        "static inline int identity(int value) {{ return value; }}\n#define resin_adapter __attribute__(({convention})) resin_adapter\n"
    );
    let inputs = ForeignInputs {
        files: BTreeMap::from([("api.h".into(), header.into_bytes().into())]),
        includes: vec!["api.h".into()],
        functions: vec![ForeignFunction {
            symbol: "resin_adapter".into(),
            name: "identity".into(),
            params: vec![int()],
            result: int(),
        }],
        ..Default::default()
    };
    let error = tools(root.path())
        .compile_foreign(
            Arc::new(inputs),
            root.path(),
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("changed the scalar ABI"),
        "{error}"
    );
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}
