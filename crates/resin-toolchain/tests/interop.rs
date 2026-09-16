use resin_executor::{Cancellation, Execution};
use resin_toolchain::{
    Environment, ForeignAnalysis, ForeignFunction, ForeignInputs, ForeignScalar, Toolchain,
};
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

fn function(name: &str) -> ForeignFunction {
    ForeignFunction {
        name: name.into(),
        params: vec![int()],
        result: int(),
    }
}

fn inputs(header: &str) -> ForeignInputs {
    ForeignInputs {
        files: BTreeMap::from([
            ("bundle/api.h".into(), Arc::from(header.as_bytes())),
            (
                "bundle/detail/types.h".into(),
                Arc::from(&b"typedef int Value;\n"[..]),
            ),
        ]),
        includes: vec!["bundle/api.h".into()],
        include_directories: vec!["bundle".into()],
        functions: vec![function("identity")],
    }
}

async fn analyze(
    inputs: ForeignInputs,
    root: &Path,
) -> Result<ForeignAnalysis, resin_toolchain::Error> {
    tools(root)
        .analyze_foreign(
            Arc::new(inputs),
            root,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
}

#[tokio::test]
async fn external_prototypes_retain_owned_metadata_after_inputs_are_removed() {
    let root = TempDir::new().unwrap();
    let analysis = analyze(
        inputs("#include <detail/types.h>\nextern Value identity(Value);\n"),
        root.path(),
    )
    .await
    .unwrap();
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    assert!(
        analysis
            .includes()
            .iter()
            .any(|path| path.ends_with("detail/types.h"))
    );
    let retained = analysis.clone();
    drop(analysis);
    drop(root);
    assert_eq!(retained.declarations()[0].name, "identity");
    assert_eq!(retained.declarations()[0].symbol, "identity");
    assert_eq!(retained.declarations()[0].c_signature, "int (int)");
}

#[tokio::test]
async fn analysis_never_assembles_header_code_or_produces_a_c_object() {
    let root = TempDir::new().unwrap();
    // Parsing this TU is valid; emitting any object from it fails in the assembler.
    let request = inputs(
        "extern int identity(int);\n__asm__(\".error \\\"C object emission forbidden\\\"\");\n",
    );
    let analysis = analyze(request, root.path()).await.unwrap();
    assert_eq!(analysis.declarations()[0].symbol, "identity");
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn prototypes_remain_linkable_when_headers_also_provide_bodies_or_macros() {
    let root = TempDir::new().unwrap();
    for header in [
        "extern int identity(int);\ninline int identity(int n) { return n; }\n",
        "extern int identity(int);\nint identity(int n) { return n; }\n",
        "extern int identity(int);\n#define identity(n) ((n) + 1)\n",
    ] {
        let analysis = analyze(inputs(header), root.path()).await.unwrap();
        assert_eq!(analysis.declarations()[0].symbol, "identity");
    }
}

#[tokio::test]
async fn macro_only_missing_static_and_header_only_functions_are_rejected() {
    let root = TempDir::new().unwrap();
    for (header, expected) in [
        ("extern int another(int);", "no C function declaration"),
        ("#define identity(n) (n)", "macro-only"),
        (
            "static int identity(int n) { return n; }",
            "external non-inline",
        ),
        (
            "static inline int identity(int n) { return n; }",
            "external non-inline",
        ),
        (
            "inline int identity(int n) { return n; }",
            "external non-inline",
        ),
        (
            "extern inline int identity(int n) { return n; }",
            "external non-inline",
        ),
        ("inline int identity(int);", "external non-inline"),
        ("int identity(int n) { return n; }", "external non-inline"),
    ] {
        let error = analyze(inputs(header), root.path()).await.unwrap_err();
        assert!(error.to_string().contains(expected), "{header}: {error}");
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }
}

#[tokio::test]
async fn declarations_require_exact_scalar_width_signedness_arity_and_result() {
    let root = TempDir::new().unwrap();
    for header in [
        "extern int identity(short);",
        "extern int identity(unsigned int);",
        "extern int identity(float);",
        "extern int identity(_Bool);",
        "extern int identity(void *);",
        "extern int identity(int, int);",
        "extern int identity(int, ...);",
        "extern int identity();",
        "extern void identity(int);",
        "extern unsigned int identity(int);",
        "extern long long identity(int);",
    ] {
        let error = analyze(inputs(header), root.path()).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not match the requested scalar ABI"),
            "{header}: {error}"
        );
    }
}

#[tokio::test]
async fn bool_pointer_float_and_narrow_integer_declarations_are_validated() {
    let root = TempDir::new().unwrap();
    let mut request = inputs(
        "extern _Bool identity(const char *, signed char, unsigned short, float, double);\n",
    );
    request.functions[0].params = vec![
        ForeignScalar::Pointer,
        ForeignScalar::Integer {
            bits: 8,
            signed: true,
        },
        ForeignScalar::Integer {
            bits: 16,
            signed: false,
        },
        ForeignScalar::Float { bits: 32 },
        ForeignScalar::Float { bits: 64 },
    ];
    request.functions[0].result = ForeignScalar::Bool;
    assert_eq!(
        analyze(request, root.path()).await.unwrap().declarations()[0].symbol,
        "identity"
    );
}

#[tokio::test]
async fn enum_declarations_use_their_underlying_integer_abi() {
    let root = TempDir::new().unwrap();
    let request = inputs(
        "enum SignedValue { negative = -1, positive = 1 };\nextern enum SignedValue identity(enum SignedValue);\n",
    );
    assert_eq!(
        analyze(request, root.path()).await.unwrap().declarations()[0].symbol,
        "identity"
    );
    let error = analyze(
        inputs("enum WideValue { wide = 0x100000000ULL };\nextern int identity(enum WideValue);\n"),
        root.path(),
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not match the requested scalar ABI"),
        "{error}"
    );
}

#[tokio::test]
async fn changed_headers_produce_independent_analysis_without_mutating_prior_metadata() {
    let root = TempDir::new().unwrap();
    let old = analyze(inputs("extern int identity(int);"), root.path())
        .await
        .unwrap();
    let mut request = inputs("extern double identity(int);");
    request.functions[0].result = ForeignScalar::Float { bits: 64 };
    let new = analyze(request, root.path()).await.unwrap();
    assert_eq!(old.declarations()[0].c_signature, "int (int)");
    assert_eq!(new.declarations()[0].c_signature, "double (int)");
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn explicit_assembler_names_resolve_to_native_import_names() {
    let root = TempDir::new().unwrap();
    let label = if cfg!(target_os = "macos") {
        "_actual_symbol"
    } else {
        "actual_symbol"
    };
    // The second prototype carries the asm label, after the ordinary declaration.
    let header =
        format!("extern int identity(int);\nextern int identity(int) __asm__(\"{label}\");\n");
    let result = analyze(inputs(&header), root.path()).await.unwrap();
    assert_eq!(result.declarations()[0].name, "identity");
    assert_eq!(result.declarations()[0].symbol, "actual_symbol");
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn raw_assembler_names_without_macho_prefix_are_rejected() {
    let root = TempDir::new().unwrap();
    let error = analyze(
        inputs("extern int identity(int) __asm__(\"raw_symbol\");"),
        root.path(),
    )
    .await
    .unwrap_err();
    assert!(
        error.to_string().contains("Mach-O underscore prefix"),
        "{error}"
    );
}

#[cfg(target_arch = "x86_64")]
#[tokio::test]
async fn nonplatform_calling_conventions_are_rejected() {
    let root = TempDir::new().unwrap();
    let convention = if cfg!(windows) { "sysv_abi" } else { "ms_abi" };
    let header = format!("extern int identity(int) __attribute__(({convention}));\n");
    let error = analyze(inputs(&header), root.path()).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not match the requested scalar ABI"),
        "{error}"
    );
}

#[tokio::test]
async fn header_diagnostics_fail_before_analysis_is_published() {
    let root = TempDir::new().unwrap();
    let error = analyze(
        inputs("#error malformed header\nextern int identity(int);"),
        root.path(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("malformed header"), "{error}");
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn malformed_paths_and_cancelled_requests_create_no_outputs() {
    let root = TempDir::new().unwrap();
    let tools = tools(root.path());
    let execution = Execution::default();
    let mut invalid = inputs("extern int identity(int);");
    invalid
        .files
        .insert("../escape.h".into(), Arc::from(&b""[..]));
    let error = tools
        .analyze_foreign(
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
        .analyze_foreign(
            Arc::new(inputs("extern int identity(int);")),
            root.path(),
            &execution,
            &cancellation,
        )
        .await
        .unwrap_err();
    assert!(error.is_cancelled());
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn portable_path_aliases_and_analysis_outputs_are_rejected_before_staging() {
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
        vec!["foreign.d"],
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
            .analyze_foreign(
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
