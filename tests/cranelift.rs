//! Execute the same verified source programs through C and the native prototype.
use resin_codegen::{NativeObject, NativeOptimization};
use resin_executor::{Cancellation, Execution};
use resin_source::{Source, SourceGraph};
use std::{collections::BTreeMap, num::NonZeroUsize, path::Path, process::Output, sync::Arc};
use tempfile::TempDir;

async fn checked(text: &str, execution: &Execution) -> Arc<resin_lir::VerifiedModule> {
    checked_profile(text, resin_lir::Profile::Host, execution).await
}

async fn checked_profile(
    text: &str,
    profile: resin_lir::Profile,
    execution: &Execution,
) -> Arc<resin_lir::VerifiedModule> {
    let cancellation = Cancellation::new();
    let source = Source::new("native-test.resin", text);
    let syntax = Arc::new(
        resin_cst::build_cst(text, None, execution, &cancellation)
            .await
            .unwrap(),
    );
    let ast = resin_ast::build_ast(syntax.clone(), execution, &cancellation)
        .await
        .unwrap();
    assert!(ast.errors.is_empty(), "{text}\n{:?}", ast.errors);
    let document = Arc::new(resin_ast::ModuleDocument {
        source: source.clone(),
        syntax,
        file: Arc::new(ast.file),
        errors: ast.errors,
    });
    let graph = SourceGraph::new(source.clone(), [source.clone()], []).unwrap();
    let program = resin_ast::build_program(
        graph,
        BTreeMap::from([(source, document)]),
        execution,
        &cancellation,
    )
    .await
    .unwrap();
    let hir = resin_hir::Hir::build(Arc::new(program), execution, &cancellation)
        .await
        .unwrap();
    let hir = hir.hir().unwrap_or_else(|error| panic!("{text}\n{error}"));
    let entry =
        resin_lir::Entry::exported(hir, "main", profile).expect("test exports its main entry");
    let lir = resin_lir::build_lir(
        hir.clone(),
        vec![entry],
        Default::default(),
        execution,
        &cancellation,
    )
    .await
    .unwrap();
    Arc::new(
        resin_lir::VerifiedModule::build(lir, execution, &cancellation)
            .await
            .unwrap(),
    )
}

fn tools(directory: &Path) -> resin_toolchain::Toolchain {
    let mut environment = resin_toolchain::Environment::capture().unwrap();
    environment.directory = directory.to_owned();
    environment.temporary = directory.to_owned();
    environment.executable = env!("CARGO_BIN_EXE_resin").into();
    environment.toolchain(None, None)
}

async fn output(executable: &resin_toolchain::Executable) -> Output {
    tokio::process::Command::new(executable.path())
        .output()
        .await
        .unwrap()
}

async fn c_executable(
    checked: Arc<resin_lir::VerifiedModule>,
    directory: &Path,
    tools: &resin_toolchain::Toolchain,
    execution: &Execution,
) -> resin_toolchain::Executable {
    let cancellation = Cancellation::new();
    let generated = resin_codegen::generate(
        checked,
        Some("main".into()),
        Arc::new(resin_codegen::NativeHeaders::default()),
        directory,
        execution,
        &cancellation,
    )
    .await
    .unwrap();
    let built = tools
        .build(
            generated.directory(),
            generated.name(),
            "main",
            resin_toolchain::CProfile::Release,
            execution,
            &cancellation,
        )
        .await
        .unwrap();
    built
        .executable(
            generated
                .program()
                .unwrap()
                .strip_prefix(generated.directory())
                .unwrap(),
        )
        .unwrap()
}

async fn native_executable(
    checked: Arc<resin_lir::VerifiedModule>,
    optimization: NativeOptimization,
    directory: &Path,
    tools: &resin_toolchain::Toolchain,
    execution: &Execution,
) -> resin_toolchain::Executable {
    let cancellation = Cancellation::new();
    let object = resin_codegen::generate_native(
        checked,
        "main".into(),
        optimization,
        execution,
        &cancellation,
    )
    .await
    .unwrap();
    tools
        .link_object(
            Arc::from(object.bytes()),
            directory,
            execution,
            &cancellation,
        )
        .await
        .unwrap()
}

async fn equivalent(text: &str, expected: i32) {
    let execution = Execution::new(NonZeroUsize::new(2).unwrap());
    let checked = checked(text, &execution).await;
    let directory = TempDir::new().unwrap();
    let tools = tools(directory.path());
    let reference =
        output(&c_executable(checked.clone(), directory.path(), &tools, &execution).await).await;
    assert_eq!(
        reference.status.code(),
        Some(expected),
        "C: {text}\n{}",
        String::from_utf8_lossy(&reference.stderr)
    );
    for optimization in [NativeOptimization::None, NativeOptimization::Speed] {
        let native = native_executable(
            checked.clone(),
            optimization,
            directory.path(),
            &tools,
            &execution,
        )
        .await;
        let native = output(&native).await;
        assert_eq!(
            native.status.code(),
            Some(expected),
            "{optimization:?}: {text}\n{}",
            String::from_utf8_lossy(&native.stderr)
        );
        assert_eq!(native.stdout, reference.stdout);
        assert_eq!(native.stderr, reference.stderr);
    }
}

#[tokio::test]
async fn unit_and_boolean_calls_preserve_short_circuit_effects() {
    equivalent(
        "export { main }; def nothing(value: ()) -> () = { value }; def main() = { nothing(()); };",
        0,
    )
    .await;
    equivalent(
        r#"export { main };
        def invert(value: bool) -> bool = { !value };
        def main() -> int = {
            var changes = 0_i;
            var left = (1 == 2) && ((changes := 9) == 9);
            var right = (1 == 1) || ((changes := 8) == 8);
            if (invert(left) && right && changes == 0) { 17 } else { 1 }
        };"#,
        17,
    )
    .await;
}

#[tokio::test]
async fn integer_widths_wrap_and_keep_signed_division_remainder_and_shifts() {
    equivalent(r#"export { main }; def main() -> int = {
        var a = 127_b + 1_b; var b = 255_ub + 2_ub;
        var c = 32767_h + 1_h; var d = 65535_uh + 2_uh;
        var e = 2147483647_i + 1_i; var f = 4294967295_ui * 4294967295_ui;
        var g = 9223372036854775807_l + 1_l; var h = 18446744073709551615_ul + 2_ul;
        if (a == -128_b && b == 1_ub && c == -32768_h && d == 1_uh &&
            e == -2147483648_i && f == 1_ui && g == -9223372036854775808_l && h == 1_ul &&
            g / -1_l == g && g % -1_l == 0_l && -17_i / 5_i == -3_i && -17_i % 5_i == -2_i &&
            (-8_i >> 2_i) == -2_i && (0xffffffff_ui >> 31_ui) == 1_ui && (1_ul << 40_ul) == 1099511627776_ul) { 23 } else { 1 }
    };"#, 23).await;
}

#[tokio::test]
async fn floating_operations_and_conversions_match_c_including_nan_and_narrowing() {
    equivalent(r#"export { main };
        def mix(a: float32, b: float64) -> float64 = { float64(a * 2_f - 0.5_f) + b / 2_d };
        def main() -> int = {
            var nan = 0_d / 0_d;
            var tiny = float32(-1e-40_d);
            var wide = 18446744073709551615_ul;
            if (mix(1.5_f, 5_d) == 5_d && int(-1.9_f) == -1_i &&
                float32(16777217_ui) == 16777216_f && float64(255_ub) == 255_d &&
                float32(1e100_d) > 1e30_f && tiny == 0_f &&
                nan != nan && !(nan < 0_d) && !(nan >= 0_d) && float64(wide) > 1e19_d) { 29 } else { 1 }
        };"#, 29).await;
}

#[tokio::test]
async fn pointers_and_reference_calls_alias_local_storage_without_copying_it() {
    equivalent(
        r#"export { main };
        def identity<T>(value: Ref<T>) -> Ref<T> = { value };
        def observe(left: Ref<int>, right: Ptr<int>) -> int = { left := 20; right.* };
        def main() -> int = {
            var value = 1_i; var alias: Ref<int> = identity(value);
            var old = alias; var address = &alias;
            identity(value) := 7;
            var roundtrip = Ptr<int>(ulong(address));
            observe(alias, roundtrip) + old + value
        };"#,
        41,
    )
    .await;
}

#[tokio::test]
async fn indirect_function_values_and_arguments_preserve_evaluation_order() {
    equivalent(
        r#"export { main };
        type Operation = (int) -> int;
        def add(value: int) -> int = { value + 2 };
        def multiply(value: int) -> int = { value * 2 };
        def choose(counter: Ref<int>) -> Operation = { counter := counter * 10 + 1; add };
        def argument(counter: Ref<int>) -> int = { counter := counter * 10 + 2; 20 };
        def apply(operation: Operation, value: int) -> int = { operation(value) };
        def main() -> int = {
            var counter = 0_i;
            var first = choose(counter)(argument(counter));
            var operation = if (counter == 12) { multiply } else { add };
            first + apply(operation, 10)
        };"#,
        42,
    )
    .await;
}

#[tokio::test]
async fn nested_loops_branch_values_and_returns_preserve_mutable_state() {
    equivalent(
        r#"export { main };
        def sum(limit: int) -> int = {
            var outer = 0_i; var total = 0_i;
            while ((outer := outer + 1) <= limit) {
                var inner = 0_i;
                while (inner < outer) {
                    total := total + if (inner == 0) { 3 } else { 2 };
                    inner := inner + 1;
                };
            };
            if (limit == 0) { outer } else { total + outer }
        };
        def main() -> int = { sum(0) + sum(3) };
    "#,
        20,
    )
    .await;
}

#[tokio::test]
async fn direct_recursion_and_generic_instances_keep_their_scalar_abis() {
    equivalent(r#"export { main };
        def identity<T>(value: T) -> T = { value };
        def next<T>(value: T) -> T = { value + 1 };
        def factorial(value: int) -> int = { if (value < 2) { 1 } else { value * factorial(value - 1) } };
        def even(value: int) -> bool = { if (value == 0) { 1 == 1 } else { odd(value - 1) } };
        def odd(value: int) -> bool = { if (value == 0) { 1 == 0 } else { even(value - 1) } };
        def main() -> int = {
            identity(());
            if (even(6) && !odd(6)) {
                factorial(identity(3_i)) + int(next(identity(7_ub))) + int(next(identity(3_f)))
            } else { 1 }
        };"#, 18).await;
}

#[tokio::test]
async fn invalid_integer_operations_fail_instead_of_silently_returning_a_value() {
    let execution = Execution::new(NonZeroUsize::new(2).unwrap());
    let directory = TempDir::new().unwrap();
    let tools = tools(directory.path());
    for expression in ["1_i / 0_i", "1_i % 0_i", "1_i << 32_i", "1_i >> -1_i"] {
        let checked = checked(
            &format!("export {{ main }}; def main() -> int = {{ {expression} }};"),
            &execution,
        )
        .await;
        assert!(
            !output(&c_executable(checked.clone(), directory.path(), &tools, &execution).await)
                .await
                .status
                .success()
        );
        for optimization in [NativeOptimization::None, NativeOptimization::Speed] {
            let native = native_executable(
                checked.clone(),
                optimization,
                directory.path(),
                &tools,
                &execution,
            )
            .await;
            assert!(
                !output(&native).await.status.success(),
                "{optimization:?}: {expression}"
            );
        }
    }
}

#[tokio::test]
async fn aggregates_owners_foreign_headers_and_shader_entries_fail_explicitly() {
    let execution = Execution::new(NonZeroUsize::new(2).unwrap());
    let sources = [
        "export { main }; def main() -> int = { var pair = (1_i, 2_i); pair.0 + pair.1 };",
        "export { main }; struct Cell { value: int }; def main() -> int = { var cell = Cell { value = 3 }; cell.value };",
        "export { main }; def main() -> int = { var values = [1_i, 2_i]; values.at(0_ul) };",
        "export { main }; intrinsic \"owner_allocate\" def allocate<T>(count: ulong, initial: T) -> StrongOwner | None; def main() -> int = { var owner = allocate(1_ul, 7_i); 0 };",
        "export { main }; extern { \"stdio.h\": {} }; def main() -> int = { 0 };",
        "export { main }; extern { \"stdlib.h\": { def abs(value: int) -> int; } }; def main() -> int = { abs(-3) };",
    ];
    for source in sources {
        let checked = checked(source, &execution).await;
        for optimization in [NativeOptimization::None, NativeOptimization::Speed] {
            let error = resin_codegen::generate_native(
                checked.clone(),
                "main".into(),
                optimization,
                &execution,
                &Cancellation::new(),
            )
            .await
            .unwrap_err();
            assert!(
                matches!(error, resin_codegen::GenerationError::Codegen { .. }),
                "{error}"
            );
            assert!(
                error.to_string().to_lowercase().contains("cranelift"),
                "{source}\n{error}"
            );
        }
    }
    let shader = checked_profile("export { main }; @compute_shader def main(index: ulong, value: Ptr<int>) = { value.* := int(index); };", resin_lir::Profile::Shader, &execution).await;
    assert!(matches!(
        resin_codegen::generate_native(
            shader,
            "main".into(),
            NativeOptimization::None,
            &execution,
            &Cancellation::new()
        )
        .await,
        Err(resin_codegen::GenerationError::Codegen { .. })
    ));
}

#[tokio::test]
async fn independent_concurrent_objects_and_executables_keep_their_own_lifetimes() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<NativeObject>();
    send_sync::<resin_lir::VerifiedModule>();
    let execution = Execution::new(NonZeroUsize::new(2).unwrap());
    let first = checked("export { main }; def main() -> int = { 17 };", &execution).await;
    let second = checked("export { main }; def main() -> int = { 29 };", &execution).await;
    let mut tasks = tokio::task::JoinSet::new();
    for (checked, optimization, expected) in [
        (first.clone(), NativeOptimization::None, 17),
        (second, NativeOptimization::Speed, 29),
        (first, NativeOptimization::Speed, 17),
    ] {
        let execution = execution.clone();
        tasks.spawn(async move {
            (
                expected,
                resin_codegen::generate_native(
                    checked,
                    "main".into(),
                    optimization,
                    &execution,
                    &Cancellation::new(),
                )
                .await
                .unwrap(),
            )
        });
    }
    let output_root = TempDir::new().unwrap();
    let tools = tools(output_root.path());
    let mut executables = Vec::new();
    while let Some(result) = tasks.join_next().await {
        let (expected, object) = result.unwrap();
        let retained = object.clone();
        drop(object);
        let executable = tools
            .link_object(
                Arc::from(retained.bytes()),
                output_root.path(),
                &execution,
                &Cancellation::new(),
            )
            .await
            .unwrap();
        drop(retained);
        executables.push((expected, executable));
    }
    for (expected, executable) in executables {
        let path = executable.path().to_owned();
        let retained = executable.clone();
        drop(executable);
        assert_eq!(output(&retained).await.status.code(), Some(expected));
        assert!(path.exists());
        drop(retained);
        assert!(
            !path.exists(),
            "final native artifact owner must clean its generation"
        );
    }
}

#[tokio::test]
async fn queued_cancellation_returns_no_object_and_releases_execution_capacity() {
    let execution = Execution::new(NonZeroUsize::new(1).unwrap());
    let checked = checked("export { main }; def main() -> int = { 7 };", &execution).await;
    let permit = execution.acquire(&Cancellation::new()).await.unwrap();
    let cancellation = Cancellation::new();
    let mut generation = Box::pin(resin_codegen::generate_native(
        checked.clone(),
        "main".into(),
        NativeOptimization::Speed,
        &execution,
        &cancellation,
    ));
    assert!(futures::poll!(&mut generation).is_pending());
    cancellation.cancel();
    let error = tokio::time::timeout(std::time::Duration::from_secs(5), generation)
        .await
        .unwrap()
        .unwrap_err();
    assert!(matches!(
        error,
        resin_codegen::GenerationError::Execution {
            error: resin_executor::Error::Cancelled
        }
    ));
    drop(permit);
    let fresh = resin_codegen::generate_native(
        checked,
        "main".into(),
        NativeOptimization::None,
        &execution,
        &Cancellation::new(),
    )
    .await
    .unwrap();
    assert!(!fresh.bytes().is_empty());
    execution.wait_idle().await;
}
