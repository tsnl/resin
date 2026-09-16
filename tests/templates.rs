mod support;

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run()
}

#[test]
fn nested_template_calls_use_context_and_preserve_numeric_widths() {
    let output = run(r#"export { main };
        import { "$/string.resin" };
        fn identity<T>(value: T) -> _  { value }
        fn increment<T>(value: T) -> T  { value + 1 }
        fn twice<U>(value: U) -> U  { increment(increment(value)) }
        fn main() -> int  {
            let mut small: ubyte; small = identity(253);
            let mut large: ulong; large = identity(4294967296);
            print(fmt("{0} {1}", (twice(small), twice(large))));
            0
        }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"255 4294967298");
}

#[test]
fn template_fields_and_layout_follow_each_nominal_argument() {
    let output = run(r#"export { main };
        import { "$/string.resin" };
        struct Small { value: int; }
        struct Large { padding: ulong; value: uint; }
        fn read<T>(value: Ptr<T>) -> _  { value.value }
        fn measure<T>() -> ulong  { size_of(T) }
        fn main() -> int  {
            let mut small = Small { value = 42 };
            let mut large = Large { padding = 3, value = 7 };
            print(fmt("{0} {1} {2} {3}", (read(&small), read(&large), measure::<Small>(), measure::<Large>())));
            0
        }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"42 7 4 16");
}

#[test]
fn generic_identity_preserves_shared_ownership() {
    let output = run(r#"export { main };
        import { "$/shared.resin" };
        fn identity<T>(value: T) -> T  { value }
        fn main() -> (int | Err<_>)  {
            let mut copy = {
                let mut owner = arc_ptr_alloc::<int>(42)?;
                identity(owner)
            };
            (copy:get().*)
        }
    "#);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn generic_results_propagate_union_errors_and_match_payloads() {
    let output = run(r#"export { main };
        struct A {}
        struct B {}
        fn propagate<T, E>(input: (T | Err<E>)) -> (_ | Err<_>)  { (input?) }
        fn recover<T, E>(input: (T | Err<E>), fallback: T) -> T  {
            match (input) { T(value) => { value }, Err(error) => { fallback } }
        }
        fn combine<T, E, F>(a: (T | Err<E>), b: (T | Err<F>)) -> (T | Err<_>)  {
            a?; (b?)
        }
        fn main() -> int  {
            let mut first = (int | Err<A>)((7));
            let mut second = (int | Err<B>)(Err(B {}));
            recover(propagate(first), 0) + recover(combine(first, second), 35)
        }
    "#);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn expected_results_select_arguments_and_concrete_casts_run_after_substitution() {
    let output = run(r#"export { main };
        fn make<T>() -> T  { 41 }
        fn as_int<T>(input: T) -> int  { int(input) }
        fn choose<T>(condition: T) -> int  { if (condition) { 1 } else { 0 } }
        fn main() -> int  {
            let mut value: uint; value = make();
            as_int(value) + choose(1 == 1)
        }
    "#);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn hir(source: &str) -> resin_hir::Module {
    support::hir(source)
}

#[test]
fn concrete_template_errors_report_the_application_chain() {
    for source in [
        "fn add<T>(value: T) -> T  { value + value } fn relay<U>(value: U) -> U  { add(value) } fn main()  { relay(1 == 1); }",
        "fn byte<T>() -> T  { 256 } fn main() -> ubyte  { byte() }",
        "fn field<T>(value: T) -> int  { value.missing } fn main() -> int  { field(1) }",
        "fn choose<T>(condition: T) -> int  { if (condition) { 1 } else { 0 } } fn main() -> int  { choose(1) }",
        "fn choose<T, U>(value: T | U) -> int  { match (value) { T(a) => { 1 }, U(b) => { 2 } } } fn main() -> int  { choose::<int, int>(1) }",
        "fn remainder<T>(value: T) -> T  { value % value } fn main()  { remainder(1.5); }",
    ] {
        let tree = hir(source);
        let error = support::frontend::lower(&tree, &[], &resin_lir::LoweringOptions::default())
            .unwrap_err()
            .remove(0);
        assert!(!error.applications.is_empty(), "{error:?}");
    }
}

#[test]
fn generic_union_matches_use_concrete_tags_after_substitution() {
    let output = run(r#"export { main };
        fn choose<T, U>(value: T | U) -> int  {
            match (value) { T(a) => { 42 }, U(b) => { 0 } }
        }
        fn main() -> int  { choose::<int, bool>(1) }
    "#);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn source_recursion_memoizes_instances_and_respects_the_configured_limit() {
    let tree = hir("fn recur<T>(value: T)  { recur(value) } fn main()  { recur(1); }");
    let module =
        support::frontend::lower(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap();
    assert_eq!(module.functions.len(), 2);
    let source = resin_source::Source::new(
        "growing.resin",
        "export { main }; fn grow<T>(value: T)  { grow(&value) } fn main()  { grow(1); }",
    );
    let mut loader = resin_source::Loader::new(resin_source::library_root());
    let output = support::frontend::analyze(source, &mut loader, None);
    let options = resin_lir::LoweringOptions {
        max_monomorphs_per_function: std::num::NonZeroUsize::new(3).unwrap(),
    };
    let message = match support::pipeline::verified_lir_with_options(
        &output,
        "main",
        resin_lir::Profile::Host,
        &options,
    ) {
        Ok(_) => panic!("expected monomorph limit"),
        Err(errors) => errors
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    };
    assert!(message.contains("limit of 3 monomorphs"), "{message}");
    assert!(message.contains("grow"), "{message}");
}
