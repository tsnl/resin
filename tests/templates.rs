mod support;

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run()
}

#[test]
fn nested_template_calls_use_context_and_preserve_numeric_widths() {
    let output = run(r#"
        export { main };
        import { "$/string.resin" };
        def identity<T>(value: T) -> _ = { value };
        def increment<T>(value: T) -> T = { value + 1 };
        def twice<U>(value: U) -> U = { increment(increment(value)) };
        def main() -> int = {
            var small: ubyte; small := identity(253);
            var large: ulong; large := identity(4294967296);
            print(fmt("{0} {1}", (twice(small), twice(large))));
            0
        };
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
    let output = run(r#"
        export { main };
        import { "$/string.resin" };
        struct Small { value: int };
        struct Large { padding: ulong, value: uint };
        def read<T>(value: Ptr<T>) -> _ = { value.value };
        def measure<T>() -> ulong = { size_of(T) };
        def main() -> int = {
            var small = Small { value = 42 };
            var large = Large { padding = 3, value = 7 };
            print(fmt("{0} {1} {2} {3}", (read(&small), read(&large), measure::<Small>(), measure::<Large>())));
            0
        };
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
    let output = run(r#"
        export { main };
        import { "$/shared.resin" };
        def identity<T>(value: T) -> T = { value };
        def main() -> Result<int, _> = {
            var copy = {
                var owner = ArcPtr<int>.alloc(42)?;
                identity(owner)
            };
            ok(copy.get().*)
        };
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
    let output = run(r#"
        export { main };
        struct A {};
        struct B {};
        def propagate<T, E>(input: Result<T, E>) -> Result<_, _> = { ok(input?) };
        def recover<T, E>(input: Result<T, E>, fallback: T) -> T = {
            match (input) { ok(value) => { value }, err(error) => { fallback } }
        };
        def combine<T, E, F>(a: Result<T, E>, b: Result<T, F>) -> Result<T, _> = {
            a?; ok(b?)
        };
        def main() -> int = {
            var first = Result<int, A>(ok(7));
            var second = Result<int, B>(err(B {}));
            recover(propagate(first), 0) + recover(combine(first, second), 35)
        };
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
    let output = run(r#"
        export { main };
        def make<T>() -> T = { 41 };
        def as_int<T>(input: T) -> int = { int(input) };
        def choose<T>(condition: T) -> int = { if (condition) { 1 } else { 0 } };
        def main() -> int = {
            var value: uint; value := make();
            as_int(value) + choose(1 == 1)
        };
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
        "def add<T>(value: T) -> T = { value + value }; def relay<U>(value: U) -> U = { add(value) }; def main() = { relay(1 == 1); };",
        "def byte<T>() -> T = { 256 }; def main() -> ubyte = { byte() };",
        "def field<T>(value: T) -> int = { value.missing }; def main() -> int = { field(1) };",
        "def choose<T>(condition: T) -> int = { if (condition) { 1 } else { 0 } }; def main() -> int = { choose(1) };",
        "def choose<T, U>(value: T | U) -> int = { match (value) { T(a) => { 1 }, U(b) => { 2 } } }; def main() -> int = { choose::<int, int>(1) };",
        "def failure<E>(value: E) -> Result<int, E> = { err(value) }; def main() = { failure(1); };",
    ] {
        let tree = hir(source);
        let error = resin_lir::build_lir(&tree, &[], &resin_lir::LoweringOptions::default())
            .unwrap_err()
            .remove(0);
        assert!(!error.applications.is_empty(), "{error:?}");
    }
}

#[test]
fn generic_union_matches_use_concrete_tags_after_substitution() {
    let output = run(r#"
        export { main };
        def choose<T, U>(value: T | U) -> int = {
            match (value) { T(a) => { 42 }, U(b) => { 0 } }
        };
        def main() -> int = { choose::<int, bool>(1) };
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
    let tree = hir("def recur<T>(value: T) = { recur(value) }; def main() = { recur(1); };");
    let module = resin_lir::build_lir(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap();
    assert_eq!(module.functions.len(), 2);
    let source = resin_source::Source::new(
        "growing.resin",
        "export { main }; def grow<T>(value: T) = { grow(&value) }; def main() = { grow(1); };",
    );
    let mut loader = resin_source::Loader::new(resin_source::library_root());
    let output = resin_hir::Hir::build(source, &mut loader, None);
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
