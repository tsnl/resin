mod support;
use support::{module, pipeline};

#[test]
fn error_wrappers_preserve_identity_payloads_and_ownership() {
    let module = module(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn boxed<T>(value: T) -> Err<T>  { Err(value) }
        fn main()  {
            let mut a: Err<i32> = Err(42);
            let mut b = Err(Err("nested"));
            let owned = string_from_str("owned");
            let c: i32 | Err<String> = boxed(owned:clone());
            let moved = c;
            { let borrowed = fmt("{0} {1} {2} {3}", (a, b, boxed(owned), moved)); print(borrowed) };
        }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout,
        b"Err(42) Err(Err(\"nested\")) Err(\"owned\") Err(\"owned\")"
    );
}

#[test]
fn error_wrappers_cannot_hide_reference_payloads() {
    let error = pipeline::source_module("struct Invalid { error: Err<Ref<i32>>, } fn main()  {}")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Ref") || error.contains("reference") || error.contains("value types"),
        "{error}"
    );
}

#[test]
fn error_wrappers_have_distinct_union_tags_and_payload_layouts() {
    let module = module(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn show(value: i32 | Err<i32>)  { match (value) {
            i32(value) => { { let borrowed = repr(value); print(borrowed) } },
            Err<i32>(error) => { { let borrowed = repr(error); print(borrowed) } },
        } }
        fn main()  { show(7); print(" "); show(Err(7)); }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"7 Err(7)");
}

#[test]
fn shaders_preserve_error_wrapper_identity() {
    let module = module(
        r#"export { kernel };
        fn error(value: i32) -> Err<i32>  { Err(value) }
        @compute_shader fn kernel(index: u64, output: Ptr<i32>)  {
            let mut value: i32 | Err<i32> = error(7);
            match (value) {
                i32(value) => { output.* = value; },
                Err<i32>(value) => { output.* = i32(value); },
            }
        }"#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

#[test]
fn plain_success_values_and_builtin_errors_propagate() {
    let module = module(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn read(fail: bool) -> i32 | Err<str>  { if (fail) { Err("bad input") } else { 42 } }
        fn work(fail: bool) -> i32 | Err<_>  { read(fail)? + 1 }
        fn main() -> i32 | Err<_>  {
            assert(work(false)? == 43);
            match (work(true)) {
                i32(_) => { Err("unexpected success") },
                Err(message) => { print(message); 0 },
            }
        }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"bad input");
}

#[test]
fn propagated_errors_widen_payloads_and_preserve_owned_values() {
    let module = module(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn failure() -> i32 | Err<String>  { Err(string_from_str("owned")) }
        fn wider() -> i32 | Err<str | String>  { failure()? }
        fn main() -> i32  { match (wider()) {
            i32(value) => { value },
            Err(error) => { { let borrowed = repr(error); print(borrowed) }; 0 },
        } }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"\"owned\"");
}

#[test]
fn entry_point_reports_propagated_error_values() {
    let module = module(
        r#"export { main };
        fn failure() -> i32 | Err<str>  { Err("broken") }
        fn main() -> i32 | Err<_>  { failure()? }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stderr, b"unhandled error: \"broken\"\n");
}

#[test]
fn inferred_errors_collect_across_plain_returns_and_propagation() {
    let module = module(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn a() -> i32 | Err<str>  { Err("text") }
        fn b() -> i32 | Err<i32>  { Err(7) }
        fn choose(flag: bool) -> i32 | Err<_>  { if (flag) { a() } else { b() } }
        fn pass(flag: bool) -> i32 | Err<_>  { choose(flag)? }
        fn main() -> i32  { match (pass(false)) {
            i32(value) => { value },
            Err(error) => { { let borrowed = repr(error); print(borrowed) }; 0 },
        } }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"7");
}

#[test]
fn question_mark_preserves_success_unions_and_destroys_exited_scopes() {
    let module = module(
        r#"export { main };
import { "$/shared.resin" };

        struct Resource { trace: Ptr<i32>, digit: i32,
            
        }
fn drop(self: Ref<Resource>)  { self.trace.* = self.trace.* * 10 + self.digit; }

        fn fail() -> i32 | Err<str>  { Err("failure") }
        fn work(trace: Ptr<i32>) -> i32 | Err<str>  {
            let mut first = Resource { trace = trace, digit = 1 };
            { let mut second = Resource { trace = trace, digit = 2 }; fail()?; };
            0
        }
        fn choice() -> i32 | str | Err<str>  { "value" }
        fn main() -> i32 | Err<_> {
            let trace_owner = arc_ptr_alloc(i32(0))?; let trace: Ref<_> = trace_owner:get().*;
            match (work(trace_owner:get())) { i32(_) => { assert(false); }, Err(_) => {} };
            assert(trace == 21);
            let mut value = choice()?;
            match (value) { i32(_) => { 1 }, str(_) => { 0 } }
        }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn one_error_pattern_handles_distinct_error_wrapper_members() {
    let module = module(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn choice() -> i32 | Err<str> | Err<i32>  { Err<i32>(7) }
        fn main()  { match (choice()) {
            Err(value) => { { let borrowed = repr(value); print(borrowed) } },
            i32(_) => {},
        } }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"7");
}

#[test]
fn shaders_propagate_and_widen_error_payloads() {
    let module = module(
        r#"export { kernel };
        fn failure() -> i32 | Err<i32>  { Err(7) }
        fn wider() -> i32 | Err<i32 | f32>  { failure()? }
        @compute_shader fn kernel(index: u64, output: Ptr<i32>)  {
            match (wider()) {
                i32(value) => { output.* = value; },
                Err(value) => { match (value) { i32(code) => { output.* = code; }, f32(_) => {} } },
            }
        }"#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

#[test]
fn propagation_checks_every_error_and_keeps_mutable_pointers_invariant() {
    for source in [
        "fn fail() -> i32 | Err<str>  { Err(\"x\") } fn wrong() -> i32  { fail()? }",
        "fn fail() -> i32 | Err<str>  { Err(\"x\") } fn wrong() -> i32 | Err<i32>  { fail()? }",
        "fn wrong(value: Ptr<Err<str>>) -> Ptr<Err<str | i32>>  { value }",
        "fn wrong(value: i32)  { match (value) { Err(_) => {} } }",
        "fn wrong(value: i32 | Err<str>)  { match (value) { Err(_) => {}, Err(_) => {}, i32(_) => {} } }",
    ] {
        assert!(pipeline::source_module(source).is_err(), "{source}");
    }
}

#[test]
fn error_only_paths_and_recursive_error_sets_complete() {
    let module = module(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn a(n: i32) -> i32 | Err<_>  { if (n == 0) { Err(i32(7)) } else { b(n - 1)? } }
        fn b(n: i32) -> i32 | Err<_>  { if (n == 0) { Err("text") } else { a(n - 1)? } }
        fn always() -> Err<str>  { Err("always") }
        fn only() -> Err<str>  { always()? }
        fn main()  {
            match (a(1)) { i32(_) => {}, Err(value) => { { let borrowed = repr(value); print(borrowed) } } };
            match (only()) { Err(value) => { print(value) } };
        }"#,
    );
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let output = project.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"\"text\"always");
}

#[test]
fn generic_functions_infer_plain_values_and_error_parameters() {
    let module = module(
        r#"export { main };
        fn propagate<T, E>(value: T | Err<E>) -> _ | Err<_>  { value? }
        fn recover<T, E>(value: T | Err<E>, fallback: T) -> T  {
            match (value) { T(value) => { value }, Err(_) => { fallback } }
        }
        fn combine<T, E, F>(a: T | Err<E>, b: T | Err<F>) -> T | Err<_>  { a?; b? }
        fn main()  {
            let mut first: i32 | Err<str> = 7;
            let mut second: i32 | Err<i32> = Err(9);
            assert(recover(propagate(first), i32(0)) == 7);
            assert(recover(combine(first, second), i32(35)) == 35);
        }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn propagation_and_union_conversions_preserve_success_inference() {
    let module = module(
        r#"export { main };
        fn number() -> i32 | Err<str>  { 42 }
        fn inferred() -> _  { number()?; i32(42) }
        fn main() -> () | Err<_>  {
            let mut explicit = (i32 | Err<str>)(7);
            assert((explicit?) == 7);
            assert(inferred()? == 42);
        }"#,
    );
    assert_eq!(
        module
            .functions
            .iter()
            .find(|function| function.name.as_deref() == Some("inferred"))
            .unwrap()
            .result,
        resin_types::Ty::union_of([
            resin_types::Ty::Int32,
            resin_types::Ty::Error {
                payload: Box::new(resin_types::Ty::Str)
            },
        ])
    );
    assert!(
        support::project::Project::new(&module, Some("main"))
            .unwrap()
            .run()
            .status
            .success()
    );
}
