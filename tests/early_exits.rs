mod support;

fn succeeds(source: &str) {
    let module = support::module(source);
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
fn returns_leave_nested_expressions_and_preserve_initialization() {
    succeeds(
        r#"export { main };
        def choose(flag: bool) -> int = {
            var value: int;
            if (flag) { return 42; } else { value := 7; };
            return value;
        };
        def both(flag: bool) -> int = { if (flag) { return 3; } else { return 4; }; };
        def nested() -> int = { 1 + { return 9; 2 } };
        def loop_return() -> int = { while (true) { return 5; }; 6 };
        def main() = {
            assert(choose(true) == 42); assert(choose(false) == 7);
            assert(both(true) == 3); assert(both(false) == 4);
            assert(nested() == 9); assert(loop_return() == 5);
            return;
            assert(false);
        };"#,
    );
}

#[test]
fn returns_drop_owners_once_in_reverse_order_and_preserve_the_result() {
    succeeds(
        r#"export { main };
        struct Resource { trace: Ptr<int>, digit: int,
            def drop(self: Ptr<Resource>) = { self.trace.* := self.trace.* * 10 + self.digit; };
        };
        def leave(trace: Ptr<int>) -> int = {
            var first = Resource { trace = trace, digit = 1 };
            { var second = Resource { trace = trace, digit = 2 }; return 42; };
        };
        def main() = { var trace = 0_i; assert(leave(&trace) == 42); assert(trace == 21); };"#,
    );
}

#[test]
fn returning_arms_do_not_participate_in_match_initialization_joins() {
    succeeds(
        r#"export { main };
        def choose(value: int | None) -> int = {
            var result: int;
            match (value) { int(number) => { result := number; }, None => { return 7; } };
            result
        };
        def main() = { assert(choose(42_i) == 42); assert(choose(None) == 7); };"#,
    );
}

#[test]
fn return_values_are_checked_even_in_unused_functions() {
    let error = support::pipeline::source_module("def unused() -> int = { return false; };")
        .unwrap_err()
        .to_string();
    assert!(error.contains("TypeMismatch"), "{error}");
}

#[test]
fn shader_returns_preserve_structured_selection_and_loops() {
    let module = support::module(
        "export { kernel }; @compute_shader def kernel(i: ulong, output: Ptr<uint>) = { if (i != 0_ul) { return; }; while (i == 0_ul) { output.* := 42_ui; return; }; };",
    );
    let project = support::project::Project::new(&module, None).unwrap();
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}
