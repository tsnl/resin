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
        fn choose(flag: bool) -> int  {
            let mut value: int;
            if (flag) { return 42; } else { value = 7; };
            return value;
        }
        fn both(flag: bool) -> int  { if (flag) { return 3; } else { return 4; }; }
        fn nested() -> int  { 1 + { return 9; 2 } }
        fn loop_return() -> int  { while (true) { return 5; }; 6 }
        fn main()  {
            assert(choose(true) == 42); assert(choose(false) == 7);
            assert(both(true) == 3); assert(both(false) == 4);
            assert(nested() == 9); assert(loop_return() == 5);
            return;
            assert(false);
        }"#,
    );
}

#[test]
fn returns_drop_owners_once_in_reverse_order_and_preserve_the_result() {
    succeeds(
        r#"export { main };
        struct Resource { trace: Ptr<int>; digit: int;
            
        }
fn drop(self: Ptr<Resource>)  { self.trace.* = self.trace.* * 10 + self.digit; }

        fn leave(trace: Ptr<int>) -> int  {
            let mut first = Resource { trace = trace, digit = 1 };
            { let mut second = Resource { trace = trace, digit = 2 }; return 42; };
        }
        fn main()  { let mut trace = 0_i; assert(leave(&trace) == 42); assert(trace == 21); }"#,
    );
}

#[test]
fn returning_arms_do_not_participate_in_match_initialization_joins() {
    succeeds(
        r#"export { main };
        fn choose(value: int | None) -> int  {
            let mut result: int;
            match (value) { int(number) => { result = number; }, None => { return 7; } };
            result
        }
        fn main()  { assert(choose(42_i) == 42); assert(choose(None) == 7); }"#,
    );
}

#[test]
fn return_values_are_checked_even_in_unused_functions() {
    let error = support::pipeline::source_module("fn unused() -> int  { return false; }")
        .unwrap_err()
        .to_string();
    assert!(error.contains("TypeMismatch"), "{error}");
}

#[test]
fn shader_returns_preserve_structured_selection_and_loops() {
    let module = support::module(
        "export { kernel }; @compute_shader fn kernel(i: ulong, output: Ptr<uint>)  { if (i != 0_ul) { return; }; while (i == 0_ul) { output.* = 42_ui; return; }; }",
    );
    let project = support::project::Project::new(&module, None).unwrap();
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

#[test]
fn loop_exits_target_the_nearest_loop_and_drop_exited_scopes() {
    succeeds(
        r#"export { main };
        struct Resource { trace: Ptr<int>; digit: int;
            
        }
fn drop(self: Ptr<Resource>)  { self.trace.* = self.trace.* * 10 + self.digit; }

        fn main()  {
            let mut trace = 0_i;
            let mut count = 0;
            while (true) {
                let mut outer = Resource { trace = &trace, digit = 1 };
                count = count + 1;
                if (count == 1) { continue; };
                while (true) {
                    let mut inner = Resource { trace = &trace, digit = 2 };
                    if (count == 2) { break; };
                    assert(false);
                };
                break;
            };
            assert(count == 2);
            assert(trace == 121);
            let mut value = 1 + { while (true) { break; }; 2 };
            assert(value == 3);
        }"#,
    );
}

#[test]
fn loop_exits_do_not_hide_initialization_errors_or_escape_conditions() {
    for source in [
        "fn unused()  { break; }",
        "fn unused()  { continue; }",
        "fn unused()  { while ({ break; true }) {}; }",
    ] {
        let error = support::pipeline::source_module(source)
            .unwrap_err()
            .to_string();
        assert!(error.contains("loop body"), "{error}");
    }
    let error = support::pipeline::source_module(
        "fn unused()  { let mut x: int; while (true) { x = 1; break; }; let mut y = x; }",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("UninitializedValue"), "{error}");
}

#[test]
fn shader_loop_exits_preserve_structured_merges() {
    let module = support::module(
        "export { kernel }; @compute_shader fn kernel(i: ulong, output: Ptr<uint>)  { let mut n = 0_ui; while (n < 8_ui) { n = n + 1_ui; if (n == 2_ui) { continue; }; while (true) { if (n == 4_ui) { break; }; break; }; if (n > 5_ui) { break; }; output.* = n; }; }",
    );
    let project = support::project::Project::new(&module, None).unwrap();
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

#[test]
fn short_circuiting_skips_conditional_exits() {
    succeeds(
        r#"export { main };
        fn main()  {
            let mut n = 0;
            while (n < 2) {
                n = n + 1;
                false && { continue; false };
                assert(n > 0);
            };
            assert(n == 2);
        }"#,
    );
}
