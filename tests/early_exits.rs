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
        fn choose(flag: bool) -> i32  {
            let mut value: i32;
            if (flag) { return 42; } else { value = 7; };
            return value;
        }
        fn both(flag: bool) -> i32  { if (flag) { return 3; } else { return 4; }; }
        fn nested() -> i32  { 1 + { return 9; 2 } }
        fn loop_return() -> i32  { while (true) { return 5; }; 6 }
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
import { "$/shared.resin" };

        struct Resource { trace: PtrMut<i32>, digit: i32,
            
        }
fn drop(self: RefMut<Resource>)  { self.trace.* = self.trace.* * 10 + self.digit; }

        fn leave(trace: PtrMut<i32>) -> i32  {
            let mut first = Resource { trace = trace, digit = 1 };
            { let mut second = Resource { trace = trace, digit = 2 }; return 42; };
        }
        fn main() -> () | Err<_> { let trace_owner = arc_ptr_alloc(i32(0))?; let trace: Ref<_> = trace_owner:get().*; assert(leave(trace_owner:get()) == 42); assert(trace == 21); }"#,
    );
}

#[test]
fn returning_arms_do_not_participate_in_match_initialization_joins() {
    succeeds(
        r#"export { main };
        fn choose(value: i32 | None) -> i32  {
            let mut result: i32;
            match (value) { i32(number) => { result = number; }, None => { return 7; } };
            result
        }
        fn main()  { assert(choose(i32(42)) == 42); assert(choose(None) == 7); }"#,
    );
}

#[test]
fn return_values_are_checked_even_in_unused_functions() {
    let error = support::pipeline::source_module("fn unused() -> i32  { return false; }")
        .unwrap_err()
        .to_string();
    assert!(error.contains("TypeMismatch"), "{error}");
}

#[test]
fn shader_returns_preserve_structured_selection_and_loops() {
    let module = support::module(
        "export { kernel }; @compute_shader fn kernel(i: u64, output: PtrMut<u32>)  { if (i != u64(0)) { return; }; while (i == u64(0)) { output.* = u32(42); return; }; }",
    );
    let project = support::project::Project::new(&module, None).unwrap();
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

#[test]
fn loop_exits_target_the_nearest_loop_and_drop_exited_scopes() {
    succeeds(
        r#"export { main };
import { "$/shared.resin" };

        struct Resource { trace: PtrMut<i32>, digit: i32,
            
        }
fn drop(self: RefMut<Resource>)  { self.trace.* = self.trace.* * 10 + self.digit; }

        fn main() -> () | Err<_> {
            let trace_owner = arc_ptr_alloc(i32(0))?; let trace: Ref<_> = trace_owner:get().*;
            let mut count = 0;
            while (true) {
                let mut outer = Resource { trace = trace_owner:get(), digit = 1 };
                count = count + 1;
                if (count == 1) { continue; };
                while (true) {
                    let mut inner = Resource { trace = trace_owner:get(), digit = 2 };
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
        "fn unused()  { let mut x: i32; while (true) { x = 1; break; }; let mut y = x; }",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("UninitializedValue"), "{error}");
}

#[test]
fn shader_loop_exits_preserve_structured_merges() {
    let module = support::module(
        "export { kernel }; @compute_shader fn kernel(i: u64, output: PtrMut<u32>)  { let mut n: u32 = 0; while (n < u32(8)) { n = n + u32(1); if (n == u32(2)) { continue; }; while (true) { if (n == u32(4)) { break; }; break; }; if (n > u32(5)) { break; }; output.* = n; }; }",
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
