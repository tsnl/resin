mod support;

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run()
}

#[test]
fn dunder_calls_references_and_operators_share_one_specialization() {
    let module = support::module(
        r#"
        export { main };
        struct Number<T> { value: T,
            def __add__(a: Number<T>, b: T) -> T = { a.value + b };
        };
        def named<T, U>(a: T, b: U) -> _ = { a.__add__(b) };
        def main() -> int = {
            var number = Number<int> { value = 40 };
            var add = Number<int>.__add__;
            if (number + 2 == number.__add__(2) && named(number, 2_i) == add(number, 2)) {
                add(number, 2)
            } else { 1 }
        };
    "#,
    );
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|f| f.name.as_deref() == Some("Number.__add__"))
            .count(),
        1
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn operators_support_generic_owners_aliases_and_distinct_operand_and_result_types() {
    let output = run(r#"
        export { main };
        struct Vector<T> { x: T, y: T,
            def __add__(a: Vector<T>, b: Vector<T>) -> Vector<T> = { Vector<T> { x = a.x + b.x, y = a.y + b.y } };
            def __neg__(a: Vector<T>) -> Vector<T> = { Vector<T> { x = -a.x, y = -a.y } };
            def __mul__(a: Vector<T>, scale: T) -> Vector<T> = { Vector<T> { x = a.x * scale, y = a.y * scale } };
            def __truediv__(a: Vector<T>, b: Vector<T>) -> T = { a.x * b.x + a.y * b.y };
            def __eq__(a: Vector<T>, b: Vector<T>) -> bool = { a.x == b.x && a.y == b.y };
        };
        type Pair<T> = Vector<T>;
        def add<T>(a: T, b: T) -> _ = { a + b };
        def multiply<T, U>(a: T, b: U) -> _ = { a * b };
        def dot<T>(a: T, b: T) -> _ = { a / b };
        def main() -> int = {
            var first = Pair<int> { x = 2, y = 3 };
            var second = Vector<int> { x = 4, y = 5 };
            var total = multiply(add(first, second), 2_i);
            if (total == Vector<int> { x = 12, y = 16 } && (-first).x == -2) {
                dot(first, second) + add(17_i, 2_i)
            } else { 1 }
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
fn operator_operands_are_evaluated_once_in_order_and_reference_values_are_read() {
    let output = run(r#"
        export { main };
        struct Number { value: int,
            def __add__(a: Number, b: Number) -> Number = { Number { value = a.value + b.value } };
            def __sub__(a: Number, b: int) -> int = { a.value - b };
        };
        def next(state: Ptr<int>, digit: int) -> Number = {
            state.* := state.* * 10 + digit;
            Number { value = digit }
        };
        def main() -> int = {
            var state = 0_i;
            var total = next(&state, 1) + next(&state, 2);
            var alias: Ref<Number> = total;
            if (state == 12 && alias.value == 3) { alias - -39 } else { 1 }
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
fn generic_overloads_lower_to_shader_calls() {
    let module = support::module(
        r#"
        export { kernel };
        struct Cell<T> { value: T,
            def __add__(a: Cell<T>, b: Cell<T>) -> Cell<T> = { Cell<T> { value = a.value + b.value } };
        };
        def add<T>(a: T, b: T) -> _ = { a + b };
        @compute_shader def kernel(index: ulong, root: Ptr<Cell<uint>>) = {
            root.* := add(root.*, Cell<uint> { value = 42 });
        };
    "#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    assert_eq!(project.generated.shaders().len(), 1);
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

#[test]
fn every_operator_symbol_dispatches_without_changing_precedence() {
    let output = run(r#"
        export { main };
        struct Bits { value: int,
            def __pos__(a: Bits) -> int = { a.value };
            def __neg__(a: Bits) -> int = { -a.value };
            def __invert__(a: Bits) -> int = { ~a.value };
            def __not__(a: Bits) -> bool = { a.value == 0 };
            def __add__(a: Bits, b: int) -> int = { a.value + b };
            def __sub__(a: Bits, b: int) -> int = { a.value - b };
            def __mul__(a: Bits, b: int) -> int = { a.value * b };
            def __truediv__(a: Bits, b: int) -> int = { a.value / b };
            def __mod__(a: Bits, b: int) -> int = { a.value % b };
            def __lshift__(a: Bits, b: int) -> int = { a.value << b };
            def __rshift__(a: Bits, b: int) -> int = { a.value >> b };
            def __and__(a: Bits, b: int) -> int = { a.value & b };
            def __or__(a: Bits, b: int) -> int = { a.value | b };
            def __xor__(a: Bits, b: int) -> int = { a.value ^ b };
            def __eq__(a: Bits, b: int) -> bool = { a.value == b };
            def __ne__(a: Bits, b: int) -> bool = { a.value != b };
            def __lt__(a: Bits, b: int) -> bool = { a.value < b };
            def __le__(a: Bits, b: int) -> bool = { a.value <= b };
            def __gt__(a: Bits, b: int) -> bool = { a.value > b };
            def __ge__(a: Bits, b: int) -> bool = { a.value >= b };
        };
        def main() -> int = {
            var x = Bits { value = 6 };
            if (+x == 6 && -x == -6 && ~x == -7 && !Bits { value = 0 }
                && x + 2 * 3 == 12 && x - 2 == 4 && x * 3 == 18
                && x / 2 == 3 && x % 4 == 2 && x << 2 == 24 && x >> 1 == 3
                && (x & 3) == 2 && (x | 1) == 7 && (x ^ 3) == 5
                && x == 6 && x != 7 && x < 7 && x <= 6 && x > 5 && x >= 6) { 42 } else { 1 }
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
fn dependent_operator_parameters_type_literals_without_numeric_defaulting() {
    let output = run(r#"
        export { main };
        struct Narrow {
            def __add__(self: Narrow, value: ubyte) -> int = { int(value) - 213 };
        };
        def add<T>(value: T) -> _ = { value + 255 };
        def increment<T>(value: T) -> T = { value + 1 };
        def main() -> int = {
            if (add(1_ub) == 0_ub && increment(41_i) == 42) { add(Narrow {}) } else { 1 }
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
fn operator_copies_and_results_use_ordinary_owner_cleanup() {
    let output = run(r#"
        export { main };
        import { "$/shared.resin" };
        struct Payload { drops: Ptr<int>,
            def drop(self: Ptr<Payload>) = { self.drops.* := self.drops.* + 1; };
        };
        struct Value { owner: ArcPtr<Payload>, value: int,
            def __add__(a: Value, b: Value) -> Value = { Value { owner = a.owner, value = a.value + b.value } };
        };
        def add<T>(a: T, b: T) -> _ = { a + b };
        def main() -> int | Err<_> = {
            var drops = 0_i;
            var answer = 0_i;
            {
                var owner = ArcPtr<Payload>.alloc(Payload { drops = &drops })?;
                drops := 0;
                var value = Value { owner = owner, value = 21 };
                var result = add(value, value);
                if (drops == 0) { answer := result.value; };
            };
            if (drops == 1) { answer } else { 1 }
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
fn dependent_operator_errors_report_the_requested_instantiation() {
    for (declaration, expression, expected) in [
        ("struct Value {};", "value + 1", "operator +"),
        (
            "struct Value { def __add__(a: Value, b: ubyte) -> int = { int(b) }; };",
            "value + 256",
            "range",
        ),
        (
            "struct Value { def __add__(a: Value, b: int) -> int = { b }; };",
            "value + (1 == 1)",
            "Bool",
        ),
    ] {
        let source = format!(
            "{declaration} def relay<T>(value: T) -> int = {{ {expression} }}; def main() = {{ relay(Value {{}}); }};"
        );
        let hir = support::hir(&source);
        let errors = support::frontend::lower(&hir, &[], &resin_lir::LoweringOptions::default())
            .unwrap_err();
        let error = &errors[0];
        assert!(error.to_string().contains(expected), "{source}\n{error}");
        assert!(
            error
                .applications
                .iter()
                .any(|a| a.function.as_ref() == "relay"),
            "{error:?}"
        );
    }
}

#[test]
fn operator_calls_obey_shader_foreign_call_and_recursion_rules() {
    for (body, expected) in [("abs(a.value)", "foreign"), ("a + b", "recursive")] {
        let source = format!(
            r#"
            export {{ kernel }};
            extern {{ "stdlib.h": {{ def abs(value: int) -> int; }}, }};
            struct Number {{ value: int,
                def __add__(a: Number, b: int) -> int = {{ {body} }};
            }};
            def add<T>(a: T) -> _ = {{ a + 1 }};
            @compute_shader def kernel(index: ulong, root: Ptr<Number>) = {{
                root.value := add(root.*);
            }};
        "#
        );
        let error = support::pipeline::shader_error(&source);
        assert!(error.contains(expected), "{source}\n{error}");
    }
}
