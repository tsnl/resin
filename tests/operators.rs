mod support;

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run()
}

#[test]
fn dunder_calls_references_and_operators_share_one_specialization() {
    let module = support::module(
        r#"export { main };
        struct Number<T> { value: T,
            
        }
fn __add__<T>(a: Ref<Number<T>>, b: T) -> T  { a.value + b }

        fn named<T, U>(a: Ref<T>, b: U) -> _  { a:__add__(b) }
        fn main() -> int  {
            let mut number = Number<int> { value = 40 };
            let mut add = __add__::<int>;
            if (number + 2 == number:__add__(2) && named(number, 2_i) == add(number, 2)) {
                add(number, 2)
            } else { 1 }
        }
    "#,
    );
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|f| f.name.as_deref() == Some("__add__"))
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
    let output = run(r#"export { main };
        struct Vector<T> { x: T, y: T,
            
            
            
            
            
        }
fn __add__<T>(a: Ref<Vector<T>>, b: Ref<Vector<T>>) -> Vector<T>  { Vector<T> { x = a.x + b.x, y = a.y + b.y } }

fn __neg__<T>(a: Ref<Vector<T>>) -> Vector<T>  { Vector<T> { x = -a.x, y = -a.y } }

fn __mul__<T>(a: Ref<Vector<T>>, scale: Ref<T>) -> Vector<T>  { Vector<T> { x = a.x * scale, y = a.y * scale } }

fn __truediv__<T>(a: Ref<Vector<T>>, b: Ref<Vector<T>>) -> T  { a.x * b.x + a.y * b.y }

fn __eq__<T>(a: Ref<Vector<T>>, b: Ref<Vector<T>>) -> bool  { a.x == b.x && a.y == b.y }

        type Pair<T> = Vector<T>;
        fn add<T>(a: Ref<T>, b: Ref<T>) -> _  { a + b }
        fn multiply<T, U>(a: Ref<T>, b: U) -> _  { a * b }
        fn dot<T>(a: Ref<T>, b: Ref<T>) -> _  { a / b }
        fn main() -> int  {
            let mut first = Pair<int> { x = 2, y = 3 };
            let mut second = Vector<int> { x = 4, y = 5 };
            let mut total = multiply(add(first, second), 2_i);
            if (total == Vector<int> { x = 12, y = 16 } && (-first).x == -2) {
                dot(first, second) + add(17_i, 2_i)
            } else { 1 }
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
fn operator_operands_are_evaluated_once_in_order_and_reference_values_are_read() {
    let output = run(r#"export { main };
import { "$/shared.resin" };

        struct Number { value: int,
            
            
        }
fn __add__(a: Number, b: Number) -> Number  { Number { value = a.value + b.value } }

fn __sub__(a: Ref<Number>, b: int) -> int  { a.value - b }

        fn next(state: Ptr<int>, digit: int) -> Number  {
            state.* = state.* * 10 + digit;
            Number { value = digit }
        }
        fn main() -> int | Err<_> {
            let state_owner = arc_ptr_alloc(0_i)?; let state: Ref<_> = state_owner:get().*;
            let mut total = next(state_owner:get(), 1) + next(state_owner:get(), 2);
            let mut alias: Ref<Number> = total;
            if (state == 12 && alias.value == 3) { alias - -39 } else { 1 }
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
fn generic_overloads_lower_to_shader_calls() {
    let module = support::module(
        r#"export { kernel };
        struct Cell<T> { value: T,
            
        }
fn __add__<T>(a: Cell<T>, b: Cell<T>) -> Cell<T>  { Cell<T> { value = a.value + b.value } }

        fn add<T>(a: T, b: T) -> _  { a + b }
        @compute_shader fn kernel(index: ulong, root: Ptr<Cell<uint>>)  {
            root.* = add(Cell<uint> { value = root.value }, Cell<uint> { value = 42 });
        }
    "#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    assert_eq!(project.generated.shaders().len(), 1);
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

#[test]
fn every_operator_symbol_dispatches_without_changing_precedence() {
    let output = run(r#"export { main };
        struct Bits { value: int,
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        }
fn __pos__(a: Ref<Bits>) -> int  { a.value }

fn __neg__(a: Ref<Bits>) -> int  { -a.value }

fn __invert__(a: Ref<Bits>) -> int  { ~a.value }

fn __not__(a: Ref<Bits>) -> bool  { a.value == 0 }

fn __add__(a: Ref<Bits>, b: int) -> int  { a.value + b }

fn __sub__(a: Ref<Bits>, b: int) -> int  { a.value - b }

fn __mul__(a: Ref<Bits>, b: int) -> int  { a.value * b }

fn __truediv__(a: Ref<Bits>, b: int) -> int  { a.value / b }

fn __mod__(a: Ref<Bits>, b: int) -> int  { a.value % b }

fn __lshift__(a: Ref<Bits>, b: int) -> int  { a.value << b }

fn __rshift__(a: Ref<Bits>, b: int) -> int  { a.value >> b }

fn __and__(a: Ref<Bits>, b: int) -> int  { a.value & b }

fn __or__(a: Ref<Bits>, b: int) -> int  { a.value | b }

fn __xor__(a: Ref<Bits>, b: int) -> int  { a.value ^ b }

fn __eq__(a: Ref<Bits>, b: int) -> bool  { a.value == b }

fn __ne__(a: Ref<Bits>, b: int) -> bool  { a.value != b }

fn __lt__(a: Ref<Bits>, b: int) -> bool  { a.value < b }

fn __le__(a: Ref<Bits>, b: int) -> bool  { a.value <= b }

fn __gt__(a: Ref<Bits>, b: int) -> bool  { a.value > b }

fn __ge__(a: Ref<Bits>, b: int) -> bool  { a.value >= b }

        fn main() -> int  {
            let mut x = Bits { value = 6 };
            if (+x == 6 && -x == -6 && ~x == -7 && !Bits { value = 0 }
                && x + 2 * 3 == 12 && x - 2 == 4 && x * 3 == 18
                && x / 2 == 3 && x % 4 == 2 && x << 2 == 24 && x >> 1 == 3
                && (x & 3) == 2 && (x | 1) == 7 && (x ^ 3) == 5
                && x == 6 && x != 7 && x < 7 && x <= 6 && x > 5 && x >= 6) { 42 } else { 1 }
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
fn dependent_operator_parameters_type_literals_without_numeric_defaulting() {
    let output = run(r#"export { main };
        struct Narrow {
            
        }
fn __add__(self: Narrow, value: ubyte) -> int  { int(value) - 213 }

        fn add<T>(value: T) -> _  { value + 255 }
        fn increment<T>(value: T) -> T  { value + 1 }
        fn main() -> int  {
            if (add(1_ub) == 0_ub && increment(41_i) == 42) { add(Narrow {}) } else { 1 }
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
fn operator_borrows_and_results_use_ordinary_owner_cleanup() {
    let output = run(r#"export { main };
        import { "$/shared.resin" };
        struct Payload { drops: Ptr<int>,
            
        }
fn drop(self: Ref<Payload>)  { self.drops.* = self.drops.* + 1; }

        struct Value { owner: ArcPtr<Payload>, value: int,
            
        }
fn __add__(a: Ref<Value>, b: Ref<Value>) -> Value  { Value { owner = a.owner:clone(), value = a.value + b.value } }

        fn add<T>(a: Ref<T>, b: Ref<T>) -> _  { a + b }
        fn main() -> int | Err<_>  {
            let drops_owner = arc_ptr_alloc(0_i)?; let drops: Ref<_> = drops_owner:get().*;
            let mut answer = 0_i;
            {
                let mut owner = arc_ptr_alloc::<Payload>(Payload { drops = drops_owner:get() })?;
                drops = 0;
                let mut value = Value { owner = owner, value = 21 };
                let mut result = add(value, value);
                if (drops == 0) { answer = result.value; };
            };
            if (drops == 1) { answer } else { 1 }
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
fn dependent_operator_errors_report_the_requested_instantiation() {
    for (declaration, expression, expected) in [
        ("struct Value {}", "value + 1", "overload of `+`"),
        (
            "struct Value {  }\nfn __add__(a: Value, b: ubyte) -> int  { int(b) }\n",
            "value + 256",
            "range",
        ),
        (
            "struct Value {  }\nfn __add__(a: Value, b: int) -> int  { b }\n",
            "value + (1 == 1)",
            "no matching overload",
        ),
    ] {
        let source = format!(
            "{declaration} fn relay<T>(value: T) -> int  {{ {expression} }} fn main()  {{ relay(Value {{}}); }}"
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
            r#"export {{ kernel }};
            extern {{ "stdlib.h": {{ fn abs(value: int) -> int; }}, }};
            struct Number {{ value: int,
                
            }}
fn __add__(a: Number, b: int) -> int  {{ {body} }}

            fn add<T>(a: T) -> _  {{ a + 1 }}
            @compute_shader fn kernel(index: ulong, root: Ptr<Number>)  {{
                root.value = add(Number {{ value = root.value }});
            }}
        "#
        );
        let error = support::pipeline::shader_error(&source);
        assert!(error.contains(expected), "{source}\n{error}");
    }
}
