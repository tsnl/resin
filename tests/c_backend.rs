use resin_types::prelude::*;
use std::fs;
use support::pipeline;
use support::toolchain;
use tempfile::TempDir;

mod support;
use support::module;

fn run_module(module: &resin_lir::Module) -> std::process::Output {
    run_entry(module, "main")
}

fn run_entry(module: &resin_lir::Module, entry: &str) -> std::process::Output {
    support::project::Project::new(module, Some(entry))
        .unwrap()
        .run()
}

#[test]
fn ownership_example_releases_memory_on_success_and_failure() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/ownership.resin");
    let m = pipeline::file_module(&path).unwrap();
    let success = run_entry(&m, "main");
    assert!(
        success.status.success(),
        "{}",
        String::from_utf8_lossy(&success.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&success.stdout).replace("\r\n", "\n"),
        "releasing allocation\nanswer = 42\n"
    );
    let failure = run_entry(&m, "failure");
    assert_eq!(failure.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&failure.stdout).replace("\r\n", "\n"),
        "releasing allocation\n"
    );
    assert_eq!(
        String::from_utf8_lossy(&failure.stderr).replace("\r\n", "\n"),
        "unhandled error: Failed\n"
    );
}

#[test]
fn array_value_projections_copy_the_element_and_destroy_the_container() {
    use resin_lir::{BasicBlock, BlockId, Instr::*, Local, Terminator};
    for projection in [
        vec![AccessStatic { index: 0 }],
        vec![
            Push {
                value: Value::UInt64 { value: 0 },
            },
            AccessDynamic,
        ],
    ] {
        let mut program = module(
            r#"export { main };
            import { "$/shared.resin" };
            struct Resource { trace: Ptr<int>, digit: int,
                
            }
fn drop(self: Ptr<Resource>)  {
                    if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; };
                }


            fn make(trace: Ptr<int>, digit: int) -> ArcPtr<Resource>  {
                let mut optional: ArcPtr<Resource> | None;
                optional = match (arc_ptr_alloc::<Resource>(Resource { trace = trace, digit = 0 })) {
                    ArcPtr<Resource>(value) => { value }, Err(error) => { None },
                };
                let mut owner = optional!;
                owner:get().digit = digit;
                owner
            }
            fn main() -> int  { 0 }
        "#,
        );
        let make = FunctionId::from_index(
            program
                .functions
                .iter()
                .position(|f| f.name.as_deref() == Some("make"))
                .unwrap(),
        );
        let element = program.functions[make.index()].result.clone();
        let payload = Ty::Defined {
            definition: TypeId::from_index(
                program
                    .types
                    .iter()
                    .position(|ty| ty.name().is_some_and(|name| name.as_ref() == "Resource"))
                    .unwrap(),
            ),
        };
        let trace = LocalId::from_index(1);
        let selected = LocalId::from_index(2);
        let int = |value| Push {
            value: Value::Int32 { value },
        };
        let mut instrs = vec![int(0), SetLocal { local: trace }];
        for digit in [1, 2] {
            instrs.extend([
                Function { function: make },
                LocalAddress { local: trace },
                int(digit),
                Call { arguments: 2 },
            ]);
        }
        instrs.push(MakeArray {
            elements: 2,
            element: element.clone(),
        });
        instrs.extend(projection);
        instrs.extend([
            SetLocal { local: selected },
            LocalAddress { local: selected },
            AccessStatic { index: 0 },
            OwnerData { pointee: payload },
            AccessStatic { index: 1 },
            Load,
            DropLocal { local: selected },
            LocalAddress { local: trace },
            Load,
            CallBuiltin {
                name: "+".into(),
                params: vec![Ty::Int32, Ty::Int32],
                result: Ty::Int32,
            },
        ]);
        let main = &mut program.functions[program.entries["main"].index()];
        main.locals = [Ty::Unit, Ty::Int32, element]
            .into_iter()
            .map(|ty| Local { name: None, ty })
            .collect();
        main.entry = BlockId::from_index(0);
        main.blocks = vec![BasicBlock {
            name: None,
            instrs,
            terminator: Terminator::Return,
        }];
        let output = run_module(&program);
        assert_eq!(
            output.status.code(),
            Some(22),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn runs(source: &str, code: i32) {
    let output = run_module(&module(source));
    assert_eq!(
        output.status.code(),
        Some(code),
        "{source}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn results_propagate_handle_payloads_and_widen_without_reordering_effects() {
    runs(
        "export { main }; struct E {} fn fail() -> (int | Err<E>)  { Err(E {}) } fn sum(a: int, b: int, c: int) -> int  { a + b + c } fn main() -> (int | Err<E>)  { (sum(1, fail()?, 3)) }",
        1,
    );
    runs(
        "export { main }; struct Bad { code: int, } fn fail(counter: Ptr<int>) -> (int | Err<Bad>)  { counter.* = counter.* + 1; Err(Bad { code = 7 }) } fn work(counter: Ptr<int>) -> (int | Err<Bad>)  { ((counter.* = counter.* + 10) + fail(counter)? + (counter.* = 1000)) } fn main() -> int  { let mut count = 0; let mut result = work(&count); match (result) { int(n) => { 99 }, Err(e) => { count + e.code } } }",
        18,
    );
    runs(
        "export { main }; struct A {} struct B { n: int, } struct C {} fn small() -> (int | Err<B>)  { Err(B { n = 42 }) } fn broad() -> (int | Err<A | B | C>)  { small() } fn main() -> int  { match (broad()) { int(n) => { n }, Err(e) => { match (e) { C(c) => { 3 }, B(b) => { b.n }, A(a) => { 1 } } } } }",
        42,
    );
    runs(
        "export { main }; struct Inner { value: int | Err<Never>, } fn nested() -> Inner | Err<Never>  { Inner { value = 42 } } fn main() -> int | Err<_>  { let mut inner = nested()?; inner.value? }",
        42,
    );
    runs(
        "export { main }; struct E {} fn main() -> (() | Err<E>)  { (()) }",
        0,
    );
    let output = run_module(&module(
        "export { main }; struct Broken {} fn main() -> (() | Err<Broken>)  { Err(Broken {}) }",
    ));
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n"),
        "unhandled error: Broken\n"
    );
    let mut m = module(
        "export { main }; struct Broken {} fn main() -> (() | Err<Broken>)  { Err(Broken {}) }",
    );
    let mut definitions = m.types.to_vec();
    let TypeDef::Nominal { name, .. } = definitions
        .iter_mut()
        .find(|ty| ty.name().is_some_and(|name| name.as_ref() == "Broken"))
        .unwrap()
    else {
        unreachable!()
    };
    *name = "quoted\"name\\value".into();
    m.types = definitions.into();
    let output = run_module(&m);
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n"),
        "unhandled error: quoted\"name\\value\n"
    );
}

#[test]
fn inferred_types_lower_to_concrete_c_and_preserve_effect_order() {
    runs(
        "export { main }; fn main() -> _  { let mut n: _; let mut p: Ptr<_>; n = 40_i; p = &n; p.* = p.* + 2; p.* }",
        42,
    );
    runs(
        "export { main }; fn wide() -> _  { let mut n = 4294967297; let mut p: Ptr<ulong>; p = &n; p.* } fn main() -> int  { if (wide() == ulong(4294967297)) { 0 } else { 1 } }",
        0,
    );
    runs(
        "export { main }; struct FieldsAB<T0, T1> { a: T0, b: T1, }\nfn main() -> _  { let mut n = 0_i; let mut pair: FieldsAB<_, _>; pair = FieldsAB<_, _> { b = (n = n + 1), a = (n = n + 1) }; pair.a * 10 + pair.b }",
        21,
    );
    runs(
        "export { main }; fn add(n: int) -> int  { n + 1 } fn select() -> (int) -> _  { add } fn main() -> _  { select()(41) }",
        42,
    );
}

#[test]
fn array_and_span_indexing_use_element_sizes() {
    runs(
        "export { main }; import { \"$/span.resin\" }; fn main () -> int  { let mut values = [10, 20, 30]; let mut p = Span<int> { data = Ptr<int>(&values), length = ulong(3) }; p:at(1) = 7; let mut end = p:at(2); p:at(1) + values(0) + end }",
        47,
    );
    runs(
        "export { main }; struct Payload { marker: uint, wide: ulong, amount: float32, } fn main () -> int  { let mut values = [Payload { marker = uint(1), wide = ulong(4294967297), amount = float32(0.5) }, Payload { marker = uint(2), wide = ulong(8589934593), amount = float32(1.5) }]; let mut p = values(0); let mut q = values(1); q.amount = q.amount + float32(2.0); if (q.wide == ulong(8589934593) && q.marker == uint(2) && q.amount == float32(3.5) && p.amount == float32(0.5)) { 0 } else { 1 } }",
        0,
    );
}

#[test]
fn numbered_examples_compile_as_strict_c11() {
    for entry in fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/examples")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "resin")
            && path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .starts_with("eg")
        {
            let module = pipeline::file_module(&path).unwrap();
            let output = run_module(&module);
            assert_eq!(
                output.status.code(),
                Some(0),
                "{}\n{}",
                path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

#[test]
fn discarded_branch_results_compile_and_preserve_effects() {
    runs(
        "export { main }; fn main () -> int  { let mut x = 0; if (1 == 1) { x = 1; () } else { () }; if (1 == 2) { (1, 2) } else { (3, 4) }; x }",
        1,
    );
}

#[test]
fn while_rechecks_conditions_and_discards_body_values() {
    runs(
        "export { main }; fn main () -> int  { let mut n = 0; let mut sum = 0; while ((n = n + 1) <= 4) { sum = sum + n }; sum + n }",
        15,
    );
    runs(
        "export { main }; fn main () -> int  { let mut n = 0; while (1 == 0) { n = 42; }; n }",
        0,
    );
    runs(
        "export { main }; fn main () -> int  { let mut n: int; while ((n = 7) == 0) {}; n }",
        7,
    );
    runs(
        "export { main }; fn main () -> int  { let mut n = 0; while (n < 1000000) { n = n + 1; }; if (n == 1000000) { 0 } else { 1 } }",
        0,
    );
}

#[test]
fn while_nests_with_branches_and_preserves_outer_values() {
    runs(
        "export { main }; fn main () -> int  { let mut i = 0; let mut total = 0; while (i < 3) { let mut j = 0; while (j < 4) { if (j < 2) { total = total + 1 } else { total = total + 2 }; j = j + 1; }; i = i + 1; }; total }",
        18,
    );
    runs(
        "export { main }; fn main () -> int  { let mut n = 0; let mut x = 7; while (n < 3) { let mut x = 10; n = n + 1; x = 20; }; x + n }",
        10,
    );
    runs(
        "export { main }; struct FieldsFirstBodyLast<T0, T1, T2> { first: T0, body: T1, last: T2, }\nfn main () -> int  { let mut n = 0; let mut r = FieldsFirstBodyLast<_, _, _> { first = 9, body = while (n < 3) { n = n + 1; }, last = n }; r.first + r.last }",
        12,
    );
    runs(
        "export { main }; fn main () -> int  { let mut n = 0; while (if (n < 3) { (1 == 1) && ((n = n + 1) < 3) } else { 1 == 0 }) {}; n }",
        3,
    );
}

#[test]
fn ordinary_functions_can_be_passed_and_selected() {
    runs(
        "export { main }; fn add (x: int, y: int) -> int  { x + y } fn apply (f: (int, int) -> int, args: (int, int)) -> int  { f(args.0, args.1) } fn main () -> int  { let mut a = add; let mut b = if (1 == 1) { a } else { add }; apply(a, (10, 3)) + b(20, 4) }",
        37,
    );
}

#[test]
fn recursive_functions_receive_state_explicitly() {
    runs(
        "export { main }; fn fact(n: int, offset: int) -> int  { if (n == 0) { offset } else { n * fact(n - 1, offset) } } fn main() -> int  { let mut offset = 2; fact(4, offset) }",
        48,
    );
}

#[test]
fn mutual_recursion_needs_no_forward_declaration() {
    runs(
        "export { main }; fn f (n: int) -> int  { if (n == 0) { 7 } else { next(n - 1) } } fn next (m: int) -> int  { f(m) } fn main () -> int  { f(3) }",
        7,
    );
}

#[test]
fn calls_take_lists_and_tuples_are_explicit_values() {
    runs(
        r#"export { main };
        fn zero() -> int  { 1 }
        fn unit(value: ()) -> int  { 2 }
        fn tuple(value: (int, int)) -> int  { value.0 + value.1 }
        fn add<T>(a: T, b: T) -> T  { a + b }
        fn apply(f: (int, int) -> int, a: int, b: int) -> int  { f(a, b) }
        fn main() -> int  {
            let mut a: () -> int; a = zero;
            let mut b: (()) -> int; b = unit;
            let mut c: ((int, int)) -> int; c = tuple;
            a() + b(()) + c((3, 4)) + apply(add::<int>, 5, 6)
        }
    "#,
        21,
    );
}

#[test]
fn arguments_are_evaluated_left_to_right_after_the_callee() {
    runs(
        r#"export { main };
        fn mark(trace: Ptr<int>, digit: int) -> int  { trace.* = trace.* * 10 + digit }
        fn consume(a: int, b: int)  {}
        fn callee(trace: Ptr<int>) -> (int, int) -> ()  { mark(trace, 1); consume }
        fn main() -> int  {
            let mut trace = 0;
            callee(&trace)(mark(&trace, 2), mark(&trace, 3));
            trace
        }
    "#,
        123,
    );
}

#[test]
fn expression_cleanup_preserves_scope_order_on_failure_and_success() {
    for (call, expected) in [
        (
            "consume(make(trace, 1), { let mut inner = make(trace, 2); fail()? })",
            21,
        ),
        (
            "[make(trace, 1), { let mut inner = make(trace, 2); fail()? }]",
            21,
        ),
        ("[make(trace, 1), make(trace, 2).get().accept(fail()?)]", 21),
        (
            "consume(make(trace, 1), { let mut inner = make(trace, 2); owned_error(trace)? })",
            215,
        ),
        (
            "last(make(trace, 1), if (make(trace, 3).get().truth()) { make(trace, 2) } else { make(trace, 4) }, fail()?)",
            231,
        ),
        (
            "consume(make(trace, 1), last(make(trace, 2), make(trace, 3), { let mut inner = make(trace, 4); succeed(trace)? }))",
            43251,
        ),
    ] {
        let declarations = r#"export { main };
            import { "$/shared.resin" };
            struct Resource { trace: Ptr<int>, digit: int,
                
                
                
            }
fn drop(self: Ptr<Resource>)  { if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; }; }

fn accept(self: Ptr<Resource>, other: ArcPtr<Resource>) -> ArcPtr<Resource>  { other }

fn truth(self: Ptr<Resource>) -> bool  { 1 == 1 }

            struct Failed {}
            struct OwnedFailed { value: ArcPtr<Resource>, }
            fn make(trace: Ptr<int>, digit: int) -> ArcPtr<Resource>  {
                let mut optional: ArcPtr<Resource> | None;
                optional = match (arc_ptr_alloc::<Resource>(Resource { trace = trace, digit = 0 })) {
                    ArcPtr<Resource>(value) => { value }, Err(error) => { None },
                };
                let mut owner = optional!;
                owner:get().digit = digit;
                owner
            }
            fn consume(a: ArcPtr<Resource>, b: ArcPtr<Resource>)  {}
            fn last(a: ArcPtr<Resource>, b: ArcPtr<Resource>, c: ArcPtr<Resource>) -> ArcPtr<Resource>  { c }
            fn fail() -> (ArcPtr<Resource> | Err<Failed>)  { Err(Failed {}) }
            fn succeed(trace: Ptr<int>) -> (ArcPtr<Resource> | Err<Failed>)  { (make(trace, 5)) }
            fn owned_error(trace: Ptr<int>) -> (ArcPtr<Resource> | Err<OwnedFailed>)  { Err(OwnedFailed { value = make(trace, 5) }) }
        "#;
        let source = format!(
            "{declarations}
            def attempt(trace: Ptr<int>) -> () | Err<_> = {{ {call}; () }};
            def main() -> int = {{
                var trace = 0;
                attempt(&trace);
                if (trace == {expected}) {{ 0 }} else {{ 1 }}
            }};"
        );
        runs(&source, 0);
    }
}

#[test]
fn tuple_projection_preserves_places_and_nested_values() {
    runs(
        r#"export { main };
        fn main() -> int  {
            let mut pair = ((1, 2), 3);
            pair.0.1 = 20;
            let mut pointer = &pair.1;
            pointer.* = 21;
            pair.0.0 + pair.0.1 + pair.1
        }
    "#,
        42,
    );
}

#[test]
fn nominal_records_preserve_source_order_and_field_layout() {
    runs(
        "export { main }; struct R { a: int, b: int, } fn main () -> int  { let mut x = 0; let mut r = R { b = (x = 1), a = (x = 2) }; r.a * 10 + r.b + x }",
        23,
    );
}

#[test]
fn loaded_values_do_not_change_after_later_stores() {
    runs(
        "export { main }; fn main () -> int  { let mut x = 1; let mut old = x; x = 2; old * 10 + x }",
        12,
    );
}

#[test]
fn short_circuiting_and_joins_preserve_effects() {
    runs(
        "export { main }; fn main () -> int  { let mut x = 0; let mut a = (1 == 2) && ((x = 1) == 1); let mut b = (1 == 1) || ((x = 2) == 2); let mut n = if (a || b) { 3 } else { 4 }; n + x }",
        3,
    );
}

#[test]
fn aliases_preserve_function_types_and_numeric_operations() {
    runs(
        "export { main }; type Meters = int; type F = (Meters) -> Meters; fn add (x: Meters) -> Meters  { x + Meters(2) } fn main () -> int  { let mut f = F(add); int(f(Meters(5))) }",
        7,
    );
    runs(
        "export { main }; type Entry = () -> int; fn start() -> int  { 23 } fn main() -> int  { let mut entry = Entry(start); entry() }",
        23,
    );
}

#[test]
fn integer_arithmetic_wraps_at_its_declared_width() {
    runs(
        "export { main }; fn main () -> int  { let mut a = sbyte(127); let mut b = a + sbyte(1); let mut c = int(2147483647) + int(1); let mut d = long(-9223372036854775808) / long(-1); if (b == sbyte(-128) && c == int(-2147483648) && d == long(-9223372036854775808)) { 0 } else { 1 } }",
        0,
    );
}

#[test]
fn signed_right_shift_and_unsigned_multiplication_are_defined() {
    runs(
        "export { main }; fn main () -> int  { let mut a = int(-8) >> int(2); let mut b = uint(4294967295) * uint(4294967295); if (a == int(-2) && b == uint(1)) { 0 } else { 1 } }",
        0,
    );
}

#[test]
fn invalid_integer_operations_fail_at_runtime() {
    for expression in ["1 / 0", "1 % 0", "1 << 32", "1 >> -1"] {
        let result = run_module(&module(&format!(
            "export {{ main }}; fn main () -> int  {{ {expression} }}"
        )));
        assert_eq!(result.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&result.stderr).contains("resin:"));
    }
}

#[test]
fn entry_selection_and_invalid_ir_have_distinct_boundaries() {
    assert!(
        support::project::Project::new(&resin_lir::Module::default(), Some("main"))
            .unwrap_err()
            .to_string()
            .contains("export { main }")
    );
    let mut m = module("export { main }; fn main () -> int  { 0 }");
    m.functions[0].blocks[0]
        .instrs
        .insert(0, resin_lir::Instr::Discard);
    assert!(
        resin_lir::VerifiedModule::new(m)
            .err()
            .unwrap()
            .to_string()
            .contains("StackUnderflow")
    );
}

#[test]
fn unused_functions_do_not_fail_strict_compilation() {
    let mut m = module("export { main }; fn main () -> int  { 0 }");
    let mut spare = m.functions[0].clone();
    spare.name = Some("unused".into());
    m.functions.push(spare);
    assert!(run_module(&m).status.success());
}

#[test]
fn failed_compilation_preserves_existing_output() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let output = temp.path().join("existing");
    fs::write(&output, b"keep me").unwrap();
    assert!(
        toolchain::compile_c(
            "not C",
            &output,
            std::ffi::OsStr::new(resin_toolchain::DEFAULT_C_COMPILER)
        )
        .is_err()
    );
    assert_eq!(fs::read(&output).unwrap(), b"keep me");
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
}

#[test]
fn loops_carry_typed_stack_values_between_iterations() {
    use resin_lir::{BasicBlock, BlockId, Instr::*, Local, Terminator::*};
    let mut m = module("export { main }; fn main () -> int  { 0 }");
    let f = m
        .functions
        .iter_mut()
        .find(|f| f.name.as_deref() == Some("main"))
        .unwrap();
    let counter = LocalId::from_index(f.locals.len());
    f.locals.push(Local {
        name: None,
        ty: Ty::Int32,
    });
    let int = |n| Push {
        value: Value::Int32 { value: n },
    };
    let op = |name: &str, result| CallBuiltin {
        name: name.into(),
        params: vec![Ty::Int32; 2],
        result,
    };
    f.blocks = vec![
        BasicBlock {
            name: None,
            instrs: vec![
                LocalAddress { local: counter },
                int(3),
                Store,
                Discard,
                int(0),
            ],
            terminator: Loop {
                condition: BlockId::from_index(1),
                body: BlockId::from_index(2),
                next: Some(BlockId::from_index(3)),
            },
        },
        BasicBlock {
            name: None,
            instrs: vec![
                int(10),
                op("+", Ty::Int32),
                LocalAddress { local: counter },
                Load,
                int(0),
                op(">", Ty::Bool),
            ],
            terminator: LoopTest,
        },
        BasicBlock {
            name: None,
            instrs: vec![
                LocalAddress { local: counter },
                Load,
                op("+", Ty::Int32),
                LocalAddress { local: counter },
                LocalAddress { local: counter },
                Load,
                int(1),
                op("-", Ty::Int32),
                Store,
                Discard,
            ],
            terminator: Continue,
        },
        BasicBlock {
            name: None,
            instrs: vec![],
            terminator: Return,
        },
    ];
    // The false condition also contributes its carried value, including when
    // the loop never enters its body.
    for (iterations, expected) in [(3, 46), (0, 10)] {
        m.functions
            .iter_mut()
            .find(|f| f.name.as_deref() == Some("main"))
            .unwrap()
            .blocks[0]
            .instrs[1] = int(iterations);
        assert_eq!(run_module(&m).status.code(), Some(expected));
    }
}

#[test]
fn array_addresses_and_dynamic_bounds_are_executable() {
    use resin_lir::{Instr::*, Local};
    let mut m = module("export { main }; fn main () -> int  { 0 }");
    let f = m
        .functions
        .iter_mut()
        .find(|f| f.name.as_deref() == Some("main"))
        .unwrap();
    let array = LocalId::from_index(f.locals.len());
    f.locals.push(Local {
        name: None,
        ty: Ty::Array {
            element: Box::new(Ty::Int32),
            length: 2,
        },
    });
    f.blocks[0].instrs = vec![
        LocalAddress { local: array },
        Push {
            value: Value::Int32 { value: 4 },
        },
        Push {
            value: Value::Int32 { value: 9 },
        },
        MakeArray {
            elements: 2,
            element: Ty::Int32,
        },
        Store,
        Discard,
        LocalAddress { local: array },
        Push {
            value: Value::Int32 { value: 1 },
        },
        AccessDynamic,
        Load,
    ];
    assert_eq!(run_module(&m).status.code(), Some(9));
    let f = m
        .functions
        .iter_mut()
        .find(|f| f.name.as_deref() == Some("main"))
        .unwrap();
    f.blocks[0].instrs[7] = Push {
        value: Value::Int32 { value: -1 },
    };
    let output = run_module(&m);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("array index"));
}

#[test]
fn indexing_addresses_evaluate_receiver_and_index_once() {
    runs(
        r#"export { main };
        import { "$/span.resin" };
        fn view(p: Ptr<int>, calls: Ptr<int>) -> Span<int>  {
            calls.* = calls.* + 1;
            Span<int> { data = p, length = ulong(3) }
        }
        fn index(calls: Ptr<int>) -> int  { calls.* = calls.* + 1; 1 }
        fn main() -> int  {
            let mut values = [10_i, 20, 30]; let mut calls = 0;
            let mut p: Ptr<int>; p = &view(Ptr<int>(&values), &calls):at(ulong(index(&calls)));
            p.* = 42;
            let mut copied = values;
            copied(0) = 9;
            let mut temporary = ([7, 8])(1);
            if (calls == 2 && values(1) == 42 && values(0) == 10 && copied(0) == 9 && temporary == 8) { 0 } else { 1 }
        }"#,
        0,
    );
}

#[test]
fn at_indexing_borrows_array_places_and_supports_field_receivers() {
    runs(
        r#"export { main };
        import { "$/span.resin" };
        struct FieldsValues<T0> { values: T0, }
struct Holder { values: Span<int>, }
        fn view(p: Ptr<int>, calls: Ptr<int>) -> Holder  {
            calls.* = calls.* + 1;
            Holder { values = Span<int> { data = p, length = 3_ul } }
        }
        fn index(calls: Ptr<int>) -> int  { calls.* = calls.* + 1; 1 }
        fn element(s: Span<int>, i: ulong) -> Ref<int>  { s:at(i) }
        fn main() -> int  {
            let mut values = [10_i, 20, 30]; let mut calls = 0;
            let mut p: Ref<int> = view(Ptr<int>(&values), &calls).values:at(ulong(index(&calls)));
            p = 42;
            let mut record = FieldsValues<_> { values = [3, 4] };
            record.values:at(0) = 8;
            let mut holder = view(Ptr<int>(&values), &calls);
            element(holder.values, 0) = 11;
            let mut temporary = [7, 8]:at(1);
            if (calls == 3 && values:at(1) == 42 && values:at(0) == 11 && record.values:at(0) == 8 && temporary == 8) { 0 } else { 1 }
        }"#,
        0,
    );
}

#[test]
fn at_indexing_checks_bounds_before_later_effects() {
    for receiver in ["values", "holder.values"] {
        for index in ["2", "18446744073709551615_ul"] {
            let output = run_module(&module(&format!(
                r#"export {{ main }};

                extern {{
                    "stdio.h": {{
                        fn puts(text: Ptr<ubyte>) -> int;
                    }},
                }};
                import {{ "$/span.resin" }};
                struct FieldsValues<T0> {{ values: T0, }}
fn main() -> int  {{
                    let mut values = [1, 2];
                    let mut holder = FieldsValues<_> {{ values = Span<int> {{ data = Ptr<int>(&values), length = 2_ul }} }}
                    {receiver}.at({index}) = 9;
                    puts("after".data); 0
                }};"#
            )));
            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
            assert!(String::from_utf8_lossy(&output.stderr).contains("index out of bounds"));
        }
    }
}

#[test]
fn array_and_span_indexing_fail_before_out_of_bounds_access() {
    for source in [
        "export { main }; fn main() -> int  { let mut xs = [1, 2]; xs(-1) }",
        "export { main }; fn main() -> int  { let mut xs = [1, 2]; xs(2) = 9; 0 }",
        "export { main }; import { \"$/span.resin\" }; fn main() -> int  { let mut xs = [1, 2]; let mut s = Span<int> { data = Ptr<int>(&xs), length = ulong(2) }; s:at(18446744073709551615_ul) }",
        "export { main }; import { \"$/span.resin\" }; fn main() -> int  { let mut s = Span<int> { data = Ptr<int>(ulong(0)), length = ulong(0) }; s:at(0) }",
    ] {
        let output = run_module(&module(source));
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("index out of bounds"),
            "{:?}",
            output
        );
    }
}

#[test]
fn decorated_functions_and_their_helpers_remain_host_callable() {
    runs(
        r#"export { main };
        fn twice(i: uint) -> uint  { i * uint(2) }
        @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { twice(i) }; }
        fn main() -> int  { let mut f = kernel; let mut output = 0_ui; f(21_ul, &output); if (output == 42_ui) { 0 } else { 1 } }
    "#,
        0,
    );
}

#[test]
fn inlined_particle_functions_execute_on_the_cpu_with_host_spans() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/particles.resin");
    let mut program = pipeline::load(&path).unwrap();
    let file = std::sync::Arc::make_mut(&mut program.modules.last_mut().unwrap().file);
    file.stmts.retain(|s| !matches!(&s.val, resin_ast::StmtKind::Function { name, .. } if name.val.as_ref() == "main"));
    // Exercise random particle generation, camera math, and the shader bodies on
    // stack-backed storage. GPU initialization stores these same generated values.
    for statement in &mut file.stmts {
        let resin_ast::StmtKind::Function { name, params, .. } = &mut statement.val else {
            continue;
        };
        if name.val.as_ref() == "apply_camera" {
            let resin_ast::TypeKind::App { args, .. } = &mut params[1].1.val else {
                panic!()
            };
            let resin_ast::TypeKind::Atom { name } = &mut args[0].val else {
                panic!()
            };
            assert_eq!(name.val.as_ref(), "HostParams");
            name.val = "Params".into();
        }
    }
    file.stmts.extend(support::parse(r#"fn main() -> int  {
            let mut state = 12345_ui | 1_ui;
            let mut particle = random_particle(&state);
            let mut particles = Span<Particle> { data = &particle, length = 1_ul };
            let mut first = particle;
            state = 12345_ui | 1_ui;
            particle = random_particle(&state);
            let mut valid = particle.x == first.x && particle.vz == first.vz;
            state = 54321_ui | 1_ui;
            particle = random_particle(&state);
            valid = valid && particle.x != first.x && particle.vx != first.vx;
            valid = valid && particle.x >= -24_f && particle.x < 24_f && particle.y >= -30_f && particle.y < 30_f && particle.z >= 0_f && particle.z < 50_f && particle.vx >= -6_f && particle.vx < 6_f && particle.vy >= -6_f && particle.vy < 6_f && particle.vz >= -6_f && particle.vz < 6_f;
            let mut params = Params { dt = 0.005_f, yaw_cos = 1_f, yaw_sin = 0_f, pitch_cos = 1_f, pitch_sin = 0_f, zoom = 1_f, aspect = 0.625_f, radius = 0.0012_f, particles = particles };
            let mut steps = 0;
            while (steps < 2000) {
                kernel(0_ul, &params);
                valid = valid && particle.x > -100_f && particle.x < 100_f && particle.y > -100_f && particle.y < 100_f && particle.z > -100_f && particle.z < 100_f;
                steps = steps + 1;
            };
            // Check every particle boundary, including float32 rounding above 2^24.
            let mut index = 0;
            while (index < 1000000) {
                valid = valid && particle_index(index * 24) == index && particle_index(index * 24 + 23) == index;
                index = index + 1;
            };
            particle = Particle { x = 0_f, y = -20_f, z = 25_f, vx = 0_f, vy = 0_f, vz = 0_f };
            let mut near = vertex(0, &params);
            let mut near_rim = vertex(1, &params);
            particle.y = 20_f;
            let mut far = vertex(0, &params);
            let mut far_rim = vertex(1, &params);
            valid = valid && fragment(near.color).b > fragment(far.color).b;
            valid = valid && near_rim.position.x - near.position.x > far_rim.position.x - far.position.x;
            // Camera controls change the projection without changing the simulation.
            let mut camera = Camera { yaw = 0_f, pitch = 0_f, zoom = 1_f };
            apply_camera(camera, &params);
            let mut before = vertex(1, &params);
            camera.zoom = 2_f;
            apply_camera(camera, &params);
            let mut zoomed = vertex(1, &params);
            valid = valid && zoomed.position.x == before.position.x * 2_f;
            move_camera(&camera, 100_d, 50_d, 0_d);
            apply_camera(camera, &params);
            let mut orbited = vertex(0, &params);
            valid = valid && orbited.position.x > 0_f && orbited.position.y < 0_f;
            let mut yaw_length = params.yaw_cos * params.yaw_cos + params.yaw_sin * params.yaw_sin;
            let mut pitch_length = params.pitch_cos * params.pitch_cos + params.pitch_sin * params.pitch_sin;
            valid = valid && yaw_length > 0.999_f && yaw_length < 1.001_f && pitch_length > 0.999_f && pitch_length < 1.001_f;
            move_camera(&camera, 0_d, 1000000_d, 1000000_d);
            valid = valid && camera.pitch == 1.4_f && camera.zoom == 3_f;
            move_camera(&camera, 0_d, -1000000_d, -1000000_d);
            valid = valid && camera.pitch == -1.4_f && camera.zoom == 0.35_f;
            camera = default_camera();
            move_camera(&camera, 0_d, 0_d, 0.5_d);
            valid = valid && camera.zoom > 1_f && camera.zoom < 1.1_f;
            move_camera(&camera, 0_d, 0_d, -0.5_d);
            valid = valid && camera.zoom > 0.999_f && camera.zoom < 1.001_f;
            let mut color = fragment(far.color);
            if (valid && particle.x != first.x && color.r >= 0_f && color.r <= 1_f && color.b >= 0_f && color.b <= 1_f) { 0 } else { 1 }
        }
    "#).stmts);
    let m = pipeline::generate_program(&program).unwrap();
    assert!(m.shaders.values().all(|entry| !entry.embedded));
    assert!(run_module(&m).status.success());
}

#[test]
fn spirv_is_only_special_on_function_declarations() {
    runs(
        "export { main }; struct FieldsSpirv<T0> { spirv: T0, }\nfn main() -> int  { let mut record = FieldsSpirv<_> { spirv = 1 }; record.spirv = 2; record.spirv }",
        2,
    );
}

#[test]
fn suffixed_literals_execute_with_their_selected_widths() {
    runs(
        r#"export { main }; fn main() -> int  {
        let mut a = -128_b; let mut b = 255_ub; let mut c = -32768_h; let mut d = 65535_uh;
        let mut e = -2147483648_i; let mut f = 4294967295_ui;
        let mut g = -9223372036854775808_l; let mut h = 18446744073709551615_ul;
        if (a < 0_b && b > 0_ub && c < 0_h && d > 0_uh && e < 0_i && f > 0_ui && g < 0_l && h == 0xffffffffffffffff_ul && 1.5_f + 2.5_f == 4_f && 1e2_d == 100_d && 1.0000000596046448_f > 1_f) { 0 } else { 1 }
    }"#,
        0,
    );
}

#[test]
fn else_if_chains_select_one_branch_and_short_circuit_conditions() {
    runs(
        r#"export { main };
    fn condition(calls: Ptr<int>, value: int, expected: int) -> bool  {
        calls.* = calls.* + 1;
        value == expected
    }
    fn classify(value: int, calls: Ptr<int>) -> int  {
        if (condition(calls, value, 0)) { 10 }
        else if (condition(calls, value, 1)) { 20 }
        else if (condition(calls, value, 2)) { 30 }
        else { 40 }
    }
    fn main() -> int  {
        let mut calls = 0;
        let mut a = classify(0, &calls);
        let mut b = classify(1, &calls);
        let mut c = classify(2, &calls);
        let mut d = classify(3, &calls);
        let mut value = 0;
        if (1 == 0) { value = 100; } else if (1 == 1) { value = value + 1; };
        if (1 == 0) { value = 100; } else if (1 == 0) { value = 100; };
        if (a == 10 && b == 20 && c == 30 && d == 40 && calls == 9 && value == 1) { 0 } else { 1 }
    }"#,
        0,
    );
}

#[test]
fn one_armed_if_evaluates_once_and_runs_branch_cleanup() {
    runs(
        r#"export { main };
    struct Add { value: Ptr<int>,
        
    }
fn drop(self: Ptr<Add>)  { self.value.* = self.value.* + 10; }


    fn condition(calls: Ptr<int>) -> bool  { calls.* = calls.* + 1; 1 == 1 }
    fn main() -> int  {
        let mut calls = 0; let mut value = 0;
        if (condition(&calls)) { let mut cleanup = Add { value = &value }; value = value + 1; };
        if (1 == 0) { value = 100; };
        if (1 == 1) { if (1 == 0) { value = 100; } else { value = value + 1; }; };
        if (calls == 1 && value == 12) { 0 } else { 1 }
    }"#,
        0,
    );
}

#[test]
fn dedicated_cleanup_bindings_retain_acquisitions_on_both_exits() {
    for fail in ["1 == 1", "1 == 2"] {
        runs(
            &format!(
                r#"export {{ main }};
            struct E {{}}
            struct Capture {{ resource: Ptr<int>, trace: Ptr<int>,
                
            }}
fn drop(self: Ptr<Capture>)  {{ self.trace.* = self.trace.* * 10 + self.resource.*; }}


            fn work(trace: Ptr<int>, fail: bool) -> (() | Err<E>)  {{
                let mut resource = 1;
                let mut captured = resource;
                let mut first = Capture {{ resource = &captured, trace = trace }};
                let mut second = Capture {{ resource = &resource, trace = trace }};
                resource = 2;
                {{ let mut captured = 9; }};
                let mut result: (() | Err<E>); result = if (fail) {{ Err(E {{}}) }} else {{ (()) }};
                result?;
                (())
            }}
            fn main() -> int  {{
                let mut trace = 0;
                match (work(&trace, {fail})) {{ ()(v) => {{}}, Err(e) => {{}} }};
                if (trace == 21) {{ 0 }} else {{ 1 }}
            }}
        "#
            ),
            0,
        );
    }
}

#[test]
fn byte_arrays_have_packed_storage_and_nested_stride() {
    runs(
        r#"export { main };
        struct FieldsBytesTail<T0, T1> { bytes: T0, tail: T1, }
fn main() -> int  {
            let mut binary = [65_ub, 66_ub];
            let mut copied = binary;
            let mut nested = [[1_ub, 2_ub], [3_ub, 4_ub]];
            let mut embedded = [65_ub, 0_ub, 66_ub];
            let mut record = FieldsBytesTail<_, _> { bytes = copied, tail = 255_ub };
            binary:at(0) = 90_ub;
            if (size_of(binary) == 2_ul && align_of(binary) == 1_ul &&
                size_of(nested) == 4_ul && size_of(embedded) == 3_ul &&
                ulong(&nested:at(1)) - ulong(&nested:at(0)) == 2_ul &&
                size_of(record) == 3_ul && ulong(&record.tail) - ulong(&record.bytes) == 2_ul &&
                copied:at(0) == 65_ub && copied:at(1) == 66_ub &&
                record.tail == 255_ub && embedded:at(1) == 0_ub) { 0 } else { 1 }
        }
    "#,
        0,
    );
}

#[test]
fn compound_control_flow_preserves_operand_order() {
    runs(include_str!("fixtures/compound_control.resin"), 0);
}

#[test]
fn shared_layout_queries_follow_padding_and_do_not_evaluate_operands() {
    runs(
        r#"export { main };
        import { "$/span.resin" };
        struct Inner { x: uint, y: ulong, z: float32, }
        struct Outer { first: uint, inner: Inner, last: float32, }
        fn main() -> int  {
            let mut side = 0_ui;
            let mut values = [1_ui, 2_ui, 3_ui];
            if (size_of(uint) == 4_ul && align_of(ulong) == 8_ul && size_of(Outer) == 40_ul &&
                align_of(Outer) == 8_ul && size_of(Span<uint>) == 16_ul &&
                size_of(values) == 12_ul && size_of(side = 1_ui) == 4_ul && side == 0_ui) { 0 } else { 1 }
        }
    "#,
        0,
    );
}

#[test]
fn numeric_conversions_check_runtime_values_and_boundaries() {
    runs(include_str!("fixtures/numeric_conversions.resin"), 0);
    runs(
        r#"export { main }; fn main() -> int  {
        let mut n = 255_ui; let mut negative = -128_i; let mut wide = 18446744073709551615_ul;
        let mut nan = float32(0.0_d / 0.0_d); let mut large = 1.0e100_d; let mut tiny = -1.0e-100_d;
        if (ubyte(n) == 255_ub && sbyte(negative) == -128_b && ulong(wide) == wide &&
            float64(n) == 255.0_d && float32(large) > 1.0e30_f && float32(tiny) == 0.0_f && nan != nan) { 0 } else { 1 }
    }"#,
        0,
    );
    for expr in [
        "ubyte(256_ui)",
        "uint(-1_i)",
        "int(2147483648_ui)",
        "uint(4294967296.0_d)",
        "long(9223372036854775808.0_d)",
        "ulong(18446744073709551616.0_d)",
        "int(0.0_d / 0.0_d)",
        "int(1.0_d / 0.0_d)",
    ] {
        let source = format!("export {{ main }}; fn main()  {{ {expr}; }}");
        let output = run_module(&module(&source));
        assert!(!output.status.success(), "{source}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("numeric conversion out of range"),
            "{source}"
        );
    }
}

#[test]
fn never_elimination_preserves_reachable_cleanup() {
    runs(include_str!("fixtures/never_elimination.resin"), 0);
}

#[path = "support/interactions.rs"]
mod interactions;

#[test]
fn interacting_features_execute_equivalently_on_cpu() {
    for source in interactions::variants() {
        runs(&source, 0);
    }
    for marker in interactions::MARKERS {
        runs(
            &format!(
                "export {{ main }}; fn main() -> int  {{ let mut n = 300; let mut bytes = Ptr<ubyte>(&n); {marker} n - 300 }}"
            ),
            0,
        );
    }
}

#[test]
fn compute_entry_preserves_ulong_indices_on_the_host() {
    runs(
        "export { main }; @compute_shader fn kernel(index: ulong, output: Ptr<ulong>)  { output.* = index; } fn main() -> int  { let mut output = 0_ul; kernel(4294967297_ul, &output); if (output == 4294967297_ul) { 0 } else { 1 } }",
        0,
    );
}

#[test]
fn structured_loops_propagate_errors_from_conditions_and_nested_bodies() {
    let m = module(include_str!("fixtures/structured_control.resin"));
    let project = support::project::Project::new(&m, Some("main")).unwrap();
    let source = fs::read_to_string(project.generated.c_source().unwrap()).unwrap();
    assert!(source.contains("while (true)"));
    assert!(!source.contains("goto "), "{source}");
    let output = project.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn sequential_conditionals_and_error_propagation_keep_constant_nesting() {
    let mut source = String::from(
        "export { main }; struct Failed {} fn step() -> (() | Err<Failed>)  { (()) } fn main() -> (int | Err<Failed>) = { let mut value = 0; ",
    );
    for _ in 0..512 {
        source.push_str("if (value == 0) { value := 1; } else { value := 0; }; step()?; ");
    }
    source.push_str("(value) };");
    let project = support::project::Project::new(&module(&source), Some("main")).unwrap();
    let c = fs::read_to_string(project.generated.c_source().unwrap()).unwrap();
    assert!(
        c.lines()
            .all(|line| line.len() - line.trim_start().len() < 32)
    );
    assert!(project.run().status.success());
}
