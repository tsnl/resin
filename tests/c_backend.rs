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
            struct Resource { trace: Ptr<i32>, digit: i32,
                
            }
fn drop(self: RefMut<Resource>)  {
                    if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; };
                }


            fn make(trace: Ptr<i32>, digit: i32) -> ArcPtr<Resource>  {
                let mut optional: ArcPtr<Resource> | None;
                optional = match (arc_ptr_alloc::<Resource>(Resource { trace = trace, digit = 0 })) {
                    ArcPtr<Resource>(value) => { value }, Err(error) => { None },
                };
                let mut owner = optional!;
                owner:get().digit = digit;
                owner
            }
            fn main() -> i32  { 0 }
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
        let mut instrs = vec![
            int(0),
            OwnerCreate { element: Ty::Int32 },
            ExcludeNone,
            SetLocal { local: trace },
        ];
        for digit in [1, 2] {
            instrs.extend([
                Function { function: make },
                LocalRef { local: trace },
                ReadOnly,
                OwnerData { pointee: Ty::Int32 },
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
            LocalRef { local: selected },
            AccessStatic { index: 0 },
            ReadOnly,
            OwnerData { pointee: payload },
            AccessStatic { index: 1 },
            Load,
            DropLocal { local: selected },
            LocalRef { local: trace },
            ReadOnly,
            OwnerData { pointee: Ty::Int32 },
            Load,
            CallBuiltin {
                name: "+".into(),
                params: vec![Ty::Int32, Ty::Int32],
                result: Ty::Int32,
            },
        ]);
        instrs.push(DropLocal { local: trace });
        let main = &mut program.functions[program.entries["main"].index()];
        main.locals = [Ty::Unit, Ty::StrongOwner, element]
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
        "export { main }; struct E {} fn fail() -> (i32 | Err<E>)  { Err(E {}) } fn sum(a: i32, b: i32, c: i32) -> i32  { a + b + c } fn main() -> (i32 | Err<E>)  { (sum(1, fail()?, 3)) }",
        1,
    );
    runs(
        "export { main };\nimport { \"$/shared.resin\" };\n struct Bad { code: i32, } fn fail(counter: Ptr<i32>) -> (i32 | Err<Bad>)  { counter.* = counter.* + 1; Err(Bad { code = 7 }) } fn work(counter: Ptr<i32>) -> (i32 | Err<Bad>)  { ({ counter.* = counter.* + 10; counter.* } + fail(counter)? + { counter.* = 1000; counter.* }) } fn main() -> i32 | Err<_> { let count_owner = arc_ptr_alloc(0)?; let count: Ref<_> = count_owner:get().*; let mut result = work(count_owner:get()); match (result) { i32(n) => { 99 }, Err(e) => { count + e.code } } }",
        18,
    );
    runs(
        "export { main }; struct A {} struct B { n: i32, } struct C {} fn small() -> (i32 | Err<B>)  { Err(B { n = 42 }) } fn broad() -> (i32 | Err<A | B | C>)  { small() } fn main() -> i32  { match (broad()) { i32(n) => { n }, Err(e) => { match (e) { C(c) => { 3 }, B(b) => { b.n }, A(a) => { 1 } } } } }",
        42,
    );
    runs(
        "export { main }; struct Inner { value: i32 | Err<Never>, } fn nested() -> Inner | Err<Never>  { Inner { value = 42 } } fn main() -> i32 | Err<_>  { let mut inner = nested()?; inner.value? }",
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
        "export { main }; fn main() -> _  { let mut n: _; n = i32(40); let p: RefMut<_> = n; p = p + 2; p }",
        42,
    );
    runs(
        "export { main }; fn wide() -> _  { let mut n = 4294967297; let p: Ref<u64> = n; p } fn main() -> i32  { if (wide() == u64(4294967297)) { 0 } else { 1 } }",
        0,
    );
    runs(
        "export { main }; struct FieldsAB<T0, T1> { a: T0, b: T1, }\nfn main() -> _  { let mut n: i32 = 0; let mut pair: FieldsAB<_, _>; pair = FieldsAB<_, _> { b = { n = n + 1; n }, a = { n = n + 1; n } }; pair.a * 10 + pair.b }",
        21,
    );
    runs(
        "export { main }; fn add(n: i32) -> i32  { n + 1 } fn select() -> (i32) -> _  { add } fn main() -> _  { select()(41) }",
        42,
    );
}

#[test]
fn array_and_span_indexing_use_element_sizes() {
    runs(
        "export { main }; import { \"$/shared.resin\", \"$/span.resin\" }; fn main () -> i32 | Err<_> { let values_owner = arc_ptr_alloc([10, 20, 30])?; let values: RefMut<_> = values_owner:get().*; let mut p = Span<i32> { data = Ptr<i32>(values_owner:get()), length = u64(3) }; p:at_mut(1) = 7; let mut end = p:at(2); p:at(1) + values(0) + end }",
        47,
    );
    runs(
        "export { main }; struct Payload { marker: u32, wide: u64, amount: f32, } fn main () -> i32  { let mut values = [Payload { marker = u32(1), wide = u64(4294967297), amount = f32(0.5) }, Payload { marker = u32(2), wide = u64(8589934593), amount = f32(1.5) }]; let p: Ref<Payload> = values:at(0); let q: RefMut<Payload> = values:at_mut(1); q.amount = q.amount + f32(2.0); if (q.wide == u64(8589934593) && q.marker == u32(2) && q.amount == f32(3.5) && p.amount == f32(0.5)) { 0 } else { 1 } }",
        0,
    );
}

#[test]
fn numbered_examples_compile_as_strict_c11() {
    for entry in fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/examples")).unwrap() {
        let path = entry.unwrap().path();
        // The interactive example has a bounded window test in mandelbrot.rs.
        if path.ends_with("eg011_mandelbrot.resin") {
            continue;
        }
        if path.extension().is_some_and(|ext| ext == "resin")
            && path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .starts_with("eg")
        {
            let module = pipeline::file_module(&path).unwrap();
            let project = support::project::Project::new(&module, Some("main")).unwrap();
            // GPU execution is covered by ray_tracing.rs; this test also runs
            // on hosts with no Vulkan device or loader.
            if path.ends_with("eg013_ray_tracing.resin") {
                project.build_executable();
                continue;
            }
            let output = project.run();
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
        "export { main }; fn main () -> i32  { let mut x = 0; if (1 == 1) { x = 1; () } else { () }; if (1 == 2) { (1, 2) } else { (3, 4) }; x }",
        1,
    );
}

#[test]
fn while_rechecks_conditions_and_discards_body_values() {
    runs(
        "export { main }; fn main () -> i32  { let mut n = 0; let mut sum = 0; while ({ n = n + 1; n } <= 4) { sum = sum + n }; sum + n }",
        15,
    );
    runs(
        "export { main }; fn main () -> i32  { let mut n = 0; while (1 == 0) { n = 42; }; n }",
        0,
    );
    runs(
        "export { main }; fn main () -> i32  { let mut n: i32; while ({ n = 7; n } == 0) {}; n }",
        7,
    );
    runs(
        "export { main }; fn main () -> i32  { let mut n = 0; while (n < 1000000) { n = n + 1; }; if (n == 1000000) { 0 } else { 1 } }",
        0,
    );
}

#[test]
fn while_nests_with_branches_and_preserves_outer_values() {
    runs(
        "export { main }; fn main () -> i32  { let mut i = 0; let mut total = 0; while (i < 3) { let mut j = 0; while (j < 4) { if (j < 2) { total = total + 1 } else { total = total + 2 }; j = j + 1; }; i = i + 1; }; total }",
        18,
    );
    runs(
        "export { main }; fn main () -> i32  { let mut n = 0; let mut x = 7; while (n < 3) { let mut x = 10; n = n + 1; x = 20; }; x + n }",
        10,
    );
    runs(
        "export { main }; struct FieldsFirstBodyLast<T0, T1, T2> { first: T0, body: T1, last: T2, }\nfn main () -> i32  { let mut n = 0; let mut r = FieldsFirstBodyLast<_, _, _> { first = 9, body = while (n < 3) { n = n + 1; }, last = n }; r.first + r.last }",
        12,
    );
    runs(
        "export { main }; fn main () -> i32  { let mut n = 0; while (if (n < 3) { (1 == 1) && ({ n = n + 1; n } < 3) } else { 1 == 0 }) {}; n }",
        3,
    );
}

#[test]
fn ordinary_functions_can_be_passed_and_selected() {
    runs(
        "export { main }; fn add (x: i32, y: i32) -> i32  { x + y } fn apply (f: (i32, i32) -> i32, args: (i32, i32)) -> i32  { f(args.0, args.1) } fn main () -> i32  { let mut a = add; let mut b = if (1 == 1) { a } else { add }; apply(a, (10, 3)) + b(20, 4) }",
        37,
    );
}

#[test]
fn recursive_functions_receive_state_explicitly() {
    runs(
        "export { main }; fn fact(n: i32, offset: i32) -> i32  { if (n == 0) { offset } else { n * fact(n - 1, offset) } } fn main() -> i32  { let mut offset = 2; fact(4, offset) }",
        48,
    );
}

#[test]
fn mutual_recursion_needs_no_forward_declaration() {
    runs(
        "export { main }; fn f (n: i32) -> i32  { if (n == 0) { 7 } else { next(n - 1) } } fn next (m: i32) -> i32  { f(m) } fn main () -> i32  { f(3) }",
        7,
    );
}

#[test]
fn calls_take_lists_and_tuples_are_explicit_values() {
    runs(
        r#"export { main };
        fn zero() -> i32  { 1 }
        fn unit(value: ()) -> i32  { 2 }
        fn tuple(value: (i32, i32)) -> i32  { value.0 + value.1 }
        fn add<T>(a: T, b: T) -> T  { a + b }
        fn apply(f: (i32, i32) -> i32, a: i32, b: i32) -> i32  { f(a, b) }
        fn main() -> i32  {
            let mut a: () -> i32; a = zero;
            let mut b: (()) -> i32; b = unit;
            let mut c: ((i32, i32)) -> i32; c = tuple;
            a() + b(()) + c((3, 4)) + apply(add::<i32>, 5, 6)
        }
    "#,
        21,
    );
}

#[test]
fn arguments_are_evaluated_left_to_right_after_the_callee() {
    runs(
        r#"export { main };
import { "$/shared.resin" };

        fn mark(trace: Ptr<i32>, digit: i32) -> i32  { trace.* = trace.* * 10 + digit; trace.* }
        fn consume(a: i32, b: i32)  {}
        fn callee(trace: Ptr<i32>) -> (i32, i32) -> ()  { mark(trace, 1); consume }
        fn main() -> i32 | Err<_> {
            let trace_owner = arc_ptr_alloc(0)?; let trace: Ref<_> = trace_owner:get().*;
            callee(trace_owner:get())(mark(trace_owner:get(), 2), mark(trace_owner:get(), 3));
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
        (
            "[make(trace, 1), { let second = make(trace, 2); second:get():accept(fail()?) }]",
            21,
        ),
        (
            "consume(make(trace, 1), { let mut inner = make(trace, 2); owned_error(trace)? })",
            215,
        ),
        (
            "last(make(trace, 1), if ({ let condition = make(trace, 3); condition:get():truth() }) { make(trace, 2) } else { make(trace, 4) }, fail()?)",
            321,
        ),
        (
            "consume(make(trace, 1), last(make(trace, 2), make(trace, 3), { let mut inner = make(trace, 4); succeed(trace)? }))",
            43251,
        ),
    ] {
        let declarations = r#"export { main };
            import { "$/shared.resin" };
            struct Resource { trace: Ptr<i32>, digit: i32,
                
                
                
            }
fn drop(self: RefMut<Resource>)  { if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; }; }

fn accept(self: Ptr<Resource>, other: ArcPtr<Resource>) -> ArcPtr<Resource>  { other }

fn truth(self: Ptr<Resource>) -> bool  { 1 == 1 }

            struct Failed {}
            struct OwnedFailed { value: ArcPtr<Resource>, }
            fn make(trace: Ptr<i32>, digit: i32) -> ArcPtr<Resource>  {
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
            fn succeed(trace: Ptr<i32>) -> (ArcPtr<Resource> | Err<Failed>)  { (make(trace, 5)) }
            fn owned_error(trace: Ptr<i32>) -> (ArcPtr<Resource> | Err<OwnedFailed>)  { Err(OwnedFailed { value = make(trace, 5) }) }
        "#;
        let source = format!(
            "{declarations}\n            fn attempt(trace: Ptr<i32>) -> () | Err<_> {{ {call}; () }}\n            fn main() -> i32 | Err<_> {{\n                let trace = arc_ptr_alloc(i32(0))?;\n                attempt(trace:get());\n                if (trace:get().* == {expected}) {{ 0 }} else {{ 1 }}\n            }}"
        );
        runs(&source, 0);
    }
}

#[test]
fn tuple_projection_preserves_places_and_nested_values() {
    runs(
        r#"export { main };
import { "$/shared.resin" };

        fn main() -> i32 | Err<_> {
            let pair_owner = arc_ptr_alloc(((1, 2), 3))?; let pair: RefMut<_> = pair_owner:get().*;
            pair.0.1 = 20;
            let mut pointer = &pair_owner:get().1;
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
        "export { main }; struct R { a: i32, b: i32, } fn main () -> i32  { let mut x = 0; let mut r = R { b = { x = 1; x }, a = { x = 2; x } }; r.a * 10 + r.b + x }",
        23,
    );
}

#[test]
fn loaded_values_do_not_change_after_later_stores() {
    runs(
        "export { main }; fn main () -> i32  { let mut x = 1; let mut old = x; x = 2; old * 10 + x }",
        12,
    );
}

#[test]
fn short_circuiting_and_joins_preserve_effects() {
    runs(
        "export { main }; fn main () -> i32  { let mut x = 0; let mut a = (1 == 2) && ({ x = 1; x } == 1); let mut b = (1 == 1) || ({ x = 2; x } == 2); let mut n = if (a || b) { 3 } else { 4 }; n + x }",
        3,
    );
}

#[test]
fn aliases_preserve_function_types_and_numeric_operations() {
    runs(
        "export { main }; type Meters = i32; type F = (Meters) -> Meters; fn add (x: Meters) -> Meters  { x + Meters(2) } fn main () -> i32  { let mut f = F(add); i32(f(Meters(5))) }",
        7,
    );
    runs(
        "export { main }; type Entry = () -> i32; fn start() -> i32  { 23 } fn main() -> i32  { let mut entry = Entry(start); entry() }",
        23,
    );
}

#[test]
fn integer_arithmetic_wraps_at_its_declared_width() {
    runs(
        "export { main }; fn main () -> i32  { let mut a: i8 = 127; let mut b = a + i8(1); let mut c = i32(2147483647) + i32(1); let mut d = i64(-9223372036854775808) / i64(-1); if (b == i8(-128) && c == i32(-2147483648) && d == i64(-9223372036854775808)) { 0 } else { 1 } }",
        0,
    );
}

#[test]
fn signed_right_shift_and_unsigned_multiplication_are_defined() {
    runs(
        "export { main }; fn main () -> i32  { let mut a = i32(-8) >> i32(2); let mut b = u32(4294967295) * u32(4294967295); if (a == i32(-2) && b == u32(1)) { 0 } else { 1 } }",
        0,
    );
}

#[test]
fn invalid_integer_operations_fail_at_runtime() {
    for expression in ["1 / 0", "1 % 0", "1 << 32", "1 >> -1"] {
        let result = run_module(&module(&format!(
            "export {{ main }}; fn main () -> i32  {{ {expression} }}"
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
    let mut m = module("export { main }; fn main () -> i32  { 0 }");
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
    let mut m = module("export { main }; fn main () -> i32  { 0 }");
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
    let mut m = module("export { main }; fn main () -> i32  { 0 }");
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
            instrs: vec![LocalRef { local: counter }, int(3), Store, Discard, int(0)],
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
                LocalRef { local: counter },
                Load,
                int(0),
                op(">", Ty::Bool),
            ],
            terminator: LoopTest,
        },
        BasicBlock {
            name: None,
            instrs: vec![
                LocalRef { local: counter },
                Load,
                op("+", Ty::Int32),
                LocalRef { local: counter },
                LocalRef { local: counter },
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
    let mut m = module("export { main }; fn main () -> i32  { 0 }");
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
        LocalRef { local: array },
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
        LocalRef { local: array },
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
        import { "$/shared.resin", "$/span.resin" };
        fn view(p: Ptr<i32>, calls: Ptr<i32>) -> Span<i32>  {
            calls.* = calls.* + 1;
            Span<i32> { data = p, length = u64(3) }
        }
        fn index(calls: Ptr<i32>) -> i32  { calls.* = calls.* + 1; 1 }
        fn main() -> i32 | Err<_> {
            let values_owner = arc_ptr_alloc([i32(10), 20, 30])?; let values: RefMut<_> = values_owner:get().*; let calls_owner = arc_ptr_alloc(0)?; let calls: Ref<_> = calls_owner:get().*;
            let span = view(Ptr<i32>(values_owner:get()), calls_owner:get());
            let mut p: Ptr<i32>; p = span:lea(u64(index(calls_owner:get())));
            p.* = 42;
            let mut copied = values;
            copied:at_mut(0) = 9;
            let pair = [7, 8]; let mut temporary = pair(1);
            if (calls == 2 && values(1) == 42 && values(0) == 10 && copied(0) == 9 && temporary == 8) { 0 } else { 1 }
        }"#,
        0,
    );
}

#[test]
fn at_indexing_borrows_array_places_and_supports_field_receivers() {
    runs(
        r#"export { main };
        import { "$/shared.resin", "$/span.resin" };
        struct FieldsValues<T0> { values: T0, }
struct Holder { values: Span<i32>, }
        fn view(p: Ptr<i32>, calls: Ptr<i32>) -> Holder  {
            calls.* = calls.* + 1;
            Holder { values = Span<i32> { data = p, length = u64(3) } }
        }
        fn index(calls: Ptr<i32>) -> i32  { calls.* = calls.* + 1; 1 }
        fn element(s: Span<i32>, i: u64) -> RefMut<i32>  { s:at_mut(i) }
        fn main() -> i32 | Err<_> {
            let values_owner = arc_ptr_alloc([i32(10), 20, 30])?; let values: RefMut<_> = values_owner:get().*; let calls_owner = arc_ptr_alloc(0)?; let calls: Ref<_> = calls_owner:get().*;
            let p: RefMut<i32> = { let borrowed = view(Ptr<i32>(values_owner:get()), calls_owner:get()).values; borrowed:at_mut(u64(index(calls_owner:get()))) };
            p = 42;
            let mut record = FieldsValues<_> { values = [3, 4] };
            record.values:at_mut(0) = 8;
            let mut holder = view(Ptr<i32>(values_owner:get()), calls_owner:get());
            element(holder.values, 0) = 11;
            let mut temporary = { let borrowed = [7, 8]; borrowed:at(1) };
            if (calls == 3 && values:at(1) == 42 && values:at(0) == 11 && record.values:at(0) == 8 && temporary == 8) { 0 } else { 1 }
        }"#,
        0,
    );
}

#[test]
fn at_indexing_checks_bounds_before_later_effects() {
    for receiver in ["values", "holder.values"] {
        for index in ["2", "u64(18446744073709551615)"] {
            let output = run_module(&module(&format!(
                r#"export {{ main }};

                extern {{
                    "stdio.h": {{
                        fn puts(text: Ptr<u8>) -> i32;
                    }},
                }};
                import {{ "$/shared.resin", "$/span.resin" }};
                struct FieldsValues<T0> {{ values: T0, }}
fn main() -> i32 | Err<_> {{
                    let values_owner = arc_ptr_alloc([1, 2])?; let values: RefMut<_> = values_owner:get().*;
                    let mut holder = FieldsValues<_> {{ values = Span<i32> {{ data = Ptr<i32>(values_owner:get()), length = u64(2) }} }};
                    {receiver}:at_mut({index}) = 9;
                    puts("after".data); 0
                }}"#
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
        "export { main }; fn main() -> i32  { let mut xs = [1, 2]; xs(-1) }",
        "export { main }; fn main() -> i32  { let mut xs = [1, 2]; xs:at_mut(2) = 9; 0 }",
        "export { main }; import { \"$/shared.resin\", \"$/span.resin\" }; fn main() -> i32 | Err<_> { let xs_owner = arc_ptr_alloc([1, 2])?; let xs: Ref<_> = xs_owner:get().*; let mut s = Span<i32> { data = Ptr<i32>(xs_owner:get()), length = u64(2) }; s:at(u64(18446744073709551615)) }",
        "export { main }; import { \"$/span.resin\" }; fn main() -> i32  { let mut s = Span<i32> { data = Ptr<i32>(u64(0)), length = u64(0) }; s:at(0) }",
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
import { "$/shared.resin" };

        fn twice(i: u32) -> u32  { i * u32(2) }
        @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { twice(i) }; }
        fn main() -> i32 | Err<_> { let mut f = kernel; let output_owner = arc_ptr_alloc(u32(0))?; let output: Ref<_> = output_owner:get().*; f(u64(21), output_owner:get()); if (output == u32(42)) { 0 } else { 1 } }
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
    // host allocations. GPU initialization stores these same generated values.
    file.stmts.extend(support::parse(r#"
import { "$/shared.resin" };
fn main() -> i32 | Err<_> {
            let state_owner = arc_ptr_alloc(u32(12345) | u32(1))?; let state: RefMut<_> = state_owner:get().*;
            let particle_owner = arc_ptr_alloc(random_particle(state))?; let particle: RefMut<_> = particle_owner:get().*;
            let mut particles = Span<Particle> { data = particle_owner:get(), length = u64(1) };
            let first = Particle { x = particle.x, y = particle.y, z = particle.z, vx = particle.vx, vy = particle.vy, vz = particle.vz };
            state = u32(12345) | u32(1);
            particle = random_particle(state);
            let mut valid = particle.x == first.x && particle.vz == first.vz;
            state = u32(54321) | u32(1);
            particle = random_particle(state);
            valid = valid && particle.x != first.x && particle.vx != first.vx;
            valid = valid && particle.x >= f32(-24) && particle.x < f32(24) && particle.y >= f32(-30) && particle.y < f32(30) && particle.z >= f32(0) && particle.z < f32(50) && particle.vx >= f32(-6) && particle.vx < f32(6) && particle.vy >= f32(-6) && particle.vy < f32(6) && particle.vz >= f32(-6) && particle.vz < f32(6);
            let params_owner = arc_ptr_alloc(Params { dt = f32(0.005), yaw_cos = f32(1), yaw_sin = f32(0), pitch_cos = f32(1), pitch_sin = f32(0), zoom = f32(1), aspect = f32(0.625), radius = f32(0.0012), particles = particles })?; let params: RefMut<_> = params_owner:get().*;
            let mut steps = 0;
            while (steps < 2000) {
                kernel(u64(0), params_owner:get());
                valid = valid && particle.x > f32(-100) && particle.x < f32(100) && particle.y > f32(-100) && particle.y < f32(100) && particle.z > f32(-100) && particle.z < f32(100);
                steps = steps + 1;
            };
            // Check every particle boundary, including float32 rounding above 2^24.
            let mut index = 0;
            while (index < 1000000) {
                valid = valid && particle_index(index * 24) == index && particle_index(index * 24 + 23) == index;
                index = index + 1;
            };
            particle = Particle { x = f32(0), y = f32(-20), z = f32(25), vx = f32(0), vy = f32(0), vz = f32(0) };
            let mut near = vertex(0, params_owner:get());
            let mut near_rim = vertex(1, params_owner:get());
            particle.y = f32(20);
            let mut far = vertex(0, params_owner:get());
            let mut far_rim = vertex(1, params_owner:get());
            valid = valid && fragment(near.color).b > fragment(far.color:clone()).b;
            valid = valid && near_rim.position.x - near.position.x > far_rim.position.x - far.position.x;
            // Camera controls change the projection without changing the simulation.
            let camera_owner = arc_ptr_alloc(Camera { yaw = f32(0), pitch = f32(0), zoom = f32(1) })?; let camera: RefMut<_> = camera_owner:get().*;
            apply_camera(camera, params);
            let mut before = vertex(1, params_owner:get());
            camera.zoom = f32(2);
            apply_camera(camera, params);
            let mut zoomed = vertex(1, params_owner:get());
            valid = valid && zoomed.position.x == before.position.x * f32(2);
            move_camera(camera, f64(100), f64(50), f64(0));
            apply_camera(camera, params);
            let mut orbited = vertex(0, params_owner:get());
            valid = valid && orbited.position.x > f32(0) && orbited.position.y < f32(0);
            let mut yaw_length = params.yaw_cos * params.yaw_cos + params.yaw_sin * params.yaw_sin;
            let mut pitch_length = params.pitch_cos * params.pitch_cos + params.pitch_sin * params.pitch_sin;
            valid = valid && yaw_length > f32(0.999) && yaw_length < f32(1.001) && pitch_length > f32(0.999) && pitch_length < f32(1.001);
            move_camera(camera, f64(0), f64(1000000), f64(1000000));
            valid = valid && camera.pitch == f32(1.4) && camera.zoom == f32(3);
            move_camera(camera, f64(0), f64(-1000000), f64(-1000000));
            valid = valid && camera.pitch == f32(-1.4) && camera.zoom == f32(0.35);
            camera = default_camera();
            move_camera(camera, f64(0), f64(0), f64(0.5));
            valid = valid && camera.zoom > f32(1) && camera.zoom < f32(1.1);
            move_camera(camera, f64(0), f64(0), f64(-0.5));
            valid = valid && camera.zoom > f32(0.999) && camera.zoom < f32(1.001);
            let mut color = fragment(far.color);
            if (valid && particle.x != first.x && color.r >= f32(0) && color.r <= f32(1) && color.b >= f32(0) && color.b <= f32(1)) { 0 } else { 1 }
        }
    "#).stmts);
    let m = pipeline::generate_program(&program).unwrap();
    assert!(m.shaders.values().all(|entry| !entry.embedded));
    assert!(run_module(&m).status.success());
}

#[test]
fn spirv_is_only_special_on_function_declarations() {
    runs(
        "export { main }; struct FieldsSpirv<T0> { spirv: T0, }\nfn main() -> i32  { let mut record = FieldsSpirv<_> { spirv = 1 }; record.spirv = 2; record.spirv }",
        2,
    );
}

#[test]
fn contextual_literals_execute_with_their_selected_widths() {
    runs(
        r#"export { main }; fn main() -> i32  {
        let mut a: i8 = -128; let mut b: u8 = 255; let mut c: i16 = -32768; let mut d: u16 = 65535;
        let mut e: i32 = -2147483648; let mut f: u32 = 4294967295;
        let mut g: i64 = -9223372036854775808; let mut h: u64 = 18446744073709551615;
        if (a < i8(0) && b > u8(0) && c < i16(0) && d > u16(0) && e < i32(0) && f > u32(0) && g < i64(0) && h == u64(0xffffffffffffffff) && f32(1.5) + f32(2.5) == f32(4) && f64(1e2) == f64(100) && f32(1.0000000596046448) > f32(1)) { 0 } else { 1 }
    }"#,
        0,
    );
}

#[test]
fn else_if_chains_select_one_branch_and_short_circuit_conditions() {
    runs(
        r#"export { main };
import { "$/shared.resin" };

    fn condition(calls: Ptr<i32>, value: i32, expected: i32) -> bool  {
        calls.* = calls.* + 1;
        value == expected
    }
    fn classify(value: i32, calls: Ptr<i32>) -> i32  {
        if (condition(calls, value, 0)) { 10 }
        else if (condition(calls, value, 1)) { 20 }
        else if (condition(calls, value, 2)) { 30 }
        else { 40 }
    }
    fn main() -> i32 | Err<_> {
        let calls_owner = arc_ptr_alloc(0)?; let calls: Ref<_> = calls_owner:get().*;
        let mut a = classify(0, calls_owner:get());
        let mut b = classify(1, calls_owner:get());
        let mut c = classify(2, calls_owner:get());
        let mut d = classify(3, calls_owner:get());
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
import { "$/shared.resin" };

    struct Add { value: Ptr<i32>,
        
    }
fn drop(self: RefMut<Add>)  { self.value.* = self.value.* + 10; }


    fn condition(calls: Ptr<i32>) -> bool  { calls.* = calls.* + 1; 1 == 1 }
    fn main() -> i32 | Err<_> {
        let calls_owner = arc_ptr_alloc(0)?; let calls: Ref<_> = calls_owner:get().*; let value_owner = arc_ptr_alloc(0)?; let value: RefMut<_> = value_owner:get().*;
        if (condition(calls_owner:get())) { let mut cleanup = Add { value = value_owner:get() }; value = value + 1; };
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
import {{ "$/shared.resin" }};

            struct E {{}}
            struct Capture {{ resource: Ptr<i32>, trace: Ptr<i32>,
                
            }}
fn drop(self: RefMut<Capture>)  {{ self.trace.* = self.trace.* * 10 + self.resource.*; }}


            fn work(trace: Ptr<i32>, fail: bool) -> () | Err<_> {{
                let resource = arc_ptr_alloc(i32(1))?;
                let captured = arc_ptr_alloc(resource:get().*)?;
                let mut first = Capture {{ resource = captured:get(), trace = trace }};
                let mut second = Capture {{ resource = resource:get(), trace = trace }};
                resource:get().* = 2;
                {{ let mut captured = 9; }};
                let mut result: (() | Err<E>); result = if (fail) {{ Err(E {{}}) }} else {{ (()) }};
                result?;
                (())
            }}
            fn main() -> i32 | Err<_> {{
                let trace_owner = arc_ptr_alloc(0)?; let trace: Ref<_> = trace_owner:get().*;
                match (work(trace_owner:get(), {fail})) {{ ()(v) => {{}}, Err(e) => {{}} }};
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
import { "$/shared.resin" };

        struct FieldsBytesTail<T0, T1> { bytes: T0, tail: T1, }
fn main() -> i32 | Err<_> {
            let mut binary = [u8(65), u8(66)];
            let mut copied = binary;
            let nested_owner = arc_ptr_alloc([[u8(1), u8(2)], [u8(3), u8(4)]])?; let nested: Ref<_> = nested_owner:get().*;
            let mut embedded = [u8(65), u8(0), u8(66)];
            let record_owner = arc_ptr_alloc(FieldsBytesTail<_, _> { bytes = copied, tail = u8(255) })?; let record: Ref<_> = record_owner:get().*;
            binary:at_mut(0) = u8(90);
            if (size_of(binary) == u64(2) && align_of(binary) == u64(1) &&
                size_of(nested) == u64(4) && size_of(embedded) == u64(3) &&
                u64(nested_owner:get():lea(1)) - u64(nested_owner:get():lea(0)) == u64(2) &&
                size_of(record) == u64(3) && u64(&record_owner:get().tail) - u64(&record_owner:get().bytes) == u64(2) &&
                copied:at(0) == u8(65) && copied:at(1) == u8(66) &&
                record.tail == u8(255) && embedded:at(1) == u8(0)) { 0 } else { 1 }
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
        struct Inner { x: u32, y: u64, z: f32, }
        struct Outer { first: u32, inner: Inner, last: f32, }
        fn main() -> i32  {
            let mut side: u32 = 0;
            let mut values = [u32(1), u32(2), u32(3)];
            if (size_of(u32) == u64(4) && align_of(u64) == u64(8) && size_of(Outer) == u64(40) &&
                align_of(Outer) == u64(8) && size_of(Span<u32>) == u64(16) &&
                size_of(values) == u64(12) && size_of({ side = u32(1); side }) == u64(4) && side == u32(0)) { 0 } else { 1 }
        }
    "#,
        0,
    );
}

#[test]
fn numeric_conversions_check_runtime_values_and_boundaries() {
    runs(include_str!("fixtures/numeric_conversions.resin"), 0);
    runs(
        r#"export { main }; fn main() -> i32  {
        let mut n: u32 = 255; let mut negative: i32 = -128; let mut wide: u64 = 18446744073709551615;
        let mut nan = f32(f64(0.0) / f64(0.0)); let mut large: f64 = 1.0e100; let mut tiny: f64 = -1.0e-100;
        if (u8(n) == u8(255) && i8(negative) == i8(-128) && u64(wide) == wide &&
            f64(n) == f64(255.0) && f32(large) > f32(1.0e30) && f32(tiny) == f32(0.0) && nan != nan) { 0 } else { 1 }
    }"#,
        0,
    );
    for expr in [
        "u8(u32(256))",
        "u32(i32(-1))",
        "i32(u32(2147483648))",
        "u32(f64(4294967296.0))",
        "i64(f64(9223372036854775808.0))",
        "u64(f64(18446744073709551616.0))",
        "i32(f64(0.0) / f64(0.0))",
        "i32(f64(1.0) / f64(0.0))",
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
                "export {{ main }};\nimport {{ \"$/shared.resin\" }};\n fn main() -> i32 | Err<_> {{ let n_owner = arc_ptr_alloc(300)?; let n: Ref<_> = n_owner:get().*; let mut bytes = Ptr<u8>(n_owner:get()); {marker} n - 300 }}"
            ),
            0,
        );
    }
}

#[test]
fn compute_entry_preserves_ulong_indices_on_the_host() {
    runs(
        "export { main };\nimport { \"$/shared.resin\" };\n @compute_shader fn kernel(index: u64, output: Ptr<u64>)  { output.* = index; } fn main() -> i32 | Err<_> { let output_owner = arc_ptr_alloc(u64(0))?; let output: Ref<_> = output_owner:get().*; kernel(u64(4294967297), output_owner:get()); if (output == u64(4294967297)) { 0 } else { 1 } }",
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
        "export { main }; struct Failed {} fn step() -> (() | Err<Failed>)  { (()) } fn main() -> (i32 | Err<Failed>) { let mut value = 0; ",
    );
    for _ in 0..512 {
        source.push_str("if (value == 0) { value = 1; } else { value = 0; }; step()?; ");
    }
    source.push_str("(value) }");
    let project = support::project::Project::new(&module(&source), Some("main")).unwrap();
    let c = fs::read_to_string(project.generated.c_source().unwrap()).unwrap();
    assert!(
        c.lines()
            .all(|line| line.len() - line.trim_start().len() < 32)
    );
    assert!(project.run().status.success());
}
