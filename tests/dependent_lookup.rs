mod support;

fn hir(source: &str) -> resin_hir::Module {
    support::hir(source)
}

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run()
}

#[test]
fn dependent_field_and_method_chains_use_each_concrete_owner() {
    let output = run(r#"export { main };
        import { "$/string.resin", "$/stdio.resin" };
        struct Small { value: i32,
            
        }
fn read(self: Small) -> i32  { self.value }

        struct Large { padding: u64, value: u64,
            
        }
fn read(self: Large) -> u64  { self.value }

        struct Holder<T> { item: T,
            
        }
fn inner<T>(self: Holder<T>) -> T  { self.item }

        fn relay_read<T>(value: T) -> _  { value:read() }
        fn field_method<T>(value: T) -> _  { value.item:read() }
        fn method_field<T>(value: T) -> _  { value:inner().value }
        fn method_method<T>(value: T) -> _  { value:inner():read() }
        fn main() -> i32  {
            let mut small = Small { value = 7 };
            let mut large = Large { padding = 3, value = 4294967296 };
            let first = Holder<Small> { item = Small { value = 7 } };
            let second = Holder<Large> { item = Large { padding = 3, value = 4294967296 } };
            { let borrowed = fmt("{0} {1} {2} {3} {4}", (
                relay_read(small), relay_read(large), field_method(first),
                method_field(Holder<Large> { item = Large { padding = 3, value = 4294967296 } }), method_method(second)
            )); print(borrowed) };
            0
        }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"7 4294967296 7 4294967296 4294967296");
}

#[test]
fn dependent_static_calls_and_references_apply_owner_and_method_arguments() {
    let source = r#"export { main };
        struct Cell<T> { value: T, }
        struct Factory<T> {
            
            
        }
fn factory_make<T, U>(value: U) -> Cell<U>  { Cell<U> { value = value } }

fn pass<T, U>(self: Factory<T>, value: U) -> Cell<U>  { Cell<U> { value = value } }

        fn direct<T, U>(value: U) -> _ { factory_make::<T, U>(value) }
        fn receiver<T, U>(owner: T, value: U) -> _  { owner:pass(value) }
        fn reference<T, U>(value: U) -> _  {
            let make = factory_make::<T, U>;
            make(value)
        }
        fn main() -> i32  {
            direct::<Factory<u64>, i32>(20).value +
                reference::<Factory<u64>, i32>(20).value +
                receiver(Factory<u64> {}, i32(2)).value
        }
    "#;
    let module =
        support::frontend::lower(&hir(source), &[], &resin_lir::LoweringOptions::default())
            .unwrap();
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| { function.name.as_deref() == Some("factory_make") })
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
fn dependent_parameters_give_unsuffixed_arguments_their_concrete_types() {
    let output = run(r#"export { main };
        import { "$/string.resin", "$/stdio.resin" };
        struct Narrow {
            
        }
fn sum(self: Narrow, first: u8, second: u16) -> u64  {
                u64(first) + u64(second)
            }

        struct Wide {
            
        }
fn sum(self: Wide, first: u64, second: u64) -> u64  { first + second }

        fn sum<T>(value: T) -> _  { value:sum(255, 65535) }
        fn main() -> i32  {
            { let borrowed = fmt("{0} {1}", (sum(Narrow {}), sum(Wide {}))); print(borrowed) };
            0
        }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"65790 65790");
}

#[test]
fn dependent_receiver_adaptation_preserves_mutation_and_evaluation_order() {
    let output = run(r#"export { main };
        import { "$/string.resin", "$/stdio.resin", "$/shared.resin" };
        struct Counter { value: i32,
            
            
        }
fn add(self: PtrMut<Counter>, amount: i32) -> i32  {
                self.value = self.value + amount;
                self.value
            }

fn read(self: PtrMut<Counter>) -> i32  { self.value }

        fn receiver<T>(trace: PtrMut<i32>, value: T) -> T  {
            trace.* = trace.* * 10 + 1;
            value
        }
        fn argument(trace: PtrMut<i32>) -> i32  { trace.* = trace.* * 10 + 2; 5 }
        fn add<T>(trace: PtrMut<i32>, value: T) -> _  {
            receiver(trace, value):add(argument(trace))
        }
        fn add_owned<T>(trace: PtrMut<i32>, value: T) -> _  {
            let owner = receiver(trace, value);
            owner:get():add(argument(trace))
        }
        fn relay_read<T>(value: T) -> _  { value:read() }
        fn main() -> (i32 | Err<_>)  {
            let trace_owner = arc_ptr_alloc(i32(0))?; let trace: Ref<_> = trace_owner:get().*;
            let local_owner = arc_ptr_alloc(Counter { value = 37 })?; let local: Ref<_> = local_owner:get().*;
            let mut shared = arc_ptr_alloc::<Counter>(Counter { value = 6 })?;
            add(trace_owner:get(), local_owner:get());
            add_owned(trace_owner:get(), shared:clone());
            { let borrowed = fmt("{0} {1} {2}", (trace, relay_read(local_owner:get()), relay_read(shared:get()))); print(borrowed) };
            (0)
        }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"1212 42 11");
}

#[test]
fn dependent_callable_fields_use_function_signatures() {
    let output = run(r#"export { main };
        struct Callback { invoke: (i32, i32) -> i32, }
        fn increment(value: i32, amount: i32) -> i32  { value + amount }
        fn call<T>(value: T) -> _  { (value.invoke)(40, 2) }
        fn main() -> i32  { call(Callback { invoke = increment }) }
    "#);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn dependent_failures_report_the_demanded_application() {
    for (declaration, body, needle) in [
        (
            "struct Owner {} fn missing(value: bool) -> i32 { 0 }",
            "value:missing()",
            "missing",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner) -> i32  { 42 }\n",
            "value:read().missing",
            "ExpectedRecord",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, n: i32) -> i32  { n }\n",
            "value:read(1 == 1)",
            "",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, n: i32) -> i32  { n }\n",
            "value:read(i32(1), i32(2))",
            "",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, n: i32) -> i32  { n }\n",
            "value:read()",
            "arguments",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, a: i32, b: i32) -> i32  { a + b }\n",
            "value:read((i32(1), i32(2)))",
            "arguments",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, a: i32, b: i32) -> i32  { a + b }\n",
            "{ struct Pair { first: i32, second: i32, } value:read(Pair { first = i32(1), second = i32(2) }) }",
            "",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, a: i32, b: i32) -> i32  { a + b }\n",
            "value:read(i32(1), i32(2), i32(3))",
            "",
        ),
        (
            "struct Owner {  }\nfn read<U>(self: Owner, n: U) -> U  { n }\n",
            "value:read::<i32, i32>(42)",
            "no matching overload",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, n: u8) -> i32  { i32(n) }\n",
            "value:read(256)",
            "range",
        ),
    ] {
        let source = format!(
            "{declaration} fn relay<T>(value: T) -> i32  {{ {body} }} fn main()  {{ relay(Owner {{}}); }}"
        );
        let error =
            support::frontend::lower(&hir(&source), &[], &resin_lir::LoweringOptions::default())
                .unwrap_err()
                .remove(0);
        assert!(error.to_string().contains(needle), "{source}: {error}");
        assert!(
            error
                .applications
                .iter()
                .any(|application| application.function.as_ref() == "relay"),
            "{source}: {error:?}"
        );
        assert!(error.span.end > error.span.start, "{error:?}");
    }
    let source = "struct FieldsCallback<T0> { callback: T0, }\nfn relay<T>(value: T) -> i32  { (value.callback)() } fn main() -> i32  { relay(FieldsCallback<_> { callback = 42 }) }";
    let error = support::frontend::lower(&hir(source), &[], &resin_lir::LoweringOptions::default())
        .unwrap_err()
        .remove(0);
    assert!(error.to_string().contains("function"), "{error}");
    assert!(
        error
            .applications
            .iter()
            .any(|application| application.function.as_ref() == "relay"),
        "{error:?}"
    );
}

#[test]
fn unused_dependent_bodies_do_not_select_methods() {
    let tree = hir(
        "export { main }; fn missing(value: i32) -> i32 { value } fn unused<T>(value: T) -> _  { value:missing().field } fn main() -> i32  { 42 }",
    );
    let module =
        support::frontend::lower(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap();
    assert!(
        module
            .functions
            .iter()
            .all(|function| function.name.as_deref() != Some("unused"))
    );
}

#[test]
fn dependent_method_calls_obey_shader_profile_rules() {
    let message = support::pipeline::shader_error(
        r#"export { kernel };

        extern {
            "stdlib.h": {
                fn abs(value: i32) -> i32;
            },
        };
        struct Owner { value: i32,
            
        }
fn read(self: PtrMut<Owner>) -> i32  { abs(self.value) }

        fn relay_read<T>(value: T) -> _  { value:read() }
        @compute_shader fn kernel(index: u64, root: PtrMut<Owner>)  {
            root.value = relay_read(root);
        }"#,
    );
    assert!(message.contains("foreign"), "{message}");
}

#[test]
fn growing_dependent_method_applications_obey_the_function_limit() {
    let tree = hir(r#"struct Grow<T> { value: T,
            
        }
fn grow<T>(self: Grow<T>)  { step(Grow<(T,)> { value = (self.value,) }); }

        fn step<T>(value: T)  { value:grow(); }
        fn main()  { step(Grow<i32> { value = 1 }); }
    "#);
    let options = resin_lir::LoweringOptions {
        max_monomorphs_per_function: std::num::NonZeroUsize::new(3).unwrap(),
    };
    let errors = support::frontend::lower(&tree, &[], &options).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| { matches!(error.kind, resin_lir::ErrorKind::MonomorphLimit { .. }) }),
        "{errors:?}"
    );
}
