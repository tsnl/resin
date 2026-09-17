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
        struct Small { value: int,
            
        }
fn read(self: Small) -> int  { self.value }

        struct Large { padding: ulong, value: ulong,
            
        }
fn read(self: Large) -> ulong  { self.value }

        struct Holder<T> { item: T,
            
        }
fn inner<T>(self: Holder<T>) -> T  { self.item }

        fn relay_read<T>(value: T) -> _  { value:read() }
        fn field_method<T>(value: T) -> _  { value.item:read() }
        fn method_field<T>(value: T) -> _  { value:inner().value }
        fn method_method<T>(value: T) -> _  { value:inner():read() }
        fn main() -> int  {
            let mut small = Small { value = 7 };
            let mut large = Large { padding = 3, value = 4294967296 };
            let first = Holder<Small> { item = Small { value = 7 } };
            let second = Holder<Large> { item = Large { padding = 3, value = 4294967296 } };
            print(fmt("{0} {1} {2} {3} {4}", (
                relay_read(small), relay_read(large), field_method(first),
                method_field(Holder<Large> { item = Large { padding = 3, value = 4294967296 } }), method_method(second)
            )));
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
        fn main() -> int  {
            direct::<Factory<ulong>, int>(20).value +
                reference::<Factory<ulong>, int>(20).value +
                receiver(Factory<ulong> {}, 2_i).value
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
fn sum(self: Narrow, first: ubyte, second: ushort) -> ulong  {
                ulong(first) + ulong(second)
            }

        struct Wide {
            
        }
fn sum(self: Wide, first: ulong, second: ulong) -> ulong  { first + second }

        fn sum<T>(value: T) -> _  { value:sum(255, 65535) }
        fn main() -> int  {
            print(fmt("{0} {1}", (sum(Narrow {}), sum(Wide {}))));
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
        struct Counter { value: int,
            
            
        }
fn add(self: Ptr<Counter>, amount: int) -> int  {
                self.value = self.value + amount;
                self.value
            }

fn read(self: Ptr<Counter>) -> int  { self.value }

        fn receiver<T>(trace: Ptr<int>, value: T) -> T  {
            trace.* = trace.* * 10 + 1;
            value
        }
        fn argument(trace: Ptr<int>) -> int  { trace.* = trace.* * 10 + 2; 5 }
        fn add<T>(trace: Ptr<int>, value: T) -> _  {
            receiver(trace, value):add(argument(trace))
        }
        fn add_owned<T>(trace: Ptr<int>, value: T) -> _  {
            receiver(trace, value):get():add(argument(trace))
        }
        fn relay_read<T>(value: T) -> _  { value:read() }
        fn main() -> (int | Err<_>)  {
            let trace_owner = arc_ptr_alloc(0_i)?; let trace: Ref<_> = trace_owner:get().*;
            let local_owner = arc_ptr_alloc(Counter { value = 37 })?; let local: Ref<_> = local_owner:get().*;
            let mut shared = arc_ptr_alloc::<Counter>(Counter { value = 6 })?;
            add(trace_owner:get(), local_owner:get());
            add_owned(trace_owner:get(), shared:clone());
            print(fmt("{0} {1} {2}", (trace, relay_read(local_owner:get()), relay_read(shared:get()))));
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
        struct Callback { invoke: (int, int) -> int, }
        fn increment(value: int, amount: int) -> int  { value + amount }
        fn call<T>(value: T) -> _  { (value.invoke)(40, 2) }
        fn main() -> int  { call(Callback { invoke = increment }) }
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
            "struct Owner {} fn missing(value: bool) -> int { 0 }",
            "value:missing()",
            "missing",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner) -> int  { 42 }\n",
            "value:read().missing",
            "ExpectedRecord",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, n: int) -> int  { n }\n",
            "value:read(1 == 1)",
            "",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, n: int) -> int  { n }\n",
            "value:read(1_i, 2_i)",
            "",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, n: int) -> int  { n }\n",
            "value:read()",
            "arguments",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, a: int, b: int) -> int  { a + b }\n",
            "value:read((1_i, 2_i))",
            "arguments",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, a: int, b: int) -> int  { a + b }\n",
            "{ struct Pair { first: int, second: int, } value:read(Pair { first = 1_i, second = 2_i }) }",
            "",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, a: int, b: int) -> int  { a + b }\n",
            "value:read(1_i, 2_i, 3_i)",
            "",
        ),
        (
            "struct Owner {  }\nfn read<U>(self: Owner, n: U) -> U  { n }\n",
            "value:read::<int, int>(42)",
            "no matching overload",
        ),
        (
            "struct Owner {  }\nfn read(self: Owner, n: ubyte) -> int  { int(n) }\n",
            "value:read(256)",
            "range",
        ),
    ] {
        let source = format!(
            "{declaration} fn relay<T>(value: T) -> int  {{ {body} }} fn main()  {{ relay(Owner {{}}); }}"
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
    let source = "struct FieldsCallback<T0> { callback: T0, }\nfn relay<T>(value: T) -> int  { (value.callback)() } fn main() -> int  { relay(FieldsCallback<_> { callback = 42 }) }";
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
        "export { main }; fn missing(value: int) -> int { value } fn unused<T>(value: T) -> _  { value:missing().field } fn main() -> int  { 42 }",
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
                fn abs(value: int) -> int;
            },
        };
        struct Owner { value: int,
            
        }
fn read(self: Ptr<Owner>) -> int  { abs(self.value) }

        fn relay_read<T>(value: T) -> _  { value:read() }
        @compute_shader fn kernel(index: ulong, root: Ptr<Owner>)  {
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
        fn main()  { step(Grow<int> { value = 1 }); }
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
