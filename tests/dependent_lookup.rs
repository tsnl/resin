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
    let output = run(r#"
        export { main };
        import { "$/string.resin" };
        struct Small { value: int,
            def read(self: Small) -> int = { self.value };
        };
        struct Large { padding: ulong, value: ulong,
            def read(self: Large) -> ulong = { self.value };
        };
        struct Holder<T> { item: T,
            def inner(self: Holder<T>) -> T = { self.item };
        };
        def read<T>(value: T) -> _ = { value.read() };
        def field_method<T>(value: T) -> _ = { value.item.read() };
        def method_field<T>(value: T) -> _ = { value.inner().value };
        def method_method<T>(value: T) -> _ = { value.inner().read() };
        def main() -> int = {
            var small = Small { value = 7 };
            var large = Large { padding = 3, value = 4294967296 };
            var first = Holder<Small> { item = small };
            var second = Holder<Large> { item = large };
            print(fmt("{0} {1} {2} {3} {4}", (
                read(small), read(large), field_method(first),
                method_field(second), method_method(second)
            )));
            0
        };
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
    let source = r#"
        export { main };
        struct Cell<T> { value: T };
        struct Factory<T> {
            def make<U>(value: U) -> Cell<U> = { Cell<U> { value = value } };
            def pass<U>(self: Factory<T>, value: U) -> Cell<U> = { Cell<U> { value = value } };
        };
        def direct<T, U>(value: U) -> _ = { T.make::<U>(value) };
        def receiver<T, U>(owner: T, value: U) -> _ = { owner.pass::<U>(value) };
        def reference<T, U>(value: U) -> _ = {
            var make = T.make::<U>;
            make(value)
        };
        def main() -> int = {
            direct::<Factory<ulong>, int>(20).value +
                reference::<Factory<ulong>, int>(20).value +
                receiver(Factory<ulong> {}, 2_i).value
        };
    "#;
    let module =
        support::frontend::lower(&hir(source), &[], &resin_lir::LoweringOptions::default())
            .unwrap();
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| { function.name.as_deref() == Some("Factory.make") })
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
    let output = run(r#"
        export { main };
        import { "$/string.resin" };
        struct Narrow {
            def sum(self: Narrow, first: ubyte, second: ushort) -> ulong = {
                ulong(first) + ulong(second)
            };
        };
        struct Wide {
            def sum(self: Wide, first: ulong, second: ulong) -> ulong = { first + second };
        };
        def sum<T>(value: T) -> _ = { value.sum(255, 65535) };
        def main() -> int = {
            print(fmt("{0} {1}", (sum(Narrow {}), sum(Wide {}))));
            0
        };
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
    let output = run(r#"
        export { main };
        import { "$/string.resin", "$/shared.resin" };
        struct Counter { value: int,
            def add(self: Ptr<Counter>, amount: int) -> int = {
                self.value := self.value + amount;
                self.value
            };
            def read(self: Counter) -> int = { self.value };
        };
        def receiver<T>(trace: Ptr<int>, value: T) -> T = {
            trace.* := trace.* * 10 + 1;
            value
        };
        def argument(trace: Ptr<int>) -> int = { trace.* := trace.* * 10 + 2; 5 };
        def add<T>(trace: Ptr<int>, value: T) -> _ = {
            receiver(trace, value).add(argument(trace))
        };
        def add_owned<T>(trace: Ptr<int>, value: T) -> _ = {
            receiver(trace, value).get().add(argument(trace))
        };
        def read<T>(value: T) -> _ = { value.read() };
        def main() -> (int | Err<_>) = {
            var trace = 0_i;
            var local = Counter { value = 37 };
            var shared = ArcPtr<Counter>.alloc(Counter { value = 6 })?;
            add(&trace, &local);
            add_owned(&trace, shared);
            print(fmt("{0} {1} {2}", (trace, read(&local), read(shared.get()))));
            (0)
        };
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
    let output = run(r#"
        export { main };
        struct Callback { invoke: (int, int) -> int };
        def increment(value: int, amount: int) -> int = { value + amount };
        def call<T>(value: T) -> _ = { (value.invoke)(40, 2) };
        def main() -> int = { call(Callback { invoke = increment }) };
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
        ("struct Owner {};", "value.missing()", "missing"),
        (
            "struct Owner { def read(self: Owner) -> int = { 42 }; };",
            "value.read().missing",
            "ExpectedRecord",
        ),
        (
            "struct Owner { def read(self: Owner, n: int) -> int = { n }; };",
            "value.read(1 == 1)",
            "",
        ),
        (
            "struct Owner { def read(self: Owner, n: int) -> int = { n }; };",
            "value.read(1_i, 2_i)",
            "",
        ),
        (
            "struct Owner { def read(self: Owner, n: int) -> int = { n }; };",
            "value.read()",
            "arguments",
        ),
        (
            "struct Owner { def read(self: Owner, a: int, b: int) -> int = { a + b }; };",
            "value.read((1_i, 2_i))",
            "arguments",
        ),
        (
            "struct Owner { def read(self: Owner, a: int, b: int) -> int = { a + b }; };",
            "value.read({ first = 1_i, second = 2_i })",
            "",
        ),
        (
            "struct Owner { def read(self: Owner, a: int, b: int) -> int = { a + b }; };",
            "value.read(1_i, 2_i, 3_i)",
            "",
        ),
        (
            "struct Owner { def read<U>(self: Owner, n: U) -> U = { n }; };",
            "value.read(42)",
            "type argument",
        ),
        (
            "struct Owner { def read(self: Owner, n: ubyte) -> int = { int(n) }; };",
            "value.read(256)",
            "range",
        ),
    ] {
        let source = format!(
            "{declaration} def relay<T>(value: T) -> int = {{ {body} }}; def main() = {{ relay(Owner {{}}); }};"
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
    let source = "def relay<T>(value: T) -> int = { (value.callback)() }; def main() -> int = { relay({ callback = 42 }) };";
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
        "export { main }; def unused<T>(value: T) -> _ = { value.missing().field }; def main() -> int = { 42 };",
    );
    let module =
        support::frontend::lower(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn dependent_method_calls_obey_shader_profile_rules() {
    let message = support::pipeline::shader_error(
        r#"
        export { kernel };

        extern {
            "stdlib.h": {
                def abs(value: int) -> int;
            },
        };
        struct Owner { value: int,
            def read(self: Owner) -> int = { abs(self.value) };
        };
        def read<T>(value: T) -> _ = { value.read() };
        @compute_shader def kernel(index: ulong, root: Ptr<Owner>) = {
            root.value := read(root);
        };"#,
    );
    assert!(message.contains("foreign"), "{message}");
}

#[test]
fn growing_dependent_method_applications_obey_the_function_limit() {
    let tree = hir(r#"
        struct Grow<T> { value: T,
            def grow(self: Grow<T>) = { step(Grow<Ptr<T>> { value = &self.value }); };
        };
        def step<T>(value: T) = { value.grow(); };
        def main() = { step(Grow<int> { value = 1 }); };
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
