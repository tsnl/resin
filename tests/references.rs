mod support;

fn result(source: &str) -> i32 {
    let output = support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run();
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
        .status
        .code()
        .expect("reference program terminated normally")
}

#[test]
fn reference_calls_preserve_aliases_while_value_bindings_copy() {
    assert_eq!(
        result(
            r#"
        export { main };
        def identity<T>(value: Ref<T>) -> Ref<T> = { value };
        def observe(left: Ref<int>, right: Ref<int>) -> int = { left := 20; right };
        def main() -> int = {
            var x: int = 1;
            var reference: Ref<_> = identity(x);
            var copied = reference;
            var address = &reference;
            identity(x) := 7;
            observe(reference, address.*) + copied + x
        };
    "#
        ),
        41
    );
}

#[test]
fn reference_returning_function_values_and_branches_select_storage() {
    assert_eq!(
        result(
            r#"
        export { main };
        def choose(a: Ref<int>, b: Ref<int>, flag: bool) -> Ref<int> = {
            if (flag) { a } else { b }
        };
        def main() -> int = {
            var a = 1_i; var b = 2_i;
            var select = choose;
            select(a, b, 1 == 0) := 41;
            a + b
        };
    "#
        ),
        42
    );
}

#[test]
fn reference_methods_and_dependent_calls_mutate_the_original() {
    assert_eq!(
        result(
            r#"
        export { main };
        struct Cell<T> { value: T,
            def get(self: Ref<Cell<T>>) -> Ref<T> = { self.value };
            def set(self: Ref<Cell<T>>, other: Ref<T>) = { self.value := other; };
        };
        def get<C>(cell: Ref<C>) -> _ = { cell.get() };
        def set<C, T>(cell: Ref<C>, value: Ref<T>) = { cell.set(value); };
        def main() -> int = {
            var cell = Cell<int> { value = 1 };
            cell.get() := 7;
            var replacement = 35_i;
            var old = get(cell);
            set(cell, replacement);
            old + cell.value
        };
    "#
        ),
        42
    );
}

#[test]
fn assignment_evaluates_a_reference_accessor_once() {
    assert_eq!(
        result(
            r#"
        export { main };
        def get(count: Ref<int>, value: Ref<int>) -> Ref<int> = {
            count := count + 1;
            value
        };
        def main() -> int = {
            var count = 0_i; var value = 0_i;
            get(count, value) := 41;
            count + value
        };
    "#
        ),
        42
    );
}

#[test]
fn references_to_pointer_slots_are_fixed_aliases() {
    assert_eq!(
        result(
            r#"
        export { main };
        def main() -> int = {
            var a = 1_i; var b = 2_i;
            var pointer = &a;
            var slot: Ref<Ptr<int>> = pointer;
            slot := &b;
            slot.* := 41;
            a + pointer.*
        };
    "#
        ),
        42
    );
}

#[test]
fn references_borrow_managed_storage_and_value_reads_copy() {
    assert_eq!(
        result(
            r#"
        export { main };
        struct Resource { drops: Ptr<int>,
            def drop(self: Ptr<Resource>) = { self.drops.* := self.drops.* + 1; };
        };
        def identity(value: Ref<Resource>) -> Ref<Resource> = { value };
        def main() -> int = {
            var drops = 0_i;
            {
                var resource = Resource { drops = &drops };
                {
                    var reference: Ref<Resource> = identity(resource);
                    { var copied = reference; };
                    if (drops != 1_i) { drops := 90; };
                    reference := Resource { drops = &drops };
                };
                if (drops != 3_i) { drops := 90; };
            };
            drops
        };
    "#
        ),
        4
    );
}

#[test]
fn dependent_reference_results_bind_but_dependent_value_temporaries_do_not() {
    for body in ["get(cell)", "cell.get()"] {
        let source = format!(
            r#"
            export {{ main }};
            struct Cell {{ value: int,
                def get(self: Ref<Cell>) -> Ref<int> = {{ self.value }};
            }};
            def get<C>(cell: Ref<C>) -> Ref<int> = {{ cell.get() }};
            def main() -> int = {{ var cell = Cell {{ value = 1 }}; {body} := 42; cell.value }};
        "#
        );
        assert_eq!(result(&source), 42);
    }
    let source = r#"
        struct Cell { value: int,
            def take(self: Ref<Cell>, value: Ref<int>) = {};
        };
        def value() -> int = { 1 };
        def call<C>(cell: Ref<C>) = { cell.take(value()); };
        def main() = { var cell = Cell { value = 1 }; call(cell); };
    "#;
    let error = support::pipeline::source_module(source).unwrap_err();
    assert!(
        error.to_string().contains("reference binding requires"),
        "{error}"
    );
}

#[test]
fn shader_references_keep_existing_local_address_restrictions() {
    for (source, diagnostic) in [
        (
            "export { kernel }; def read(value: Ref<uint>) -> uint = { value }; @compute_shader def kernel(i: ulong, output: Ptr<uint>) = { var local = 1_ui; output.* := read(local); };",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; def local() -> Ref<uint> = { var value = 1_ui; value }; @compute_shader def kernel(i: ulong, output: Ptr<uint>) = { output.* := local(); };",
            "cannot return a local address",
        ),
    ] {
        let error = support::pipeline::shader_error(source);
        assert!(error.to_string().contains(diagnostic), "{error}");
    }
}

#[test]
fn stored_function_signatures_preserve_reference_parameters_and_results() {
    assert_eq!(
        result(
            r#"
        export { main };
        type Access = (Ref<int>) -> Ref<int>;
        struct Accessor { call: Access };
        def identity(value: Ref<int>) -> Ref<int> = { value };
        def invoke<F>(function: F, value: Ref<int>) -> Ref<int> = { function(value) };
        def main() -> int = {
            var accessor = Accessor { call = identity };
            var value = 1_i;
            invoke(accessor.call, value) := 41;
            (accessor.call)(value) := value + 1;
            value
        };
    "#
        ),
        42
    );
}
