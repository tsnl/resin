mod support;

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run()
}

#[test]
fn constructors_and_nominal_arguments_infer_function_parameters() {
    let output = run(r#"
        export { main };
        struct Pair<T> { first: T, second: T };
        def sum<T>(pair: Pair<T>) -> T = { pair.first + pair.second };
        def make<T>(first: T, second: T) -> Pair<T> = {
            Pair<T> { first = first, second = second }
        };
        def main() -> int = {
            var small = Pair<ubyte> { first = 250, second = 5 };
            var large = make(4294967296_ul, 2_ul);
            print(fmt("{0} {1}", (sum(small), sum(large))));
            0
        };
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"255 4294967298");
}

#[test]
fn pointers_to_generic_records_preserve_field_places() {
    let output = run(r#"
        export { main };
        struct Cell<T> { value: T };
        def replace<T>(cell: Ptr<Cell<T>>, value: T) -> T = {
            var previous = cell.value;
            cell.value := value;
            previous
        };
        def main() -> int = {
            var cell = Cell<int> { value = 7 };
            var previous = replace(&cell, 35);
            previous + cell.value
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
fn nongeneric_wrappers_can_nest_generic_nominal_storage() {
    let output = run(r#"
        export { main };
        struct Cell<T> { value: T };
        struct Wrapped { cell: Cell<int> };
        struct Outer { wrapped: Wrapped };
        def main() -> int = {
            var outer = Outer {
                wrapped = Wrapped { cell = Cell<int> { value = 7 } }
            };
            outer.wrapped.cell.value := 42;
            outer.wrapped.cell.value
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
fn nongeneric_wrapper_layout_queries_wait_for_nominal_specialization() {
    let output = run(r#"
        export { main };
        struct Cell<T> { value: T };
        struct Wrapped { cell: Cell<int> };
        struct Outer { wrapped: Wrapped };
        def main() -> int = { int(size_of(Wrapped) + size_of(Outer) + align_of(Outer)) };
    "#);
    assert_eq!(
        output.status.code(),
        Some(12),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn array_and_span_index_calls_preserve_generic_field_places() {
    let output = run(r#"
        export { main };
        struct Cell<T> { value: T };
        def copy_through_array<T>(value: T) -> T = {
            var cells = [Cell<T> { value = value }];
            cells(0_ul).value
        };
        def main() -> int = {
            var cells = [Cell<int> { value = 7 }, Cell<int> { value = 35 }];
            var view = Span<Cell<int>> {
                data = Ptr<Cell<int>>(&cells), length = 2
            };
            cells(0_ul).value := copy_through_array(cells(0_ul).value) + view(1_ul).value;
            cells(0_ul).value
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
fn recursive_generic_pointers_reuse_their_concrete_identity() {
    let output = run(r#"
        export { main };
        struct Node<T> { value: T, next: Ptr<Node<T>> };
        def read<T>(node: Ptr<Node<T>>) -> T = { node.value };
        def main() -> int = {
            var tail = Node<int> { value = 35, next = Ptr<Node<int>>(0_ul) };
            var head = Node<int> { value = 7, next = &tail };
            read(&head) + read(head.next)
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
fn generic_nominal_error_types_keep_result_payloads() {
    let output = run(r#"
        export { main };
        struct Failure<T> { value: T };
        def fail<T>(value: T) -> Result<T, Failure<T>> = {
            err(Failure<T> { value = value })
        };
        def relay<T>(value: T) -> Result<T, _> = { ok(fail(value)?) };
        def main() -> int = {
            match (relay(42_i)) {
                ok(value) => { 0 },
                err(error) => { error.value }
            }
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
fn shader_fields_specialize_generic_nominal_storage() {
    let module = support::module(
        r#"
        export { kernel };
        struct Cell<T> { value: T };
        def increment<T>(cell: Ptr<Cell<T>>) = { cell.value := cell.value + 1; };
        @compute_shader def kernel(index: ulong, root: Ptr<Cell<uint>>) = {
            increment(root);
        };
    "#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    assert_eq!(project.generated.shaders().len(), 1);
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

#[test]
fn imported_aliases_preserve_nominal_identity_and_instance_reuse() {
    let directory = tempfile::tempdir().unwrap();
    for (name, source) in [
        (
            "pair.resin",
            "export { Pair }; struct Pair<T> { first: T, second: T };",
        ),
        (
            "alias.resin",
            "export { Renamed }; import { \"pair.resin\" }; type Renamed<U> = Pair<U>;",
        ),
        (
            "main.resin",
            r#"
            export { main };
            import { "pair.resin", "alias.resin" };
            def first<T>(pair: Pair<T>) -> T = { pair.first };
            def main() -> int = {
                first(Pair<int> { first = 20, second = 0 }) +
                first(Renamed<int> { first = 22, second = 0 })
            };
        "#,
        ),
    ] {
        std::fs::write(directory.path().join(name), source).unwrap();
    }
    let program = support::pipeline::load(&directory.path().join("main.resin")).unwrap();
    let module = support::pipeline::generate_program(&program).unwrap();
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| function.name.as_deref() == Some("first"))
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
fn unused_generic_arguments_do_not_demand_recursive_layouts() {
    let source = "struct Recursive<T> { next: Recursive<T> }; def marker<T>() -> int = { 42 }; def main() -> int = { marker::<Recursive<int>>() };";
    let module = support::module(source);
    assert!(!module.types.iter().any(|definition| {
        definition
            .name()
            .is_some_and(|name| name.starts_with("Recursive<"))
    }));
}

#[test]
fn demanding_an_infinite_generic_layout_reports_the_application() {
    let source = "struct Recursive<T> { next: Recursive<T> }; def measure<T>() -> ulong = { size_of(T) }; def main() -> ulong = { measure::<Recursive<int>>() };";
    let hir = resin_hir::generate(&support::parse(source)).unwrap();
    let error = resin_lir::generate(&hir).unwrap_err();
    assert!(!error.applications.is_empty(), "{error}");
    assert!(
        error.applications.iter().any(|application| application
            .arguments
            .iter()
            .any(|argument| argument.as_ref() == "Recursive<int>")),
        "{error:?}"
    );
}

#[test]
fn generic_constructor_literals_are_range_checked_after_substitution() {
    let source = "struct Cell<T> { value: T }; def make<T>() -> Cell<T> = { Cell<T> { value = 256 } }; def main() -> Cell<ubyte> = { make() };";
    let hir = resin_hir::generate(&support::parse(source)).unwrap();
    let error = resin_lir::generate(&hir).unwrap_err();
    assert!(!error.applications.is_empty(), "{error}");
    assert!(error.to_string().contains("out of range"), "{error}");
}

#[test]
fn local_structs_capture_outer_types_and_specialize_each_layout() {
    let output = run(r#"
        export { main };
        def pair<T>(value: T) -> _ = {
            struct Local<U> { outer: T, inner: U };
            Local<int> { outer = value, inner = 35 }
        };
        def measure<T>(value: T) -> ulong = { size_of(T) };
        def main() -> int = {
            var small = pair(7_i);
            var large = pair(4294967296_ul);
            print(fmt("{0} {1} {2} {3} {4} {5}", (
                small.outer, large.outer, small.inner, large.inner,
                measure(small), measure(large)
            )));
            0
        };
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"7 4294967296 35 35 8 16");
}

#[test]
fn optional_generic_structs_preserve_the_payload_after_unwrapping() {
    let output = run(r#"
        export { main };
        struct Cell<T> { value: T };
        def present<T>(value: T) -> Cell<T> | None = { Cell<T> { value = value } };
        def main() -> int = { present(42_i)!.value };
    "#);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn nongeneric_wrappers_copy_shared_generic_storage_and_destroy_it_once() {
    let output = run(r#"
        export { main };
        struct Resource { trace: Ptr<int>, answer: int,
            def drop(self: Ptr<Resource>) = { self.trace.* := self.trace.* + 1; };
        };
        struct Cell<T> { value: T };
        struct Envelope { owner: Arc<Cell<Resource>> };
        def main() -> int = {
            var trace = 0_i;
            {
                var owner = Arc<Cell<Resource>>(Cell<Resource> {
                    value = Resource { trace = &trace, answer = 42 }
                });
                var first = Envelope { owner = owner };
                var second = first;
                if (second.owner.value.answer != 42) { trace := 100; };
            };
            trace + 41
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
fn foreign_pointer_signatures_keep_generic_pointees() {
    let output = run(r#"
        export { main };
        struct Cell<T> { value: T };
        extern "stdlib.h" def free(value: Ptr<Cell<int>>);
        def main() -> int = { free(Ptr<Cell<int>>(0_ul)); 42 };
    "#);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
