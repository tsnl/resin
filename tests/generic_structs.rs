mod support;

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run()
}

#[test]
fn constructors_and_nominal_arguments_infer_function_parameters() {
    let output = run(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        struct Pair<T> { first: T, second: T, }
        fn sum<T>(pair: Pair<T>) -> T  { pair.first + pair.second }
        fn make<T>(first: T, second: T) -> Pair<T>  {
            Pair<T> { first = first, second = second }
        }
        fn main() -> i32  {
            let mut small = Pair<u8> { first = 250, second = 5 };
            let mut large = make(u64(4294967296), u64(2));
            { let borrowed = fmt("{0} {1}", (sum(small), sum(large))); print(borrowed) };
            0
        }
    "#,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"255 4294967298");
}

#[test]
fn pointers_to_generic_records_preserve_field_places() {
    let output = run(r#"export { main };
import { "$/shared.resin" };

        struct Cell<T> { value: T, }
        fn replace<T>(cell: PtrMut<Cell<T>>, value: T) -> T  {
            let mut previous = cell.value;
            cell.value = value;
            previous
        }
        fn main() -> i32 | Err<_> {
            let cell_owner = arc_ptr_alloc(Cell<i32> { value = 7 })?; let cell: Ref<_> = cell_owner:get().*;
            let mut previous = replace(cell_owner:get(), 35);
            previous + cell.value
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
fn nongeneric_wrappers_can_nest_generic_nominal_storage() {
    let output = run(r#"export { main };
        struct Cell<T> { value: T, }
        struct Wrapped { cell: Cell<i32>, }
        struct Outer { wrapped: Wrapped, }
        fn main() -> i32  {
            let mut outer = Outer {
                wrapped = Wrapped { cell = Cell<i32> { value = 7 } }
            };
            outer.wrapped.cell.value = 42;
            outer.wrapped.cell.value
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
fn nongeneric_wrapper_layout_queries_wait_for_nominal_specialization() {
    let output = run(r#"export { main };
        struct Cell<T> { value: T, }
        struct Wrapped { cell: Cell<i32>, }
        struct Outer { wrapped: Wrapped, }
        fn main() -> i32  { i32(size_of(Wrapped) + size_of(Outer) + align_of(Outer)) }
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
    let output = run(r#"export { main };
        import { "$/shared.resin", "$/span.resin" };
        struct Cell<T> { value: T, }
        fn copy_through_array<T>(value: T) -> T  {
            let mut cells = [Cell<T> { value = value }];
            cells(u64(0)).value
        }
        fn main() -> i32 | Err<_> {
            let cells_owner = arc_ptr_alloc([Cell<i32> { value = 7 }, Cell<i32> { value = 35 }])?; let cells: RefMut<_> = cells_owner:get().*;
            let mut view = SpanMut<Cell<i32>> {
                data = PtrMut<Cell<i32>>(cells_owner:get()), length = 2
            };
            cells:at_mut(u64(0)).value = copy_through_array(cells(u64(0)).value) + view:at(u64(1)).value;
            cells(u64(0)).value
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
fn recursive_generic_pointers_reuse_their_concrete_identity() {
    let output = run(r#"export { main };
import { "$/shared.resin" };

        struct Node<T> { value: T, next: PtrMut<Node<T>>, }
        fn read<T>(node: PtrMut<Node<T>>) -> T  { node.value }
        fn main() -> i32 | Err<_> {
            let tail_owner = arc_ptr_alloc(Node<i32> { value = 35, next = PtrMut<Node<i32>>(u64(0)) })?; let tail: Ref<_> = tail_owner:get().*;
            let head_owner = arc_ptr_alloc(Node<i32> { value = 7, next = tail_owner:get() })?; let head: Ref<_> = head_owner:get().*;
            read(head_owner:get()) + read(head.next)
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
fn generic_nominal_error_types_keep_result_payloads() {
    let output = run(r#"export { main };
        struct Failure<T> { value: T, }
        fn fail<T>(value: T) -> (T | Err<Failure<T>>)  {
            Err(Failure<T> { value = value })
        }
        fn relay<T>(value: T) -> (T | Err<_>)  { (fail(value)?) }
        fn main() -> i32  {
            match (relay(i32(42))) {
                i32(value) => { 0 },
                Err(error) => { error.value }
            }
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
fn shader_fields_specialize_generic_nominal_storage() {
    let module = support::module(
        r#"export { kernel };
        struct Cell<T> { value: T, }
        fn increment<T>(cell: PtrMut<Cell<T>>)  { cell.value = cell.value + 1; }
        @compute_shader fn kernel(index: u64, root: PtrMut<Cell<u32>>)  {
            increment(root);
        }
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
            "export { Pair }; struct Pair<T> { first: T, second: T, }",
        ),
        (
            "alias.resin",
            "export { Renamed }; import { \"pair.resin\" }; type Renamed<U> = Pair<U>;",
        ),
        (
            "main.resin",
            r#"export { main };
            import { "pair.resin", "alias.resin" };
            fn first<T>(pair: Pair<T>) -> T  { pair.first }
            fn main() -> i32  {
                first(Pair<i32> { first = 20, second = 0 }) +
                first(Renamed<i32> { first = 22, second = 0 })
            }
        "#,
        ),
    ] {
        std::fs::write(directory.path().join(name), source).unwrap();
    }
    let module = support::pipeline::file_module(&directory.path().join("main.resin")).unwrap();
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
    let source = "struct Recursive<T> { next: Recursive<T>, } fn marker<T>() -> i32  { 42 } fn main() -> i32  { marker::<Recursive<i32>>() }";
    let module = support::module(source);
    assert!(!module.types.iter().any(|definition| {
        definition
            .name()
            .is_some_and(|name| name.starts_with("Recursive<"))
    }));
}

#[test]
fn demanding_an_infinite_generic_layout_reports_the_application() {
    let source = "struct Recursive<T> { next: Recursive<T>, } fn measure<T>() -> u64  { size_of(T) } fn main() -> u64  { measure::<Recursive<i32>>() }";
    let hir = support::hir(source);
    let error = support::frontend::lower(&hir, &[], &resin_lir::LoweringOptions::default())
        .unwrap_err()
        .remove(0);
    assert!(!error.applications.is_empty(), "{error}");
    assert!(
        error.applications.iter().any(|application| application
            .arguments
            .iter()
            .any(|argument| argument.as_ref() == "Recursive<i32>")),
        "{error:?}"
    );
}

#[test]
fn generic_constructor_literals_are_range_checked_after_substitution() {
    let source = "struct Cell<T> { value: T, } fn make<T>() -> Cell<T>  { Cell<T> { value = 256 } } fn main() -> Cell<u8>  { make() }";
    let hir = support::hir(source);
    let error = support::frontend::lower(&hir, &[], &resin_lir::LoweringOptions::default())
        .unwrap_err()
        .remove(0);
    assert!(!error.applications.is_empty(), "{error}");
    assert!(error.to_string().contains("out of range"), "{error}");
}

#[test]
fn local_structs_capture_outer_types_and_specialize_each_layout() {
    let output = run(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn pair<T>(value: T) -> _  {
            struct Local<U> { outer: T, inner: U, }
            Local<i32> { outer = value, inner = 35 }
        }
        fn measure<T>(value: T) -> u64  { size_of(T) }
        fn main() -> i32  {
            let mut small = pair(i32(7));
            let mut large = pair(u64(4294967296));
            { let borrowed = fmt("{0} {1} {2} {3} {4} {5}", (
                small.outer, large.outer, small.inner, large.inner,
                measure(small), measure(large)
            )); print(borrowed) };
            0
        }
    "#,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"7 4294967296 35 35 8 16");
}

#[test]
fn optional_generic_structs_preserve_the_payload_after_unwrapping() {
    let output = run(r#"export { main };
        struct Cell<T> { value: T, }
        fn present<T>(value: T) -> Cell<T> | None  { Cell<T> { value = value } }
        fn main() -> i32  { present(i32(42))!.value }
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
    let output = run(r#"export { main };
        import { "$/shared.resin" };
        struct Resource { trace: PtrMut<i32>, answer: i32,
            
        }
fn drop(self: RefMut<Resource>)  { if (self.answer != 0) { self.trace.* = self.trace.* + 1; }; }

        struct Cell<T> { value: T, }
        struct Envelope { owner: ArcPtr<Cell<Resource>>, }
        fn main() -> i32 | Err<_> {
            let trace_owner = arc_ptr_alloc(i32(0))?; let trace: RefMut<_> = trace_owner:get().*;
            {
                let mut optional: ArcPtr<Cell<Resource>> | None;
                optional = match (arc_ptr_alloc::<Cell<Resource>>(Cell<Resource> {
                    value = Resource { trace = trace_owner:get(), answer = 0 }
                })) { ArcPtr<Cell<Resource>>(value) => { value }, Err(error) => { None } };
                let mut owner = optional!;
                owner:get().value.answer = 42;
                let mut first = Envelope { owner = owner };
                let mut second = first;
                if (second.owner:get().value.answer != 42) { trace = 100; };
            };
            trace + 41
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
fn foreign_pointer_signatures_keep_generic_pointees() {
    let output = run(r#"export { main };

        extern {
            "stdlib.h": {
                fn free(value: PtrMut<Cell<i32>>);
            },
        };
        struct Cell<T> { value: T, }
        fn main() -> i32  { free(PtrMut<Cell<i32>>(u64(0))); 42 }"#);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
