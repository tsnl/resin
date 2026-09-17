mod support;

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run()
}

#[test]
fn generic_free_operations_support_direct_and_receiver_calls() {
    let output = run(r#"export { main };
        import { "$/string.resin", "$/stdio.resin" };
        struct Cell<T> { value: T,
            
            
            
        }
fn cell_make<T>(value: T) -> Cell<T>  { Cell<T> { value = value } }

fn read<T>(self: Ref<Cell<T>>) -> T  { self.value }

fn with<T, U>(self: Ref<Cell<T>>, value: U) -> Cell<U>  { Cell<U> { value = value } }

        fn main() -> int  {
            let mut first = cell_make::<int>(7);
            let mut second = first:with::<_, ulong>(4294967296);
            let mut third = with::<int, ubyte>(first, 255);
            print(fmt("{0} {1} {2}", (first:read(), second:read(), third:read())));
            0
        }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"7 4294967296 255");
}

#[test]
fn factory_method_arguments_follow_expected_results_and_explicit_holes() {
    let output = run(r#"export { main };
        import { "$/string.resin", "$/stdio.resin" };
        struct Cell<T> { value: T, }
        struct Factory {
            
        }
fn create<T>(self: Ref<Factory>) -> Cell<T>  { Cell<T> { value = 41 } }

        fn main() -> int  {
            let mut factory = Factory {};
            let mut first: Cell<int>; first = factory:create();
            let mut second = factory:create::<ulong>();
            let mut third: Cell<ubyte>; third = factory:create::<_>();
            print(fmt("{0} {1} {2}", (first.value, second.value, third.value)));
            first.value + 1
        }
    "#);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"41 41 41");
}

#[test]
fn free_function_references_retain_all_explicit_binders() {
    let output = run(r#"export { main };
        struct Cell<T> { value: T,
            
            
        }
fn cell_make<T>(value: T) -> Cell<T>  { Cell<T> { value = value } }

fn select<T, U>(cell: Ref<Cell<T>>, value: U) -> U  { value }

        fn main() -> int  {
            let mut make = cell_make::<int>;
            let select_int = select::<int, int>;
            let mut cell = make(7);
            select_int(cell, 35) + cell.value
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
fn recursive_methods_complete_results_without_changing_owner_or_method_binders() {
    let output = run(r#"export { main };
        struct Cell<T> { value: T,
            
            
        }
fn read<T>(self: Ref<Cell<T>>, depth: int) -> _  {
                if (depth == 0) { self.value } else { self:read(depth - 1) }
            }

fn choose<T, U>(self: Ref<Cell<T>>, value: U, depth: int) -> _  {
                if (depth == 0) { value } else { self:choose(value, depth - 1) }
            }

        fn main() -> int  {
            let mut cell = Cell<int> { value = 7 };
            cell:read(3) + cell:choose(35, 3)
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
fn generic_drop_hooks_use_each_owner_argument_and_reverse_scope_order() {
    let output = run(r#"export { main };
import { "$/shared.resin" };

        struct Tracked<T> { trace: Ptr<ulong>, value: T,
            
            
            
        }
fn tracked_make<T>(trace: Ptr<ulong>, value: T) -> Tracked<T>  {
                Tracked<T> { trace = trace, value = value }
            }

fn read<T>(self: Ref<Tracked<T>>) -> T  { self.value }

fn drop<T>(self: Ref<Tracked<T>>)  {
                self.trace.* = self.trace.* * 10 + size_of(T);
            }

        fn main() -> int | Err<_> {
            let trace_owner = arc_ptr_alloc(0_ul)?; let trace: Ref<_> = trace_owner:get().*;
            {
                let mut first = tracked_make::<int>(trace_owner:get(), 7);
                let mut second = tracked_make::<ulong>(trace_owner:get(), 35);
                if (first:read() != 7 || second:read() != 35) { trace = 100; };
            };
            int(trace)
        }
    "#);
    assert_eq!(
        output.status.code(),
        Some(84),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn generic_drop_hooks_run_on_result_propagation() {
    let output = run(r#"export { main };
import { "$/shared.resin" };

        struct Failed {}
        struct Tracked<T> { trace: Ptr<ulong>, value: T,
            
        }
fn drop<T>(self: Ref<Tracked<T>>)  {
                self.trace.* = self.trace.* * 10 + size_of(T);
            }

        fn fail() -> (() | Err<Failed>)  { Err(Failed {}) }
        fn work<T>(trace: Ptr<ulong>, value: T) -> (() | Err<_>)  {
            let mut local = Tracked<T> { trace = trace, value = value };
            fail()?;
            (())
        }
        fn main() -> int | Err<_> {
            let trace_owner = arc_ptr_alloc(0_ul)?; let trace: Ref<_> = trace_owner:get().*;
            work(trace_owner:get(), 1_i);
            work(trace_owner:get(), 1_ub);
            int(trace)
        }
    "#);
    assert_eq!(
        output.status.code(),
        Some(41),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn imported_generic_aliases_share_free_operation_instances() {
    let directory = tempfile::tempdir().unwrap();
    for (name, source) in [
        (
            "cell.resin",
            r#"export { Cell , cell_make, read };
            struct Cell<T> { value: T,
                
                
            }
fn cell_make<T>(value: T) -> Cell<T>  { Cell<T> { value = value } }

fn read<T>(self: Ref<Cell<T>>) -> T  { self.value }

        "#,
        ),
        (
            "alias.resin",
            "export { Renamed }; import { \"cell.resin\" }; type Renamed<T> = Cell<T>;",
        ),
        (
            "main.resin",
            r#"export { main };
            import { "cell.resin", "alias.resin" };
            fn main() -> int  { cell_make::<int>(20):read() + cell_make::<int>(22):read() }
        "#,
        ),
    ] {
        std::fs::write(directory.path().join(name), source).unwrap();
    }
    let module = support::pipeline::file_module(&directory.path().join("main.resin")).unwrap();
    for name in ["cell_make", "read"] {
        assert_eq!(
            module
                .functions
                .iter()
                .filter(|function| function.name.as_deref() == Some(name))
                .count(),
            1
        );
    }
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
fn shader_receivers_use_the_specialized_generic_owner_method() {
    let module = support::module(
        r#"export { kernel };
        struct Cell<T> { value: T,
            
        }
fn increment<T>(self: Ptr<Cell<T>>)  { self.value = self.value + 1; }

        @compute_shader fn kernel(index: ulong, root: Ptr<Cell<uint>>)  { root:increment(); }
    "#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    assert_eq!(project.generated.shaders().len(), 1);
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}
