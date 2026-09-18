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

        fn main() -> i32  {
            let mut first = cell_make::<i32>(7);
            let mut second = first:with::<_, u64>(4294967296);
            let mut third = with::<i32, u8>(first, 255);
            { let borrowed = fmt("{0} {1} {2}", (first:read(), second:read(), third:read())); print(borrowed) };
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

        fn main() -> i32  {
            let mut factory = Factory {};
            let mut first: Cell<i32>; first = factory:create();
            let mut second = factory:create::<u64>();
            let mut third: Cell<u8>; third = factory:create::<_>();
            { let borrowed = fmt("{0} {1} {2}", (first.value, second.value, third.value)); print(borrowed) };
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

        fn main() -> i32  {
            let mut make = cell_make::<i32>;
            let select_int = select::<i32, i32>;
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
fn read<T>(self: Ref<Cell<T>>, depth: i32) -> _  {
                if (depth == 0) { self.value } else { self:read(depth - 1) }
            }

fn choose<T, U>(self: Ref<Cell<T>>, value: U, depth: i32) -> _  {
                if (depth == 0) { value } else { self:choose(value, depth - 1) }
            }

        fn main() -> i32  {
            let mut cell = Cell<i32> { value = 7 };
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

        struct Tracked<T> { trace: PtrMut<u64>, value: T,
            
            
            
        }
fn tracked_make<T>(trace: PtrMut<u64>, value: T) -> Tracked<T>  {
                Tracked<T> { trace = trace, value = value }
            }

fn read<T>(self: Ref<Tracked<T>>) -> T  { self.value }

fn drop<T>(self: RefMut<Tracked<T>>)  {
                self.trace.* = self.trace.* * 10 + size_of(T);
            }

        fn main() -> i32 | Err<_> {
            let trace_owner = arc_ptr_alloc(u64(0))?; let trace: RefMut<_> = trace_owner:get().*;
            {
                let mut first = tracked_make::<i32>(trace_owner:get(), 7);
                let mut second = tracked_make::<u64>(trace_owner:get(), 35);
                if (first:read() != 7 || second:read() != 35) { trace = 100; };
            };
            i32(trace)
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
        struct Tracked<T> { trace: PtrMut<u64>, value: T,
            
        }
fn drop<T>(self: RefMut<Tracked<T>>)  {
                self.trace.* = self.trace.* * 10 + size_of(T);
            }

        fn fail() -> (() | Err<Failed>)  { Err(Failed {}) }
        fn work<T>(trace: PtrMut<u64>, value: T) -> (() | Err<_>)  {
            let mut local = Tracked<T> { trace = trace, value = value };
            fail()?;
            (())
        }
        fn main() -> i32 | Err<_> {
            let trace_owner = arc_ptr_alloc(u64(0))?; let trace: Ref<_> = trace_owner:get().*;
            work(trace_owner:get(), i32(1));
            work(trace_owner:get(), u8(1));
            i32(trace)
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
            fn main() -> i32  { let left = cell_make::<i32>(20); let right = cell_make::<i32>(22); left:read() + right:read() }
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
fn increment<T>(self: PtrMut<Cell<T>>)  { self.value = self.value + 1; }

        @compute_shader fn kernel(index: u64, root: PtrMut<Cell<u32>>)  { root:increment(); }
    "#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    assert_eq!(project.generated.shaders().len(), 1);
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}
