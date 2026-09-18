mod support;

#[test]
fn generic_aliases_borrow_storage_and_resolve_free_operations() {
    let module = support::module(
        r#"export { main };
import { "$/shared.resin" };

        struct FieldsDataLength<T0, T1> { data: T0, length: T1, }
type View<T> = FieldsDataLength<PtrMut<T>, u64>;
        struct Item { value: i32,
            
        }
fn read(self: Ref<Item>) -> i32  { self.value }

        type Renamed<T> = Item;
        fn first<T>(view: View<T>) -> Ref<T>  { view.data.* }
        fn main() -> i32 | Err<_> {
            let item_owner = arc_ptr_alloc(Renamed<bool> { value = 42 })?; let item: Ref<_> = item_owner:get().*;
            first(View<Item> { data = item_owner:get(), length = 1 }):read()
        }
    "#,
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
fn aliases_reuse_the_same_function_instance() {
    let module = support::module(
        "type First<T> = PtrMut<T>; type Second<U> = First<U>; fn measure<T>() -> u64  { size_of(T) } fn main() -> u64  { measure::<First<i32>>() + measure::<Second<i32>>() }",
    );
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| function.name.as_deref() == Some("measure"))
            .count(),
        1
    );
}
