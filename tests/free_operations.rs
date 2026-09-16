mod support;

#[test]
fn generic_relations_and_operators_select_visible_signatures() {
    let source = r#"
        export { main };
        struct Left { value: int; }
        struct Right { value: int; }
        fn relate(a: Left, b: Right) -> int { a.value + b.value }
        fn relate(a: Right, b: Left) -> int { a.value - b.value }
        fn relate(a: int, b: int) -> int { a + b }
        fn relation<A, B>(a: A, b: B) -> _ { a:relate(b) }
        fn __add__(a: Left, b: Right) -> int { a.value + b.value }
        fn sum<A, B>(a: A, b: B) -> _ { a + b }
        fn main() -> int {
            let a = relation(Left { value= 19 }, Right { value= 23 });
            let b = relation(Right { value= 23 }, Left { value= 19 });
            let c = sum(Left { value= 19 }, Right { value= 23 });
            let d = sum(19_i, 23_i);
            if (a == 42 && b == 4 && c == 42 && d == 42) { 0 } else { 1 }
        }
    "#;
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
