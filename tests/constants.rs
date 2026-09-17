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
        .expect("constant program exited normally")
}

#[test]
fn constants_and_iota_execute_with_explicit_initializers_and_fixed_types() {
    assert_eq!(
        result(
            r#"export { main };
        const (
            first, second: u32 = 1 << iota, 2 << iota;
            _, _ = iota, iota;
            third, fourth: u32 = 1 << iota, 2 << iota;
        );
        const reset = iota;
        const answer: i32 = i32(first + second + third + fourth) + 27;
        fn main() -> i32  {
            const ( a = iota + 1; b = iota + 1; );
            const half: f32 = 1.0 / 2.0;
            const text = "ok";
            const yes = answer == 42 && half == f32(0.5);
            if (yes && text.length == u64(2) && reset == 0 && a == 1 && b == 2) { answer } else { 1 }
        }
    "#
        ),
        42
    );
}

#[test]
fn sizeof_matches_layout_after_generic_specialization() {
    assert_eq!(
        result(
            r#"export { main };
        struct Pair<T> { first: T, second: T, }
        struct Node { next: Ptr<Node>, value: i32, }
        const pair_bytes = sizeof(Pair<i32>);
        const node_bytes = sizeof(Node);
        fn size<T>() -> u64  { sizeof(T) }
        fn main() -> i32  {
            if (pair_bytes == u64(8) && pair_bytes == size::<Pair<i32>>() && node_bytes == size::<Node>()) { 42 } else { 1 }
        }
    "#
        ),
        42
    );
}

#[test]
fn integer_boundaries_float_rounding_and_conversions_are_evaluated_at_declaration() {
    assert_eq!(
        result(
            r#"export { main };
        const maximum = ~u64(0);
        const minimum = -9223372036854775808;
        const rounded = (f32(16777216) + f32(1)) - f32(16777216);
        const truncated = i32(f64(4.75));
        const shifted = -8 >> 2;
        fn main() -> i32  {
            if (maximum == u64(18446744073709551615) && minimum < 0 && rounded == f32(0) && truncated == 4 && shifted == -2) { 42 } else { 1 }
        }
    "#
        ),
        42
    );
}

#[test]
fn sizeof_matches_native_c_representations() {
    let directory = tempfile::tempdir().unwrap();
    let header = directory.path().join("sizes.h");
    std::fs::write(
        &header,
        r#"#include <stdint.h>
        #include <stdbool.h>
        #include "resin_runtime.h"
        struct TestRecord { int8_t a; double b; uint16_t c; };
        struct TestStr { uint8_t *data; uint64_t length; };
        struct TestUnion { uint32_t tag; union { int32_t a; int32_t b; } payload; };
        static inline uint64_t native_size(uint32_t index) {
            const uint64_t sizes[] = {
                sizeof(bool), sizeof(int8_t), sizeof(uint16_t), sizeof(double),
                sizeof(struct TestStr), sizeof(uint8_t), sizeof(uint8_t), sizeof(uint32_t),
                sizeof(struct TestRecord), sizeof(struct TestUnion), sizeof(struct TestUnion),
                sizeof(ResinGpuPtr), sizeof(ResinGpuPipelineContract), sizeof(ResinArc *),
                sizeof(void (*)(void))
            };
            return sizes[index];
        }
    "#,
    )
    .unwrap();
    let types = [
        "bool",
        "i8",
        "u16",
        "f64",
        "str",
        "()",
        "None",
        "Never",
        "Record",
        "First | Second",
        "(i32 | Err<Failure>)",
        "GpuView",
        "GpuPipelineContract",
        "GpuArguments",
        "(i32) -> i32",
    ];
    let declarations = types
        .iter()
        .enumerate()
        .map(|(i, ty)| format!("const bytes_{i} = sizeof({ty});"))
        .collect::<Vec<_>>()
        .join("\n");
    let conditions = types
        .iter()
        .enumerate()
        .map(|(i, ty)| format!("bytes_{i} == native_size({i}) && sizeof({ty}) == bytes_{i}"))
        .collect::<Vec<_>>()
        .join(" && ");
    assert_eq!(
        result(&format!(
            r#"export {{ main }};
        extern {{ "{}": {{ fn native_size(index: u32) -> u64; }} }};
        struct Record {{ a: i8, b: f64, c: u16, }}
        struct First {{ value: i32, }}
        struct Second {{ value: i32, }}
        struct Failure {{ value: i32, }}
        {declarations}
        fn main() -> i32  {{ if ({conditions}) {{ 42 }} else {{ 1 }} }}
    "#,
            header.display()
        )),
        42
    );
}

#[test]
fn constants_lower_to_shader_literals_without_runtime_arithmetic() {
    let module = support::module(
        r#"export { kernel };
        const ( first: u32 = 1 << iota; second: u32 = 1 << iota; );
        const answer: u32 = (first + second) * 14;
        @compute_shader fn kernel(index: u64, output: Ptr<u32>)  {
            const bytes = sizeof(f64);
            let mut native_bytes = sizeof(f64);
            output.* = answer;
        }
    "#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    let shader = project.generated.shaders()[0].unoptimized_spirv();
    let bytes = std::fs::read(shader).unwrap();
    let uint_type = support::shaders::instructions(&bytes, 21)
        .find(|args| args[1..] == [32, 0])
        .unwrap()[0];
    assert!(
        support::shaders::instructions(&bytes, 43)
            .any(|args| args[0] == uint_type && args[2..] == [42])
    );
    // The entry wrapper can perform 64-bit index arithmetic; the constant's
    // 32-bit arithmetic must already be gone in unoptimized SPIR-V.
    for opcode in [128, 132, 196] {
        assert!(!support::shaders::instructions(&bytes, opcode).any(|args| args[0] == uint_type));
    }
    support::shaders::validate(shader);
}

#[test]
fn boolean_literals_work_in_constants_and_runtime_control_flow() {
    assert_eq!(
        result(
            r#"export { main };
        const yes: bool = true;
        const no = false;
        fn invert(value: bool) -> bool  { !value }
        fn main() -> i32  {
            let mut running = true;
            let mut count = 0;
            while (running) { count = count + 1; running = false; };
            if (yes && !no && invert(false) && count == 1) { 42 } else { 1 }
        }"#
        ),
        42
    );
}
