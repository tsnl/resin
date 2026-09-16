//! C adapters preserve scalar ABI details while Resin functions use their own ABI.
mod support;

fn run(header: &str, declarations: &str, body: &str, expected: i32) {
    let directory = tempfile::TempDir::new().unwrap();
    let path = directory.path().join("native_fixture.h");
    std::fs::write(&path, header).unwrap();
    let path = path.to_str().unwrap().replace('\\', "/");
    let source =
        format!("export {{ main }}; extern {{ \"{path}\": {{ {declarations} }} }}; {body}");
    let module = support::module(&source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert_eq!(
        output.status.code(),
        Some(expected),
        "{source}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}

#[test]
fn narrow_signed_unsigned_and_boolean_values_cross_c_adapters() {
    run(
        r#"#include <stdint.h>
        #include <stdbool.h>
        static inline int8_t signed_byte(int8_t value) { return value == -120 ? -117 : 1; }
        static inline uint8_t unsigned_byte(uint8_t value) { return value == 250 ? 247 : 1; }
        static inline int16_t signed_half(int16_t value) { return value == -30000 ? -29000 : 1; }
        static inline uint16_t unsigned_half(uint16_t value) { return value == 65000 ? 64000 : 1; }
        static inline bool negate(bool value) { return !value; }
        "#,
        r#"def signed_byte(value: sbyte) -> sbyte;
        def unsigned_byte(value: ubyte) -> ubyte;
        def signed_half(value: short) -> short;
        def unsigned_half(value: ushort) -> ushort;
        def negate(value: bool) -> bool;"#,
        r#"def main() -> int = {
            if (signed_byte(-120_b) == -117_b && unsigned_byte(250_ub) == 247_ub &&
                signed_half(-30000_h) == -29000_h && unsigned_half(65000_uh) == 64000_uh &&
                negate(1 == 2) && !negate(1 == 1)) { 37 } else { 1 }
        };"#,
        37,
    );
}

#[test]
fn void_pointer_and_indirect_static_inline_calls_keep_resin_evaluation_order() {
    run(
        r#"#include <stdint.h>
        static inline void change(int32_t *value, int32_t replacement) { *value = replacement; }
        static inline int32_t *same(int32_t *value) { return value; }
        #define add_values(left, right) ((left) + (right))
        "#,
        r#"def change(value: Ptr<int>, replacement: int);
        def same(value: Ptr<int>) -> Ptr<int>;
        def add_values(left: int, right: int) -> int;"#,
        r#"def unit(value: ()) = { value };
        def main() -> int = {
            var value = 0_i;
            var indirect = change;
            unit(indirect(&value, 42));
            var pointer = same(&value);
            pointer.* := add_values(pointer.*, 5);
            value
        };"#,
        47,
    );
}

#[test]
fn wide_integer_and_float_calls_preserve_all_argument_and_result_bits() {
    run(
        r#"#include <stdint.h>
        static inline int64_t signed_word(int64_t value) { return value + 3; }
        static inline uint64_t unsigned_word(uint64_t value) { return value - 3; }
        static inline float single(float left, float right) { return left * right; }
        static inline double double_value(double left, double right) { return left / right; }
        "#,
        r#"def signed_word(value: long) -> long;
        def unsigned_word(value: ulong) -> ulong;
        def single(left: float32, right: float32) -> float32;
        def double_value(left: float64, right: float64) -> float64;"#,
        r#"def main() -> int = {
            if (signed_word(-9223372036854775807_l) == -9223372036854775804_l &&
                unsigned_word(18446744073709551615_ul) == 18446744073709551612_ul &&
                single(1.5_f, 4_f) == 6_f && double_value(11_d, 2_d) == 5.5_d) { 53 } else { 1 }
        };"#,
        53,
    );
}
