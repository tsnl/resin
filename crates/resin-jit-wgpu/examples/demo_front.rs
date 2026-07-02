//! Demo of DSL `debug_print` output for various node types (Python `demo_front`).

use std::io::{self, Write};

use resin_core::{AxisIndex, ElementOperator, F2, F4};
use resin_dsl::{const_bytes, constant, param, zeros, View};

fn section(title: &str) {
    let _ = writeln!(io::stderr(), "\n=== {title} ===");
}

/// F2 pack for the fp16 demo only (no `ConstData` for f16 yet).
fn f2_const(shape: &[u32], data: &[f32]) -> View {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&x| f32_to_f16_bits(x).to_le_bytes())
        .collect();
    const_bytes(
        shape.to_vec().into_boxed_slice(),
        F2,
        bytes.into_boxed_slice(),
    )
}

fn f32_to_f16_bits(f: f32) -> u16 {
    // Sufficient for small demo integers (0..6); not a full IEEE rounder.
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign;
    }
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

fn print_view(v: &View) {
    let mut s = String::new();
    resin_dsl::debug_print(v, &mut s).unwrap();
    let _ = write!(io::stderr(), "{s}");
}

fn main() {
    section("Scalar constant");
    print_view(&42.0f32.into());

    section("1D constant");
    print_view(&constant([3], &[1.0f32, 2.0, 3.0]));

    section("2D constant");
    print_view(&constant([2, 2], &[1.0f32, 2.0, 3.0, 4.0]));

    section("fp16 constant");
    print_view(&f2_const(&[2], &[1.0, 2.0]));

    section("U4 constant");
    print_view(&constant([3], &[1u32, 2, 3]));

    section("Add two tensors");
    let t1 = constant([2], &[1.0f32, 2.0]);
    let t2 = constant([2], &[3.0f32, 4.0]);
    print_view(&(&t1 + &t2));

    section("Chained ops: (t1 + t2) * t3");
    let t3 = constant([2], &[5.0f32, 6.0]);
    print_view(&(&(&t1 + &t2) * &t3));

    section("Chained exp/log");
    print_view(&constant([2], &[1.0f32, 2.0]).exp().log());

    section("Deep chain: exp.log.exp.log");
    print_view(
        &constant([4], &[1.0f32, 2.0, 3.0, 4.0])
            .exp()
            .log()
            .exp()
            .log(),
    );

    section("Add scalar");
    let t = constant([2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
    let ten: View = 10.0f32.into();
    print_view(&(&t + &ten));

    section("Sum reduction axis=1");
    print_view(
        &constant([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reduce(&[1], ElementOperator::Add)
            .unwrap(),
    );

    section("Integer index");
    print_view(
        &constant([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .index((0, ..))
            .unwrap(),
    );

    section("Slice with step");
    // Step ≠ 1 still uses AxisIndex (std Range is step-1 only).
    print_view(
        &constant([6], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .index(AxisIndex::slice(None, None, 2))
            .unwrap(),
    );

    section("Multi-dim index");
    print_view(
        &constant([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .index((1, 1..3))
            .unwrap(),
    );

    section("Permute (transpose)");
    print_view(
        &constant([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .permute(&[1, 0])
            .unwrap(),
    );

    section("Compact after slice");
    let t = constant([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    print_view(
        &t.index((AxisIndex::slice(None, None, 2), ..))
            .unwrap()
            .copy(None),
    );

    section("Max with scalar");
    let zero = zeros([], F4);
    print_view(&constant([3], &[1.0f32, 2.0, 3.0]).max_elem(&zero).unwrap());

    section("Greater than (method)");
    let a = constant([2], &[1.0f32, 2.0]);
    let b = constant([2], &[2.0f32, 1.0]);
    print_view(&a.gt(&b).unwrap());

    section("Shared subexpressions");
    let t1 = constant([2], &[1.0f32, 2.0]);
    let t2 = constant([2], &[3.0f32, 4.0]);
    let s = &t1 + &t2;
    print_view(&(&s * &s));

    section("Param node");
    print_view(&param([4], F4));
}
