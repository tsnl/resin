//! Demo of DSL `debug_print` output for various node types (Python `demo_front.py`).

use std::io::{self, Write};

use resin::dsl::{debug_print, ElementType, Tensor};

fn section(title: &str) {
    let _ = writeln!(io::stderr(), "\n=== {title} ===");
}

fn print_tensor(tensor: &Tensor) {
    let mut out = String::new();
    debug_print(tensor, &mut out).unwrap();
    let _ = write!(io::stderr(), "{out}");
}

fn main() {
    section("Scalar constant");
    print_tensor(&Tensor::zeros(&[], ElementType::F32));

    section("1D constant");
    print_tensor(&Tensor::constant_f32(&[3], &[1.0, 2.0, 3.0]));

    section("2D constant");
    print_tensor(&Tensor::constant_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));

    section("Add two tensors");
    let t1 = Tensor::constant_f32(&[2], &[1.0, 2.0]);
    let t2 = Tensor::constant_f32(&[2], &[3.0, 4.0]);
    print_tensor(&(t1.clone() + t2.clone()));

    section("Chained ops: (t1 + t2) * t3");
    let t3 = Tensor::constant_f32(&[2], &[5.0, 6.0]);
    print_tensor(&((t1 + t2) * t3));

    section("Chained exp/log");
    print_tensor(
        &Tensor::constant_f32(&[2], &[1.0, 2.0])
            .exp()
            .log(),
    );

    section("Deep chain: exp.log.exp.log");
    print_tensor(
        &Tensor::constant_f32(&[4], &[1.0, 2.0, 3.0, 4.0])
            .exp()
            .log()
            .exp()
            .log(),
    );

    section("Add scalar");
    let t = Tensor::constant_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let ten = Tensor::full(&[], 10.0, ElementType::F32);
    print_tensor(&(t + ten.clone()));

    section("Sum reduction axis=1");
    print_tensor(
        &Tensor::constant_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).sum_axes(&[1]),
    );

    section("Permute (transpose)");
    print_tensor(&Tensor::constant_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).transpose());

    section("Shared subexpressions");
    let t1 = Tensor::constant_f32(&[2], &[1.0, 2.0]);
    let t2 = Tensor::constant_f32(&[2], &[3.0, 4.0]);
    let s = t1 + t2;
    print_tensor(&(s.clone() * s));

    section("Param node");
    print_tensor(&Tensor::parameter(&[4], ElementType::F32));
}