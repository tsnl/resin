//! Demo of DSL `debug_print` output for various node types.

use std::io::{self, Write};

use resin::dsl::{Tensor, debug_print};

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
    print_tensor(&Tensor::scalar(0.0));

    section("1D constant");
    print_tensor(&Tensor::constant(&[3], &[1.0, 2.0, 3.0]));

    section("2D constant");
    print_tensor(&Tensor::constant(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));

    section("Add two tensors");
    let t1 = Tensor::constant(&[2], &[1.0, 2.0]);
    let t2 = Tensor::constant(&[2], &[3.0, 4.0]);
    print_tensor(&(t1.clone() + t2.clone()));

    section("Chained ops: (t1 + t2) * t3");
    let t3 = Tensor::constant(&[2], &[5.0, 6.0]);
    print_tensor(&((t1 + t2) * t3));

    section("Chained exp/log");
    print_tensor(&Tensor::constant(&[2], &[1.0, 2.0]).exp().log());

    section("Add scalar (implicit broadcast)");
    let t = Tensor::constant(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    print_tensor(&(t + Tensor::scalar(10.0)));

    section("Sum reduction axis=1");
    print_tensor(&Tensor::constant(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).sum_axes(&[1]));

    section("Transpose");
    print_tensor(&Tensor::constant(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).transpose());

    section("Matmul");
    let a = Tensor::parameter(&[2, 4]);
    let b = Tensor::parameter(&[4, 3]);
    print_tensor(&a.matmul(&b));

    section("Shared subexpressions");
    let t1 = Tensor::constant(&[2], &[1.0, 2.0]);
    let t2 = Tensor::constant(&[2], &[3.0, 4.0]);
    let s = t1 + t2;
    print_tensor(&(s.clone() * s));

    section("Param node");
    print_tensor(&Tensor::parameter(&[4]));
}
