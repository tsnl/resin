use half::f16;
use resin::tensor::Tensor;

fn main() {
    let x = Tensor::from([1.0]) * Tensor::from([2.0]);
    eprintln!("{:#?}", x);

    let lt = Tensor::from([[1.0, 0.0], [0.0, 1.0]]);
    let rt = Tensor::from([[2.0, 1.0], [0.0, 2.0]]);
    let x = lt.matmul(rt);
    eprintln!("{:#?}", x);
}
