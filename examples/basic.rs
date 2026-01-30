use resin::expr::Expr;

fn main() {
    let x = Expr::from([1.0]) * Expr::from([2.0]);
    eprintln!("{:#?}", x);

    let lt = Expr::from([[1.0, 0.0], [0.0, 1.0]]);
    let rt = Expr::from([[2.0, 1.0], [0.0, 2.0]]);
    let x = lt.matmul(rt);
    eprintln!("{:#?}", x);
}
