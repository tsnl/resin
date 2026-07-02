use resin_core::ElementOperator;
use resin_dsl::View;

pub fn softmax(x: &View, axes: &[usize]) -> Result<View, String> {
    let max_x = x.reduce(axes, ElementOperator::Max)?;
    let shifted = x - &max_x;
    let exp_x = shifted.exp();
    let sum_exp = exp_x.reduce(axes, ElementOperator::Add)?;
    Ok(&exp_x / &sum_exp)
}

pub fn cross_entropy(y_hat: &View, y: &View) -> Result<View, String> {
    assert_eq!(y_hat.shape(), y.shape());
    let axis = y_hat.rank() - 1;
    let eps: View = 1e-7f32.into();
    let safe = y_hat.max_elem(&eps)?;
    let prod = y * &safe.log();
    let summed = prod.reduce(&[axis], ElementOperator::Add)?;
    Ok(-summed.squeeze(&[axis])?)
}

pub fn mean(n: &View) -> Result<View, String> {
    let count = n.shape().iter().map(|&d| u64::from(d)).product::<u64>() as f32;
    let mut reduced = n.sum(None)?;
    for axis in (0..reduced.rank()).rev() {
        reduced = reduced.squeeze(&[axis])?;
    }
    Ok(&reduced / &count.into())
}
