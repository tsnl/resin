#![allow(dead_code)]

pub fn source(ty: &str) -> String {
    include_str!("../fixtures/integer_arithmetic.resin")
        .replace("type Integer = i64;", &format!("type Integer = {ty};"))
}

// (left, right, operation, result); None means the operation must fail.
pub fn cases(ty: &str) -> Vec<(i128, i128, u32, Option<i128>)> {
    let bits: u32 = ty[1..].parse().unwrap();
    let signed = ty.starts_with('i');
    let high = (1_i128 << (bits - u32::from(signed))) - 1;
    let mut cases = vec![
        (7, 3, 0, Some(2)),
        (7, 3, 1, Some(1)),
        (high, 1, 0, Some(high)),
        (high, 2, 1, Some(1)),
        (3, 2, 2, Some(12)),
        (high, 1, 3, Some(high / 2)),
        (7, 0, 2, Some(7)),
        (7, 0, 3, Some(7)),
        (7, 0, 0, None),
        (7, 0, 1, None),
        (7, i128::from(bits), 2, None),
        (7, i128::from(bits), 3, None),
    ];
    if signed {
        let low = -high - 1;
        cases.extend([
            (-7, 3, 0, Some(-2)),
            (-7, 3, 1, Some(-1)),
            (7, -3, 0, Some(-2)),
            (7, -3, 1, Some(1)),
            (low, -1, 0, Some(low)),
            (low, -1, 1, Some(0)),
            (low, 2, 0, Some(low / 2)),
            (-7, 1, 3, Some(-4)),
            (-1, i128::from(bits - 1), 3, Some(-1)),
            (1, i128::from(bits - 1), 2, Some(low)),
            (-1, 1, 2, Some(-2)),
            (7, -1, 2, None),
            (7, -1, 3, None),
        ]);
    } else {
        cases.extend([
            (high, 2, 0, Some(high / 2)),
            (1, i128::from(bits - 1), 2, Some(1_i128 << (bits - 1))),
            (high, i128::from(bits - 1), 3, Some(1)),
            (high, 1, 2, Some(high - 1)),
        ]);
    }
    cases
}
