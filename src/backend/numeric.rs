use crate::ir::Ty;

fn integer(ty: &Ty) -> Option<(u32, bool)> {
    Some(match ty {
        Ty::Int8 => (8, true),
        Ty::UInt8 => (8, false),
        Ty::Int16 => (16, true),
        Ty::UInt16 => (16, false),
        Ty::Int32 => (32, true),
        Ty::UInt32 => (32, false),
        Ty::Int64 => (64, true),
        Ty::UInt64 => (64, false),
        _ => return None,
    })
}

fn range(bits: u32, signed: bool) -> (i128, i128) {
    if signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    }
}

/// A predicate for a conversion that must trap. Floating comparisons use exact
/// power-of-two bounds and truncation, avoiding rounded integer maxima and NaNs.
pub(super) fn invalid(
    from: &Ty,
    to: &Ty,
    expr: &str,
    cast: impl Fn(&Ty, &str) -> String,
    trunc: &str,
) -> Option<String> {
    let (bits, signed) = integer(to)?;
    if let Some((source_bits, source_signed)) = integer(from) {
        let (lo, hi) = range(bits, signed);
        let (source_lo, source_hi) = range(source_bits, source_signed);
        let literal = |n: i128| {
            if n >= 0 {
                format!("{n}ul")
            } else {
                n.to_string()
            }
        };
        let mut conditions = Vec::new();
        if lo > source_lo {
            conditions.push(format!("({expr}) < {}", cast(from, &literal(lo))));
        }
        if hi < source_hi {
            conditions.push(format!("({expr}) > {}", cast(from, &literal(hi))));
        }
        if conditions.is_empty() {
            None
        } else {
            Some(conditions.join(" || "))
        }
    } else {
        let low = if signed {
            -(2f64.powi(bits as i32 - 1))
        } else {
            0.0
        };
        let high = 2f64.powi(bits as i32 - i32::from(signed));
        let suffix = if *from == Ty::Float32 { "f" } else { "" };
        Some(format!(
            "!({trunc}({expr}) >= {low:.1}{suffix} && {trunc}({expr}) < {high:.1}{suffix})"
        ))
    }
}
