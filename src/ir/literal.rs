//! Numeric literal classification and case-sensitive primitive type suffixes.
use super::Ty;

pub fn split(text: &str) -> (&str, Option<Ty>) {
    let hex = is_hex(text);
    let Some(suffix) = text.chars().last() else {
        return (text, None);
    };
    // b/B/d/f are hexadecimal digits, so they never denote a hex suffix.
    let ty = match suffix {
        'b' if !hex => Ty::Int8,
        'B' if !hex => Ty::UInt8,
        'h' => Ty::Int16,
        'H' => Ty::UInt16,
        'i' => Ty::Int32,
        'I' => Ty::UInt32,
        'l' => Ty::Int64,
        'L' => Ty::UInt64,
        'f' if !hex => Ty::Float32,
        'd' if !hex => Ty::Float64,
        _ => return (text, None),
    };
    (&text[..text.len() - 1], Some(ty))
}

/// The fallback type for an unsuffixed literal, before any contextual inference.
/// Hexadecimal e/E digits are not decimal exponents.
pub fn unsuffixed_type(text: &str) -> Ty {
    if text.contains('.') || (!is_hex(text) && text.contains(['e', 'E'])) {
        Ty::Float64
    } else {
        Ty::Int32
    }
}

fn is_hex(text: &str) -> bool {
    let magnitude = text.strip_prefix('-').unwrap_or(text);
    magnitude.starts_with("0x") || magnitude.starts_with("0X")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_digits_and_signs_do_not_change_suffix_or_exponent_rules() {
        for (text, body, suffix, default) in [
            ("-0xdead", "-0xdead", None, Ty::Int32),
            ("0XAB", "0XAB", None, Ty::Int32),
            ("-0xFFL", "-0xFF", Some(Ty::UInt64), Ty::Int32),
            ("-12b", "-12", Some(Ty::Int8), Ty::Int32),
            ("12B", "12", Some(Ty::UInt8), Ty::Int32),
            ("-1e2", "-1e2", None, Ty::Float64),
            ("1E2f", "1E2", Some(Ty::Float32), Ty::Float64),
            ("-1.5d", "-1.5", Some(Ty::Float64), Ty::Float64),
        ] {
            assert_eq!(split(text), (body, suffix), "{text}");
            assert_eq!(unsuffixed_type(body), default, "{text}");
        }
    }
}
