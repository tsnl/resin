//! Numeric suffixes are case insensitive and select an exact primitive type.
use super::Ty;

pub fn split(text: &str) -> (&str, Option<Ty>) {
    let hex = is_hex(text);
    let Some(width) = text.as_bytes().last().map(u8::to_ascii_lowercase) else {
        return (text, None);
    };
    let mut end = text.len() - 1;
    let unsigned = end > 0 && text.as_bytes()[end - 1].eq_ignore_ascii_case(&b'u');
    if unsigned {
        end -= 1;
    }
    // Hex b/B is a digit unless an underscore or unsigned qualifier separates it.
    // Hex d/D/f/F always remain digits; hexadecimal floats are unsupported.
    let separated = end > 0 && text.as_bytes()[end - 1] == b'_';
    let ty = match (width, unsigned) {
        (b'b', false) if !hex || separated => Ty::Int8,
        (b'b', true) => Ty::UInt8,
        (b'h', false) => Ty::Int16,
        (b'h', true) => Ty::UInt16,
        (b'i', false) => Ty::Int32,
        (b'i', true) => Ty::UInt32,
        (b'l', false) => Ty::Int64,
        (b'l', true) => Ty::UInt64,
        (b'f', false) if !hex => Ty::Float32,
        (b'd', false) if !hex => Ty::Float64,
        _ => return (text, None),
    };
    (text[..end].trim_end_matches('_'), Some(ty))
}

/// Normalize only the suffix; keep digit grouping and hexadecimal digit case.
pub fn format(text: &str) -> String {
    let (digits, ty) = split(text);
    let suffix = match ty {
        Some(Ty::Int8) => "b",
        Some(Ty::UInt8) => "ub",
        Some(Ty::Int16) => "h",
        Some(Ty::UInt16) => "uh",
        Some(Ty::Int32) => "i",
        Some(Ty::UInt32) => "ui",
        Some(Ty::Int64) => "l",
        Some(Ty::UInt64) => "ul",
        Some(Ty::Float32) => "f",
        Some(Ty::Float64) => "d",
        _ => return text.into(),
    };
    format!("{digits}_{suffix}")
}

/// The fallback type for an unsuffixed literal, before any contextual inference.
/// Hexadecimal e/E digits are not decimal exponents.
pub fn unsuffixed_type(text: &str) -> Ty {
    if text.contains('.') || (!is_hex(text) && text.contains(['e', 'E'])) {
        Ty::Float64
    } else {
        Ty::Int64
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
            ("-0xdead", "-0xdead", None, Ty::Int64),
            ("0XAB", "0XAB", None, Ty::Int64),
            ("-0xFF_ul", "-0xFF", Some(Ty::UInt64), Ty::Int64),
            ("-12_b", "-12", Some(Ty::Int8), Ty::Int64),
            ("12_ub", "12", Some(Ty::UInt8), Ty::Int64),
            ("-1e2", "-1e2", None, Ty::Float64),
            ("1E2_f", "1E2", Some(Ty::Float32), Ty::Float64),
            ("-1.5_d", "-1.5", Some(Ty::Float64), Ty::Float64),
        ] {
            assert_eq!(split(text), (body, suffix), "{text}");
            assert_eq!(unsuffixed_type(body), default, "{text}");
        }
    }
}
