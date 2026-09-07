//! Numeric suffixes are case-sensitive and select an exact primitive type.
use super::Ty;

pub fn split(text: &str) -> (&str, Option<Ty>) {
    let magnitude = text.strip_prefix('-').unwrap_or(text);
    let hex = magnitude.starts_with("0x") || magnitude.starts_with("0X");
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
