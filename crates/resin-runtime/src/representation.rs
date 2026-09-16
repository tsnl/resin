//! Render initialized values through explicit native-layout descriptors.
use crate::host::fail;
use std::{
    ffi::{CStr, c_char, c_void},
    io::{self, Write},
    ptr,
};

const UNIT: u32 = 0;
const NONE: u32 = 1;
const BOOL: u32 = 2;
const I8: u32 = 3;
const I16: u32 = 4;
const I32: u32 = 5;
const I64: u32 = 6;
const U8: u32 = 7;
const U16: u32 = 8;
const U32: u32 = 9;
const U64: u32 = 10;
const F32: u32 = 11;
const F64: u32 = 12;
const STR: u32 = 13;
const POINTER: u32 = 14;
const OPAQUE: u32 = 15;
const NAMED: u32 = 16;
const ARRAY: u32 = 17;
const TUPLE: u32 = 18;
const RECORD: u32 = 19;
const UNION: u32 = 20;
const TEXT: u32 = 21;
const ERROR: u32 = 22;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ResinReprValue {
    ty: *const ResinReprType,
    data: *const c_void,
}

#[repr(C)]
pub struct ResinReprField {
    name: *const c_char,
    ty: *const ResinReprType,
    offset: usize,
    tag: u32,
}

#[repr(C)]
pub struct ResinReprType {
    kind: u32,
    name: *const c_char,
    size: usize,
    length: usize,
    fields: *const ResinReprField,
    field_count: usize,
    text: Option<unsafe extern "C" fn(*const c_void) -> crate::print::ResinPrintBytes>,
}

/// # Safety
/// The descriptor graph must describe the initialized value at `data` accurately.
/// Names are NUL terminated; fields and literal bytes must be readable throughout.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_repr(
    ty: *const ResinReprType,
    data: *const c_void,
) -> *mut crate::shared::ResinArc {
    let mut bytes = Vec::new();
    unsafe { render(&mut bytes, ResinReprValue { ty, data }) }
        .unwrap_or_else(|_| fail("representation failed"));
    crate::print::owned_bytes(&bytes)
}

/// # Safety
/// Same initialized-value and descriptor requirements as `resin_repr`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_report_error(ty: *const ResinReprType, data: *const c_void) {
    let mut bytes = b"unhandled error: ".to_vec();
    unsafe { render(&mut bytes, ResinReprValue { ty, data }) }
        .unwrap_or_else(|_| fail("representation failed"));
    bytes.push(b'\n');
    unsafe {
        crate::print::resin_stream_write(1, bytes.as_ptr(), bytes.len());
    }
}

pub(crate) unsafe fn render(out: &mut impl Write, value: ResinReprValue) -> io::Result<()> {
    unsafe { value_at(out, &*value.ty, value.data.cast(), 0) }
}

unsafe fn name<'a>(text: *const c_char) -> &'a [u8] {
    unsafe { CStr::from_ptr(text).to_bytes() }
}

unsafe fn fields<'a>(ty: &ResinReprType) -> &'a [ResinReprField] {
    if ty.field_count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ty.fields, ty.field_count) }
    }
}

unsafe fn value_at(
    out: &mut impl Write,
    ty: &ResinReprType,
    data: *const u8,
    depth: usize,
) -> io::Result<()> {
    if depth >= 128 {
        return out.write_all(b"...");
    }
    // Values may be packed. Never create aligned Rust references into native storage.
    unsafe {
        match ty.kind {
            UNIT => out.write_all(b"()"),
            NONE => out.write_all(b"None"),
            BOOL => write!(out, "{}", ptr::read_unaligned(data) != 0),
            I8 => write!(out, "{}", ptr::read_unaligned(data.cast::<i8>())),
            I16 => write!(out, "{}", ptr::read_unaligned(data.cast::<i16>())),
            I32 => write!(out, "{}", ptr::read_unaligned(data.cast::<i32>())),
            I64 => write!(out, "{}", ptr::read_unaligned(data.cast::<i64>())),
            U8 => write!(out, "{}", ptr::read_unaligned(data)),
            U16 => write!(out, "{}", ptr::read_unaligned(data.cast::<u16>())),
            U32 => write!(out, "{}", ptr::read_unaligned(data.cast::<u32>())),
            U64 => write!(out, "{}", ptr::read_unaligned(data.cast::<u64>())),
            F32 => write!(out, "{}", ptr::read_unaligned(data.cast::<f32>())),
            F64 => write!(out, "{}", ptr::read_unaligned(data.cast::<f64>())),
            STR => {
                let bytes = ptr::read_unaligned(data.cast::<*const u8>());
                let length =
                    ptr::read_unaligned(data.add(std::mem::size_of::<usize>()).cast::<usize>());
                quoted(
                    out,
                    if length == 0 {
                        &[]
                    } else {
                        std::slice::from_raw_parts(bytes, length)
                    },
                )
            }
            POINTER => write!(out, "0x{:x}", ptr::read_unaligned(data.cast::<usize>())),
            OPAQUE => {
                out.write_all(b"<")?;
                out.write_all(name(ty.name))?;
                out.write_all(b">")
            }
            NAMED => {
                out.write_all(name(ty.name))?;
                let field = &fields(ty)[0];
                let body = &*field.ty;
                if body.kind == UNIT
                    || (matches!(body.kind, TUPLE | RECORD) && body.field_count == 0)
                {
                    return Ok(());
                }
                if body.kind != RECORD {
                    out.write_all(b"(")?;
                }
                value_at(out, body, data.add(field.offset), depth + 1)?;
                if body.kind != RECORD {
                    out.write_all(b")")?;
                }
                Ok(())
            }
            ARRAY => {
                out.write_all(b"[")?;
                let field = &fields(ty)[0];
                let element = &*field.ty;
                for index in 0..ty.length {
                    if index != 0 {
                        out.write_all(b", ")?;
                    }
                    value_at(
                        out,
                        element,
                        data.add(field.offset + element.size * index),
                        depth + 1,
                    )?;
                }
                out.write_all(b"]")
            }
            TUPLE | RECORD => {
                let tuple = ty.kind == TUPLE;
                out.write_all(if tuple { b"(" } else { b" { " })?;
                for (index, field) in fields(ty).iter().enumerate() {
                    if index != 0 {
                        out.write_all(b", ")?;
                    }
                    if !tuple {
                        out.write_all(name(field.name))?;
                        out.write_all(b" = ")?;
                    }
                    value_at(out, &*field.ty, data.add(field.offset), depth + 1)?;
                }
                if tuple && ty.field_count == 1 {
                    out.write_all(b",")?;
                }
                out.write_all(if tuple { b")" } else { b" }" })
            }
            UNION => {
                let tag = ptr::read_unaligned(data.cast::<u32>());
                let field = fields(ty)
                    .iter()
                    .find(|field| field.tag == tag)
                    .unwrap_or_else(|| fail("invalid representation tag"));
                let label = name(field.name);
                if !label.is_empty() {
                    out.write_all(label)?;
                    out.write_all(b"(")?;
                }
                value_at(out, &*field.ty, data.add(field.offset), depth + 1)?;
                if !label.is_empty() {
                    out.write_all(b")")?;
                }
                Ok(())
            }
            TEXT => quoted(out, text(ty, data)),
            ERROR => {
                out.write_all(b"Err(")?;
                let field = &fields(ty)[0];
                value_at(out, &*field.ty, data.add(field.offset), depth + 1)?;
                out.write_all(b")")
            }
            _ => fail("invalid representation kind"),
        }
    }
}

fn quoted(out: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    out.write_all(b"\"")?;
    for chunk in bytes.utf8_chunks() {
        for &byte in chunk.valid().as_bytes() {
            match byte {
                b'"' => out.write_all(b"\\\"")?,
                b'\\' => out.write_all(b"\\\\")?,
                b'\n' => out.write_all(b"\\n")?,
                b'\r' => out.write_all(b"\\r")?,
                b'\t' => out.write_all(b"\\t")?,
                0 => out.write_all(b"\\0")?,
                1..=31 | 127 => write!(out, "\\x{byte:02x}")?,
                _ => out.write_all(&[byte])?,
            }
        }
        for byte in chunk.invalid() {
            write!(out, "\\x{byte:02x}")?;
        }
    }
    out.write_all(b"\"")
}

unsafe fn text<'a>(ty: &ResinReprType, data: *const u8) -> &'a [u8] {
    let bytes = unsafe { (ty.text.unwrap())(data.cast()) };
    if bytes.length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(bytes.data, bytes.length) }
    }
}

pub(crate) unsafe fn format(out: &mut impl Write, value: ResinReprValue) -> io::Result<()> {
    let ty = unsafe { &*value.ty };
    if ty.kind == TEXT {
        out.write_all(unsafe { text(ty, value.data.cast()) })
    } else {
        unsafe { render(out, value) }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn quoted_bytes_preserve_unicode_and_escape_invalid_text() {
        let mut out = Vec::new();
        super::quoted(&mut out, b"a\n\0\xff\xc3\xa9").unwrap();
        assert_eq!(out, "\"a\\n\\0\\xffé\"".as_bytes());
    }
}
