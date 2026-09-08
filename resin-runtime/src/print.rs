use std::io::{self, Write};

use crate::host::fail;

const UNIT: u32 = 0;
const BOOL: u32 = 1;
const SIGNED: u32 = 2;
const UNSIGNED: u32 = 3;
const FLOAT32: u32 = 4;
const FLOAT64: u32 = 5;
const BYTES: u32 = 6;
const POINTER: u32 = 7;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ResinPrintBytes {
    data: *const u8,
    length: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union ResinPrintData {
    signed_value: i64,
    unsigned_value: u64,
    float_value: f64,
    bytes: ResinPrintBytes,
}

#[repr(C)]
pub struct ResinPrintArg {
    kind: u32,
    value: ResinPrintData,
}

/// Write bytes to stdout and flush, without interpreting braces or adding a newline.
/// # Safety
/// `data` must be readable for `length` bytes; NULL is allowed for an empty buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_print(data: *const u8, length: usize) {
    if unsafe { resin_stream_write(0, data, length) } != 0 {
        fail("stdout write failed");
    }
}

/// Write and flush stdout (0) or stderr (1). Returns zero on success.
/// # Safety
/// `data` must be readable for `length` bytes; NULL is allowed for an empty buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_stream_write(stream: u32, data: *const u8, length: usize) -> i32 {
    let bytes = unsafe { slice(data, length) };
    fn write(mut stream: impl Write, bytes: &[u8]) -> io::Result<()> {
        stream.write_all(bytes).and_then(|()| stream.flush())
    }
    let result = match stream {
        0 => write(io::stdout().lock(), bytes),
        1 => write(io::stderr().lock(), bytes),
        _ => return -1,
    };
    if result.is_ok() { 0 } else { -1 }
}

/// Format into an Arc allocation containing a span followed by its byte storage.
/// The span layout matches Resin's `Span<ubyte>` on the supported 64-bit targets.
/// # Safety
/// `format` and `args` must be readable for their given lengths, as must each
/// byte argument. Union members must match their tags; NULL is allowed for empty buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_format(
    format: *const u8,
    length: usize,
    args: *const ResinPrintArg,
    count: usize,
) -> *mut crate::shared::ResinArc {
    let format = unsafe { slice(format, length) };
    let args = unsafe { slice(args, count) };
    let parts = parse(format, count).unwrap_or_else(|message| fail(message));
    if args.iter().any(|arg| arg.kind > POINTER) {
        fail("invalid format argument tag");
    }
    let mut bytes = Vec::new();
    unsafe { render(&mut bytes, &parts, args) }
        .unwrap_or_else(|error| fail(&format!("format: {error}")));
    let size = size_of::<ResinPrintBytes>()
        .checked_add(bytes.len())
        .and_then(|n| n.checked_add(1))
        .unwrap_or_else(|| fail("formatted string is too large"));
    unsafe extern "C" fn destroy(_: *mut std::ffi::c_void) {}
    let owner = crate::shared::resin_arc_new(size, align_of::<ResinPrintBytes>(), destroy);
    unsafe {
        let payload = crate::shared::resin_arc_data(owner).cast::<ResinPrintBytes>();
        let data = payload.add(1).cast::<u8>();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
        *data.add(bytes.len()) = 0;
        payload.write(ResinPrintBytes {
            data,
            length: bytes.len(),
        });
    }
    owner
}

unsafe fn slice<'a, T>(data: *const T, length: usize) -> &'a [T] {
    if length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, length) }
    }
}

#[derive(Debug, PartialEq)]
enum Part<'a> {
    Text(&'a [u8]),
    Arg(usize),
}

fn parse(format: &[u8], count: usize) -> Result<Vec<Part<'_>>, &'static str> {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < format.len() {
        match format[i] {
            b'{' | b'}' if format.get(i + 1) == Some(&format[i]) => {
                parts.push(Part::Text(&format[i..i + 1]));
                i += 2;
            }
            b'{' => {
                i += 1;
                let start = i;
                let mut index = 0usize;
                while let Some(digit @ b'0'..=b'9') = format.get(i) {
                    index = index
                        .checked_mul(10)
                        .and_then(|n| n.checked_add(usize::from(digit - b'0')))
                        .ok_or("format index is too large")?;
                    i += 1;
                }
                if i == start || format.get(i) != Some(&b'}') {
                    return Err("format expects {0}, {1}, ... or escaped {{ and }} braces");
                }
                if index >= count {
                    return Err("format index out of range");
                }
                parts.push(Part::Arg(index));
                i += 1;
            }
            b'}' => return Err("unescaped } in format"),
            _ => {
                let start = i;
                while i < format.len() && !matches!(format[i], b'{' | b'}') {
                    i += 1;
                }
                parts.push(Part::Text(&format[start..i]));
            }
        }
    }
    Ok(parts)
}

unsafe fn render(
    out: &mut impl Write,
    parts: &[Part<'_>],
    args: &[ResinPrintArg],
) -> io::Result<()> {
    for part in parts {
        match part {
            Part::Text(text) => out.write_all(text)?,
            Part::Arg(index) => {
                let arg = &args[*index];
                unsafe {
                    match arg.kind {
                        UNIT => out.write_all(b"()")?,
                        BOOL => write!(out, "{}", arg.value.unsigned_value != 0)?,
                        SIGNED => write!(out, "{}", arg.value.signed_value)?,
                        UNSIGNED => write!(out, "{}", arg.value.unsigned_value)?,
                        FLOAT32 => write!(out, "{}", arg.value.float_value as f32)?,
                        FLOAT64 => write!(out, "{}", arg.value.float_value)?,
                        BYTES => {
                            let bytes = arg.value.bytes;
                            out.write_all(slice(bytes.data, bytes.length))?;
                        }
                        POINTER => write!(out, "0x{:x}", arg.value.unsigned_value)?,
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_fields_and_escaped_braces() {
        assert_eq!(
            parse(b"{{{1}}} {0}{1}", 2).unwrap(),
            vec![
                Part::Text(b"{"),
                Part::Arg(1),
                Part::Text(b"}"),
                Part::Text(b" "),
                Part::Arg(0),
                Part::Arg(1)
            ]
        );
        assert!(parse(b"", 0).unwrap().is_empty());
    }

    #[test]
    fn malformed_formats_are_rejected() {
        for format in [
            "{",
            "}",
            "{}",
            "{a}",
            "{0",
            "{0:02}",
            "{1}",
            "{999999999999999999999999999999}",
        ] {
            assert!(parse(format.as_bytes(), 1).is_err(), "{format}");
        }
        assert!(parse(b"{0}", 0).is_err());
    }
}
