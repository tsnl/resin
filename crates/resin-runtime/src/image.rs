//! Host-side PNG encode/decode for the C ABI.

use std::ffi::{CStr, c_char, c_void};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::ptr;
use std::slice;

use crate::ResinStatus;

/// # Safety
/// `path` must be a valid C string; `pixels` must hold `height` rows of
/// `row_stride` (or `width * channels`) packed 8-bit samples.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_image_write_png(
    path: *const c_char,
    width: u32,
    height: u32,
    channels: u32,
    pixels: *const c_void,
    row_stride: usize,
) -> ResinStatus {
    let Some(path) = c_path(path) else {
        return ResinStatus::InvalidArgument;
    };
    let Some(color) = color_type(channels) else {
        return ResinStatus::InvalidArgument;
    };
    if width == 0 || height == 0 || pixels.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let packed = (width as usize).saturating_mul(channels as usize);
    if packed == 0 {
        return ResinStatus::InvalidArgument;
    }
    let stride = if row_stride == 0 { packed } else { row_stride };
    if stride < packed {
        return ResinStatus::InvalidArgument;
    }
    let total = stride.saturating_mul(height as usize);
    let src = unsafe { slice::from_raw_parts(pixels.cast::<u8>(), total) };

    let packed_buf;
    let image: &[u8] = if stride == packed {
        src
    } else {
        packed_buf = pack_rows(src, height as usize, packed, stride);
        &packed_buf
    };

    match encode_png(path, width, height, color, image) {
        Ok(()) => ResinStatus::Success,
        Err(status) => status,
    }
}

/// # Safety
/// `path` must be a valid C string. Output pointers must be valid.
/// `*out_pixels` is allocated with malloc and must be freed with
/// [`resin_image_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_image_read_png(
    path: *const c_char,
    requested_channels: u32,
    out_width: *mut u32,
    out_height: *mut u32,
    out_channels: *mut u32,
    out_pixels: *mut *mut c_void,
) -> ResinStatus {
    if out_width.is_null()
        || out_height.is_null()
        || out_channels.is_null()
        || out_pixels.is_null()
        || (requested_channels != 0 && color_type(requested_channels).is_none())
    {
        return ResinStatus::InvalidArgument;
    }
    unsafe {
        *out_width = 0;
        *out_height = 0;
        *out_channels = 0;
        *out_pixels = ptr::null_mut();
    }
    let Some(path) = c_path(path) else {
        return ResinStatus::InvalidArgument;
    };

    let decoded = match read_png(path, requested_channels) {
        Ok(decoded) => decoded,
        Err(status) => return status,
    };

    let Some(ptr) = malloc_copy(&decoded.pixels) else {
        return ResinStatus::OutOfMemory;
    };
    unsafe {
        *out_width = decoded.width;
        *out_height = decoded.height;
        *out_channels = decoded.channels;
        *out_pixels = ptr;
    }
    ResinStatus::Success
}

/// # Safety
/// `pixels` must be null or a pointer from [`resin_image_read_png`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_image_free(pixels: *mut c_void) {
    if !pixels.is_null() {
        unsafe { libc::free(pixels) };
    }
}

/// Packed 8-bit pixels, top-left origin. `channels` is 1=Y, 2=YA, 3=RGB, 4=RGBA.
pub fn write_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    channels: u32,
    pixels: &[u8],
) -> Result<(), ResinStatus> {
    let Some(color) = color_type(channels) else {
        return Err(ResinStatus::InvalidArgument);
    };
    if width == 0 || height == 0 {
        return Err(ResinStatus::InvalidArgument);
    }
    let packed = (width as usize).saturating_mul(channels as usize);
    if packed == 0 || pixels.len() < packed.saturating_mul(height as usize) {
        return Err(ResinStatus::InvalidArgument);
    }
    encode_png(path.as_ref(), width, height, color, pixels)
}

pub struct PngImage {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub pixels: Vec<u8>,
}

/// Packed 8-bit pixels. `requested_channels` 0 keeps the file's channel count.
pub fn read_png(path: impl AsRef<Path>, requested_channels: u32) -> Result<PngImage, ResinStatus> {
    if requested_channels != 0 && color_type(requested_channels).is_none() {
        return Err(ResinStatus::InvalidArgument);
    }
    let decoded = decode_png(path.as_ref())?;
    if requested_channels == 0 || requested_channels == decoded.channels {
        return Ok(decoded);
    }
    let pixels = convert_channels(
        &decoded.pixels,
        decoded.channels,
        requested_channels,
        decoded.width as usize * decoded.height as usize,
    );
    Ok(PngImage {
        width: decoded.width,
        height: decoded.height,
        channels: requested_channels,
        pixels,
    })
}

fn encode_png(
    path: &Path,
    width: u32,
    height: u32,
    color: png::ColorType,
    pixels: &[u8],
) -> Result<(), ResinStatus> {
    let file = File::create(path).map_err(|_| ResinStatus::IoError)?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|_| ResinStatus::IoError)?;
    writer
        .write_image_data(pixels)
        .map_err(|_| ResinStatus::IoError)
}

fn decode_png(path: &Path) -> Result<PngImage, ResinStatus> {
    let file = File::open(path).map_err(|_| ResinStatus::IoError)?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|_| ResinStatus::IoError)?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|_| ResinStatus::IoError)?;
    buf.truncate(info.buffer_size());
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err(ResinStatus::Unsupported),
    };
    Ok(PngImage {
        width: info.width,
        height: info.height,
        channels,
        pixels: buf,
    })
}

fn color_type(channels: u32) -> Option<png::ColorType> {
    match channels {
        1 => Some(png::ColorType::Grayscale),
        2 => Some(png::ColorType::GrayscaleAlpha),
        3 => Some(png::ColorType::Rgb),
        4 => Some(png::ColorType::Rgba),
        _ => None,
    }
}

fn c_path<'a>(path: *const c_char) -> Option<&'a Path> {
    if path.is_null() {
        return None;
    }
    let cstr = unsafe { CStr::from_ptr(path) };
    let s = cstr.to_str().ok()?;
    if s.is_empty() {
        return None;
    }
    Some(Path::new(s))
}

fn pack_rows(src: &[u8], height: usize, packed: usize, stride: usize) -> Vec<u8> {
    let mut out = vec![0u8; packed * height];
    for y in 0..height {
        let dst = y * packed;
        let src_row = y * stride;
        out[dst..dst + packed].copy_from_slice(&src[src_row..src_row + packed]);
    }
    out
}

fn convert_channels(src: &[u8], src_ch: u32, dst_ch: u32, pixels: usize) -> Vec<u8> {
    let src_ch = src_ch as usize;
    let dst_ch = dst_ch as usize;
    let mut out = vec![0u8; pixels * dst_ch];
    for i in 0..pixels {
        let s = &src[i * src_ch..];
        let (r, g, b, a) = match src_ch {
            1 => (s[0], s[0], s[0], 255),
            2 => (s[0], s[0], s[0], s[1]),
            3 => (s[0], s[1], s[2], 255),
            4 => (s[0], s[1], s[2], s[3]),
            _ => (0, 0, 0, 255),
        };
        let d = &mut out[i * dst_ch..];
        match dst_ch {
            1 => d[0] = luma(r, g, b),
            2 => {
                d[0] = luma(r, g, b);
                d[1] = a;
            }
            3 => {
                d[0] = r;
                d[1] = g;
                d[2] = b;
            }
            4 => {
                d[0] = r;
                d[1] = g;
                d[2] = b;
                d[3] = a;
            }
            _ => {}
        }
    }
    out
}

fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((u32::from(r) * 77 + u32::from(g) * 150 + u32::from(b) * 29) / 256) as u8
}

fn malloc_copy(bytes: &[u8]) -> Option<*mut c_void> {
    if bytes.is_empty() {
        return None;
    }
    let ptr = unsafe { libc::malloc(bytes.len()) };
    if ptr.is_null() {
        return None;
    }
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
    }
    Some(ptr)
}
