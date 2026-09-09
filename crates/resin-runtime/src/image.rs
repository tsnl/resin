//! Host-side PNG encode/decode for the C ABI.

use std::borrow::Cow;
use std::ffi::{CStr, c_char, c_void};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::ptr;
use std::slice;

use crate::ResinStatus;

struct PixelLayout {
    width: u32,
    height: u32,
    color: png::ColorType,
    row_bytes: usize,
    stride: usize,
    byte_len: usize,
}

impl PixelLayout {
    fn new(width: u32, height: u32, channels: u32, stride: usize) -> Result<Self, ResinStatus> {
        let color = color_type(channels).ok_or(ResinStatus::InvalidArgument)?;
        let row_bytes = (width as usize)
            .checked_mul(channels as usize)
            .ok_or(ResinStatus::InvalidArgument)?;
        let stride = if stride == 0 { row_bytes } else { stride };
        let byte_len = stride
            .checked_mul(height as usize)
            .ok_or(ResinStatus::InvalidArgument)?;
        if width == 0 || height == 0 || stride < row_bytes || byte_len > isize::MAX as usize {
            return Err(ResinStatus::InvalidArgument);
        }
        Ok(Self {
            width,
            height,
            color,
            row_bytes,
            stride,
            byte_len,
        })
    }

    fn packed_pixels<'a>(&self, pixels: &'a [u8]) -> Result<Cow<'a, [u8]>, ResinStatus> {
        let pixels = pixels
            .get(..self.byte_len)
            .ok_or(ResinStatus::InvalidArgument)?;
        if self.stride == self.row_bytes {
            return Ok(Cow::Borrowed(pixels));
        }
        let mut packed = Vec::with_capacity(self.row_bytes * self.height as usize);
        for row in pixels.chunks_exact(self.stride) {
            packed.extend_from_slice(&row[..self.row_bytes]);
        }
        Ok(Cow::Owned(packed))
    }
}

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
    if pixels.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let layout = match PixelLayout::new(width, height, channels, row_stride) {
        Ok(layout) => layout,
        Err(status) => return status,
    };
    let pixels = unsafe { slice::from_raw_parts(pixels.cast::<u8>(), layout.byte_len) };
    match write_png(path, &layout, pixels) {
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

    let decoded = match image_read_png(path, requested_channels) {
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
pub fn image_write_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    channels: u32,
    pixels: &[u8],
) -> Result<(), ResinStatus> {
    let layout = PixelLayout::new(width, height, channels, 0)?;
    write_png(path.as_ref(), &layout, pixels)
}

pub struct PngImage {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub pixels: Vec<u8>,
}

/// Packed 8-bit pixels. `requested_channels` 0 keeps the file's channel count.
pub fn image_read_png(
    path: impl AsRef<Path>,
    requested_channels: u32,
) -> Result<PngImage, ResinStatus> {
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

fn write_png(path: &Path, layout: &PixelLayout, pixels: &[u8]) -> Result<(), ResinStatus> {
    let pixels = layout.packed_pixels(pixels)?;
    let file = File::create(path).map_err(|_| ResinStatus::IoError)?;
    encode_png(BufWriter::new(file), layout, &pixels)
}

fn encode_png(output: impl Write, layout: &PixelLayout, pixels: &[u8]) -> Result<(), ResinStatus> {
    let mut encoder = png::Encoder::new(output, layout.width, layout.height);
    encoder.set_color(layout.color);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|_| ResinStatus::IoError)?;
    writer
        .write_image_data(pixels)
        .map_err(|_| ResinStatus::IoError)?;
    writer.finish().map_err(|_| ResinStatus::IoError)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::io;
    use std::path::PathBuf;

    struct FlushError;

    impl Write for FlushError {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("output is full"))
        }
    }

    fn test_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("resin-png-{name}-{}.png", std::process::id()))
    }

    #[test]
    fn encoder_reports_flush_errors() {
        let layout = PixelLayout::new(1, 1, 4, 0).unwrap();
        assert_eq!(
            encode_png(FlushError, &layout, &[255, 0, 0, 255]),
            Err(ResinStatus::IoError)
        );
    }

    #[test]
    fn rust_writer_uses_only_the_image_bytes() {
        let path = test_path("extra-bytes");
        let pixels = [255, 0, 0, 255, 42, 42, 42, 42];
        image_write_png(&path, 1, 1, 4, &pixels).unwrap();
        let decoded = image_read_png(&path, 0).unwrap();
        assert_eq!((decoded.width, decoded.height, decoded.channels), (1, 1, 4));
        assert_eq!(decoded.pixels, pixels[..4]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn c_writer_removes_row_padding() {
        let path = test_path("row-padding");
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let pixels: [u8; 12] = [1, 2, 3, 4, 90, 90, 5, 6, 7, 8, 90, 90];
        assert_eq!(
            unsafe { resin_image_write_png(path_c.as_ptr(), 1, 2, 4, pixels.as_ptr().cast(), 6) },
            ResinStatus::Success
        );
        let decoded = image_read_png(&path, 0).unwrap();
        assert_eq!((decoded.width, decoded.height, decoded.channels), (1, 2, 4));
        assert_eq!(decoded.pixels, [1, 2, 3, 4, 5, 6, 7, 8]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_images_preserve_the_output_file() {
        let path = test_path("invalid-image");
        std::fs::write(&path, b"existing contents").unwrap();
        for (width, height, channels) in [
            (0, 1, 4),
            (1, 0, 4),
            (1, 1, 0),
            (1, 1, 5),
            (u32::MAX, u32::MAX, 4),
            (2, 1, 4),
        ] {
            assert_eq!(
                image_write_png(&path, width, height, channels, &[0; 4]),
                Err(ResinStatus::InvalidArgument)
            );
            assert_eq!(std::fs::read(&path).unwrap(), b"existing contents");
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_c_layouts_are_rejected_before_reading_pixels() {
        let path = test_path("invalid-c-layout");
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        std::fs::write(&path, b"existing contents").unwrap();
        for (width, height, channels, stride) in [
            (0, 1, 4, 0),
            (1, 0, 4, 0),
            (1, 1, 0, 0),
            (1, 1, 5, 0),
            (1, 1, 4, 3),
            (1, 1, 4, usize::MAX),
            (1, 2, 4, usize::MAX),
            (u32::MAX, u32::MAX, 4, 0),
        ] {
            assert_eq!(
                unsafe {
                    resin_image_write_png(
                        path_c.as_ptr(),
                        width,
                        height,
                        channels,
                        ptr::dangling(),
                        stride,
                    )
                },
                ResinStatus::InvalidArgument
            );
            assert_eq!(std::fs::read(&path).unwrap(), b"existing contents");
        }
        std::fs::remove_file(path).unwrap();
    }
}
