//! Linear floating-point images. File codecs stay behind the standard image ABI.
use crate::{FloatImage, ResinStatus};
use std::{
    ffi::{CStr, c_char},
    path::Path,
    ptr, slice,
};

fn sample_count(width: u32, height: u32) -> Result<usize, ResinStatus> {
    if width == 0 || height == 0 {
        return Err(ResinStatus::InvalidArgument);
    }
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|count| count.checked_mul(4))
        .filter(|count| *count <= isize::MAX as usize / size_of::<f32>())
        .ok_or(ResinStatus::InvalidArgument)
}

fn allocate(width: usize, height: usize) -> Result<FloatImage, ResinStatus> {
    let width = u32::try_from(width).map_err(|_| ResinStatus::InvalidArgument)?;
    let height = u32::try_from(height).map_err(|_| ResinStatus::InvalidArgument)?;
    let count = sample_count(width, height)?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(count)
        .map_err(|_| ResinStatus::OutOfMemory)?;
    pixels.resize(count, 0.0);
    Ok(FloatImage {
        width,
        height,
        pixels,
    })
}

pub(crate) fn read_exr(path: &Path) -> Result<FloatImage, ResinStatus> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _| allocate(resolution.width(), resolution.height()),
        |output, position, (r, g, b, a): (f32, f32, f32, f32)| {
            if let Ok(output) = output {
                let index = (position.y() * output.width as usize + position.x()) * 4;
                output.pixels[index..index + 4].copy_from_slice(&[r, g, b, a]);
            }
        },
    )
    .map_err(|_| ResinStatus::IoError)?;
    image.layer_data.channel_data.pixels
}

pub(crate) fn read_hdr(path: &Path) -> Result<FloatImage, ResinStatus> {
    let reader = ::image::ImageReader::open(path).map_err(|_| ResinStatus::IoError)?;
    let reader = reader
        .with_guessed_format()
        .map_err(|_| ResinStatus::IoError)?;
    if reader.format() != Some(::image::ImageFormat::Hdr) {
        return Err(ResinStatus::IoError);
    }
    let image = reader
        .decode()
        .map_err(|_| ResinStatus::IoError)?
        .into_rgba32f();
    let (width, height) = image.dimensions();
    sample_count(width, height)?;
    Ok(FloatImage {
        width,
        height,
        pixels: image.into_raw(),
    })
}

pub(crate) fn write_exr(
    path: &Path,
    width: u32,
    height: u32,
    pixels: &[f32],
) -> Result<(), ResinStatus> {
    if pixels.len() != sample_count(width, height)? {
        return Err(ResinStatus::InvalidArgument);
    }
    exr::prelude::write_rgba_file(path, width as usize, height as usize, |x, y| {
        let index = (y * width as usize + x) * 4;
        (
            pixels[index],
            pixels[index + 1],
            pixels[index + 2],
            pixels[index + 3],
        )
    })
    .map_err(|_| ResinStatus::IoError)
}

// Both decoders publish only complete images. The caller can always free a
// successful result through resin_image_free, independent of the source codec.
unsafe fn read_image(
    path: *const c_char,
    width: *mut u32,
    height: *mut u32,
    pixels: *mut *mut f32,
    decoder: fn(&Path) -> Result<FloatImage, ResinStatus>,
) -> ResinStatus {
    if width.is_null() || height.is_null() || pixels.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe {
        *width = 0;
        *height = 0;
        *pixels = ptr::null_mut();
    }
    if path.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let Ok(path) = (unsafe { CStr::from_ptr(path) }).to_str() else {
        return ResinStatus::InvalidArgument;
    };
    let image = match decoder(Path::new(path)) {
        Ok(image) => image,
        Err(status) => return status,
    };
    let output = unsafe { libc::malloc(image.pixels.len() * size_of::<f32>()) }.cast::<f32>();
    if output.is_null() {
        return ResinStatus::OutOfMemory;
    }
    unsafe {
        ptr::copy_nonoverlapping(image.pixels.as_ptr(), output, image.pixels.len());
        *width = image.width;
        *height = image.height;
        *pixels = output;
    }
    ResinStatus::Success
}

/// # Safety
/// Path is a C string; output pointers are writable. Free successful pixels with resin_image_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_image_read_exr(
    path: *const c_char,
    width: *mut u32,
    height: *mut u32,
    pixels: *mut *mut f32,
) -> ResinStatus {
    unsafe { read_image(path, width, height, pixels, read_exr) }
}

/// # Safety
/// Path is a C string; output pointers are writable. Free successful pixels with resin_image_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_image_read_hdr(
    path: *const c_char,
    width: *mut u32,
    height: *mut u32,
    pixels: *mut *mut f32,
) -> ResinStatus {
    unsafe { read_image(path, width, height, pixels, read_hdr) }
}

/// # Safety
/// Path is a C string. Pixels covers sample_count initialized float32 samples.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_image_write_exr(
    path: *const c_char,
    width: u32,
    height: u32,
    pixels: *const f32,
    count: usize,
) -> ResinStatus {
    if path.is_null() || pixels.is_null() || sample_count(width, height) != Ok(count) {
        return ResinStatus::InvalidArgument;
    }
    let Ok(path) = (unsafe { CStr::from_ptr(path) }).to_str() else {
        return ResinStatus::InvalidArgument;
    };
    ResinStatus::from_result(write_exr(Path::new(path), width, height, unsafe {
        slice::from_raw_parts(pixels, count)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exr_roundtrip_preserves_hdr_negative_and_alpha_samples() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("samples.exr");
        let pixels = [32.0, -0.5, 0.125, 0.25, 0.0, 2.0, 4.0, 1.0];
        write_exr(&path, 2, 1, &pixels).unwrap();
        assert_eq!(
            read_exr(&path).unwrap(),
            FloatImage {
                width: 2,
                height: 1,
                pixels: pixels.to_vec()
            }
        );
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            write_exr(&path, 3, 1, &pixels),
            Err(ResinStatus::InvalidArgument)
        );
        assert_eq!(
            write_exr(&path, u32::MAX, u32::MAX, &pixels),
            Err(ResinStatus::InvalidArgument)
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn rgb_exr_supplies_alpha_and_failed_ffi_reads_clear_outputs() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("rgb.exr");
        exr::prelude::write_rgb_file(&path, 1, 1, |_, _| (8.0_f32, 0.5_f32, -1.0_f32)).unwrap();
        assert_eq!(read_exr(&path).unwrap().pixels, [8.0, 0.5, -1.0, 1.0]);
        let mut width = 100;
        let mut height = 100;
        let mut pixels = ptr::dangling_mut::<f32>();
        assert_eq!(
            unsafe { resin_image_read_exr(ptr::null(), &mut width, &mut height, &mut pixels) },
            ResinStatus::InvalidArgument
        );
        assert_eq!((width, height, pixels), (0, 0, ptr::null_mut()));
    }

    #[test]
    fn radiance_rgbe_values_decode_without_tone_mapping() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("environment.hdr");
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n".to_vec();
        bytes.extend_from_slice(&[128, 64, 32, 132]);
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(
            read_hdr(&path).unwrap(),
            FloatImage {
                width: 1,
                height: 1,
                pixels: vec![8.0, 4.0, 2.0, 1.0]
            }
        );
    }
}
