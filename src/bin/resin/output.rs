use std::{
    io::{self, Write},
    path::Path,
};

pub(super) fn write(bytes: &[u8], destination: Option<&Path>) -> super::Result<i32> {
    if let Some(path) = destination {
        resin::toolchain::write_output(bytes, path)?;
    } else {
        io::stdout().lock().write_all(bytes)?;
    }
    Ok(0)
}
