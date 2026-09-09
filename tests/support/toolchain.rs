#![allow(dead_code)]
use resin_toolchain::Environment;
use resin_toolchain::Toolchain;
use std::ffi::OsStr;

pub fn c(compiler: &OsStr) -> Toolchain {
    Environment::capture()
        .unwrap()
        .toolchain(Some(compiler), None)
}

pub fn glsl(compiler: &OsStr) -> Toolchain {
    Environment::capture()
        .unwrap()
        .toolchain(None, Some(compiler))
}
