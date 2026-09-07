#![allow(dead_code)]
use resin::{cli::Environment, toolchain::Settings};
use std::ffi::OsStr;

pub fn c(compiler: &OsStr) -> Settings {
    Environment::capture()
        .unwrap()
        .toolchain(Some(compiler), None)
}

pub fn glsl(compiler: &OsStr) -> Settings {
    Environment::capture()
        .unwrap()
        .toolchain(None, Some(compiler))
}
