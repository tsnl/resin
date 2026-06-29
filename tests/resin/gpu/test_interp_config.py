"""Config validation tests for resin_rt_pybind.Interp."""

import pytest

import resin_rt_pybind


def test_unknown_config_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown config key"):
        _ = resin_rt_pybind.Interp("wgpu", {"bogus": 1})


def test_device_name_must_be_string() -> None:
    with pytest.raises(ValueError, match="device_name must be a string"):
        _ = resin_rt_pybind.Interp("wgpu", {"device_name": 42})


def test_unknown_device_name_raises() -> None:
    with pytest.raises(ValueError, match="no GPU adapter found with device_name"):
        _ = resin_rt_pybind.Interp(
            "wgpu",
            {"device_name": "nonexistent-adapter-name-xyz"},
        )
