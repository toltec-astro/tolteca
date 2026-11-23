"""The comparison must not conceal shifts or calibrate away differences."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).parents[2] / "examples/tune_model_comparison/compare_maps.py"
SPEC = importlib.util.spec_from_file_location("compare_maps", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_shared_axis_preserves_sky_coordinates():
    left, right = np.arange(-6.0, 7.0, 2.0), np.arange(-2.0, 11.0, 2.0)
    a, b = MODULE.shared_axis(left, right)
    np.testing.assert_array_equal(left[a], right[b])
    np.testing.assert_array_equal(a, [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(b, [0, 1, 2, 3, 4])


@pytest.mark.parametrize("right", [np.arange(4.0) + 0.5, np.arange(4.0) * 2])
def test_rejects_interpolation_or_rescaling(right):
    with pytest.raises(ValueError):
        MODULE.shared_axis(np.arange(4.0), right)


def test_difference_uses_common_mask_without_rescaling():
    signal = np.arange(9.0, dtype=float).reshape(3, 3)
    left = {
        "signal": signal,
        "weight": np.ones((3, 3)),
        "coverage": np.ones((3, 3)),
        "x": np.arange(-1.0, 2.0),
        "y": np.arange(-1.0, 2.0),
        "header": {"UNIT": "mJy/beam"},
    }
    weight = np.ones((3, 3))
    weight[0, 0] = 0
    right = dict(left, signal=signal + 3, weight=weight)
    _, _, a, b, metrics = MODULE.compare(left, right)
    assert metrics["common_pixels"] == 8
    assert metrics["difference_rms"] == 3
    assert metrics["difference_max_abs"] == 3
    assert np.isnan(a[0, 0]) and np.isnan(b[0, 0])
    right["header"] = {"UNIT": "Jy/beam"}
    with pytest.raises(ValueError, match="units differ"):
        MODULE.compare(left, right)


def test_opposite_storage_direction_uses_wcs_not_image_registration():
    left = np.arange(-4.0, 5.0, 2.0)[::-1]
    right = np.arange(-2.0, 7.0, 2.0)
    a, b = MODULE.shared_axis(left, right)
    np.testing.assert_array_equal(left[a], right[b])
    np.testing.assert_array_equal(left[a], [-2.0, 0.0, 2.0, 4.0])
    np.testing.assert_array_equal(a, [3, 2, 1, 0])
