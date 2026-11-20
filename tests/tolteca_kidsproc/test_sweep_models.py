"""All sweep models transform measured IQ without subtracting neighbours."""

from dataclasses import replace

import numpy as np
import pytest

from tolteca_kidsproc.models import (
    KidsSweepGainWithLinTrend,
    KidsSweepModel,
    KidsSweepWithAffineGain,
    KidsSweepWithDelayGain,
)


@pytest.mark.parametrize("family", ["affine", "delay", "gainlintrend"])
def test_common_transforms_retain_neighbour_and_noise_without_detector_states(family):
    f = np.linspace(499.95e6, 500.08e6, 71)
    fr, r, c = 500e6, 5e-5, 2e-5 + 3e-6j
    circle = 0.5 / (r + 1j * (f / fr - 1))
    neighbour = (0.3 + 0.1j) * 0.5 / (3e-5 + 1j * (f / 500.025e6 - 1))
    noise = np.random.default_rng(123).normal(size=(2, len(f)))
    coordinate = circle + neighbour + noise[0] + 1j * noise[1]
    origin, h0 = 499.99e6, 2 + 3j
    if family == "affine":
        model = KidsSweepWithAffineGain(fr, r, c, origin, h0, 1e-6 - 2e-6j)
        gain = h0 + (1e-6 - 2e-6j) * (f - origin)
        measured = gain * (1 - 2 * c * coordinate)
        state_free = replace(model, fr=np.nan, r=np.nan)
    elif family == "delay":
        model = KidsSweepWithDelayGain(fr, r, c, origin, h0, 2e-7, 30e-9)
        gain = h0 * np.exp((2e-7 - 2j * np.pi * 30e-9) * (f - origin))
        measured = gain * (1 - 2 * c * coordinate)
        state_free = replace(model, fr=np.nan, r=np.nan)
    else:
        model = KidsSweepGainWithLinTrend(
            fr, 1 / (2 * r), 2, 3, origin, 1e-4, -2e-4, 4, 5
        )
        measured = (2 + 3j) * coordinate + (1e-4 - 2e-4j) * (f - origin) + 4 + 5j
        state_free = replace(model, fr=np.nan, Qr=np.nan)
    transformed = model.derotate(measured, f)
    np.testing.assert_allclose(transformed, coordinate, rtol=1e-12)
    np.testing.assert_allclose(
        transformed - circle, neighbour + noise[0] + 1j * noise[1], rtol=1e-12
    )
    np.testing.assert_array_equal(state_free.derotate(measured, f), transformed)
    np.testing.assert_allclose(model.rotate(transformed, f), measured, rtol=1e-12)
    # Historical (f-fr)/fr and physical f/fr-1 round differently near resonance.
    np.testing.assert_allclose(model.derotate(model.eval(f), f), circle, rtol=5e-12)


@pytest.mark.parametrize(
    "model_class", [KidsSweepWithAffineGain, KidsSweepWithDelayGain]
)
def test_physical_broadcasting_and_zero_gain_or_coupling(model_class):
    values = {
        "fr": np.array([500e6, 500.02e6]),
        "r": np.array([5e-5, 4e-5]),
        "c": np.array([2e-5 + 1e-6j, 1e-5 - 1e-6j]),
        "f_g": 500e6,
        "H0": 1 + 2j,
    }
    if model_class is KidsSweepWithAffineGain:
        model = model_class(**values, H1=1e-6j)
    else:
        model = model_class(**values, a=1e-6, tau=20e-9)
    f = np.linspace(499.95e6, 500.05e6, 7)[:, None]
    measured = np.full((7, 2), 0.3 + 0.6j)
    np.testing.assert_allclose(model.rotate(model.derotate(measured, f), f), measured)
    assert model.eval(f).shape == (7, 2)
    for invalid in (
        replace(
            model, H0=0, **({"H1": 0} if model_class is KidsSweepWithAffineGain else {})
        ),
        replace(model, c=0),
    ):
        with pytest.raises(ValueError, match="zero gain or coupling"):
            invalid.derotate(measured, f)
    missing = replace(model, c=np.array([2e-5, np.nan]))
    transformed = missing.derotate(measured, f)
    assert np.isfinite(transformed[:, 0]).all()
    assert np.isnan(transformed[:, 1]).all()


def test_pair_selection_requires_an_existing_association_and_is_a_snapshot():
    f = np.linspace(499.9e6, 500.2e6, 25)
    group = KidsSweepModel(
        [500e6, 500.03e6], [1e4, 2e4], [f, f], associations=np.eye(2, dtype=bool)
    )
    params = group.make_params()
    model = group.pair_model(params, 1, 1)
    before = model.eval(f)
    params["c1_h0_re"].value = 5
    np.testing.assert_array_equal(model.eval(f), before)
    assert not np.allclose(group.pair_model(params, 1, 1).eval(f), before)
    with pytest.raises(ValueError, match="not included"):
        group.pair_model(params, 0, 1)
    with pytest.raises(IndexError, match="outside"):
        group.pair_model(params, -1, 0)
