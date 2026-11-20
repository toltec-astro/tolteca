"""Check the power convention and sampling estimator for the r reference."""

import numpy as np
import pytest
from scipy.signal import csd, welch

from tolteca_kidsproc._noise_reference import rx_noise_budget, rx_periodograms


def test_power_budget_and_unresolved_excess():
    """ASD subtraction and clipping must not masquerade as photon inference."""
    result = rx_noise_budget([1, 1, 1], [11, 1, 0.5])
    np.testing.assert_allclose(result["excess_x"], [10, 0, -0.5])
    assert result["kappa"][0] == pytest.approx(np.sqrt(10))
    assert result["penalty"][0] == pytest.approx(np.sqrt(1.1))
    assert np.isnan(result["kappa"][1:]).all()
    assert np.isnan(result["penalty"][1:]).all()
    assert rx_noise_budget(1, 11, beta=2)["excess_x"] == 9
    with pytest.raises(ValueError, match="positive"):
        rx_noise_budget(0, 1)


def test_periodograms_match_complex_cross_scipy_welch():
    """Preserve cross-spectrum sign, one-sided normalization and leading axes."""
    rng = np.random.default_rng(302)
    w = rng.normal(size=(2, 8192)) + 1j * rng.normal(size=(2, 8192))
    f, spectra = rx_periodograms(w, 128, step=512)
    ff, pr = welch(w.real, 128, nperseg=1024)
    _, cross = csd(w.real, w.imag, 128, nperseg=1024)
    np.testing.assert_allclose(f, ff)
    np.testing.assert_allclose(spectra[..., 0, 0].mean(axis=-2).real, pr)
    np.testing.assert_allclose(spectra[..., 1, 0].mean(axis=-2), cross)


def test_known_photon_and_circular_readout_noise():
    """Recover injected excess and show equal solved readout under rotation."""
    rng = np.random.default_rng(112)
    n = 1024 * 512
    readout = rng.normal(size=n) + 1j * rng.normal(size=n)
    # Any complex local inverse rotates and scales circular readout equally.
    readout *= 2 * np.exp(0.7j)
    optical = 1j * np.sqrt(40) * rng.normal(size=n)
    _, spectra = rx_periodograms(readout + optical, 128)
    powers = spectra[..., 1:, :, :].mean(axis=(0, 1)).diagonal().real
    result = rx_noise_budget(*powers)
    assert result["kappa"] == pytest.approx(np.sqrt(10), rel=0.01)
    assert result["penalty"] == pytest.approx(np.sqrt(1.1), rel=0.002)
