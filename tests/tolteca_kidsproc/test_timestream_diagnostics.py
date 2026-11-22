"""Post-solving diagnostics preserve the fitted model and correlated noise."""

import numpy as np
import pytest

from tolteca_kidsproc.timestream_diagnostics import coordinate_spectra, spectral_matrix

from .test_joint_continuation import case


@pytest.mark.parametrize("strength", [0.0, 0.4])
def test_group_evaluation_preserves_solving_response_and_coordinates(strength):
    solver, _, fr, r = case(b=2 * strength * np.array([3e-5, 4e-5]) ** 3)
    evaluated = solver.evaluate(fr[:2], r[:2])
    x = solver.probes[None, :, None] / fr[:2, None, :] - 1
    xp = evaluated.x_prime.values
    np.testing.assert_allclose(
        xp - solver.b / (r[:2, None, :] ** 2 + xp**2), x, atol=1e-15
    )
    actual = solver(evaluated.I.values, evaluated.Q.values)
    assert not actual[2].any()
    np.testing.assert_allclose(actual[0], fr[:2], rtol=0, atol=0.02)
    np.testing.assert_allclose(actual[1], r[:2], rtol=0, atol=5e-12)
    assert evaluated.resonator.values.tolist() == ["r7", "r2"]
    assert evaluated.chan.values.tolist() == [19, 3, 27]
    if not strength:
        np.testing.assert_allclose(evaluated.x_prime, evaluated.x_opt, rtol=3e-16)
        np.testing.assert_array_equal(evaluated.D, np.ones(x.shape))
        np.testing.assert_array_equal(evaluated.mu, np.zeros(x.shape))


def test_noise_cross_term_is_signed_and_required_for_propagation():
    rng = np.random.default_rng(731)
    xp = rng.normal(0, 1e-7, 8192)
    r = -0.6 * xp + rng.normal(0, 2e-8, len(xp))
    D, mu = 1.4, 1.1
    xo = D * xp + mu * r
    result = coordinate_spectra(xp, xo, r, 128, D=D, mu=mu)
    actual = result.cross_real.sel(coordinate="x_opt", coordinate2="x_opt")
    np.testing.assert_allclose(result.linear_predicted, actual, rtol=1e-12)
    np.testing.assert_allclose(result.exact_reconstruction, actual, rtol=1e-12)
    assert (result.linear_terms.sel(term="cross") < 0).all()
    without_cross = result.linear_terms.sel(term=["x_prime", "loss"]).sum("term")
    assert float(without_cross.mean()) > 2 * float(actual.mean())
    assert float(result.linear_remainder.max()) < 1e-45


def test_cross_spectrum_convention_and_complete_sample_requirement():
    time = np.arange(4096) / 128
    signals = np.column_stack(
        [np.cos(2 * np.pi * 8 * time), np.sin(2 * np.pi * 8 * time)]
    )
    frequency, cross = spectral_matrix(signals, 128)
    k = np.argmax(cross[:, 0, 0].real)
    assert frequency[k] == 8
    assert cross[k, 0, 1].imag < 0
    np.testing.assert_allclose(cross, cross.swapaxes(1, 2).conj())
    signals[5, 0] = np.nan
    with pytest.raises(ValueError, match="complete real"):
        spectral_matrix(signals, 128)
    with pytest.raises(ValueError, match="two windows"):
        spectral_matrix(np.ones((30, 2)), 128, nperseg=32)


def test_local_linear_prediction_does_not_hide_nonlinear_remainder():
    time = np.arange(4096) / 128
    r = 2e-5 * (1 + 0.1 * np.sin(2 * np.pi * 7 * time))
    xp = 1e-5 + 4e-6 * np.sin(2 * np.pi * 11 * time)
    b = 2 * 0.4 * (2e-5) ** 3
    xo = xp - b / (r * r + xp * xp)
    den = (2e-5) ** 2 + (1e-5) ** 2
    result = coordinate_spectra(
        xp, xo, r, 128, D=1 + 2 * b * 1e-5 / den**2, mu=2 * b * 2e-5 / den**2
    )
    actual = result.cross_real.sel(coordinate="x_opt", coordinate2="x_opt")
    np.testing.assert_allclose(
        result.exact_reconstruction, actual, rtol=1e-11, atol=1e-28
    )
    assert float(result.linear_remainder.sum()) > 1e-15
    assert not np.allclose(result.linear_predicted, actual, rtol=0.001, atol=1e-30)
