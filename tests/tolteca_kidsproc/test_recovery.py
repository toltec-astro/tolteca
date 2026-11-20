"""Recovery diagnostics distinguish missing information from noisy information."""

import numpy as np
import pytest

from tolteca_kidsproc.recovery import SampleFlag, recovery_diagnostics


def test_duplicate_response_is_unresolved_even_with_enough_channels():
    jac = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]])
    result = recovery_diagnostics(jac, np.eye(4))
    assert result["rank"] == 2
    assert result["flag"] == SampleFlag.singular_response
    assert np.isinf(result["condition"])
    assert np.isnan(result["covariance"]).all()
    # Independent probes recover all four states with no deblending cost.
    independent = recovery_diagnostics(np.eye(4), np.eye(4))
    np.testing.assert_allclose(independent["noise_amplification"], 1)
    assert independent["flag"] == 0


def test_weak_response_warning_preserves_uncertainty_and_unit_invariance():
    jac = np.diag([1, 1, 0.01, 0.01])
    base = recovery_diagnostics(jac, np.eye(4), condition_max=20)
    assert base["rank"] == 4
    assert base["flag"] == SampleFlag.condition_high
    np.testing.assert_allclose(np.diag(base["covariance"]), [1, 1, 10000, 10000])
    # Change signal units in one readout only; whitened diagnostics cannot change.
    units = np.diag([3, 2, 3, 2])
    rescaled = recovery_diagnostics(units @ jac, units @ units.T, condition_max=20)
    np.testing.assert_allclose(rescaled["covariance"], base["covariance"])
    assert rescaled["condition"] == pytest.approx(base["condition"])
    assert base["calibration_uncertainty"] == "not_propagated"
    assert base["acceptance"] == "not_assessed"
    assert recovery_diagnostics(jac, np.eye(4))["flag"] == 0


def test_correlated_noise_matches_monte_carlo_recovered_errors():
    rng = np.random.default_rng(37)
    jac = np.array([[1, 0, 0.5, 0], [0.4, 0, 1, 0], [0, 1, 0, 0.4], [0, 0.2, 0, 1]])
    noise = np.array([[1, 0.3, 0, 0], [0.3, 2, 0, 0], [0, 0, 1, -0.2], [0, 0, -0.2, 1]])
    result = recovery_diagnostics(jac, noise)
    measured = rng.multivariate_normal(np.zeros(4), noise, 100000)
    states = np.linalg.solve(jac, measured.T).T
    np.testing.assert_allclose(
        np.cov(states, rowvar=False), result["covariance"], atol=0.03
    )
    assert np.all(result["noise_amplification"] >= 1)


@pytest.mark.parametrize("noise", [np.zeros((4, 4)), np.diag([1, 1, 1, -1])])
def test_invalid_noise_is_not_replaced_with_identity(noise):
    with pytest.raises(ValueError, match="positive definite"):
        recovery_diagnostics(np.eye(4), noise)
