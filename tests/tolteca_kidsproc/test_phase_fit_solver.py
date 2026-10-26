"""Numerical identities, units, covariance and failure behavior of phase fitting."""

from pathlib import Path

import numpy as np
import pytest

from tolteca_kidsproc._phase_fit_solver import _PhaseModel, fit_phase


@pytest.mark.parametrize("quality", [-20000.0, 1e12])
def test_nonlinear_window_rejects_empty_or_undersampled_data(quality):
    from tolteca_kidsproc._phase_fit import ResonanceFitterSingleTone

    frequency = 6.4e8 + np.arange(-20, 21) * 1000
    signal = 2 + np.exp(1j * np.linspace(-2, 2, len(frequency)))
    # Call the phase stage directly: negative or excessively large reference Q
    # yields an empty or single-sample window before any optimizer is reached.
    fitter = object.__new__(ResonanceFitterSingleTone)
    with pytest.raises(ValueError, match="at least four samples"):
        fitter.fitphasenlin_c2(
            frequency, signal, {"f0": 6.4e8, "Q": quality, "phi": 0, "r": 1}
        )


def problem(nonlinear):
    frequency = 6.4e8 + np.linspace(-24000, 24000, 41)
    radius = 2.3
    angle = np.linspace(-2.4, 2.1, len(frequency))
    signal = 1.2 + 1.8j + radius * np.exp(1j * angle)
    initial = np.array([20000.0, 6.4e8 + 500, 0.3])
    if nonlinear:
        initial = np.insert(initial, 1, 3000 / (2 * radius) ** 2)
    sigma = np.linspace(0.7, 1.5, len(frequency))
    return frequency, initial, sigma, radius, signal


def physical_phase(frequency, parameters, radius, signal):
    if len(parameters) == 4:
        quality, alpha, frequency0, phase = parameters
        shifted = (
            frequency0
            - alpha * np.abs(signal - radius * np.exp(1j * (phase + np.pi))) ** 2
        )
    else:
        quality, shifted, phase = parameters
    return phase + 2 * np.arctan(-2 * quality * (frequency - shifted) / shifted)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_model_identity_and_central_difference_jacobian(nonlinear):
    frequency, initial, sigma, radius, signal = problem(nonlinear)
    observed = physical_phase(frequency, initial, radius, signal)
    model = _PhaseModel(frequency, observed, initial, sigma, radius, signal)
    for displacement in (0.0, 0.03, -0.02):
        parameters = model.initial + displacement
        physical = parameters * model.scales + model.offsets
        expected = (
            physical_phase(frequency, physical, radius, signal) - observed
        ) / sigma
        # Calling the Jacobian first also checks the shared-intermediate cache.
        analytic = model.jacobian(parameters)
        np.testing.assert_allclose(model.residual(parameters), expected, atol=2e-10)
        central = np.empty_like(analytic)
        for column in range(len(parameters)):
            delta = np.zeros_like(parameters)
            delta[column] = 1e-6
            central[:, column] = (
                model.residual(parameters + delta) - model.residual(parameters - delta)
            ) / (2e-6)
        np.testing.assert_allclose(analytic, central, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_fit_and_covariance_in_physical_units(nonlinear):
    frequency, truth, sigma, radius, signal = problem(nonlinear)
    observed = physical_phase(frequency, truth, radius, signal)
    observed += 1e-4 * np.random.default_rng(710).normal(size=len(frequency))
    start = truth.copy()
    start[0] *= 0.97
    start[-2] += 150
    start[-1] += 0.01
    fitted, covariance, info = fit_phase(
        frequency, observed, start, sigma=sigma, radius=radius, signal=signal
    )
    np.testing.assert_allclose(
        physical_phase(frequency, fitted, radius, signal), observed, atol=3e-4
    )
    assert info["status"] in (1, 2, 3, 4)
    assert info["nfev"] < 30
    assert not info["retried"]
    # Independent physical-coordinate derivatives check covariance units,
    # including Hz and alpha, and relative-sigma/dof normalization.
    steps = np.maximum(abs(fitted) * 1e-5, 1e-6)
    steps[-2] = 0.1
    jacobian = np.empty((len(frequency), len(fitted)))
    for column, step in enumerate(steps):
        delta = np.zeros_like(fitted)
        delta[column] = step
        jacobian[:, column] = (
            physical_phase(frequency, fitted + delta, radius, signal)
            - physical_phase(frequency, fitted - delta, radius, signal)
        ) / (2 * step * sigma)
    expected = (
        np.linalg.inv(jacobian.T @ jacobian)
        * info["cost"]
        / (len(frequency) - len(fitted))
    )
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    np.testing.assert_allclose(covariance / scale, expected / scale, atol=2e-4)


def test_signal_unit_change_transforms_alpha_and_covariance():
    frequency, start, sigma, radius, signal = problem(nonlinear=True)
    observed = physical_phase(frequency, start, radius, signal)
    observed += 1e-4 * np.sin(np.arange(len(frequency)))
    baseline, covariance, _ = fit_phase(
        frequency, observed, start, sigma=sigma, radius=radius, signal=signal
    )
    factor = 1e4
    conversion = np.array([1, factor**-2, 1, 1])
    changed, changed_covariance, _ = fit_phase(
        frequency,
        observed,
        start * conversion,
        sigma=sigma,
        radius=radius * factor,
        signal=signal * factor,
    )
    np.testing.assert_allclose(changed, baseline * conversion, rtol=1e-8)
    np.testing.assert_allclose(
        changed_covariance, covariance * np.outer(conversion, conversion), rtol=1e-7
    )


@pytest.mark.parametrize("nonlinear", [False, True])
def test_bounded_fit_and_explicit_evaluation_limit(nonlinear):
    frequency, truth, sigma, radius, signal = problem(nonlinear)
    observed = physical_phase(frequency, truth, radius, signal)
    start = truth.copy()
    start[0] *= 0.9
    start[-2] += 300
    start[-1] += 0.02
    widths = abs(truth) * 0.5
    widths[-2] = 2000
    fitted, _, _ = fit_phase(
        frequency,
        observed,
        start,
        sigma=sigma,
        radius=radius,
        signal=signal,
        bounds=(truth - widths, truth + widths),
    )
    np.testing.assert_allclose(fitted, truth, rtol=2e-6)
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_phase(frequency, observed, start, radius=radius, signal=signal, maxfev=1)


@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
def test_invalid_budget(limit):
    with pytest.raises(ValueError, match="positive integer"):
        fit_phase([1, 2, 3], [0, 0, 0], [1000, 2, 0], maxfev=limit)


def test_invalid_samples_and_weights():
    with pytest.raises(ValueError, match="Too few"):
        fit_phase([1, 2], [0, 0], [1000, 2, 0])
    with pytest.raises(ValueError, match="finite"):
        fit_phase([1, 2, 3], [0, np.nan, 0], [1000, 2, 0])
    with pytest.raises(ValueError, match="positive standard deviations"):
        fit_phase([1, 2, 3], [0, 0, 0], [1000, 2, 0], sigma=[1, 0, 1])


def test_optional_retry_recovers_lower_cost_basin():
    with np.load(Path(__file__).parent / "data/channel586-phase.npz") as data:
        kwargs = {
            "sigma": data["sigma"],
            "radius": data["r"].item(),
            "signal": data["z"],
        }
        primary, _, primary_info = fit_phase(data["f"], data["y"], data["p0"], **kwargs)
        retried, _, retry_info = fit_phase(
            data["f"], data["y"], data["p0"], retry_a_bounds=(-0.2, 1.5), **kwargs
        )
        radius = kwargs["radius"]
    primary_a = primary[1] * (2 * radius) ** 2 * primary[0] / primary[2]
    retried_a = retried[1] * (2 * radius) ** 2 * retried[0] / retried[2]
    assert primary_a > 3
    assert 0.12 < retried_a < 0.14
    assert retry_info["retried"] and not primary_info["retried"]
    assert retry_info["cost"] < 0.1 * primary_info["cost"]
    assert primary_info["nfev"] < retry_info["nfev"] < 400


def test_channel596_is_repeatable_and_retry_is_explicit():
    from tolteca_kids.drive_fit.fit import find_best_drive, fit_channel

    with np.load(Path(__file__).parent / "data/channel596.npz") as source:
        data = dict(source)
    signal = data["I"] + 1j * data["Q"]
    for retry, expected in [(False, 5.4932), (True, 6.3408)]:
        selected = []
        for multiplier in (1, 1 + 1e-15, 1 - 1e-14, 1 + 1e-12):
            results, errors = fit_channel(
                data["frequency"],
                signal * multiplier,
                tone_frequency=data["tone_frequency"].item(),
                tau=-36.2,
                window_Qr=12000,
                phase_fit_retry=retry,
            )
            assert not any(errors)
            chosen = find_best_drive(
                data["power"],
                [r["bif"] for r in results],
                [r["Q"] for r in results],
                np.zeros(len(results)),
            )
            assert chosen[3] == 0
            selected.append(-chosen[0])
        assert np.ptp(selected) < 1e-6
        np.testing.assert_allclose(selected, expected, atol=1e-4, rtol=0)


def test_failed_analytic_reference_propagates_without_sentinel_parameters():
    from tolteca_kids.drive_fit.fit import fit_channel

    with np.load(Path(__file__).parent / "data/channel596.npz") as source:
        results, errors = fit_channel(
            source["frequency"],
            source["I"] + 1j * source["Q"],
            tone_frequency=source["tone_frequency"].item(),
            tau=-36.2,
            window_Qr=12000,
            phase_fit_maxfev=1,
        )
    assert results == [None] * 7
    assert "did not converge" in errors[0]
    assert all("reference fit failed" in error for error in errors[1:])
