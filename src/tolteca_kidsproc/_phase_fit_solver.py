"""Centered phase models with analytic derivatives and bounded fit work.

The physical model is Sam Walker's phase model in linewidth and circle-radius
units. MINPACK still performs the unbounded Levenberg-Marquardt solve. Keeping
the shared model/Jacobian intermediates avoids paying for them twice.
"""

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, least_squares, leastsq


class _PhaseModel:
    """A weighted phase objective in dimensionless fit coordinates."""

    def __init__(self, frequency, phase, initial, sigma, radius, signal):  # noqa: PLR0913, PLR0917 -- Physical model inputs.
        self.frequency = np.asarray(frequency, dtype=float)
        self.phase = np.asarray(phase, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if (
            self.frequency.ndim != 1
            or self.phase.shape != self.frequency.shape
            or initial.shape not in {(3,), (4,)}
            or not np.isfinite(self.frequency).all()
            or not np.isfinite(self.phase).all()
            or not np.isfinite(initial).all()
        ):
            msg = "Phase fit requires finite, aligned vectors"
            raise ValueError(msg)
        if len(self.phase) < len(initial):
            msg = "Too few phase samples for the fit parameters"
            raise ValueError(msg)
        self.nonlinear = len(initial) == 4  # noqa: PLR2004 -- Q, alpha, f0, phi.
        self.q_scale = max(abs(initial[0]), 1.0)
        center = float(np.median(self.frequency))
        if center <= 0:
            msg = "Phase fit requires positive RF frequencies"
            raise ValueError(msg)
        width = center / self.q_scale
        self.x = (self.frequency - center) / width
        self.frequency_ratio = 1.0 + self.x / self.q_scale
        if self.nonlinear:
            if radius is None or not np.isfinite(radius) or radius <= 0:
                msg = "Nonlinear phase fit requires a positive radius"
                raise ValueError(msg)
            signal = np.asarray(signal)
            if signal.shape != self.x.shape or not np.isfinite(signal).all():
                msg = "Nonlinear phase samples must align with frequencies"
                raise ValueError(msg)
            self.signal = signal / (2.0 * radius)
            self.scales = np.array([self.q_scale, width / (2 * radius) ** 2, width, 1])
            self.offsets = np.array([0, 0, center, 0])
        else:
            self.scales = np.array([self.q_scale, width, 1])
            self.offsets = np.array([0, center, 0])
        self.initial = (initial - self.offsets) / self.scales
        sigma = np.ones_like(self.phase) if sigma is None else np.asarray(sigma)
        if sigma.shape not in {(), self.phase.shape} or not (
            np.isfinite(sigma).all() and (sigma > 0).all()
        ):
            msg = "Phase weights must be finite positive standard deviations"
            raise ValueError(msg)
        self.weight = np.broadcast_to(1.0 / sigma, self.phase.shape)
        self._parameters = None

    def _evaluate(self, parameters):
        key = tuple(parameters)
        if key == self._parameters:
            return
        self._parameters = key
        q, *_, phi = parameters
        if self.nonlinear:
            _, b, d, _ = parameters
            self.rotation = 0.5 * np.exp(1j * phi)
            self.amplitude = self.signal + self.rotation
            self.power = np.abs(self.amplitude) ** 2
            h = d - b * self.power
        else:
            h = parameters[1]
        self.denominator = 1.0 + h / self.q_scale
        self.detuning = (self.x - h) / self.denominator
        self.argument = -2.0 * q * self.detuning
        self.prediction = phi + 2.0 * np.arctan(self.argument)

    def residual(self, parameters):
        """Return phase residuals divided by the supplied standard deviations."""
        self._evaluate(parameters)
        return (self.prediction - self.phase) * self.weight

    def jacobian(self, parameters):
        """Return the analytic derivative of the weighted residuals."""
        self._evaluate(parameters)
        factor = 2.0 * self.weight / (1.0 + self.argument**2)
        dh = 2.0 * parameters[0] * self.frequency_ratio / self.denominator**2
        jac = np.empty((len(self.x), len(parameters)))
        jac[:, 0] = -2.0 * self.detuning * factor
        if self.nonlinear:
            power_derivative = 2.0 * np.real(
                np.conj(self.amplitude) * 1j * self.rotation
            )
            jac[:, 1] = -self.power * dh * factor
            jac[:, 2] = dh * factor
            jac[:, 3] = self.weight - parameters[1] * power_derivative * dh * factor
        else:
            jac[:, 1] = dh * factor
            jac[:, 2] = self.weight
        return jac


def _physical_result(model, result, cost, nfev, retry):
    """Transform parameters and relative-sigma covariance to physical units."""
    parameters, covariance = result[:2]
    dof = len(model.phase) - len(parameters)
    if covariance is None or not np.isfinite(covariance).all() or dof <= 0:
        covariance = np.full((len(parameters), len(parameters)), np.inf)
        warnings.warn(
            "Covariance of the phase parameters could not be estimated",
            OptimizeWarning,
            stacklevel=3,
        )
    else:
        covariance = covariance * (cost / dof)
    covariance = covariance * np.outer(model.scales, model.scales)
    return (
        parameters * model.scales + model.offsets,
        covariance,
        {"nfev": nfev, "status": result[4], "cost": cost, "retried": retry},
    )


def fit_phase(  # noqa: PLR0913 -- Explicit physical inputs and solver controls.
    frequency,
    phase,
    initial,
    *,
    sigma=None,
    radius=None,
    signal=None,
    maxfev=200,
    bounds=None,
    retry_a_bounds=None,
):
    """Fit linear or nonlinear phase, returning physical parameters and covariance.

    ``maxfev`` limits each solve. When ``retry_a_bounds`` is supplied, a failed
    nonlinear solve or an out-of-range nonlinearity permits one extra solve:
    zero initial nonlinearity and half the reference Q (a broader resonance).
    Only a converged candidate with a lower weighted residual sum can replace
    the first result. This optional search can change scientific selections.

    The covariance is transformed back to Q, alpha, frequency in Hz and phase
    (omitting alpha for a linear fit), with curve_fit's relative-sigma scaling.
    The third return value records status, total evaluations, cost and retry use.
    """
    if (
        isinstance(maxfev, bool)
        or not isinstance(maxfev, (int, np.integer))
        or maxfev < 1
    ):
        msg = "maxfev must be a positive integer"
        raise ValueError(msg)
    model = _PhaseModel(frequency, phase, initial, sigma, radius, signal)
    scaled_bounds = (
        None
        if bounds is None
        else tuple((np.asarray(b) - model.offsets) / model.scales for b in bounds)
    )

    def solve(start):
        if scaled_bounds is None:
            return leastsq(
                model.residual,
                start,
                Dfun=model.jacobian,
                full_output=True,
                maxfev=maxfev,
            )
        # Preserve the optional bounded interface; the telescope path uses LM.
        solved = least_squares(
            model.residual,
            start,
            jac=model.jacobian,
            bounds=scaled_bounds,
            max_nfev=maxfev,
        )
        # Same rank truncation as curve_fit for its bounded least-squares path.
        _, singular, vectors = np.linalg.svd(solved.jac, full_matrices=False)
        keep = singular > np.finfo(float).eps * max(solved.jac.shape) * singular[0]
        covariance = (vectors[keep].T / singular[keep] ** 2) @ vectors[keep]
        return (
            solved.x,
            covariance,
            {"nfev": solved.nfev, "fvec": solved.fun},
            solved.message,
            solved.status if solved.success else 5,
        )

    result = solve(model.initial)
    nfev = result[2]["nfev"]
    cost = float(np.dot(result[2]["fvec"], result[2]["fvec"]))
    success = (
        result[4] in (1, 2, 3, 4) and np.isfinite(result[0]).all() and np.isfinite(cost)
    )
    retry = False
    if model.nonlinear and retry_a_bounds is not None:
        low, high = retry_a_bounds
        if not np.isfinite([low, high]).all() or low >= high:
            msg = "Retry nonlinearity bounds must be finite and increasing"
            raise ValueError(msg)
        q, b, d, _ = result[0]
        nonlinearity = q * b / (1 + d / model.q_scale)
        retry = not success or not low <= nonlinearity <= high
    if retry:
        alternate = model.initial.copy()
        alternate[0] *= 0.5
        alternate[1] = 0.0
        if scaled_bounds is not None:
            alternate = np.clip(alternate, *scaled_bounds)
        other = solve(alternate)
        nfev += other[2]["nfev"]
        other_cost = float(np.dot(other[2]["fvec"], other[2]["fvec"]))
        # Avoid selecting numerically indistinguishable duplicates of one basin.
        if (
            other[4] in (1, 2, 3, 4)
            and np.isfinite(other[0]).all()
            and np.isfinite(other_cost)
            and (not success or other_cost < cost * (1.0 - 1e-8))
        ):
            result, cost, success = other, other_cost, True
    if not success or not np.isfinite(result[0]).all() or not np.isfinite(cost):
        msg = f"Phase fit did not converge after {nfev} evaluations: {result[3]}"
        raise RuntimeError(msg)
    return _physical_result(model, result, cost, nfev, retry)
