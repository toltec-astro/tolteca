"""Stationary softening response with an explicit sweep history."""

import numpy as np
from scipy.optimize import brentq

NONLINEARITY_UNIQUE_MAX = 0.75
NONLINEARITY_CRITICAL = 4 / (3 * np.sqrt(3))


def validate_nonlinearity(a, branch):
    """Validate nonlinearity against the declared stationary response domain."""
    if branch not in ("unique", "increasing", "decreasing"):
        msg = "nonlinear_branch must be unique, increasing or decreasing"
        raise ValueError(msg)
    a = np.asarray(a)
    if np.any(~np.isfinite(a) | (a < 0)):
        msg = "Nonlinearity requires finite nonnegative a_nl"
        raise ValueError(msg)
    if branch == "unique" and np.any(a > NONLINEARITY_UNIQUE_MAX):
        msg = "Nonlinearity requires 0 <= a_nl <= 0.75 for unique response"
        raise ValueError(msg)


def _branch_brackets(y0, a, branch):
    low, high = y0.copy(), y0 + a
    for value in np.unique(a[a > NONLINEARITY_CRITICAL]):
        middle = -1 / (2 * np.sqrt(3))

        def derivative(y, strength=value):
            return 1 + 8 * strength * y / (1 + 4 * y * y) ** 2

        left = brentq(derivative, -(value + 1), middle, xtol=1e-14)
        right = brentq(derivative, middle, 0, xtol=1e-14)
        turn = left if branch == "increasing" else right
        threshold = turn - value / (1 + 4 * turn * turn)
        selected = a == value
        lower = selected & (
            (y0 < threshold) if branch == "increasing" else (y0 <= threshold)
        )
        upper = selected & ~lower
        high = np.where(lower, np.minimum(high, left), high)
        low = np.where(upper, np.maximum(low, right), low)
    return low, high


def nonlinear_detuning(y0, a, *, branch="unique"):
    """Solve y = y0 + a/(1+4*y²) with a bracketed Newton iteration.

    Arrays broadcast. The default unique response requires 0 <= a <= 0.75.
    Increasing/decreasing selects the stable quasistatic response reached from
    below/above resonance, including the jump at the corresponding fold. It is
    a declared history, not a history inferred from the evaluation point order.
    Return y and its implicit derivative denominator. Derivatives are local to
    a stable branch and do not describe motion of the jump between samples.
    """
    y0, a = np.broadcast_arrays(np.asarray(y0, float), np.asarray(a, float))
    validate_nonlinearity(a, branch)
    if not np.any(a):
        return y0.copy(), np.ones_like(y0)
    shape = y0.shape
    if branch != "unique":
        # Use the same ndarray arithmetic for scalar, grouped and paired views.
        y0, a = y0.reshape(-1), a.reshape(-1)
    low, high = _branch_brackets(y0, a, branch)
    y = np.clip(y0 + a / (1 + 4 * y0**2), low, high)
    for _ in range(60):
        q = 1 + 4 * y**2
        residual = y - y0 - a / q
        derivative = 1 + 8 * a * y / q**2
        converged = np.abs(residual) <= 2e-14 * (1 + np.abs(y0))
        if np.all(converged):
            break
        high = np.where(residual > 0, y, high)
        low = np.where(residual < 0, y, low)
        with np.errstate(divide="ignore", invalid="ignore"):
            trial = y - residual / derivative
        inside = (trial >= low) & (trial <= high)
        updated = np.where(inside, trial, (low + high) / 2)
        # Freeze converged directional roots individually. Near a fold, extra
        # iterations caused by unrelated samples can change a root noticeably.
        y = (
            np.where(converged & (a > NONLINEARITY_CRITICAL), y, updated)
            if branch != "unique"
            else updated
        )
    derivative = 1 + 8 * a * y / (1 + 4 * y**2) ** 2
    return y.reshape(shape), derivative.reshape(shape)
