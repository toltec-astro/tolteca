"""Drive selection and sequential phase fitting on numerical sweep arrays."""

from __future__ import annotations

# ruff: noqa: N803 -- Retain physical Qr names.
import numpy as np

from tolteca_kidsproc._phase_fit import ResonanceFitterSingleTone

# Commissioning delays used by the production kid_phase_fit driver, in ns.
CABLE_DELAY_NS = (
    -36.2,
    -34.4,
    -34.8,
    -34.7,
    -34.6,
    -41.3,
    -39.2,
    -32.3,
    -31.0,
    -31.0,
    -31.0,
    -37.4,
    -39.3,
)
A_MIN = -0.2
A_MAX = 1.5


def find_best_drive(power, nonlinearity, Qr, fit_flag, *, a_ref=0.2):
    """Select drive power using the production recipe's first crossing.

    Parameters
    ----------
    power : array-like
        Negative drive attenuation, in dB, one value per fit.
    nonlinearity, Qr : array-like
        Fitted nonlinearity and loaded quality factor.
    fit_flag : array-like
        Fit execution flags; zero means the fit completed. Phase-residual
        quality flags are separate and do not enter this selection.
    a_ref : float, optional
        Desired nonlinearity, default 0.2.

    Returns
    -------
    tuple
        Selected power, nonlinearity, Qr and flag. Flag 0 means an accepted
        crossing; 1 means an endpoint, one valid point, or no valid points.
        No valid points returns NaN for all three scientific values.
    """
    power = np.asarray(power)
    nonlinearity = np.asarray(nonlinearity)
    Qr = np.asarray(Qr)
    fit_flag = np.asarray(fit_flag)
    if not (
        power.ndim == 1
        and power.shape == nonlinearity.shape == Qr.shape == fit_flag.shape
    ):
        msg = "Drive selection requires aligned one-dimensional arrays"
        raise ValueError(msg)
    valid = (fit_flag == 0) & (nonlinearity <= A_MAX) & (nonlinearity >= A_MIN)
    p = power[valid]
    a = nonlinearity[valid]
    q = Qr[valid]
    if len(p) == 0:
        return np.nan, np.nan, np.nan, 1
    if len(p) == 1:
        return p[0], a[0], q[0], 1
    order = np.argsort(p)
    p, a, q = p[order], a[order], q[order]
    best, offset, flag = _first_crossing(p, a - a_ref, tolerance=0.01)
    return best, offset + a_ref, np.interp(best, p, q), flag


def _first_crossing(x, y, *, tolerance):
    """Preserve the v2 endpoint and crossing order, including its tolerance."""
    if y[0] >= 0:
        return x[0], y[0], 1
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        x0, x1 = x[i], x[i + 1]
        if np.isclose(y0, 0, atol=tolerance):
            return x0, y0, 0
        if y0 < 0 and y1 >= 0:
            dx = x1 - x0
            dy = y1 - y0
            return x0 - y0 * dx / dy, 0.0, 0
    return x[-1], y[-1], 1


def fit_channel(  # noqa: PLR0913 -- Scientific and solver settings are explicit.
    frequency,
    signal,
    *,
    tone_frequency,
    tau,
    window_Qr,
    phase_fit_method="analytic",
    phase_fit_maxfev=200,
    phase_fit_retry=False,
):
    """Fit a channel in ascending power order using the first sweep as reference."""
    results = []
    errors = []
    reference = None
    solver = {
        "phase_fit_method": phase_fit_method,
        "phase_fit_maxfev": phase_fit_maxfev,
        "phase_fit_retry_a_bounds": (A_MIN, A_MAX) if phase_fit_retry else None,
    }
    for i, (f, z) in enumerate(zip(frequency, signal, strict=True)):
        if i > 0 and reference is None:
            results.append(None)
            errors.append("The lowest-power reference fit failed")
            continue
        try:
            if i == 0:
                reference = ResonanceFitterSingleTone(
                    f,
                    z,
                    tau,
                    1,
                    tone_frequency,
                    tone_frequency / window_Qr,
                    weight_type="lorentz",
                    **solver,
                ).result
            correction = signal[0, -1] / z[-1]
            fitted = ResonanceFitterSingleTone(
                f,
                z * correction,
                tau,
                1,
                tone_frequency,
                tone_frequency / window_Qr,
                pherr_threshold_num=10,
                pherr_threshold=0.25,
                result0=reference,
                weight_type="lorentz",
                **solver,
            )
        except (
            TypeError,  # MINPACK reports too few phase samples as TypeError.
            ValueError,
            RuntimeError,
            IndexError,
            FloatingPointError,
            ZeroDivisionError,
            np.linalg.LinAlgError,
        ) as exc:
            results.append(None)
            errors.append(f"{type(exc).__name__}: {exc}")
        else:
            results.append(fitted.result)
            errors.append("")
    return results, errors
