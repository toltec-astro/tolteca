"""Conditional loss/detuning regression for independently measured sweeps."""

import numpy as np
from scipy.optimize import least_squares


def fit_loss_response(x, r, covariance):
    """Fit ``r = intercept + slope * (x - center)`` with errors in both axes.

    Parameters
    ----------
    x, r : array_like
        Paired dimensionless detuning and loaded loss from independent sweeps.
    covariance : array_like, shape (n, 2, 2)
        Conditional covariance in the same coordinates, including their cross
        term. Each matrix must be positive definite. Cross-acquisition errors
        are assumed independent; model inadequacy is not included.

    Returns
    -------
    dict
        Slope ``R_r/R_x``, its conditional standard error, intercept, center,
        chi-square and residual degrees of freedom. The conventional frequency
        to dissipation ratio beta is the reciprocal slope, not the slope itself.
    """
    minimum_samples, parameter_count = 4, 2
    x, r = np.asarray(x, float), np.asarray(r, float)
    cov = np.asarray(covariance, float)
    if (
        x.ndim != 1
        or x.shape != r.shape
        or len(x) < minimum_samples
        or cov.shape != (len(x), 2, 2)
        or not np.isfinite([x, r]).all()
        or not np.isfinite(cov).all()
        or not np.allclose(cov, cov.swapaxes(1, 2), rtol=1e-10, atol=0)
        or np.any(np.linalg.eigvalsh(cov) <= 0)
        or np.ptp(x) <= 0
    ):
        msg = "Response needs finite varying pairs and positive covariance"
        raise ValueError(msg)
    # Numerical scaling only; input/output coordinates remain dimensionless.
    center = float(x.mean())
    scale = float(np.ptp(x))
    design = np.column_stack([np.ones(len(x)), (x - center) / scale])
    y = r / scale
    c = cov / scale**2
    initial = np.linalg.lstsq(design, y, rcond=None)[0]

    def residual(parameters):
        slope = parameters[1]
        variance = c[:, 1, 1] + slope**2 * c[:, 0, 0] - 2 * slope * c[:, 0, 1]
        return (y - design @ parameters) / np.sqrt(variance)

    fit = least_squares(
        residual, initial, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000
    )
    if not fit.success or np.linalg.matrix_rank(fit.jac) < parameter_count:
        msg = "Response fit did not converge with full rank"
        raise ValueError(msg)
    uncertainty = np.linalg.inv(fit.jac.T @ fit.jac)
    return {
        "center": center,
        "intercept": float(fit.x[0] * scale),
        "slope": float(fit.x[1]),
        "slope_error": float(np.sqrt(uncertainty[1, 1])),
        "chi2": float(np.dot(fit.fun, fit.fun)),
        "dof": len(x) - 2,
    }
