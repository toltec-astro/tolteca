"""Local identifiability and noise diagnostics for simultaneous detector recovery."""

from enum import IntFlag

import numpy as np


class SampleFlag(IntFlag):
    """Sample validity bits; condition_high is a warning, not failure.

    The other nonzero bits indicate an invalid inverse result. Zero means no
    numerical warning, not an accepted physical calibration.
    Native and Python recovery preserve the same bit values in uint16 arrays.
    iteration_limit
    and stalled accompany recovery_failed. outside_domain identifies a finite
    state outside positive physical loss/frequency or supplied frequency limits.
    """

    valid = 0
    nonfinite_input = 1
    singular_response = 2
    nonfinite_output = 4
    invalid_model = 8
    recovery_failed = 16
    condition_high = 32
    residual_high = 64
    ambiguous_solution = 128
    outside_domain = 256
    iteration_limit = 512
    stalled = 1024
    unstable_response = 2048


def recovery_diagnostics(
    jacobian, iq_covariance, *, rank_rtol=1e-8, condition_max=None
):
    """Measure detector-state observability at the actual probes.

    Parameters
    ----------
    jacobian : array_like
        Real response derivatives, shape ``(2*n_channels, 2*n_resonators)``.
        Rows are all I then all Q. Columns are paired detector-state coordinates;
        scale them explicitly before calling. Native joint recovery uses
        ``dr/r_reference, dfr/(2*r_reference*fr_reference)`` for each resonator.
    iq_covariance : array_like
        Positive definite, symmetric IQ covariance in the same signal units and
        row order as the Jacobian. This function does not estimate it from a sweep.
    rank_rtol : float, optional
        Relative singular-value cutoff in (0, 1), default 1e-8.
    condition_max : float or None, optional
        Condition warning threshold greater than one. None disables the warning;
        numerical rank failure is always reported. No scientific cut is assumed.

    Returns
    -------
    dict
        ``rank``, ``condition``, ``singular_values``, ``flag``, ``covariance`` in
        supplied state coordinates, and ``noise_amplification`` per state.
        Amplification compares joint recovery with the same probes/noise while
        all other detectors are known. Rank-deficient covariance/amplification
        are NaN, not a pseudoinverse with falsely small uncertainty.
        Gain/coupling calibration uncertainty is excluded and recorded explicitly.

    Raises
    ------
    ValueError
        Shapes, finite values, covariance or cutoffs are invalid.
    """
    jac = np.asarray(jacobian, float)
    noise = np.asarray(iq_covariance, float)
    matrix_ndim = 2
    if (
        jac.ndim != matrix_ndim
        or not all(jac.shape)
        or any(n % 2 for n in jac.shape)
        or not np.isfinite(jac).all()
        or noise.shape != (jac.shape[0], jac.shape[0])
        or not np.isfinite(noise).all()
        or np.linalg.norm(noise - noise.T) > 1e-12 * np.linalg.norm(noise)
        or not 0 < rank_rtol < 1
        or (
            condition_max is not None
            and (not np.isfinite(condition_max) or condition_max <= 1)
        )
    ):
        msg = (
            "Require finite paired response axes, symmetric IQ covariance "
            "and valid cutoffs"
        )
        raise ValueError(msg)
    try:
        whitened = np.linalg.solve(np.linalg.cholesky(noise), jac)
    except np.linalg.LinAlgError as exc:
        msg = "IQ covariance must be positive definite"
        raise ValueError(msg) from exc
    _, singular, vt = np.linalg.svd(whitened, full_matrices=False)
    rank = int(np.sum(singular > rank_rtol * singular[0]))
    full_rank = rank == jac.shape[1]
    condition = float(singular[0] / singular[-1]) if full_rank else np.inf
    flag = SampleFlag.valid if full_rank else SampleFlag.singular_response
    covariance = np.full((jac.shape[1], jac.shape[1]), np.nan)
    amplification = np.full(jac.shape[1], np.nan)
    if full_rank:
        inverse = vt.T / singular
        covariance = inverse @ inverse.T
        for j in range(0, jac.shape[1], 2):
            _, ss, vv = np.linalg.svd(whitened[:, j : j + 2], full_matrices=False)
            local_inverse = vv.T / ss
            local_covariance = local_inverse @ local_inverse.T
            amplification[j : j + 2] = np.sqrt(
                np.diag(covariance)[j : j + 2] / np.diag(local_covariance)
            )
        if condition_max is not None and condition > condition_max:
            flag |= SampleFlag.condition_high
    return {
        "rank": rank,
        "condition": condition,
        "singular_values": singular,
        "flag": int(flag),
        "covariance": covariance,
        "noise_amplification": amplification,
        "rank_rtol": rank_rtol,
        "condition_max": condition_max,
        "calibration_uncertainty": "not_propagated",
        "acceptance": "not_assessed",
    }
