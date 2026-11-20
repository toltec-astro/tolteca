"""Compare repeated fits of identical measured sweeps."""

import numpy as np
import xarray as xr

__all__ = ["compare_kids_model_fits"]


def compare_kids_model_fits(candidates, *, cost_relative_tolerance=0.05):
    """Compare starting values or constraints without selecting a calibration.

    Parameters
    ----------
    candidates : mapping of str to KidsModelView
        Named completed fits with identical samples, masks, uncertainties,
        channel-resonator associations, gain family/origins and sweep history.
        Keep their source files open until this function returns. Bounds and
        fixed/tied parameters may differ and remain in each source fit.
    cost_relative_tolerance : float, default 0.05
        Fraction above the smallest converged training cost included in the
        ``near_best`` comparison. This is a descriptive tolerance, not a
        confidence interval or a calibration acceptance threshold.

    Returns
    -------
    xarray.Dataset
        Candidate-by-group convergence, cost, evaluations and complex residual
        RMS on fitted and excluded valid samples. Candidate-by-row physical
        parameters retain the source fit row order. ``near_best_count`` and
        parameter minima/maxima describe converged candidates within the cost
        tolerance. One candidate alone cannot establish start stability.
        No converged candidate gives count zero, an empty best name, and NaN
        parameter ranges. Detector identity and calibration acceptance are not
        assessed or changed.

    Raises
    ------
    ValueError
        No candidates, an invalid tolerance, or incompatible measured inputs
        or model associations were supplied.
    """
    if (
        not candidates
        or not np.isfinite(cost_relative_tolerance)
        or cost_relative_tolerance < 0
    ):
        msg = "Provide candidates and a finite nonnegative cost tolerance"
        raise ValueError(msg)
    views = list(candidates.values())
    reference = views[0]
    sample_keys = [
        "group_id",
        "chan",
        "sample_index",
        "frequency",
        "I",
        "Q",
        "fit_selected",
        "data_valid",
        "sigma_I",
        "sigma_Q",
    ]
    row_keys = [
        "group_id",
        "chan",
        "resonator_id",
        "sample_start",
        "sample_stop",
        "gain_family",
        "f_g",
    ]
    for view in views:
        view.validate()
        compatible = (
            view.samples[sample_keys].equals(reference.samples[sample_keys])
            and view.fit[row_keys].equals(reference.fit[row_keys])
            and view.joint.group_id.equals(reference.joint.group_id)
            and view.fit.attrs.get("nonlinear_branch", "unique")
            == reference.fit.attrs.get("nonlinear_branch", "unique")
        )
        if not compatible:
            msg = (
                "Comparison requires identical measurements, masks, weights "
                "and model associations"
            )
            raise ValueError(msg)
    groups = reference.joint.group_id.values
    costs, fitted_rms, excluded_rms = [], [], []
    for view in views:
        cost, fit_rms, test_rms = [], [], []
        for group in groups:
            rows = view.fit.group_id.values == group
            cost.append(float(view.fit.cost.values[rows][0]))
            samples = view.samples
            valid = (samples.group_id.values == group) & samples.data_valid.values
            residual = samples.I_residual.values + 1j * samples.Q_residual.values
            for selected, output in [(True, fit_rms), (False, test_rms)]:
                values = residual[valid & (samples.fit_selected.values == selected)]
                output.append(
                    float(np.sqrt(np.mean(abs(values) ** 2))) if len(values) else np.nan
                )
        costs.append(cost)
        fitted_rms.append(fit_rms)
        excluded_rms.append(test_rms)
    cost = np.asarray(costs)
    converged = np.stack(
        [v.joint.fit_status.values == "optimizer_converged" for v in views]
    )
    eligible = converged & np.isfinite(cost)
    best = np.min(np.where(eligible, cost, np.inf), axis=0)
    near = eligible & (cost <= best * (1 + cost_relative_tolerance))
    names = np.asarray(list(candidates), dtype=str)
    result = xr.Dataset(
        {
            "group_id": ("group", groups),
            "converged": (("candidate", "group"), converged),
            "cost": (("candidate", "group"), cost),
            "fit_rms": (("candidate", "group"), fitted_rms),
            "excluded_rms": (("candidate", "group"), excluded_rms),
            "nfev": (
                ("candidate", "group"),
                np.stack([v.joint.nfev.values for v in views]),
            ),
            "near_best": (("candidate", "group"), near),
            "near_best_count": ("group", near.sum(axis=0)),
            "best_candidate": (
                "group",
                np.where(
                    np.isfinite(best),
                    names[np.argmin(np.where(eligible, cost, np.inf), axis=0)],
                    "",
                ),
            ),
        },
        coords={"candidate": names},
        attrs={
            "cost_relative_tolerance": cost_relative_tolerance,
            "cost_tolerance_meaning": "descriptive_not_statistical_confidence",
            "acceptance": "not_assessed",
            "identity_action": "none",
            "nonlinear_branch": reference.fit.attrs.get("nonlinear_branch", "unique"),
            "parameter_ranges": (
                "converged_candidates_within_cost_tolerance; "
                "singleton_is_not_stability_evidence"
            ),
        },
    )
    row_group = np.array(
        [np.flatnonzero(groups == g)[0] for g in reference.fit.group_id.values]
    )
    for key in ("group_id", "chan", "resonator_id"):
        result[f"row_{key}"] = ("fit_row", reference.fit[key].values)
    for key in (
        "fr",
        "r",
        "Qr",
        "a_nl",
        "c_re",
        "c_im",
        "H0_re",
        "H0_im",
        "H1_re",
        "H1_im",
        "gain_a",
        "tau",
    ):
        values = np.stack([v.fit[key].values for v in views])
        result[key] = (("candidate", "fit_row"), values, reference.fit[key].attrs)
        included = near[:, row_group] & np.isfinite(values)
        for suffix, fill, reducer in [
            ("min", np.inf, np.min),
            ("max", -np.inf, np.max),
        ]:
            bounds = reducer(np.where(included, values, fill), axis=0)
            result[f"{key}_{suffix}"] = (
                "fit_row",
                np.where(included.any(axis=0), bounds, np.nan),
                reference.fit[key].attrs,
            )
    return result
