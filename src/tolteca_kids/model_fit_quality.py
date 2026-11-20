"""Saved fit diagnostics and explicit, configurable quality flags."""

import json
from enum import IntFlag

import numpy as np
import xarray as xr
from pydantic import Field
from tollan.config import FrozenBaseModel

from tolteca_kidsproc.models import resonance_feature_frequency


class ModelFitBitMask(IntFlag):
    """ModelFit quality/1 bits; independent of native kids.fit/1 flag values.

    Zero means no listed warning, not acceptance for timestream inversion.
    Group warnings propagate to every contributing channel. Selection and
    recovery warnings belong to channels; power warnings require a valid match.
    """

    not_converged = 1 << 0
    at_bound = 1 << 1
    rank_deficient = 1 << 2
    condition_high = 1 << 3
    residual_high = 1 << 4
    resonance_outside_window = 1 << 5
    initialization_fallback = 1 << 6
    no_fit = 1 << 7
    selection_required = 1 << 8
    recovery_unresolved = 1 << 9
    probe_outside_window = 1 << 10
    nonlinear_limit = 1 << 11
    power_high = 1 << 12
    power_low = 1 << 13
    power_reference_unmatched = 1 << 14
    zero_drive_outside_window = 1 << 15


class ModelFitQualityConfig(FrozenBaseModel):
    """Optional warning thresholds; None disables that numerical cut.

    Parameters
    ----------
    condition_max : float or None, optional
        Maximum condition in scaled optimizer coordinates, default None.
    residual_fraction_max : float or None, optional
        Maximum sqrt(sum(abs(residual)**2)/sum(abs(IQ)**2)), default None.
        Evaluated on samples actually used in each joint fit.
    power_tolerance_db : float, optional
        Allowed difference from a valid drive-fit reference, default 1 dB.
        This is a diagnostic tolerance, not a calibrated usability boundary.
    """

    condition_max: float | None = Field(default=None, gt=1, allow_inf_nan=False)
    residual_fraction_max: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    power_tolerance_db: float = Field(default=1.0, gt=0, allow_inf_nan=False)


def flag_names(value):
    """Return the names of all set ModelFit bits, or 'none'."""
    return ", ".join(f.name for f in ModelFitBitMask if int(value) & f) or "none"


def model_fit_quality(view, config, power=None):
    """Summarize saved samples and solver outcomes without changing a fit.

    Parameters
    ----------
    view : KidsModelView
        Completed Python analysis, including failed individual groups.
    config : ModelFitQualityConfig
        Explicit warning thresholds.
    power : xarray.Dataset or None, optional
        Channel-aligned power diagnostics from power_setup.

    Returns
    -------
    xarray.Dataset
        Group metrics and channel masks, with bit definitions and thresholds.
        All acquired channels remain present in their original order.
    """
    fit, samples, joint, channels = view.fit, view.samples, view.joint, view.chan_prop
    flags, fractions, redchi, counts, f_centers = [], [], [], [], []
    feature, kind = resonance_feature_frequency(
        fit.fr.values,
        fit.Qr.values,
        fit.a_nl.values,
        nonlinear_branch=fit.attrs.get("nonlinear_branch", "unique"),
    )
    feature = xr.DataArray(feature, dims="row", coords={"row": fit.row})
    for i, group in enumerate(joint.group_id.values):
        rows = fit.isel(row=np.flatnonzero(fit.group_id.values == group))
        ss = samples.isel(
            sample=np.flatnonzero(
                (samples.group_id.values == group) & samples.fit_selected.values
            )
        )
        residual = ss.I_residual.values + 1j * ss.Q_residual.values
        measured = ss.I.values + 1j * ss.Q.values
        denominator = np.sum(abs(measured) ** 2)
        fraction = (
            np.sqrt(np.sum(abs(residual) ** 2) / denominator) if denominator else np.nan
        )
        dof = 2 * len(residual) - int(joint.variable_count[i])
        weighted = np.all(np.isfinite(ss.sigma_I) & (ss.sigma_I > 0)) and np.all(
            np.isfinite(ss.sigma_Q) & (ss.sigma_Q > 0)
        )
        chi = (
            float(
                np.sum(
                    (ss.I_residual / ss.sigma_I) ** 2
                    + (ss.Q_residual / ss.sigma_Q) ** 2
                )
            )
            if weighted
            else np.nan
        )
        first = int(joint.parameter_start[i])
        last = first + int(joint.parameter_count[i])
        optimizer = joint.isel(parameter=slice(first, last))
        nonlinearity = np.array(
            [str(n).endswith("_a_nl") for n in optimizer.optimizer_name.values]
        )
        varying = (
            optimizer.optimizer_vary.values
            if "optimizer_vary" in optimizer
            else np.ones(last - first, bool)
        )
        upper_hit = (
            nonlinearity
            & varying
            & np.isclose(
                optimizer.optimizer_value.values,
                optimizer.optimizer_max.values,
                atol=1e-8,
                rtol=1e-6,
            )
        )
        feature_rows = feature.isel(row=np.flatnonzero(fit.group_id.values == group))
        flag = ModelFitBitMask(0)
        checks = {
            "not_converged": joint.fit_status[i].item() != "optimizer_converged",
            "at_bound": bool(rows.at_bound.any()),
            "rank_deficient": int(joint["rank"][i]) < int(joint.variable_count[i]),
            "condition_high": config.condition_max is not None
            and float(joint.condition[i]) > config.condition_max,
            "residual_high": config.residual_fraction_max is not None
            and fraction > config.residual_fraction_max,
            "resonance_outside_window": any(
                not bool(
                    ((feature_rows >= rows.f_min) & (feature_rows <= rows.f_max))
                    .where(rows.resonator_id == res, other=False)
                    .any()
                )
                for res in np.unique(rows.resonator_id)
            ),
            "zero_drive_outside_window": any(
                not bool(
                    ((rows.fr >= rows.f_min) & (rows.fr <= rows.f_max))
                    .where(rows.resonator_id == res, other=False)
                    .any()
                )
                for res in np.unique(rows.resonator_id)
            ),
            "initialization_fallback": bool(
                np.any(rows.initialization_fallback.values != "")
            ),
            "nonlinear_limit": bool(upper_hit.any()),
        }
        for name, enabled in checks.items():
            if enabled:
                flag |= ModelFitBitMask[name]
        flags.append(int(flag))
        fractions.append(fraction)
        redchi.append(chi / dof if dof > 0 else np.nan)
        counts.append(len(residual))
        f_centers.append(float(rows.fr.median()))
    chan_flags = np.zeros(channels.sizes["chan"], dtype=np.uint32)
    by_group = dict(zip(joint.group_id.values, flags, strict=True))
    for i, chan in enumerate(channels.chan.values):
        groups = set(fit.group_id.values[fit.chan.values == chan])
        for group in groups:
            chan_flags[i] |= by_group[group]
        reason = channels.recovery_reason[i].item()
        for name, enabled in {
            "no_fit": not groups,
            "selection_required": len(groups) > 1,
            "recovery_unresolved": reason == "insufficient_simultaneous_measurements",
            "probe_outside_window": reason == "probe_outside_fit_window",
        }.items():
            if enabled:
                chan_flags[i] |= ModelFitBitMask[name]
    if power is not None and "reference_status" in power:
        if not np.array_equal(power.chan, channels.chan):
            msg = "Power diagnostics must retain the acquired channel order"
            raise ValueError(msg)
        matched = power.reference_status.values == "matched"
        delta = power.power_delta_db.values
        chan_flags[~matched] |= np.uint32(ModelFitBitMask.power_reference_unmatched)
        chan_flags[matched & (delta > config.power_tolerance_db)] |= np.uint32(
            ModelFitBitMask.power_high
        )
        chan_flags[matched & (delta < -config.power_tolerance_db)] |= np.uint32(
            ModelFitBitMask.power_low
        )
    return xr.Dataset(
        {
            "f_feature": ("fit_row", feature.values, {"units": "Hz"}),
            "feature_kind": ("fit_row", kind),
            "group_id": ("group", joint.group_id.values),
            "group_flag": ("group", np.asarray(flags, np.uint32)),
            "residual_fraction": ("group", fractions),
            "reduced_chi2": ("group", redchi),
            "sample_count": ("group", counts),
            "frequency": ("group", f_centers, {"units": "Hz"}),
            "channel_flag": ("chan", chan_flags),
        },
        coords={"chan": channels.chan.values},
        attrs={
            "schema": "model_fit_quality/1",
            "feature_rows": "aligned_with_model_fit_rows",
            "resonance_outside_window": (
                "canonical_response_feature_outside_all_associated_windows"
            ),
            "zero_drive_outside_window": "zero_drive_fr_outside_all_associated_windows",
            "nonlinear_limit": "variable_a_nl_at_its_recorded_optimizer_upper_bound",
            "thresholds": config.model_dump(mode="json"),
            "flag_definitions": json.dumps({f.name: int(f) for f in ModelFitBitMask}),
            "acceptance": "not_assessed",
            "reduced_chi2_assumption": "independent repeat means and IQ quadratures",
        },
    )
