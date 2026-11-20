"""Adapt measured TolTEC sweep blocks to the generic fixed-group fitter."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tolteca_kidsproc.analysis import KidsModelAnalysis
from tolteca_kidsproc.segmentation import group_intervals

from ._model_history import bind_analysis
from ._model_peak_links import make_peak_links
from .kids_find import KidsFind
from .model_fit_quality import model_fit_quality
from .power_setup import power_setup_diagnostics
from .sweep import ReducedSweepView
from .sweep_check import SweepCheck

_SWEEP_NDIM = 2
_MIN_SAMPLES = 3


def _sweep_arrays(sweep):
    """Read aligned I/Q and frequency while retaining acquired channel labels."""
    frequency, real, imag = sweep.frequency, sweep.I, sweep.Q
    if frequency is None or real is None or imag is None:
        msg = "Model fitting requires reduced I/Q and frequency arrays"
        raise ValueError(msg)
    return (
        frequency.u.to("Hz").values,
        real.values + 1j * imag.values,
        real.coords[real.dims[0]].values,
    )


def _plan_inputs(sweep, finding, fit_n_fwhm, mask):
    """Validate inputs independently of group iteration."""
    if not np.isfinite(fit_n_fwhm) or fit_n_fwhm <= 0:
        msg = "fit_n_fwhm must be finite and positive"
        raise ValueError(msg)
    frequency, signal, channels = _sweep_arrays(sweep)
    if frequency.ndim != _SWEEP_NDIM or frequency.shape != signal.shape:
        msg = "Plan construction requires channel-by-sample sweep arrays"
        raise ValueError(msg)
    usable = np.isfinite(frequency) & (frequency > 0) & np.isfinite(signal)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != frequency.shape:
            msg = "Plan mask must be boolean with the sweep shape"
            raise ValueError(msg)
        usable &= mask
    detected = finding.detected
    fr = detected["f"].to_value("Hz")
    qr = np.asarray(detected["Qr"])
    res_ids, groups = np.asarray(detected["idx"]), np.asarray(detected["group"])
    if len(set(res_ids)) != len(res_ids) or not np.all(
        np.isfinite(fr) & (fr > 0) & np.isfinite(qr) & (qr > 0)
    ):
        msg = "Finding candidates require unique IDs and positive fr/Qr"
        raise ValueError(msg)
    return frequency, usable, channels, fr, qr, res_ids, groups


def make_plan(sweep, finding, *, fit_n_fwhm, mask, frequency_bounds=None):
    """Close finding groups over support before selecting recorded channel samples."""
    frequency, usable, channels, fr, qr, res_ids, initial_groups = _plan_inputs(
        sweep, finding, fit_n_fwhm, mask
    )
    channel_min = np.min(np.where(usable, frequency, np.inf), axis=1)
    channel_max = np.max(np.where(usable, frequency, -np.inf), axis=1)
    half_widths = fit_n_fwhm * fr / qr
    bounds = (
        np.column_stack((fr - half_widths, fr + half_widths))
        if frequency_bounds is None
        else np.asarray(frequency_bounds, dtype=float)
    )
    groups = group_intervals(bounds, initial_groups)
    if np.any((fr < bounds[:, 0]) | (fr > bounds[:, 1])):
        msg = "Frequency support must contain each candidate's initial position"
        raise ValueError(msg)
    rows = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        candidates = np.flatnonzero(
            (channel_min <= np.max(bounds[members, 1]))
            & (channel_max >= np.min(bounds[members, 0]))
        )
        for ci in candidates:
            chan = channels[ci]
            segments = []
            union = np.zeros(frequency.shape[1], dtype=bool)
            for j in members:
                inside = (
                    (frequency[ci] >= bounds[j, 0])
                    & (frequency[ci] <= bounds[j, 1])
                    & usable[ci]
                )
                # A component with no usable samples in its own window does not
                # establish a measured channel association.
                if not np.any(inside & usable[ci]):
                    continue
                union |= inside
                edges = np.diff(np.r_[False, inside, False].astype(int))
                for start, stop in zip(
                    np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
                ):
                    segments.append(
                        {
                            "group_id": str(group),
                            "chan": chan,
                            "resonator_id": str(res_ids[j]),
                            "sample_start": start,
                            "sample_stop": stop,
                            "fr_initial": fr[j],
                            "r_initial": 1 / (2 * qr[j]),
                            "finding_group_id": str(initial_groups[j]),
                            "frequency_min": bounds[j, 0],
                            "frequency_max": bounds[j, 1],
                        }
                    )
            if np.count_nonzero(union & usable[ci]) >= _MIN_SAMPLES:
                rows.extend(segments)
    columns = {
        "group_id": str,
        "chan": channels.dtype,
        "resonator_id": str,
        "sample_start": int,
        "sample_stop": int,
        "fr_initial": float,
        "r_initial": float,
        "finding_group_id": str,
        "frequency_min": float,
        "frequency_max": float,
    }
    plan = xr.Dataset(
        {
            key: ("row", np.asarray([r[key] for r in rows], dtype=dtype))
            for key, dtype in columns.items()
        }
    )
    plan.fr_initial.attrs["units"] = "Hz"
    plan.r_initial.attrs["units"] = ""
    for name in ("frequency_min", "frequency_max"):
        plan[name].attrs["units"] = "Hz"
    plan.attrs.update(
        segmentation="overlapping_support",
        support_source="finding_linewidth" if frequency_bounds is None else "supplied",
        fit_n_fwhm=fit_n_fwhm,
        status="available"
        if rows
        else "no_usable_samples"
        if len(fr)
        else "no_candidates",
    )
    return plan


def fit_python(block, config, plan):
    """Compute first, then return a self-contained result for atomic installation."""
    sweep = ReducedSweepView(block.tree["sweep"])
    frequency, signal, _ = _sweep_arrays(sweep)
    mask = np.isfinite(frequency) & np.isfinite(signal)
    check_path = SweepCheck.result_path(alias=config.check_alias)
    if check_path in block.tree.children:
        check = SweepCheck.view(block.tree, alias=config.check_alias)
        mask &= ~check.mask_spike & ~check.mask_chan_bad[:, None]
    elif config.check_alias is not None:
        msg = "The selected SweepCheck result is absent"
        raise ValueError(msg)
    axis = sweep.chan_axis_table
    mask &= np.asarray(axis["tone_mask"], dtype=bool)[:, None]
    sigma = None
    if config.use_uncertainty:
        if sweep.unc_I is None or sweep.unc_Q is None:
            msg = "Weighted ModelFit requires sweep repeat uncertainties"
            raise ValueError(msg)
        probe = block.probe_axis
        mapper = probe.mapper
        counts = (
            mapper.get_arr(probe.data_source, mapper.schema.sample_end).values
            - mapper.get_arr(probe.data_source, mapper.schema.sample_start).values
        )
        if counts.shape != (signal.shape[-1],) or np.any(counts < 1):
            msg = "Sweep uncertainties require aligned positive repeat counts"
            raise ValueError(msg)
        sigma = np.stack((sweep.unc_I.values, sweep.unc_Q.values), axis=-1)
        sigma = sigma / np.sqrt(counts)[None, :, None]
        mask &= np.all(np.isfinite(sigma) & (sigma > 0), axis=-1)
    if plan is None:
        finding = KidsFind.view(block.tree, alias=config.finding_alias)
        plan = make_plan(sweep, finding, fit_n_fwhm=config.fit_n_fwhm, mask=mask)
        origin = "kids_find"
    else:
        origin = "supplied"
    result = xr.DataTree()
    analysis, history = bind_analysis(block, config.analysis or KidsModelAnalysis())
    # Preserve measured IQ units and phase. No normalized or interpolated
    # quality-check samples are used as independent measurements.
    payload = sweep.data_source.assign_coords(frequency=sweep.frequency)
    analysis(
        payload.kids.sweep,
        plan,
        mask=mask,
        sigma=sigma,
        probe_frequency=axis["f_chan"].to_value("Hz"),
        output=result,
    )
    result.attrs["sweep_history"] = history
    result["plan"] = xr.DataTree(dataset=plan.copy(deep=True))
    if origin == "kids_find":
        result["peak_links"] = make_peak_links(finding, result.kids.model_fit.fit)
    power = None
    if config.power_setup is not None:
        power = power_setup_diagnostics(
            axis["f_chan"].to_value("Hz"),
            axis["f_tone"].to_value("Hz"),
            axis["tone_amp"],
            axis["tone_phase"],
            axis["tone_mask"],
            meta=block.meta,
            config=config.power_setup,
            channels=axis["chan"],
        )
        power.attrs["purpose"] = "acquired_sweep"
        result["power"] = power
    result["quality"] = model_fit_quality(result.kids.model_fit, config.quality, power)
    result.attrs.update({"plan_source": origin, "config": config.model_dump_json()})
    return result
